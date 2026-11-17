"""Regression checks for scientific monitoring and unchanged optimization."""

import copy
import csv
import gzip
import json
import random
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset
from tensorboard.backend.event_processing.event_accumulator import (
    EventAccumulator,
)

from criterions.pretrain_masked_criterion import PretrainMaskedCriterion
from preprocessors import STFTPreprocessor
from runner import Runner, restore_cuda_rng
from tasks.spec_pretrain import SpecPretrain
from util.experiments.lifecycle import FitRun
from util.experiments.monitoring import (
    Accumulator,
    TrainingMonitor,
    batch_statistics,
    embedding_rank,
    summarize,
)
from util.mask_utils import fixed_mask_inputs, mask_inputs, variable_mask

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("device", ["cuda", "cuda:0"])
def test_runner_pins_device_before_memory_limits(tmp_path, device):
    """Per-device allocator limits must not redirect the primary replica."""
    current = [0]
    placements = []
    model = torch.nn.Linear(2, 2)
    model.to = lambda value: placements.append(value) or model
    cfg = OmegaConf.create({
        "device": device, "multi_gpu": True,
        "gpu_memory_fraction": 0.2, "artifact_dir": str(tmp_path),
        "output_tb": False, "quiet": True, "optim": "SGD", "lr": 0.001,
        "total_steps": 2,
        "scheduler": {"name": "ramp_up", "warmup": 0.5, "total_steps": 2},
    })
    task = SimpleNamespace(cfg=SimpleNamespace(name="device_smoke"))
    with (
        patch("torch.cuda.current_device", side_effect=lambda: current[0]),
        patch("torch.cuda.device_count", return_value=4),
        patch(
            "torch.cuda.set_per_process_memory_fraction",
            side_effect=lambda fraction, index: current.__setitem__(0, index),
        ),
        patch("torch.cuda.reset_peak_memory_stats"),
        patch("torch.nn.DataParallel", side_effect=lambda value: value),
    ):
        runner = Runner(cfg, task, model, None)
    try:
        assert current[0] == 3
        assert runner.device == torch.device("cuda:0")
        assert placements == [torch.device("cuda:0")]
    finally:
        runner.progress.close()


@pytest.mark.parametrize("saved,visible", [(1, 4), (4, 1), (4, 4)])
def test_cuda_rng_restoration_across_device_counts(saved, visible):
    """Changed GPU counts keep available streams without invalid devices."""
    states = [torch.tensor([i], dtype=torch.uint8) for i in range(saved)]
    with (
        patch("torch.cuda.device_count", return_value=visible),
        patch("torch.cuda.manual_seed_all") as seed,
        patch("torch.cuda.set_rng_state_all") as restore,
    ):
        if saved != visible:
            with pytest.warns(RuntimeWarning, match="Stochastic replay"):
                restore_cuda_rng(states)
            seed.assert_called_once_with(torch.initial_seed())
        else:
            restore_cuda_rng(states)
            seed.assert_not_called()
    actual = restore.call_args.args[0]
    assert len(actual) == min(saved, visible)
    for expected, restored in zip(states, actual):
        assert torch.equal(expected, restored)


def test_cuda_rng_restoration_rejects_empty_checkpoint():
    """An empty CUDA state must not silently reset a resumed experiment."""
    with pytest.raises(ValueError, match="random-state list is empty"):
        restore_cuda_rng([])


def test_tensorboard_resume_is_one_run_with_rollback(tmp_path):
    """Resume joins validation history and replaces unsaved training points."""
    shared = tmp_path / "tensorboard"
    first = TrainingMonitor(
        tmp_path / "initial", np.arange(4.0), 2.0, 0.1, 1,
        "recording_zscore", tensorboard_root=shared,
    )
    first.writer.add_scalar("validation/overall/total_loss", 3.0, 1)
    first.writer.add_scalar("train/overall/total_loss", 4.0, 1)
    first.writer.add_scalar("train/overall/total_loss", 9.0, 101)
    first.close()
    resumed = TrainingMonitor(
        tmp_path / "resumed", np.arange(4.0), 2.0, 0.1, 1,
        "recording_zscore", tensorboard_root=shared, purge_step=2,
    )
    resumed.writer.add_scalar("train/overall/total_loss", 2.0, 101)
    resumed.writer.add_scalar("validation/overall/total_loss", 1.0, 1001)
    resumed.close()
    events = EventAccumulator(str(shared))
    events.Reload()
    validation = events.Scalars("validation/overall/total_loss")
    training = events.Scalars("train/overall/total_loss")
    assert [event.step for event in validation] == [1, 1001]
    assert [(event.step, event.value) for event in training] == [
        (1, 4.0), (101, 2.0),
    ]
    parents = {
        path.parent for path in tmp_path.rglob("events.out.tfevents.*")
    }
    assert parents == {shared}


def test_status_groups_progress_by_stage(tmp_path):
    """Keep preparation, training, and evaluation counts distinct."""
    run = object.__new__(FitRun)
    run.fit_id = "status-smoke"
    run.logs = tmp_path
    (tmp_path / "status").mkdir()
    run.status(
        "fit_preparation", completed_sessions=2, total_sessions=3
    )
    run.status(
        "training", completed_steps=10, total_steps=20,
        completed_epochs=0, total_epochs=2,
    )
    run.status("validation", completed_batches=4, total_batches=8)
    status = json.loads((tmp_path / "status/status.json").read_text())
    assert status["phase"] == "validation"
    assert status["stages"]["fit_preparation"] == {
        "completed_sessions": 2, "total_sessions": 3,
    }
    assert status["stages"]["training"]["completed_steps"] == 10
    assert status["stages"]["validation"] == {
        "completed_batches": 4, "total_batches": 8,
    }
    with pytest.raises(ValueError, match="metric fields"):
        run.status("training", validation={"loss": 1.0})


def sample_stats(repeats=1):
    """Make nine known groups with one absent content-aware subset."""
    target = torch.tensor([2.0, 0.5, -2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    target = target.repeat(repeats).reshape(repeats, 1, 9)
    error = torch.arange(9).repeat(repeats).reshape_as(target)
    provenance = {
        "selection": torch.tensor(
            [1, 1, 1, 2, 2, 2, 3, 3, 3], dtype=torch.uint8
        ).expand_as(target),
        "action": torch.tensor([1, 2, 0] * 3, dtype=torch.uint8).expand_as(
            target
        ),
    }
    return batch_statistics(
        target + error,
        target,
        torch.ones_like(target),
        torch.zeros((repeats, 1), dtype=torch.bool),
        provenance,
        1.0,
    )


def outs(stats):
    return {
        "monitor_statistics": stats,
        "loss": 3.0,
        "l1_loss": 1.0,
        "content_aware_loss": 2.0,
        "content_l1": 1.0,
        "zero_baseline_l1": 2.0,
        "content_zero_baseline_l1": 2.0,
        "grad_norm": 2.0,
    }


def test_group_denominators_and_pooling():
    stats = sample_stats()
    for i, row in enumerate(stats["sums"]):
        result = summarize(row, 2.0)
        assert result["count"] == 1
        assert result["l1"] == i
        assert result["content_l1"] == (None if i == 1 else i)
        assert "diagnostic_objective" not in result
        assert result["relative_l1"] == pytest.approx(
            i / abs([2.0, 0.5, -2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0][i])
        )
        if i == 1:
            assert result["relative_content_l1"] is None
        else:
            assert result["relative_content_l1"] == result["relative_l1"]
        assert result["target_std"] == result["prediction_std"] == 0
    pooled = Accumulator()
    pooled.add(outs(stats), 1.0)
    pooled.add(outs(sample_stats(3)), 1.0)
    np.testing.assert_allclose(pooled.sums, sample_stats(4)["sums"])
    for a, b in zip(pooled.sums, stats["sums"]):
        assert summarize(a, 2.0)["l1"] == summarize(b, 2.0)["l1"]
    assert summarize(np.zeros(10), 2.0)["l1"] is None
    assert pooled.clipped == 2


def test_padding_exclusion_and_invalid_provenance():
    target = torch.tensor([[[2.0, 3.0], [float("nan"), float("nan")]]])
    provenance = {
        "selection": torch.ones_like(target, dtype=torch.uint8),
        "action": torch.ones_like(target, dtype=torch.uint8),
    }
    stats = batch_statistics(
        target,
        target,
        torch.ones_like(target),
        torch.tensor([[False, True]]),
        provenance,
        1.0,
    )
    assert stats["sums"][:, 0].sum() == 2
    assert stats["coverage"].sum() == 2
    np.testing.assert_array_equal(stats["exposure"], [1, 1])
    provenance["selection"][0, 0, 0] = 0
    with pytest.raises(ValueError, match="provenance differs"):
        batch_statistics(
            target,
            target,
            torch.ones_like(target),
            torch.tensor([[False, True]]),
            provenance,
            1.0,
        )


@pytest.mark.parametrize("kind", ["fixed", "variable"])
def test_optional_provenance_preserves_randomness(kind):
    cfg = OmegaConf.load(ROOT / f"conf/task/{kind}_mask_pretrain.yaml")
    data = torch.arange(186 * 40).reshape(186, 40).float() / 128
    for seed in range(5):
        random.seed(seed)
        before = torch.get_rng_state()
        plain = mask_inputs(data, cfg)
        expected_rng = random.getstate()
        random.seed(seed)
        detailed = mask_inputs(data, cfg, return_provenance=True)
        assert all(torch.equal(a, b) for a, b in zip(plain, detailed[:2]))
        assert random.getstate() == expected_rng
        assert torch.equal(before, torch.get_rng_state())
        assert torch.equal(detailed[1].bool(), detailed[2]["selection"].bool())


@pytest.mark.parametrize(
    "spectral_dice, expected", [(0.0, 1), (0.15, 2), (0.9, 1)]
)
def test_fixed_overwrite_and_noop(spectral_dice, expected):
    cfg = OmegaConf.load(ROOT / "conf/task/fixed_mask_pretrain.yaml")
    data = torch.arange(32).reshape(8, 4).float()
    with patch(
        "util.mask_utils.create_masked_intervals",
        side_effect=[([(1, 3)], range(6)), ([(1, 3)], range(2))],
    ), patch(
        "util.mask_utils.random.random", side_effect=[0.9, spectral_dice]
    ), patch(
        "util.mask_utils.random.randint", return_value=0
    ):
        masked, _, metadata = fixed_mask_inputs(data, cfg, True)
    assert torch.all(metadata["selection"][1:3, 1:3] == 3)
    assert torch.all(metadata["action"][1:3, 1:3] == expected)
    if spectral_dice < 0.1:
        assert torch.all(masked[1:3, 1:3] == 0)
    elif spectral_dice < 0.2:
        assert torch.equal(masked[:, 1:3], data[:, :2])


def test_variable_noop_retains_temporal_action():
    x = torch.zeros(2, 2)
    labels = torch.ones_like(x)
    temporal = {"selection": labels.byte(), "action": labels.byte()}
    spectral = {
        "selection": labels.byte() * 2,
        "action": torch.zeros_like(x).byte(),
    }
    with patch(
        "util.mask_utils.variable_mask_time",
        return_value=(x, labels.clone(), temporal),
    ), patch(
        "util.mask_utils.variable_mask_freq",
        return_value=(x, labels.clone(), spectral),
    ):
        _, _, metadata = variable_mask(x, None, True)
    assert torch.all(metadata["selection"] == 3)
    assert torch.all(metadata["action"] == 1)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(40, 40)
        self.cfg = OmegaConf.create({"input_dim": 40})

    def forward(self, x, padding):
        return self.linear(x), torch.zeros_like(x)

    def save_model_weights(self, states):
        states["model"] = self.state_dict()
        return states


def example():
    x = torch.arange(24 * 40).reshape(24, 40).float() / 128
    cfg = OmegaConf.load(ROOT / "conf/task/fixed_mask_pretrain.yaml")
    random.seed(5)
    masked, labels, metadata = mask_inputs(x, cfg, True)
    return {
        "masked_input": masked,
        "target": x,
        "mask_label": labels,
        "mask_provenance": metadata,
        "length": 24,
        "wav": np.ones(32),
    }


def test_criterion_and_optimizer_are_unchanged():
    from tasks.batch_utils import spec_collator

    batch = spec_collator([example()])
    criterion = PretrainMaskedCriterion()
    criterion.build_criterion(OmegaConf.create({"alpha": 2.0}))
    model = TinyModel()
    twin = copy.deepcopy(model)
    before = torch.get_rng_state()
    loss, logged = criterion(model, batch, "cpu")
    plain = dict(batch)
    plain.pop("mask_provenance")
    reference, _ = criterion(twin, plain, "cpu")
    assert torch.equal(loss, reference)
    assert torch.equal(before, torch.get_rng_state())
    for net, objective in ((model, loss), (twin, reference)):
        optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
        objective.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()
    for a, b in zip(model.parameters(), twin.parameters()):
        assert torch.equal(a.grad, b.grad)
        assert torch.equal(a, b)
    assert "monitor_statistics" in logged


def read_rows(root, name):
    with (root / name).open() as stream:
        return list(csv.DictReader(stream))


@pytest.mark.parametrize(
    "total_steps, log_step, checkpoint_step, epoch_end, "
    "train_monitor_step, expected_steps, expected_valid_steps, "
    "expected_batches, expected_starts",
    [
        (4, 2, 2, False, None, [1, 3, 4], [1, 3, 4],
         [1, 2, 1], [1, 2, 4]),
        (5, 3, 2, False, None, [1, 3, 4, 5], [1, 3, 4, 5],
         [1, 2, 1, 1], [1, 2, 4, 5]),
        (5, 3, 2, True, None, [1, 2, 3, 4, 5],
         [1, 2, 3, 4, 5], [1, 1, 1, 1, 1], [1, 2, 3, 4, 5]),
        (5, 4, 4, False, 2, [1, 3, 5], [1, 5],
         [1, 2, 2], [1, 2, 4]),
    ],
)
def test_training_csv_tensorboard_and_best_checkpoint(
    tmp_path, total_steps, log_step, checkpoint_step, epoch_end,
    train_monitor_step, expected_steps, expected_valid_steps,
    expected_batches, expected_starts,
):
    torch.set_num_threads(1)
    task = SpecPretrain(OmegaConf.create({"name": "spec_pretrain"}))
    extractor = STFTPreprocessor(
        OmegaConf.load(ROOT / "conf/preprocessor/stft.yaml")
    )
    np.testing.assert_allclose(
        extractor.frequency_axis(), np.arange(40) * 5.12
    )
    task.dataset = SimpleNamespace(extracter=extractor)
    examples = [example() for _ in range(4)]
    task.train_set = Subset(examples, [0, 1])
    task.valid_set = Subset(examples, [2])
    task.test_set = Subset(examples, [3])
    criterion = PretrainMaskedCriterion()
    criterion.build_criterion(OmegaConf.create({"alpha": 2.0}))
    cfg = OmegaConf.create(
        {
            "artifact_dir": str(tmp_path / "checkpoints"),
            "device": "cpu",
            "multi_gpu": False,
            "output_tb": True,
            "quiet": True,
            "optim": "SGD",
            "lr": 0.001,
            "total_steps": total_steps,
            "log_step": log_step,
            "checkpoint_step": checkpoint_step,
            "epoch_end_validation": epoch_end,
            "train_batch_size": 1,
            "valid_batch_size": 1,
            "num_workers": 0,
            "shuffle": False,
            "train_monitor_step": train_monitor_step,
            "grad_clip": 1.0,
            "scheduler": {
                "name": "ramp_up",
                "warmup": 0.5,
                "total_steps": total_steps,
            },
        }
    )
    runner = Runner(cfg, task, TinyModel(), criterion)
    status_events = []
    runner.status_callback = lambda phase, **details: status_events.append(
        (phase, details)
    )
    if checkpoint_step != log_step:
        shared_validation = task.valid_set
        task.valid_set = [example()]
        with pytest.raises(ValueError, match="same source dataset"):
            runner.train()
        task.valid_set = shared_validation
    try:
        best = runner.train()
        metrics = runner.test(best)
        assert "monitor_accumulator" not in metrics
        assert any(
            phase == "training"
            and details["completed_steps"] == total_steps
            for phase, details in status_events
        )
        assert any(
            phase == "validation"
            and details["completed_batches"] == 1
            and details["total_batches"] == 1
            for phase, details in status_events
        )
        assert any(
            phase == "test"
            and details["completed_batches"] == 1
            and details["total_batches"] == 1
            for phase, details in status_events
        )
        assert all(
            "validation" not in details
            and "best_validation" not in details
            for _, details in status_events
        )
    finally:
        runner.monitor.close()
        runner.progress.close()
    root = tmp_path / "monitoring"
    with (root / "metadata.json").open() as stream:
        metadata = json.load(stream)
    assert metadata["stft_normalization_mode"] == "recording_zscore"
    rows = read_rows(root, "metrics.csv")
    train = [row for row in rows if row["phase"] == "train"]
    validation = [row for row in rows if row["phase"] == "validation"]
    assert [int(row["step"]) for row in train] == expected_steps
    assert [int(row["step"]) for row in validation] == (
        expected_valid_steps
    )
    assert [int(row["batches"]) for row in train] == expected_batches
    assert all("diagnostic_objective" not in row for row in rows)
    assert all(float(row["relative_l1"]) >= 0 for row in train)
    assert all(float(row["original_relative_l1"]) >= 0 for row in train)
    assert [int(row["interval_start"]) for row in train] == expected_starts
    if epoch_end:
        epochs = read_rows(root, "epoch_metrics.csv")
        assert [int(row["epoch"]) for row in epochs] == [1, 1, 2, 2, 3, 3]
        assert [int(row["step"]) for row in epochs] == [2, 2, 4, 4, 5, 5]
        assert [row["complete"] for row in epochs] == [
            "True", "True", "True", "True", "False", "False"
        ]
        assert [int(row["batches"]) for row in epochs] == [
            2, 1, 2, 1, 1, 1
        ]
        assert len(read_rows(root, "epoch_mask_groups.csv")) == (
            len(epochs) * 15
        )
        assert len(read_rows(root, "epoch_frequency_coverage.csv")) == (
            len(epochs) * 6 * 40
        )
    checkpoint = torch.load(
        tmp_path / "checkpoints/checkpoint_best.pth", weights_only=False
    )
    assert int(rows[-1]["checkpoint_step"]) == checkpoint["completed_steps"]
    events = EventAccumulator(str(root / "tensorboard"))
    events.Reload()
    events_loss = events.Scalars("train/overall/l1")
    np.testing.assert_allclose(
        [row.value for row in events_loss],
        [float(row["l1"]) for row in train],
        rtol=1e-6,
    )
    for phase, phase_rows in (("train", train), ("validation", validation)):
        events_loss = events.Scalars(f"{phase}/overall/total_loss")
        expected = (
            expected_steps if phase == "train" else expected_valid_steps
        )
        assert [point.step for point in events_loss] == expected
        np.testing.assert_allclose(
            [point.value for point in events_loss],
            [float(row["original_loss"]) for row in phase_rows],
            rtol=1e-6,
        )
    scalar_tags = events.Tags()["scalars"]
    assert "train/masks/l1" in events.Tags()["images"]
    assert "train/masks/all/zero/l1" in scalar_tags
    assert "train/masks/temporal_only/all/content_count" in scalar_tags
    assert "train/masks/temporal_only/zero/l1" not in scalar_tags
    assert "train/masks/all/zero/absolute_error_sum" not in scalar_tags
    assert "train/overall/absolute_error_sum" not in scalar_tags
    assert "train/overall/original_loss" not in scalar_tags
    assert "train/overall/total_loss" in scalar_tags
    assert "validation/overall/total_loss" in scalar_tags
    groups = read_rows(root, "mask_groups.csv")
    assert len(groups) == len(rows) * 15
    with gzip.open(root / "validation_frequency.csv.gz", "rt") as stream:
        frequency_rows = list(csv.DictReader(stream))
    assert len(frequency_rows) == len(validation) * 4 * 40
    coverage = read_rows(root, "frequency_coverage.csv")
    final = [
        r
        for r in coverage
        if r["phase"] == "train"
        and r["step"] == str(total_steps)
        and r["scope"] == "cumulative"
        and r["strategy"] == "union"
    ]
    assert sum(int(r["selected_cells"]) for r in final) == sum(
        int(r["count"]) for r in train
    )


def test_incomplete_fit_resumes_from_two_checkpoints(tmp_path):
    """Preserve an incomplete fit only with matched recovery states."""
    import random

    scientific = {"runner": {"total_steps": 5, "shuffle": False}}
    root = tmp_path / "artifacts"
    logs = tmp_path / "logs"
    with FitRun(root, logs, "recovery", scientific) as first:
        checkpoints = first.path / "checkpoints"
        checkpoints.mkdir()
        state = {
            "completed_steps": 3,
            "best_step": 3,
            "best_validation_loss": 1.0,
            "model": {},
            "optim": {},
            "scheduler": {},
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
        }
        torch.save(state, checkpoints / "checkpoint_last.pth")
        best_path = checkpoints / "checkpoint_best.pth"
        torch.save(state, best_path)
        previous = checkpoints / "checkpoint_best.previous.pth"
        previous.write_bytes(best_path.read_bytes())
        torch.save({**state, "completed_steps": 4}, best_path)
    with FitRun(root, logs, "recovery", scientific) as resumed:
        assert resumed.resumed
        assert resumed.path == first.path
        assert not resumed.reused
    assert (first.path / "checkpoints/checkpoint_last.pth").is_file()
    assert not previous.exists()
    assert torch.load(best_path, weights_only=False)["completed_steps"] == 3


def test_completion_integrity_and_legacy_reuse(tmp_path):
    for monitoring in (False, True):
        scientific = {"monitoring_fixture": monitoring}
        with FitRun(
            tmp_path / "fits_root", tmp_path / "logs", "test", scientific
        ) as run:
            checkpoints = run.path / "checkpoints"
            checkpoints.mkdir()
            for name in ("best", "last"):
                (checkpoints / f"checkpoint_{name}.pth").write_bytes(b"test")
            if monitoring:
                monitor = TrainingMonitor(
                    run.path / "monitoring", np.arange(9), 2.0, 1.0,
                    1, "recording_zscore", False,
                )
                accumulator = Accumulator()
                accumulator.add(outs(sample_stats()))
                monitor.record("train", 1, accumulator, learning_rates=[0.01])
                monitor.close()
            run.complete({"loss": 1.0})
        with FitRun(
            tmp_path / "fits_root", tmp_path / "logs", "test", scientific
        ) as reused:
            assert reused.reused
        if monitoring:
            with (run.path / "monitoring/metrics.csv").open("a") as stream:
                stream.write("corrupt\n")
            with pytest.raises(ValueError, match="checksum"):
                with FitRun(
                    tmp_path / "fits_root",
                    tmp_path / "logs",
                    "test",
                    scientific,
                ):
                    pass


def test_frequency_errors_use_each_bin_own_selected_count():
    target = torch.tensor([[[2.0, 4.0, 8.0]]])
    predicted = torch.tensor([[[1.0, 2.0, 4.0]]])
    provenance = {
        "selection": torch.tensor([[[1, 2, 3]]], dtype=torch.uint8),
        "action": torch.ones((1, 1, 3), dtype=torch.uint8),
    }
    stats = batch_statistics(
        predicted, target, torch.ones_like(target),
        torch.zeros((1, 1), dtype=torch.bool), provenance, 1.0,
        include_frequency=True,
    )
    frequency = stats["frequency"]
    np.testing.assert_array_equal(frequency[:, :, 0], np.eye(3))
    np.testing.assert_array_equal(
        frequency[np.arange(3), np.arange(3), 1], [1, 2, 4]
    )
    np.testing.assert_array_equal(
        frequency[np.arange(3), np.arange(3), 4], [2, 4, 8]
    )


def test_validation_rank_and_frequency_from_one_forward():
    from models.masked_tf_model import MaskedTFModel
    from tasks.batch_utils import spec_collator

    cfg = OmegaConf.create({
        "input_dim": 40, "hidden_dim": 40, "nhead": 4,
        "layer_dim_feedforward": 80, "layer_activation": "gelu",
        "encoder_num_layers": 1,
    })
    model = MaskedTFModel()
    model.build_model(cfg)
    criterion = PretrainMaskedCriterion()
    criterion.build_criterion(OmegaConf.create({"alpha": 2.0}))
    task = SpecPretrain(OmegaConf.create({"name": "spec_pretrain"}))
    loader = DataLoader(
        [example(), example()], batch_size=2, collate_fn=spec_collator
    )
    outs = task.get_valid_outs(
        model, loader, criterion, "cpu", rank_indices=[0, 1],
        collect_frequency=True,
    )
    assert outs["rank_embeddings"].shape == (2, 40)
    frequency = outs["monitor_accumulator"].frequency
    assert frequency.shape == (3, 40, 6)
    assert int(frequency[:, :, 0].sum()) == int(
        outs["monitor_accumulator"].sums[:, 0].sum()
    )
    assert np.isfinite(frequency).all()

    orthogonal = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
    rank = embedding_rank(orthogonal)
    assert rank["effective_rank"] == pytest.approx(2)
    assert rank["effective_rank_fraction"] == pytest.approx(1)
    assert rank["pca95_components"] == 2
    with pytest.warns(UserWarning, match="zero centered variance"):
        collapsed = embedding_rank(np.ones((4, 2)))
    assert collapsed["effective_rank"] == 0
    assert collapsed["pca95_components"] == 0


def test_nonfinite_statistics_rejected():
    with pytest.raises(ValueError, match="finite"):
        summarize(np.full(9, np.nan), 2.0)


def test_tensorboard_toggle_keeps_scientific_identity():
    from util.experiments.lifecycle import scientific_config

    cfg = OmegaConf.create(
        {
            "exp": {"runner": {"lr": 0.01, "output_tb": False}},
            "data": {"name": "ajile12"},
            "model": {"name": "masked_tf_model"},
            "task": {"name": "spec_pretrain"},
            "criterion": {"alpha": 2.0},
            "preprocessor": {"name": "stft"},
            "protocol": {"revision": "test"},
        }
    )
    dataset = SimpleNamespace(inventory=[])
    before = scientific_config(cfg, dataset, {})
    cfg.exp.runner.output_tb = True
    cfg.exp.runner.test_diagnostics = ["representation_rank", "frequency_errors"]
    cfg.exp.runner.reports = {
        "formats": ["png", "csv"], "splits": ["test"],
        "figures": ["mask_margins"], "tables": ["overall_metrics"],
    }
    assert before == scientific_config(cfg, dataset, {})


def test_spectral_content_weights_and_gradient():
    """Weight only selected content and normalise by their total weight."""
    from criterions.pretrain_masked_criterion import spectral_content_loss

    errors = torch.tensor([[[2., 100., 4.]]], requires_grad=True)
    content = torch.tensor([[[True, False, True]]])
    frequencies = torch.tensor([0., 99.84, 199.68])
    loss = spectral_content_loss(errors, content, frequencies, 1., 2.)
    assert loss.item() == pytest.approx(10 / 3)
    loss.backward()
    torch.testing.assert_close(errors.grad, torch.tensor([[[1/3, 0., 2/3]]]))
    assert spectral_content_loss(
        errors, content, frequencies, 0., 2.,
    ).item() == 3.


def test_epoch_patience_and_retained_snapshots(tmp_path):
    """Four non-improving epochs stop and persist all completed snapshots."""
    task = SpecPretrain(OmegaConf.create({"name": "spec_pretrain"}))
    extractor = STFTPreprocessor(
        OmegaConf.load(ROOT / "conf/preprocessor/stft.yaml")
    )
    task.dataset = SimpleNamespace(extracter=extractor)
    examples = [example() for _ in range(4)]
    task.train_set = Subset(examples, [0, 1])
    task.valid_set = Subset(examples, [2])
    task.test_set = Subset(examples, [3])
    criterion = PretrainMaskedCriterion()
    criterion.build_criterion(OmegaConf.create({"alpha": 2.0}))
    cfg = OmegaConf.create({
        "artifact_dir": str(tmp_path / "checkpoints"),
        "device": "cpu", "multi_gpu": False, "output_tb": False,
        "quiet": True, "optim": "SGD", "lr": .001,
        "total_steps": 40, "epochs": 20, "log_step": 1000,
        "checkpoint_step": 1000, "epoch_end_validation": True,
        "train_batch_size": 1, "valid_batch_size": 1, "num_workers": 0,
        "shuffle": False, "grad_clip": 1.,
        "early_stopping_patience": 4, "retain_epoch_checkpoints": True,
        "scheduler": {"name": "ramp_up", "warmup": .1, "total_steps": 40},
    })
    runner = Runner(cfg, task, TinyModel(), criterion)
    actual_valid = runner.get_valid_outs

    def constant_validation():
        result = actual_valid()
        result["loss"] = 1.
        return result

    runner.get_valid_outs = constant_validation
    try:
        runner.train()
        assert runner.completed_steps == 10
        assert runner.stale_epochs == 4
        assert runner.stopped_early
        assert len(list((tmp_path / "checkpoints/epochs").glob("*.pth"))) == 5
        last = torch.load(tmp_path / "checkpoints/checkpoint_last.pth",
                          weights_only=False)
        assert last["stale_epochs"] == 4 and last["stopped_early"]
        assert not last["pending_fixed_validation"]
    finally:
        runner.progress.close()
        runner.monitor.close()
