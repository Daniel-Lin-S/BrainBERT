"""Regression checks for scientific monitoring and unchanged optimization."""

import copy
import csv
import json
import random
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import Subset
from tensorboard.backend.event_processing.event_accumulator import (
    EventAccumulator,
)

from criterions.pretrain_masked_criterion import PretrainMaskedCriterion
from preprocessors import STFTPreprocessor
from runner import Runner
from tasks.spec_pretrain import SpecPretrain
from util.experiments.lifecycle import FitRun
from util.experiments.monitoring import (
    Accumulator,
    TrainingMonitor,
    batch_statistics,
    summarize,
)
from util.mask_utils import fixed_mask_inputs, mask_inputs, variable_mask

ROOT = Path(__file__).resolve().parents[2]


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
    "total_steps, log_step, checkpoint_step, expected_steps, "
    "expected_batches, expected_starts",
    [
        (4, 2, 2, [1, 3, 4], [1, 2, 1], [1, 2, 4]),
        (5, 3, 2, [1, 3, 4, 5], [1, 2, 1, 1], [1, 2, 4, 5]),
    ],
)
def test_training_csv_tensorboard_and_best_checkpoint(
    tmp_path, total_steps, log_step, checkpoint_step, expected_steps,
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
            "train_batch_size": 1,
            "valid_batch_size": 1,
            "num_workers": 0,
            "shuffle": False,
            "grad_clip": 1.0,
            "scheduler": {
                "name": "ramp_up",
                "warmup": 0.5,
                "total_steps": total_steps,
            },
        }
    )
    runner = Runner(cfg, task, TinyModel(), criterion)
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
    assert [int(row["step"]) for row in validation] == expected_steps
    assert [int(row["batches"]) for row in train] == expected_batches
    assert all("diagnostic_objective" not in row for row in rows)
    assert all(float(row["relative_l1"]) >= 0 for row in train)
    assert all(float(row["original_relative_l1"]) >= 0 for row in train)
    assert [int(row["interval_start"]) for row in train] == expected_starts
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
        events_loss = events.Scalars(f"{phase}/overall/original_loss")
        assert [point.step for point in events_loss] == expected_steps
        np.testing.assert_allclose(
            [point.value for point in events_loss],
            [float(row["original_loss"]) for row in phase_rows],
            rtol=1e-6,
        )
    assert "train/masks/l1" in events.Tags()["images"]
    groups = read_rows(root, "mask_groups.csv")
    assert len(groups) == len(rows) * 15
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
    assert before == scientific_config(cfg, dataset, {})
