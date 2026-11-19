"""Cheap reporting and validation/test diagnostic parity regressions."""

import gzip
from types import SimpleNamespace
from unittest.mock import Mock
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from runner import Runner

from util.experiments.lifecycle import OPERATIONAL_RUNNER
from util.experiments.monitoring import (
    Accumulator,
    TrainingMonitor,
    rank_positions,
    diagnostic_names,
)
from util.experiments.reporting import Report, load_history, read_table, render
from util.experiments.report_settings import report_settings, reports_requested

ROOT = Path(__file__).resolve().parents[2]


def test_recover_flushed_gzip_without_mutation(tmp_path):
    path = tmp_path / "frequency.csv.gz"
    payload = gzip.compress(b"step,value\n1,2.5\n")[:-8]
    path.write_bytes(payload)
    with pytest.warns(UserWarning, match="Missing gzip footer"):
        frame = read_table(path)
    assert frame.value.tolist() == [2.5]
    assert path.read_bytes() == payload
    path.write_bytes(gzip.compress(b"step,value\n1,2")[:-8])
    with (
        pytest.warns(UserWarning),
        pytest.raises(ValueError, match="Incomplete"),
    ):
        read_table(path)


def test_resume_drops_unsaved_rows(tmp_path):
    root = tmp_path / "monitoring"
    root.mkdir()
    rows = pd.DataFrame(
        {
            "phase": ["train", "train"],
            "step": [1, 101],
            "interval_start": [1, 2],
            "original_loss": [4.0, 9.0],
            "prediction_std": [0.5, 0.6],
            "target_std": [1.0, 1.0],
        }
    )
    rows.to_csv(root / "metrics.csv", index=False)
    segment = root / "segments/resumed"
    segment.mkdir(parents=True)
    resumed = rows.iloc[[1]].copy()
    resumed["original_loss"] = 3.0
    resumed.to_csv(segment / "metrics.csv", index=False)
    frame = load_history(tmp_path)["metrics"]
    assert frame.step.tolist() == [1, 101]
    assert frame.total_loss.tolist() == [4.0, 3.0]


def test_test_and_validation_share_diagnostic_writers(tmp_path):
    monitor = TrainingMonitor(
        tmp_path / "monitoring",
        np.arange(4.0),
        2.0,
        1.0,
        1,
        "recording_zscore",
        tensorboard=False,
    )
    stats = Accumulator()
    stats.frequency = np.ones((3, 4, 6))
    embeddings = np.arange(24.0).reshape(8, 3)
    try:
        for phase in ("validation", "test"):
            monitor.record_frequency(7, stats, phase=phase)
            monitor.record_rank(7, embeddings, phase=phase)
    finally:
        monitor.close()
    for name, suffix in (("rank", ".csv"), ("frequency", ".csv.gz")):
        pd.testing.assert_frame_equal(
            read_table(tmp_path / "monitoring" / f"test_{name}{suffix}"),
            read_table(tmp_path / "monitoring" / f"validation_{name}{suffix}"),
        )
    positions = rank_positions(32768)
    assert len(positions) == 2048
    assert positions[0] == 0 and positions[-1] == 32767


def test_operational_defaults_and_ajile_enablement():
    with initialize_config_dir(
        config_dir=str(ROOT / "conf"), version_base=None
    ):
        for path in sorted((ROOT / "conf/exp").glob("*.yaml")):
            cfg = compose(config_name=None, overrides=[f"+exp={path.stem}"])
            enabled = path.stem.startswith("ajile12")
            runner = cfg.exp.get("runner", {})
            assert (
                bool(diagnostic_names(runner.get("test_diagnostics")))
                == enabled
            )
            assert reports_requested(runner.get("reports")) == enabled
            if not enabled:
                assert "test_diagnostics" not in runner
                assert "reports" not in runner
    assert {"test_diagnostics", "reports"}.issubset(OPERATIONAL_RUNNER)


def test_ready_outputs_survive_pending_and_failed_diagnostics(tmp_path):
    report = Report(
        tmp_path,
        Path("fit-identity"),
        7,
        [7],
        settings={
            "formats": ["png", "csv"],
            "splits": ["test"],
            "figures": ["frequency_errors"],
            "tables": ["objective_components"],
        },
    )
    frame = pd.DataFrame({"step": [7], "total_loss": [2.0]})
    report.table("test/overall/objective", frame, "Saved test objective")
    ready = tmp_path / "tables/test/overall/objective.csv"
    original = ready.read_bytes()
    report.register(
        "figures/test/frequency/reconstruction.png",
        "Pending test diagnostics",
        "pending",
    )
    report.register(
        "figures/test/frequency/reconstruction.png",
        "Evaluation unavailable",
        "failed",
    )
    assert ready.read_bytes() == original
    index = pd.read_csv(tmp_path / "output_index.csv")
    assert index.state.tolist() == ["ready", "failed"]
    assert not (tmp_path / "figures/test/frequency/reconstruction.png").exists()


@pytest.mark.parametrize("value", [None, {}, [], ""])
def test_empty_settings_produce_nothing(tmp_path, value):
    assert not diagnostic_names(value)
    assert not reports_requested(value)
    render(tmp_path, "unused", settings=report_settings(value))
    assert not (tmp_path / "campaigns").exists()


@pytest.mark.parametrize("value", [True, False])
def test_boolean_settings_are_rejected(value):
    with pytest.raises(ValueError):
        diagnostic_names(value)
    with pytest.raises(ValueError):
        report_settings(value)


def test_explicit_family_format_and_split_selection(tmp_path):
    report = Report(
        tmp_path,
        Path("fit-id"),
        7,
        [],
        settings={
            "formats": ["png"],
            "splits": ["validation"],
            "figures": ["mask_margins"],
            "tables": [],
        },
    )
    assert report.wants("figures", "validation/mask_conditions/margins/history")
    assert not report.wants(
        "figures", "validation/mask_conditions/full_grid/history"
    )
    assert not report.wants("figures", "test/mask_conditions/margins/history")
    assert not report.wants(
        "tables", "validation/mask_conditions/margins/metrics"
    )


@pytest.mark.parametrize(
    "names",
    [
        [],
        ["representation_rank"],
        ["frequency_errors"],
        ["representation_rank", "frequency_errors"],
    ],
)
def test_runner_activates_only_named_diagnostics(names):
    runner = object.__new__(Runner)
    runner.cfg = OmegaConf.create(
        {
            "test_diagnostics": names,
            "valid_batch_size": 2,
            "num_workers": 0,
        }
    )
    output = {"loss": 0.5, "monitor_accumulator": object()}
    if "representation_rank" in names:
        output["rank_embeddings"] = np.ones((8, 3))
    runner.task = SimpleNamespace(
        cfg=SimpleNamespace(name="spec_pretrain"),
        test_set=list(range(8)),
        get_batch_iterator=Mock(return_value=[{}]),
        get_valid_outs=Mock(return_value=output),
    )
    runner.monitor = Mock()
    runner.report_progress = Mock()
    runner.criterion = object()
    runner.device = "cpu"
    runner.completed_steps = 10
    runner.best_step = 7
    result = runner.test(Mock())
    options = runner.task.get_valid_outs.call_args.kwargs
    assert options["collect_frequency"] == ("frequency_errors" in names)
    assert (options["rank_indices"] is not None) == (
        "representation_rank" in names
    )
    assert runner.monitor.record_rank.called == ("representation_rank" in names)
    assert runner.monitor.record_frequency.called == (
        "frequency_errors" in names
    )
    assert result == {"loss": 0.5}


def test_figure_layout_groups_related_outputs():
    from util.experiments.reporting import figure_output

    assert figure_output("optimization/dynamics") == (
        "figures/training/optimization_dynamics.png"
    )
    for split in ("training", "validation", "test"):
        for family in ("margins", "full_grid"):
            logical = f"{split}/mask_conditions/{family}/history"
            output = figure_output(logical)
            assert output == (
                f"figures/{split}/mask_conditions/{family}_history.png"
            )
            assert figure_output(output[8:-4]) == output
        assert figure_output(f"{split}/representation/rank") == (
            f"figures/{split}/representation_rank.png"
        )


@pytest.mark.parametrize(
    "policy",
    [
        {"zero": 1, "random": 0, "unchanged": 0},
        {"zero": 0, "random": 1, "unchanged": 0},
    ],
)
def test_test_policy_preserves_training_and_excludes_unchanged(policy):
    import random
    import torch
    from util.mask_utils import mask_inputs, test_masked_dataset

    cfg = OmegaConf.create(
        {
            "mask_type": "variable",
            "mask_p": 0.1,
            "min_f": 0.01,
            "max_f": 250,
            "n_freq_steps": 40,
        }
    )
    dataset = SimpleNamespace(task_cfg=cfg)
    test = test_masked_dataset(dataset, policy)
    assert "replacement_probabilities" not in dataset.task_cfg
    assert test.task_cfg.replacement_probabilities == policy
    state = random.getstate()
    try:
        random.seed(4)
        _, mask, provenance = mask_inputs(
            torch.ones(160, 40),
            test.task_cfg,
            return_provenance=True,
        )
        assert mask.bool().any()
        assert (provenance["action"][mask.bool()] > 0).all()
    finally:
        random.setstate(state)


def test_named_evaluations_keep_separate_monitors(tmp_path):
    import json
    from types import MethodType

    runner = object.__new__(Runner)
    runner.cfg = OmegaConf.create(
        {
            "test_evaluations": {
                "zero": {"zero": 1, "random": 0, "unchanged": 0},
                "random": {"zero": 0, "random": 1, "unchanged": 0},
            }
        }
    )
    original_cfg = runner.cfg
    monitor = TrainingMonitor(
        tmp_path / "monitoring",
        np.arange(4.0),
        2,
        1,
        1,
        "recording_zscore",
        tensorboard=False,
    )
    runner.monitor = monitor
    observed = []

    def evaluate(self, model, index, count):
        observed.append((dict(self.cfg.test_masking), self.monitor.root))
        assert count == 2
        return {"loss": float(index)}

    runner._test_once = MethodType(evaluate, runner)
    try:
        output = runner.test(object())
        assert len(output["test_evaluations"]) == 2
        assert runner.cfg is original_cfg and runner.monitor is monitor
        for policy, root in observed:
            assert (
                json.loads((root.parent / "policy.json").read_text()) == policy
            )
        assert observed[0][1] != observed[1][1]
    finally:
        monitor.close()


def test_parent_history_cutoff_and_current_test_isolation(tmp_path):
    """Continuation retains ancestors only through the restored checkpoint."""
    import json
    import torch
    from util.experiments.storage import file_digest

    parent, child = tmp_path / "parent", tmp_path / "child"
    for fit, steps in [(parent, [1, 10, 20]), (child, [11, 20])]:
        (fit / "monitoring").mkdir(parents=True)
        rows = [
            {"phase": phase, "step": step, "interval_start": step,
             "original_loss": float(step), "prediction_std": .5,
             "target_std": 1.}
            for phase in ["train", "validation", "test"] for step in steps
        ]
        pd.DataFrame(rows).to_csv(fit / "monitoring/metrics.csv", index=False)
    checkpoint = parent / "checkpoints/last.pth"
    checkpoint.parent.mkdir()
    torch.save({"completed_steps": 10}, checkpoint)
    provenance = {"invocation": {"exp": {"runner": {
        "parent_checkpoint": str(checkpoint),
        "parent_checkpoint_sha256": file_digest(checkpoint),
    }}}}
    (child / "provenance.json").write_text(json.dumps(provenance))
    rows = load_history(child)["metrics"]
    assert rows.query("phase == 'train'").step.tolist() == [1, 10, 11, 20]
    assert rows.query("phase == 'validation'").step.tolist() == [1, 10, 11, 20]
    assert rows.query("phase == 'test'").step.tolist() == [11, 20]
    assert not rows.duplicated(["phase", "step"]).any()
    assert load_history(child, include_parent=False)["metrics"].step.min() == 11
    torch.save({"completed_steps": 20}, checkpoint)
    with pytest.raises(ValueError, match="checksum"):
        load_history(child)


def test_unmasked_plots_omit_missing_validation_steps(tmp_path, monkeypatch):
    from util.experiments.reporting import unmasked_figures

    metrics = pd.DataFrame({
        "phase": ["validation"] * 3, "step": [10, 20, 30],
        "total_loss": [1., 1., 1.], "l1": [1., 1., 1.],
        "content_l1": [1., 1., 1.], "relative_l1": [1., 1., 1.],
        "relative_content_l1": [1., 1., 1.], "std_ratio": [1., 1., 1.],
    })
    monitor = pd.DataFrame({"step": [10, 30], **{
        f"{metric}_{key}": [0.2, 0.4]
        for metric in ("feature_sd", "normalized_attention_entropy")
        for key in ("mean", "median", "q1", "q3", "minimum")
    }})
    report = Report(tmp_path, Path("fit"), 30, [], settings={
        "formats": ["png"], "splits": ["validation"],
        "figures": ["representation_rank"], "tables": [],
    })
    captured = {}
    def capture(name, figure, meaning):
        if name.endswith(("feature_sd", "normalized_attention_entropy")):
            captured[name] = figure
    monkeypatch.setattr(report, "figure", capture)
    unmasked_figures(report, {"metrics": metrics,
                              "validation_unmasked": monitor})
    assert len(captured) == 2
    for fig in captured.values():
        assert len(fig.axes[0].lines) >= 1
        for line in fig.axes[0].lines[:1]:
            assert list(line.get_xdata()) == [10, 20, 30]
            assert np.isnan(line.get_ydata()[1])
    import matplotlib.pyplot as plt
    plt.close("all")
