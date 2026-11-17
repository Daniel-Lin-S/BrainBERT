"""Cheap invariants for paired fixed-mask test diagnostics."""

import pytest
import torch

from util.experiments.fixed_mask_diagnostics import (
    balanced_bins, fixed_masks, reconstruction_statistics,
)


def test_nested_masks_and_balanced_assignment():
    """All conditions retain pairing and exact widths without padding."""
    bins = balanced_bins(80, 4, 3)
    assert torch.equal(torch.bincount(bins), torch.full((4,), 20))
    padding = torch.zeros(80, 20, dtype=torch.bool)
    masks = [fixed_masks(
        padding, 4, torch.arange(80), duration, 14, bins, 3,
    ) for duration in (4, 8, 14)]
    for (temporal, spectral), duration in zip(masks, (4, 8, 14)):
        assert temporal.sum() == 80 * duration * 4
        assert spectral.sum() == 80 * 20
        assert (temporal & spectral).sum() == 80 * duration
    assert not (masks[0][0] & ~masks[1][0]).any()
    assert not (masks[1][0] & ~masks[2][0]).any()
    assert torch.equal(masks[0][1], masks[2][1])
    assert torch.equal(bins, balanced_bins(80, 4, 3))


def test_distinct_overall_and_frequency_baselines():
    """Overall zero and frequency training means have different errors."""
    target = torch.tensor([[[2., 4.], [6., 8.]]])
    selected = torch.ones_like(target, dtype=torch.bool)
    objective, sums = reconstruction_statistics(
        torch.zeros_like(target), target, selected, selected,
        torch.tensor([1., 3.]), 2.,
    )
    assert torch.equal(objective, torch.tensor([5., 5., 15., 5., 5., 15.]))
    assert torch.equal(sums[:, 2], torch.tensor([6., 6.]))
    assert torch.equal(sums[:, 1], torch.tensor([8., 12.]))


def test_short_windows_rejected():
    """Never clip the fixed temporal duration to fit a short window."""
    with pytest.raises(ValueError, match="shorter"):
        fixed_masks(
            torch.zeros(1, 3, dtype=torch.bool), 1,
            torch.tensor([0]), 4, 14, torch.tensor([0]), 0,
        )


def test_target_only_copy_preserves_original_recipe(monkeypatch):
    """Diagnostic copies skip masks without changing shared configuration."""
    from types import SimpleNamespace
    from omegaconf import OmegaConf
    from util.mask_utils import test_masked_dataset as clone
    from datasets.masked_tf_dataset import prepare_example

    original = SimpleNamespace(task_cfg=OmegaConf.create({}))
    diagnostic = clone(original, targets_only=True)
    assert "diagnostic_targets_only" not in original.task_cfg
    features = torch.tensor([[2., 3.], [4., 5.]])

    def unexpected_mask(*args, **kwargs):
        raise AssertionError("Unused random masks must not be generated")

    monkeypatch.setattr(
        "datasets.masked_tf_dataset.mask_inputs", unexpected_mask,
    )
    result = prepare_example(
        None, None, diagnostic.task_cfg, features=features,
    )
    assert torch.equal(result["target"], features)
    assert not result["mask_label"].any()


class _DiagnosticModel(torch.nn.Module):
    """Deterministic inexpensive reconstruction for scheduler checks."""

    def forward(self, values, padding):
        return values * 0.5, None


class _DiagnosticTask:
    """Use ordinary dictionary collation for tiny synthetic targets."""

    def get_batch_iterator(self, dataset, size, **kwargs):
        return torch.utils.data.DataLoader(dataset, size, **kwargs)


def test_shared_stream_matches_serial_and_resumes(tmp_path):
    """Device scheduling and resumption preserve batch reduction order."""
    from util.experiments.fixed_mask_diagnostics import evaluate_conditions

    torch.set_num_threads(1)
    dataset = [{
        "target": torch.arange(80).reshape(20, 4).float() + 2,
        "attn_mask": torch.zeros(20, dtype=torch.bool),
    } for _ in range(80)]
    conditions = [("spectral", "spectral", 0),
                  ("temporal_4", "temporal", 4),
                  ("joint_4", "joint", 4)]
    bins = balanced_bins(80, 4, 0)

    def run(devices, root):
        return evaluate_conditions(
            _DiagnosticTask(), dataset, _DiagnosticModel(), devices,
            conditions, bins, torch.ones(4), [4, 8, 14], 0, 2.,
            4, 0, root, {"test": "paired"}, lambda *args: None, None,
        )

    serial = run(["cpu"], tmp_path / "serial")
    parallel = run(["cpu", "cpu:0"], tmp_path / "parallel")
    resumed = run(["cpu"], tmp_path / "parallel")
    assert serial == parallel == resumed


def test_training_baseline_partial_resume(tmp_path):
    """Resumed training means equal a direct mean over all valid targets."""
    from util.experiments.fixed_mask_diagnostics import training_baseline

    dataset = [{
        "target": torch.full((3, 2), float(i)),
        "attn_mask": torch.zeros(3, dtype=torch.bool),
    } for i in range(80)]
    args = (_DiagnosticTask(), dataset, 4, 0, tmp_path, {"test": "mean"},
            lambda *args: None, None)
    first = training_baseline(*args)
    resumed = training_baseline(*args)
    assert first == resumed
    assert first["mean"] == [39.5, 39.5]
    assert first["frame_count"] == 240



def test_temporal_interpolation_and_boundary_hold():
    """Use only the nearest visible endpoints and count edge gaps."""
    from util.experiments.fixed_mask_diagnostics import linear_temporal_errors

    target = torch.tensor([
        [[0.], [1.], [2.], [3.], [4.]],
        [[2.], [3.], [4.], [5.], [6.]],
    ])
    mask = torch.zeros_like(target, dtype=torch.bool)
    mask[0, 1:4] = True
    mask[1, 0:2] = True
    errors, edge = linear_temporal_errors(
        target, torch.zeros(2, 5, dtype=torch.bool), mask,
    )
    assert edge == 1
    assert torch.equal(errors, torch.tensor([[3., 3.]], dtype=torch.float64))
    altered = target.clone()
    altered[0, 1:4] += 10
    changed, _ = linear_temporal_errors(
        altered, torch.zeros(2, 5, dtype=torch.bool), mask,
    )
    assert torch.equal(
        changed, torch.tensor([[33., 33.]], dtype=torch.float64),
    )


def test_temporal_interpolation_rejects_invalid_masks():
    """All masked cells must have available visible context."""
    from util.experiments.fixed_mask_diagnostics import linear_temporal_errors

    target = torch.ones(1, 4, 2)
    with pytest.raises(ValueError, match="endpoint"):
        linear_temporal_errors(
            target, torch.zeros(1, 4, dtype=torch.bool),
            torch.ones_like(target, dtype=torch.bool),
        )


def test_frequency_losses_use_content_denominators_and_combined_baseline():
    """Total-loss ratios are ratios of objectives, not sums of ratios."""
    from util.experiments.fixed_mask_diagnostics import frequency_losses
    import numpy as np

    losses = frequency_losses([[10, 20, 40, 2, 8, 12]], 2.)
    np.testing.assert_allclose(losses["model"], [[2, 4, 10]])
    np.testing.assert_allclose(losses["constant"], [[4, 6, 16]])
    np.testing.assert_allclose(losses["relative"], [[.5, 2 / 3, .625]])
    with pytest.raises(ValueError, match="content cells"):
        frequency_losses([[10, 20, 40, 0, 0, 0]], 2.)
    with pytest.raises(ValueError, match="finite"):
        frequency_losses([[10, float("nan"), 40, 2, 8, 12]], 2.)


def test_interpolation_content_errors_exclude_noncontent_targets():
    """The content baseline uses the same target-defined subset as training."""
    from util.experiments.fixed_mask_diagnostics import linear_temporal_errors

    target = torch.tensor([[[0.], [.5], [3.], [0.]]])
    selected = torch.tensor([[[False], [True], [True], [False]]])
    errors, edges = linear_temporal_errors(
        target, torch.zeros(1, 4, dtype=torch.bool), selected,
    )
    assert edges == 0
    assert torch.equal(errors, torch.tensor([[3.5, 3.]], dtype=torch.float64))


def test_training_content_counts_only_selected_cells(tmp_path, monkeypatch):
    """Subset estimates preserve RNG state and exclude unselected/padded cells."""
    import random
    from types import SimpleNamespace
    from util.experiments.fixed_mask_diagnostics import training_content_counts

    dataset = [{
        "target": torch.tensor([[2., .5], [.5, 3.], [8., 8.]]),
        "attn_mask": torch.tensor([False, False, True]),
    } for _ in range(8)]
    task = _DiagnosticTask()
    task.train_set = dataset
    monkeypatch.setattr("util.mask_utils.test_masked_dataset",
                        lambda x, **kwargs: x)

    def masks(values, cfg):
        random.random()
        return values, torch.tensor([[True, True], [True, False]])

    monkeypatch.setattr("util.mask_utils.mask_inputs", masks)
    options = {"sample_windows": 4, "selection_seed": 0, "mask_seed": 3}
    state = random.getstate()
    result = training_content_counts(
        task, SimpleNamespace(), options, tmp_path, {"fit": "test"},
        0, 2, lambda *args: None,
    )
    assert random.getstate() == state
    assert result["selected_count"] == [8, 4]
    assert result["content_count"] == [4, 0]
    assert result["windows"] == 4
    assert not result["historical_masks"]
