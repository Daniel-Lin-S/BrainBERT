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
