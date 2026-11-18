"""Verify reusable group quotas, cache priority, and frozen membership."""

import numpy as np
import pytest

from util.experiments.sampling import (
    allocate, extend_group, select_group, select_or_load
)


def test_quota_probability_and_exact_budget() -> None:
    """Check equal-group and size-proportional allocations."""
    assert allocate([10, 90], 11, 0) == [6, 5]
    assert allocate([10, 90], 100, 1) == [10, 90]


def test_cached_examples_fill_budget_first() -> None:
    """Reuse cached examples before drawing previously unseen examples."""
    rng = np.random.default_rng(4)
    first = select_group(20, 6, {1, 3}, rng, True)
    assert len(first) == len(set(first)) == 6
    assert {1, 3}.issubset(first)
    second = select_group(20, 3, set(first), rng, True)
    assert len(second) == 3
    assert set(second).issubset(first)


def test_repeats_only_after_exhaustion() -> None:
    """Use every eligible example before repeating any."""
    drawn = select_group(
        5, 8, {1, 2}, np.random.default_rng(1), True
    )
    assert len(drawn) == 8
    assert set(drawn) == set(range(5))
    assert len(set(drawn[:5])) == 5


def test_selection_freezes_cache_snapshot(tmp_path) -> None:
    """A cached-example change cannot alter a saved protocol selection."""
    class Dataset:
        selection_namespace = "test"
        selection_revision = "test-v1"
        inventory = [{"content_id": "source"}]
        partitions = {
            "train": ["train"],
            "validation": ["valid"],
            "test": ["test"],
        }
        records = [
            {"session": "train", "eligible": 20},
            {"session": "valid", "eligible": 20},
            {"session": "test", "eligible": 20},
        ]
        ends = [20, 40, 60]
        cached = {"train": {1, 3}, "valid": {0}, "test": {2}}

        def group_id(self, record: dict) -> str:
            return record["session"]

        def cache_settings(self) -> dict:
            return {"nperseg": 400}

        def cached_local_indices(self, record: dict) -> set[int]:
            return self.cached[record["session"]]

    dataset = Dataset()
    settings = {
        "sample_budget": 6,
        "alpha": 0,
        "seed": 7,
        "validation_batches": 1,
        "test_batches": 1,
        "valid_batch_size": 4,
    }
    first = select_or_load(tmp_path, dataset, settings)
    assert select_or_load(
        tmp_path, dataset, {**settings, "alpha": 0.0}
    ) == first
    assert {1, 3}.issubset(first["indices"]["train"])
    dataset.cached["train"] = set(range(20))
    assert select_or_load(tmp_path, dataset, settings) == first
    dataset.cached["train"] = set(first["indices"]["train"])
    larger = select_or_load(
        tmp_path, dataset, {**settings, "sample_budget": 10}
    )
    assert set(first["indices"]["train"]).issubset(
        larger["indices"]["train"]
    )


    nested = select_or_load(
        tmp_path, dataset, {**settings, "sample_budget": 12}, parent=first
    )
    assert len(set(nested["indices"]["train"])) == 12
    assert set(first["indices"]["train"]).issubset(
        nested["indices"]["train"]
    )
    for phase in ("validation", "test"):
        assert nested["indices"][phase] == first["indices"][phase]
    dataset.cached["train"] = set(range(20))
    assert select_or_load(
        tmp_path, dataset, {**settings, "sample_budget": 12}, parent=first
    ) == nested
    with pytest.raises(ValueError, match="sampling differs"):
        select_or_load(
            tmp_path, dataset,
            {**settings, "sample_budget": 14, "seed": 8}, parent=first,
        )
    with pytest.raises(ValueError, match="exceed the parent"):
        select_or_load(tmp_path, dataset, settings, parent=first)


def test_nested_group_retains_parent_and_prioritizes_cache() -> None:
    """An unrelated cache cannot displace mandatory parent windows."""
    retained = {1, 3}
    cached = {4, 5, 6, 7}
    selected = extend_group(
        20, 5, retained, cached, np.random.default_rng(7)
    )
    assert len(set(selected)) == 5
    assert retained.issubset(selected)
    assert set(selected) - retained <= cached
    with pytest.raises(ValueError, match="Nested quota"):
        extend_group(20, 1, retained, cached, np.random.default_rng(7))


def test_raw_cache_reuses_extraction(tmp_path) -> None:
    """A second access loads the raw tensor without extracting again."""
    import torch
    from omegaconf import OmegaConf

    from preprocessors.cache import (
        cached_raw_positions, cached_raw_stft, raw_stft_path
    )

    cfg = OmegaConf.create({
        "feature_cache": str(tmp_path),
        "freq_channel_cutoff": 3,
        "nperseg": 8,
        "normalizing": "recording_zscore",
    })
    calls = []

    def extract() -> torch.Tensor:
        calls.append(1)
        return torch.ones((5, 3), dtype=torch.float32)

    first = cached_raw_stft(
        tmp_path, "dataset_raw", "source", 4, cfg, extract
    )
    second = cached_raw_stft(
        tmp_path, "dataset_raw", "source", 4, cfg, extract
    )
    assert len(calls) == 1
    torch.testing.assert_close(first, second)
    assert raw_stft_path(
        tmp_path, "dataset_raw", "source", 4, cfg
    ).exists()
    assert cached_raw_positions(
        tmp_path, "dataset_raw", "source", cfg
    ) == {4}
