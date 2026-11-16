"""Verify budget integration with the training runner."""

import numpy as np
import pytest
from omegaconf import OmegaConf


def test_budget_overrides_configured_steps(tmp_path, monkeypatch) -> None:
    """Warn and derive runner and scheduler steps from sample budget."""
    from util.experiments.execution import configure_budgeted

    class Dataset:
        selection_namespace = "test"
        selection_revision = "test-v1"
        records = []
        partitions = {}
        ends = []
        inventory = []

        def group_id(self, record: dict) -> str:
            return record["session"]

        def cached_local_indices(self, record: dict) -> set[int]:
            return set()

        def cache_settings(self) -> dict:
            return {}

        def __len__(self) -> int:
            return 3

        def __getitem__(self, index: int) -> int:
            return index

        def configure_selection(self, selection: dict) -> None:
            self.selection = selection

    class Task:
        dataset = Dataset()

    cfg = OmegaConf.create({
        "task": {"sampling": {
            "mode": "temperature",
            "sample_budget": 682000,
            "alpha": 0,
            "seed": 0,
            "validation_batches": 128,
            "test_batches": 128,
        }},
        "data": {"name": "ajile12"},
        "preprocessor": {"normalizing": "recording_zscore"},
        "storage": {"artifacts": str(tmp_path)},
        "exp": {"runner": {
            "train_batch_size": 256,
            "valid_batch_size": 256,
            "shuffle": False,
            "total_steps": 400000,
            "scheduler": {"name": "ramp_up"},
        }},
    })
    selection = {"indices": {
        "train": [0], "validation": [1], "test": [2],
    }}
    monkeypatch.setattr(
        "util.experiments.sampling.select_or_load",
        lambda root, dataset, settings: selection,
    )
    with pytest.warns(UserWarning, match="total_steps is ignored"):
        configure_budgeted(cfg, Task())
    assert cfg.exp.runner.total_steps == 2665
    assert cfg.exp.runner.scheduler.total_steps == 2665


def test_ajile_evaluation_balances_electrodes(tmp_path) -> None:
    """Keep cached examples while balancing selected electrode windows."""
    from types import SimpleNamespace

    from data.ajile.sampling import select_evaluation_group

    session = "synthetic"
    folder = tmp_path / "sessions" / session / "blocks"
    folder.mkdir(parents=True)
    valid = np.array([
        [True, True],
        [True, True],
        [True, False],
        [False, True],
    ])
    np.savez_compressed(
        folder / "part.npz", reasons=(~valid).astype(np.uint8)
    )
    record = {
        "session": session,
        "eligible": int(valid.sum()),
        "metadata": {
            "samples": 4 * 2500,
            "channels": 2,
        },
    }
    selected = select_evaluation_group(
        SimpleNamespace(audit=tmp_path),
        record,
        quota=4,
        cached={1},
        rng=np.random.default_rng(3),
    )
    assert 1 in selected
    assert len(selected) == len(set(selected)) == 4
    positions = np.flatnonzero(valid)
    counts = np.bincount(positions[selected] % 2, minlength=2)
    np.testing.assert_array_equal(counts, [2, 2])
