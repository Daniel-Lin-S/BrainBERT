"""Verify budget integration with the training runner."""

import numpy as np
import pytest
from omegaconf import OmegaConf


@pytest.mark.parametrize("name, batch, rate", [
    ("spec2vec", 256, 1e-4),
    ("ajile12", 256, 1e-4),
    ("ajile12_followup", 256, 1e-4),
    ("seeg_wav2vec", 16, 1e-4),
])
def test_experiment_rates_preserve_defaults_and_scale(name, batch, rate):
    """Check composed experiment defaults, scaling, and explicit rates."""
    from pathlib import Path

    from hydra import compose, initialize_config_dir

    from util.experiments.lifecycle import resolve_batch_scaled_lr

    config_dir = Path(__file__).resolve().parents[2] / "conf"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        for overrides, expected in (
            ([], rate),
            ([f"exp.runner.train_batch_size={batch * 2}"], rate * 2),
            ([f"exp.runner.train_batch_size={batch * 2}",
              "exp.runner.lr=0.003"], 0.003),
        ):
            cfg = compose(overrides=[f"+exp={name}", *overrides])
            resolve_batch_scaled_lr(cfg)
            assert cfg.exp.runner.lr == pytest.approx(expected)


@pytest.mark.parametrize("name", ["finetune", "feature_extract"])
def test_downstream_rate_does_not_scale(name):
    """Downstream configurations retain their independent learning rate."""
    from pathlib import Path

    from hydra import compose, initialize_config_dir

    from util.experiments.lifecycle import resolve_batch_scaled_lr

    config_dir = Path(__file__).resolve().parents[2] / "conf"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(overrides=[
            f"+exp={name}", "exp.runner.train_batch_size=128",
        ])
        resolve_batch_scaled_lr(cfg)
        assert cfg.exp.runner.lr == pytest.approx(1e-3)


@pytest.mark.parametrize("batch, expected", [
    (128, 5e-5), (256, 1e-4), (512, 2e-4),
])
def test_pretraining_rate_scales_optimizer_batch(batch, expected):
    """Scale the optimizer batch and preserve explicit overrides."""
    from util.experiments.lifecycle import resolve_batch_scaled_lr

    cfg = OmegaConf.create({"exp": {
        "base_lr": 1e-4,
        "runner": {
            "lr": None, "train_batch_size": batch, "microbatch_size": 32,
        },
    }})
    resolve_batch_scaled_lr(cfg)
    assert cfg.exp.runner.lr == pytest.approx(expected)
    cfg.exp.runner.lr = 0.003
    resolve_batch_scaled_lr(cfg)
    assert cfg.exp.runner.lr == 0.003


def test_pretraining_rate_rejects_invalid_batch():
    """Invalid optimizer batches must not produce a learning rate."""
    from util.experiments.lifecycle import resolve_batch_scaled_lr

    cfg = OmegaConf.create({"exp": {
        "base_lr": 1e-4, "runner": {"lr": None, "train_batch_size": 0},
    }})
    with pytest.raises(ValueError, match="train_batch_size"):
        resolve_batch_scaled_lr(cfg)


def test_budget_and_epoch_duration_are_independent(tmp_path, monkeypatch):
    """Resolve either run length after freezing AJILE membership."""
    import json
    from torch.utils.data import Subset

    from util.experiments.execution import (
        configure_budgeted, resolve_training_steps,
    )

    class Dataset:
        selection_namespace = "test"
        selection_revision = "test-v1"
        records = [
            {"session": phase, "content_id": phase, "eligible": 3}
            for phase in ("train", "validation", "test")
        ]
        partitions = {
            phase: [phase] for phase in ("train", "validation", "test")
        }
        ends = [3, 6, 9]
        inventory = []

        def group_id(self, record: dict) -> str:
            return record["session"]

        def cached_local_indices(self, record: dict) -> set[int]:
            return set()

        def cache_settings(self) -> dict:
            return {}

        def __len__(self) -> int:
            return 9

        def __getitem__(self, index: int) -> int:
            return index

        def configure_selection(self, selection: dict) -> None:
            self.selection = selection

    class Task:
        def __init__(self):
            self.dataset = Dataset()
            self.train_set = Subset(self.dataset, [0, 1, 2])

    pinned = tmp_path / "evaluation.json"
    pinned.write_text(json.dumps({
        "content_ids": {
            phase: phase for phase in ("train", "validation", "test")
        },
        "indices": {"validation": [3], "test": [6]},
    }))
    cfg = OmegaConf.create({
        "task": {"sampling": {
            "mode": "temperature", "sample_budget": 2,
            "alpha": 0, "seed": 0,
            "validation_batches": 1, "test_batches": 1,
        }},
        "data": {"pinned_evaluation": str(pinned)},
        "storage": {"artifacts": str(tmp_path)},
        "exp": {"runner": {
            "train_batch_size": 2, "valid_batch_size": 1,
            "shuffle": False, "total_steps": 5,
            "scheduler": {"name": "ramp_up"},
        }},
    })
    selection = {"indices": {
        "train": [0, 1], "validation": [3], "test": [6],
    }}
    monkeypatch.setattr(
        "util.experiments.sampling.select_or_load",
        lambda root, dataset, settings: selection,
    )
    budgeted = Task()
    configure_budgeted(cfg, budgeted)
    resolve_training_steps(cfg, budgeted)
    assert len(budgeted.train_set) == 2
    assert cfg.exp.runner.total_steps == 5
    assert cfg.exp.runner.scheduler.total_steps == 5

    cfg.task.sampling.pop("sample_budget")
    cfg.exp.runner.total_steps = None
    cfg.exp.runner.epochs = 2
    full = Task()
    configure_budgeted(cfg, full)
    resolve_training_steps(cfg, full)
    assert len(full.train_set) == 3
    assert cfg.exp.runner.total_steps == 4
    assert cfg.exp.runner.scheduler.total_steps == 4


def test_equivalent_duration_and_pin_paths_share_fit_identity() -> None:
    """Hash resolved science, not duration syntax or local placement."""
    from types import SimpleNamespace

    from util.experiments.lifecycle import scientific_config

    cfg = OmegaConf.create({
        "exp": {"runner": {
            "total_steps": 10, "epochs": None,
            "epoch_end_validation": True,
        }},
        "protocol": {"revision": "epoch-v1"},
        "model": {"name": "masked_tf_model"},
        "task": {"name": "spec_pretrain"},
        "criterion": {"alpha": 2.0},
        "preprocessor": {"name": "stft"},
        "data": {
            "name": "ajile12", "pinned_evaluation": "/tmp/first.json",
        },
    })
    dataset = SimpleNamespace(
        inventory=[{"session": "source", "content_id": "digest"}],
        selection_manifest={
            "indices": {"train": [], "validation": [1], "test": [2]}
        },
    )
    splits = {"train": ["a"], "validation": ["b"], "test": ["c"]}
    first = scientific_config(cfg, dataset, splits)
    cfg.exp.runner.epochs = 5
    cfg.data.pinned_evaluation = "/tmp/other.json"
    second = scientific_config(cfg, dataset, splits)
    assert first == second


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
