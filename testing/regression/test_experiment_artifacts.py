"""Regression checks for storage and identity using disposable artifacts."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from util.experiments.storage import atomic_write, digest, write_json
from util.experiments.lifecycle import FitRun, scientific_config
from preprocessors.cache import cached_stft


def test_atomic_failure_preserves_old_file(tmp_path):
    path = tmp_path / "artifact.json"
    write_json(path, {"old": 1})

    def fail(candidate):
        raise ValueError("bad candidate")

    with pytest.raises(ValueError, match="bad candidate"):
        atomic_write(path, lambda p: p.write_text("bad"), fail)
    assert json.loads(path.read_text()) == {"old": 1}


def complete_stub(run):
    checkpoints = run.path / "checkpoints"
    checkpoints.mkdir()
    for name in ("best", "last"):
        torch.save({"model": {}}, checkpoints / f"checkpoint_{name}.pth")
    run.complete({"loss": 1.0})


def test_reuse_across_campaigns_and_protect_corruption(tmp_path):
    args = [str(tmp_path / "artifacts"), str(tmp_path / "logs")]
    with FitRun(*args, "first", {"batch_size": 2}) as first:
        complete_stub(first)
        identity = first.fit_id
    with FitRun(*args, "second", {"batch_size": 2}) as second:
        assert second.reused
        assert second.fit_id == identity
    checkpoint = second.path / "checkpoints" / "checkpoint_best.pth"
    checkpoint.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        with FitRun(*args, "third", {"batch_size": 2}):
            pass
    assert checkpoint.read_bytes() == b"corrupt"


def test_lock_and_quarantine(tmp_path):
    args = [
        str(tmp_path / "artifacts"),
        str(tmp_path / "logs"),
        "campaign",
        {"batch_size": 2},
    ]
    with FitRun(*args) as first:
        with pytest.raises(RuntimeError, match="already running"):
            with FitRun(*args):
                pass
        (first.path / "partial").write_text("preserve")
    with FitRun(*args) as retry:
        assert not retry.reused
        old = list((retry.root / "quarantine").rglob("partial"))
        assert len(old) == 1 and old[0].read_text() == "preserve"
        assert not (retry.path / "partial").exists()


def test_scientific_identity_ignores_invocation(tmp_path):
    cfg = OmegaConf.create(
        {
            "exp": {
                "runner": {
                    "lr": 0.1,
                    "optim": "LAMB",
                    "log_step": 1,
                    "checkpoint_step": 2,
                    "device": "cpu",
                    "num_workers": 0,
                }
            },
            "protocol": {"revision": "original"},
            "data": {
                "name": "masked_tf_dataset",
                "data": "/first",
                "pinned_test": "/first/test.json",
            },
            "model": {"hidden_dim": 8},
            "task": {"mask_type": "fixed"},
            "criterion": {"alpha": 1},
            "preprocessor": {"name": "stft", "feature_cache": "/first/cache"},
        }
    )
    dataset = SimpleNamespace(inventory=[{"sample": "one", "sha256": "abc"}])
    splits = {"train": ["one"], "test": ["two"]}
    before = digest(scientific_config(cfg, dataset, splits))
    cfg.exp.runner.device = "cuda"
    cfg.exp.runner.num_workers = 12
    cfg.data.data = "/second"
    cfg.data.pinned_test = "/second/test.json"
    cfg.preprocessor.feature_cache = "/second/cache"
    assert digest(scientific_config(cfg, dataset, splits)) == before
    cfg.exp.runner.lr = 0.2
    assert digest(scientific_config(cfg, dataset, splits)) != before
    cfg.exp.runner.lr = 0.1
    cfg.exp.runner.log_step = 5
    assert digest(scientific_config(cfg, dataset, splits)) != before


def test_cache_content_identity_and_corruption(tmp_path):
    cfg = OmegaConf.create({"feature_cache": str(tmp_path), "nperseg": 400})
    calls = []

    def extract():
        calls.append(1)
        return torch.ones(3, 2)

    wav = np.arange(100, dtype="float32")
    assert torch.equal(cached_stft(wav, cfg, extract), torch.ones(3, 2))
    cached_stft(wav, cfg, extract)
    assert len(calls) == 1
    wav[0] = 10
    cached_stft(wav, cfg, extract)
    assert len(calls) == 2
    for path in tmp_path.rglob("*.npy"):
        np.save(path, np.full((3, 2), np.nan))
    with pytest.raises(ValueError, match="Invalid cached"):
        cached_stft(wav, cfg, extract)


def test_original_stft_exactly_matches_cache(tmp_path):
    from preprocessors.stft import STFTPreprocessor

    cfg = OmegaConf.create(
        {
            "name": "stft",
            "freq_channel_cutoff": 40,
            "nperseg": 400,
            "noverlap": 350,
            "normalizing": "zscore",
        }
    )
    waveform = np.random.default_rng(1).normal(size=10000).astype("float32")
    baseline = STFTPreprocessor(cfg)(waveform)
    cfg.feature_cache = str(tmp_path)
    first = STFTPreprocessor(cfg)(waveform)
    second = STFTPreprocessor(cfg)(waveform)
    assert torch.equal(baseline, first)
    assert torch.equal(first, second)


def test_machine_settings_cannot_override_science():
    from util.experiments.execution import execute

    cfg = OmegaConf.create(
        {
            "campaign": "guard",
            "protocol": {"revision": "test"},
            "data": {"pinned_test": "unused"},
            "exp": {"runner": {"checkpoint_step": 1}},
            "runtime": {"lr": 0.1},
        }
    )
    with pytest.raises(ValueError, match="cannot override scientific"):
        execute(cfg)


def test_post_install_audit_restores_backup(tmp_path):
    path = tmp_path / "artifact.json"
    write_json(path, {"old": 1})

    def audit(candidate):
        if candidate == path:
            raise ValueError("installed audit failed")

    with pytest.raises(ValueError, match="installed audit"):
        atomic_write(path, lambda p: p.write_text("bad"), audit)
    assert json.loads(path.read_text()) == {"old": 1}
