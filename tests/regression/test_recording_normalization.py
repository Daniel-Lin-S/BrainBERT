"""Verify split-local recording normalization on synthetic AJILE windows."""

import json
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import pytest
from omegaconf import OmegaConf

from data.ajile.core import OUTPUT, POLICY, WINDOW, waveform
from data.ajile.normalization import recording_statistics
from datasets.ajile_dataset import AJILE12
from preprocessors.stft import STFTPreprocessor

ROOT = Path(__file__).resolve().parents[2]


def test_recording_moments_use_only_eligible_electrode_windows(tmp_path):
    rng = np.random.default_rng(11)
    samples = 3 * WINDOW
    values = rng.normal(size=(samples, 2))
    values[WINDOW : 2 * WINDOW, 0] *= 1000
    source = tmp_path / "sample.nwb"
    with h5py.File(source, "w") as saved:
        saved.create_dataset("acquisition/ElectricalSeries/data", data=values)
    record = {
        "session": "synthetic",
        "source": source.name,
        "content_id": "synthetic-source-and-exclusions",
        "eligible": 4,
        "metadata": {
            "samples": samples,
            "channels": 2,
            "conversion": 1.0,
            "channel_conversion": None,
            "offset": 0.0,
        },
    }
    valid = np.array([[True, True], [False, True], [True, False]])
    cfg = OmegaConf.load(ROOT / "conf/preprocessor/stft.yaml")
    cfg.feature_cache = str(tmp_path / "cache")
    preprocessor = STFTPreprocessor(cfg)
    result = recording_statistics(
        tmp_path,
        record,
        "train",
        valid,
        preprocessor,
    )
    assert result["count"].tolist() == [372, 372]
    for channel in range(2):
        spectra = np.concatenate(
            [
                preprocessor.unnormalized(
                    waveform(
                        source, record["metadata"], index, channel, valid
                    )[: OUTPUT - 1]
                ).numpy()
                for index in np.flatnonzero(valid[:, channel])
            ]
        )
        np.testing.assert_allclose(
            result["mean"][channel], spectra.mean(axis=0), rtol=1e-6
        )
        np.testing.assert_allclose(
            result["std"][channel], spectra.std(axis=0), rtol=1e-6
        )
        for index in np.flatnonzero(valid[:, channel]):
            wav = waveform(source, record["metadata"], index, channel, valid)[
                : OUTPUT - 1
            ]
            normalized = preprocessor(
                wav,
                {
                    "mean": result["mean"][channel],
                    "std": result["std"][channel],
                    "identity": f"{result['identity']}:electrode:{channel}",
                },
            )
            assert normalized.shape == (186, 40)
            assert np.isfinite(normalized.numpy()).all()
    with patch(
        "data.ajile.normalization._channel_moments",
        side_effect=AssertionError("recomputed"),
    ):
        reused = recording_statistics(
            tmp_path,
            record,
            "train",
            valid,
            preprocessor,
        )
    np.testing.assert_array_equal(reused["mean"], result["mean"])
    held_out = recording_statistics(
        tmp_path,
        record,
        "test",
        valid,
        preprocessor,
    )
    assert held_out["identity"] != result["identity"]
    np.testing.assert_allclose(held_out["mean"], result["mean"])


def test_missing_recording_statistics_fail_clearly():
    cfg = OmegaConf.load(ROOT / "conf/preprocessor/stft.yaml")
    preprocessor = STFTPreprocessor(cfg)
    with pytest.raises(ValueError, match="fitted frequency statistics"):
        preprocessor(np.ones(OUTPUT - 1, dtype=np.float32))
    cfg.normalizing = "window_zscore"
    window = STFTPreprocessor(cfg)(np.linspace(0, 1, OUTPUT - 1))
    assert window.shape == (186, 40)


def test_ajile_dataset_applies_cached_recording_statistics(tmp_path):
    source_root = tmp_path / "source"
    source_root.mkdir()
    audit = tmp_path / "audit"
    session = "synthetic-session"
    blocks = audit / "sessions" / session / "blocks"
    blocks.mkdir(parents=True)
    rng = np.random.default_rng(42)
    source = source_root / "sample.nwb"
    with h5py.File(source, "w") as saved:
        saved.create_dataset(
            "acquisition/ElectricalSeries/data",
            data=rng.normal(size=(3 * WINDOW, 2)),
        )
    valid = np.array([[True, True], [False, True], [True, False]])
    with (blocks / "block.npz").open("wb") as stream:
        np.savez_compressed(stream, reasons=(~valid).astype(np.uint8))
    record = {
        "session": session,
        "source": source.name,
        "content_id": "synthetic-content",
        "eligible": int(valid.sum()),
        "metadata": {
            "samples": 3 * WINDOW,
            "channels": 2,
            "conversion": 1.0,
            "channel_conversion": None,
            "offset": 0.0,
        },
    }
    (audit / "inventory.json").write_text(
        json.dumps(
            {
                "policy": POLICY,
                "sessions": [record],
            }
        )
    )
    (audit / "splits.json").write_text(
        json.dumps(
            {
                "train": [session],
                "validation": [],
                "test": [],
            }
        )
    )
    (audit / "sessions" / session / "source.json").write_text(
        json.dumps(
            {
                "size": source.stat().st_size,
                "mtime_ns": source.stat().st_mtime_ns,
            }
        )
    )
    preprocessing = OmegaConf.load(ROOT / "conf/preprocessor/stft.yaml")
    preprocessing.feature_cache = str(tmp_path / "feature_cache")
    task = OmegaConf.load(ROOT / "conf/task/fixed_mask_pretrain.yaml")
    dataset = AJILE12(
        OmegaConf.create({"data": str(source_root), "audit": str(audit)}),
        task_cfg=task,
        preprocessor_cfg=preprocessing,
    )
    first = dataset[0]
    stats = dataset.statistics
    assert len(dataset) == 4
    assert stats["count"].tolist() == [372, 372]
    assert first["target"].shape == (186, 40)
    assert first["mask_label"].shape == first["target"].shape
    another = dataset[3]
    assert another["target"].shape == first["target"].shape


def test_dataset_moments_are_electrode_specific_and_train_only(tmp_path):
    import json

    import torch

    import models
    from data.ajile.core import POLICY
    from data.ajile.normalization import dataset_statistics
    from datasets.ajile_dataset import AJILE12
    from tasks.spec_pretrain import SpecPretrain

    source_root = tmp_path / "source"
    source_root.mkdir()
    audit = tmp_path / "audit"
    rng = np.random.default_rng(23)
    records = []
    assignments = {"train": [], "validation": [], "test": []}
    for number, scale, partition in (
        (0, 1.0, "train"),
        (1, 4.0, "train"),
        (2, 50.0, "test"),
    ):
        session = f"session-{number}"
        source = source_root / f"{session}.nwb"
        with h5py.File(source, "w") as saved:
            saved.create_dataset(
                "acquisition/ElectricalSeries/data",
                data=rng.normal(size=(2 * WINDOW, 2)) * scale,
            )
        valid = np.ones((2, 2), dtype=bool)
        folder = audit / "sessions" / session
        (folder / "blocks").mkdir(parents=True)
        with (folder / "blocks" / "block.npz").open("wb") as stream:
            np.savez_compressed(
                stream, reasons=np.zeros((2, 2), dtype=np.uint8)
            )
        (folder / "source.json").write_text(
            json.dumps(
                {
                    "size": source.stat().st_size,
                    "mtime_ns": source.stat().st_mtime_ns,
                }
            )
        )
        record = {
            "session": session,
            "subject": "sub-01",
            "source": source.name,
            "content_id": f"source-{number}",
            "eligible": 4,
            "metadata": {
                "samples": 2 * WINDOW,
                "channels": 2,
                "ids": [100, 101],
                "conversion": 1.0,
                "channel_conversion": None,
                "offset": 0.0,
            },
        }
        records.append(record)
        assignments[partition].append(session)
    (audit / "inventory.json").write_text(
        json.dumps(
            {
                "policy": POLICY,
                "sessions": records,
            }
        )
    )
    (audit / "splits.json").write_text(json.dumps(assignments))
    cfg = OmegaConf.load(ROOT / "conf/preprocessor/stft.yaml")
    cfg.normalizing = "dataset_zscore"
    cfg.feature_cache = str(tmp_path / "feature_cache")
    preprocessor = STFTPreprocessor(cfg)
    training = records[:2]
    fitted = dataset_statistics(source_root, audit, training, preprocessor)
    assert len(fitted["keys"]) == 2
    assert fitted["count"].tolist() == [4 * 186, 4 * 186]
    expected = {100: [], 101: []}
    for record in training:
        valid = np.ones((2, 2), dtype=bool)
        for channel, electrode_id in enumerate((100, 101)):
            for window in range(2):
                wav = waveform(
                    source_root / record["source"],
                    record["metadata"],
                    window,
                    channel,
                    valid,
                )[: OUTPUT - 1]
                expected[electrode_id].append(
                    preprocessor.unnormalized(wav).numpy()
                )
    for row, key in enumerate(fitted["keys"]):
        _, electrode_id = json.loads(str(key))
        frames = np.concatenate(expected[electrode_id])
        np.testing.assert_allclose(
            fitted["mean"][row], frames.mean(axis=0), rtol=1e-6
        )
        np.testing.assert_allclose(
            fitted["std"][row], frames.std(axis=0), rtol=1e-6
        )
    with patch(
        "data.ajile.normalization._prepare_dataset",
        side_effect=AssertionError("recomputed"),
    ):
        reused = dataset_statistics(source_root, audit, training, preprocessor)
    np.testing.assert_array_equal(reused["mean"], fitted["mean"])

    task = SpecPretrain(
        OmegaConf.load(ROOT / "conf/task/fixed_mask_pretrain.yaml")
    )
    task.dataset = AJILE12(
        OmegaConf.create({"data": str(source_root), "audit": str(audit)}),
        task_cfg=task.cfg,
        preprocessor_cfg=cfg,
    )
    model_cfg = OmegaConf.create(
        {
            "name": "masked_tf_model",
            "hidden_dim": 40,
            "input_dim": 40,
            "nhead": 4,
            "layer_dim_feedforward": 80,
            "layer_activation": "gelu",
            "encoder_num_layers": 1,
        }
    )
    model = task.build_model(model_cfg)
    assert len(model.stft_count) == 2
    assert model.stft_mean.shape == (2, 40)
    state = model.state_dict()
    restored = models.build_model(model_cfg)
    restored.load_weights(state)
    for key in ("stft_mean", "stft_std", "stft_count", "stft_keys_utf8"):
        assert torch.equal(state[key], restored.state_dict()[key])
    test_index = sum(record["eligible"] for record in training)
    example = task.dataset[test_index]
    assert example["target"].shape == (186, 40)
    test_record = records[2]
    raw = waveform(
        source_root / test_record["source"],
        test_record["metadata"],
        0,
        0,
        np.ones((2, 2), dtype=bool),
    )[: OUTPUT - 1]
    raw_stft = preprocessor.unnormalized(raw)
    normalized = restored.normalize_stft(raw_stft, "sub-01", 100)
    torch.testing.assert_close(
        normalized, example["target"], rtol=1e-6, atol=1e-6
    )
    with pytest.raises(ValueError, match="no training statistics"):
        restored.normalize_stft(raw_stft, "sub-01", 999)
    stale = models.build_model(model_cfg)
    different = dict(fitted)
    different["mean"] = fitted["mean"] + 1
    stale.set_stft_statistics(different)
    with pytest.raises(ValueError, match="statistics differ"):
        stale.load_weights(state)
    # Runner's checkpoint writer must accept the persistent identity/count
    # buffers as well as floating point moments.
    from criterions.pretrain_masked_criterion import PretrainMaskedCriterion
    from runner import Runner

    criterion = PretrainMaskedCriterion()
    criterion.build_criterion(OmegaConf.create({"alpha": 2.0}))
    runner_cfg = OmegaConf.create(
        {
            "artifact_dir": str(tmp_path / "checkpoints"),
            "device": "cpu",
            "multi_gpu": False,
            "output_tb": False,
            "quiet": True,
            "optim": "SGD",
            "lr": 0.001,
            "total_steps": 4,
            "log_step": 2,
            "checkpoint_step": 2,
            "train_batch_size": 1,
            "valid_batch_size": 1,
            "num_workers": 0,
            "shuffle": False,
            "grad_clip": 1.0,
            "scheduler": {
                "name": "ramp_up",
                "warmup": 0.5,
                "total_steps": 4,
            },
        }
    )
    runner = Runner(runner_cfg, task, model, criterion)
    try:
        runner.save_checkpoints(best_val=True)
    finally:
        runner.progress.close()
        runner.monitor.close()
    checkpoint = torch.load(
        tmp_path / "checkpoints/checkpoint_best.pth", weights_only=False
    )
    assert torch.equal(checkpoint["model"]["stft_mean"], state["stft_mean"])
