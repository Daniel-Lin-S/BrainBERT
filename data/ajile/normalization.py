"""Cache AJILE per-electrode STFT moments by recording or dataset.

Inputs are a completed audit's eligibility blocks and the source NWB. Only
eligible five-second electrode windows in the recording's assigned split
contribute. The exact waveform trim and unnormalized STFT used by training
are applied. Cache files under the configured feature root contain:
``count`` (channels,): contributing STFT frames per electrode;
``mean`` and ``std`` (channels, frequencies): population moments;
``identity`` and ``partition``: deterministic provenance. One partial file
per electrode permits restart without revisiting completed electrodes.
Dataset-level mode additionally pools each subject/electrode across training
recordings and stores a small subject/electrode-to-moment table. Validation
and test sources never contribute to that table. No waveform or spectrogram
corpus is stored.
"""

import fcntl
import json
from pathlib import Path

import h5py
import numpy as np
from omegaconf import OmegaConf

from data.ajile.core import OUTPUT, POLICY, waveform
from util.experiments.storage import atomic_write, digest

REVISION = "ajile-stft-recording-moments-v1"


def eligibility(audit: Path, record: dict) -> np.ndarray:
    """Read audited eligibility, shape (windows, electrodes), exactly."""
    folder = audit / "sessions" / record["session"] / "blocks"
    blocks = []
    for path in sorted(folder.glob("*.npz")):
        with np.load(path, allow_pickle=False) as saved:
            blocks.append(saved["reasons"] == 0)
    if not blocks:
        raise ValueError(f"Missing eligibility blocks: {folder.resolve()}")
    valid = np.concatenate(blocks)
    expected = (
        record["metadata"]["samples"] // 2500,
        record["metadata"]["channels"],
    )
    if valid.shape != expected or int(valid.sum()) != record["eligible"]:
        raise ValueError(
            "AJILE eligibility shape or count differs from audit."
        )
    return valid


def _write_npz(path: Path, fields: dict) -> None:
    """Atomically save a small cache and verify its exact contents."""

    def write(candidate: Path) -> None:
        with candidate.open("wb") as stream:
            np.savez_compressed(stream, **fields)

    def audit(candidate: Path) -> None:
        with np.load(candidate, allow_pickle=False) as saved:
            if set(saved.files) != set(fields):
                raise ValueError("Recording-statistics cache fields differ.")
            for name, expected in fields.items():
                equal_nan = np.issubdtype(expected.dtype, np.floating)
                if not np.array_equal(
                    saved[name], expected, equal_nan=equal_nan
                ):
                    raise ValueError(
                        f"Recording-statistics cache field differs: {name}"
                    )

    atomic_write(path, write, audit)


def _channel_moments(
    source: h5py.File,
    path: Path,
    record: dict,
    valid: np.ndarray,
    channel: int,
    preprocessor: object,
) -> tuple[int, np.ndarray, np.ndarray]:
    """Pool unmasked STFT frames of eligible windows for one electrode."""
    frequencies = len(preprocessor.frequency_axis())
    mean = np.zeros(frequencies, dtype=np.float64)
    m2 = np.zeros(frequencies, dtype=np.float64)
    count = 0
    for index in np.flatnonzero(valid[:, channel]):
        wav = waveform(
            path, record["metadata"], int(index), channel, valid, source
        )
        spectrum = (
            preprocessor.unnormalized(wav[:-1])
            .numpy()
            .astype(np.float64, copy=False)
        )
        n = spectrum.shape[0]
        chunk_mean = spectrum.mean(axis=0)
        chunk_m2 = ((spectrum - chunk_mean) ** 2).sum(axis=0)
        delta = chunk_mean - mean
        total = count + n
        mean += delta * n / total
        m2 += chunk_m2 + delta**2 * count * n / total
        count = total
    return count, mean, m2


def _cache_key(record: dict, partition: str, preprocessor: object) -> str:
    """Identify source, exclusions, split assignment and STFT numerics."""
    settings = OmegaConf.to_container(preprocessor.cfg, resolve=True)
    settings.pop("feature_cache", None)
    return digest(
        {
            "revision": REVISION,
            "policy": POLICY,
            "source": record["content_id"],
            "partition": partition,
            "preprocessor": settings,
        }
    )


def recording_statistics(
    source_root: Path,
    record: dict,
    partition: str,
    valid: np.ndarray,
    preprocessor: object,
) -> dict:
    """Return cached per-electrode moments, computing missing parts once.

    Parameters
    ----------
    source_root : pathlib.Path
        Local NWB root.
    record : dict
        Audited recording identity and metadata.
    partition : str
        Assigned train, validation, or test split.
    valid : numpy.ndarray
        Eligible electrode windows, shape (windows, electrodes).
    preprocessor : STFTPreprocessor
        Configured extractor using recording_zscore.

    Returns
    -------
    dict
        Mean and standard deviation arrays (electrodes, frequencies),
        per-electrode frame counts, and normalization identity.
    """
    if preprocessor.cfg.normalizing not in {
        "recording_zscore",
        "dataset_zscore",
    }:
        raise ValueError("Expected fitted STFT normalization mode.")
    if partition not in {"train", "validation", "test"}:
        raise ValueError(f"Unknown AJILE split: {partition!r}")
    cache_root = preprocessor.cfg.get("feature_cache")
    if not cache_root:
        raise ValueError(
            "Recording-wise AJILE STFT requires a feature_cache root."
        )
    key = _cache_key(record, partition, preprocessor)
    frames = preprocessor.unnormalized(
        np.zeros(OUTPUT - 1, dtype=np.float32)
    ).shape[0]
    folder = Path(cache_root).expanduser().resolve() / "recording_stats" / key
    folder.mkdir(parents=True, exist_ok=True)
    complete = folder / "complete.npz"
    with (folder / "prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not complete.exists():
            _prepare(
                source_root / record["source"],
                record,
                partition,
                valid,
                preprocessor,
                folder,
                key,
                frames,
            )
        with np.load(complete, allow_pickle=False) as saved:
            if (
                str(saved["identity"]) != key
                or str(saved["partition"]) != partition
            ):
                raise ValueError(
                    "Recording-statistics cache identity differs."
                )
            result = {
                name: saved[name].copy() for name in ("count", "mean", "std")
            }
    if (
        result["mean"].shape
        != (valid.shape[1], len(preprocessor.frequency_axis()))
        or result["std"].shape != result["mean"].shape
    ):
        raise ValueError("Recording-statistics cache shape differs.")
    expected_counts = valid.sum(axis=0) * frames
    active = expected_counts > 0
    if not np.array_equal(result["count"], expected_counts):
        raise ValueError("Recording statistics used a different window set.")
    if (
        np.any(result["count"][active] <= 0)
        or not np.isfinite(result["mean"][active]).all()
        or not np.isfinite(result["std"][active]).all()
        or np.any(result["std"][active] < 0)
        or (
            preprocessor.cfg.normalizing == "recording_zscore"
            and np.any(result["std"][active] == 0)
        )
    ):
        raise ValueError("Recording-statistics cache has invalid moments.")
    result["identity"] = key
    return result


def _prepare(
    source_path: Path,
    record: dict,
    partition: str,
    valid: np.ndarray,
    preprocessor: object,
    folder: Path,
    key: str,
    frames: int,
) -> None:
    """Resume at electrode granularity and commit one complete cache."""
    channels = valid.shape[1]
    frequencies = len(preprocessor.frequency_axis())
    counts = np.zeros(channels, dtype=np.int64)
    means = np.full((channels, frequencies), np.nan, dtype=np.float64)
    stds = np.full_like(means, np.nan)
    with h5py.File(source_path, "r") as source:
        for channel in range(channels):
            indices = np.flatnonzero(valid[:, channel])
            if not len(indices):
                continue
            partial = folder / f"electrode_{channel:03d}.npz"
            if not partial.exists():
                count, mean, m2 = _channel_moments(
                    source, source_path, record, valid, channel, preprocessor
                )
                _write_npz(
                    partial,
                    {
                        "identity": np.array(key),
                        "channel": np.array(channel),
                        "windows": np.array(len(indices)),
                        "count": np.array(count),
                        "mean": mean,
                        "m2": m2,
                    },
                )
            with np.load(partial, allow_pickle=False) as saved:
                if (
                    str(saved["identity"]) != key
                    or int(saved["channel"]) != channel
                    or int(saved["windows"]) != len(indices)
                ):
                    raise ValueError(
                        "Electrode-statistics cache identity differs."
                    )
                count = int(saved["count"])
                if count != len(indices) * frames:
                    raise ValueError(
                        "Electrode statistics used a different window set."
                    )
                mean = saved["mean"].copy()
                m2 = saved["m2"].copy()
            if (
                count <= 0
                or mean.shape != (frequencies,)
                or m2.shape != mean.shape
                or not np.isfinite(mean).all()
                or not np.isfinite(m2).all()
                or np.any(m2 < 0)
            ):
                raise ValueError(
                    "Electrode-statistics cache has invalid moments."
                )
            counts[channel] = count
            means[channel] = mean
            stds[channel] = np.sqrt(m2 / count)
    _write_npz(
        folder / "complete.npz",
        {
            "identity": np.array(key),
            "partition": np.array(partition),
            "count": counts,
            "mean": means,
            "std": stds,
        },
    )


def dataset_statistics(
    source_root: Path,
    audit: Path,
    records: list,
    preprocessor: object,
) -> dict:
    """Fit electrode/frequency moments from training recordings only.

    Parameters
    ----------
    source_root, audit : pathlib.Path
        Local NWB and completed audit roots.
    records : list
        Audited training-session dictionaries only; no validation/test rows.
    preprocessor : STFTPreprocessor
        Configured dataset_zscore extractor.

    Returns
    -------
    dict
        Sorted subject/electrode keys, (electrodes, frequencies) means/stds,
        per-electrode frame counts, and a deterministic cache identity.
    """
    if preprocessor.cfg.normalizing != "dataset_zscore":
        raise ValueError("Expected dataset_zscore STFT configuration.")
    if not records or not preprocessor.cfg.get("feature_cache"):
        raise ValueError(
            "Dataset-level STFT needs training recordings and a cache root."
        )
    settings = OmegaConf.to_container(preprocessor.cfg, resolve=True)
    settings.pop("feature_cache", None)
    sources = sorted(
        (record["session"], record["content_id"]) for record in records
    )
    if len({session for session, _ in sources}) != len(records):
        raise ValueError("Duplicate training recording identity.")
    key = digest(
        {
            "revision": REVISION,
            "policy": POLICY,
            "training_sources": sources,
            "preprocessor": settings,
            "pooling": "subject-electrode-frequency-training-only",
        }
    )
    root = Path(preprocessor.cfg.feature_cache).expanduser().resolve()
    folder = root / "dataset_stats" / key
    folder.mkdir(parents=True, exist_ok=True)
    complete = folder / "complete.npz"
    with (folder / "prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not complete.exists():
            _prepare_dataset(
                source_root, audit, records, preprocessor, complete, key
            )
        with np.load(complete, allow_pickle=False) as saved:
            if str(saved["identity"]) != key:
                raise ValueError("Dataset-statistics cache identity differs.")
            result = {
                name: saved[name].copy()
                for name in ("keys", "count", "mean", "std")
            }
    frequencies = len(preprocessor.frequency_axis())
    if (
        result["mean"].shape != (len(result["keys"]), frequencies)
        or result["std"].shape != result["mean"].shape
        or result["count"].shape != (len(result["keys"]),)
        or np.any(result["count"] <= 0)
        or not np.isfinite(result["mean"]).all()
        or not np.isfinite(result["std"]).all()
        or np.any(result["std"] <= 0)
    ):
        raise ValueError("Dataset-statistics cache has invalid moments.")
    result["identity"] = key
    return result


def _prepare_dataset(
    source_root: Path,
    audit: Path,
    records: list,
    preprocessor: object,
    complete: Path,
    key: str,
) -> None:
    """Pool per-recording partials by stable subject/electrode identity."""
    pooled = {}
    frequencies = len(preprocessor.frequency_axis())
    for record in sorted(records, key=lambda item: item["session"]):
        valid = eligibility(audit, record)
        statistics = recording_statistics(
            source_root, record, "train", valid, preprocessor
        )
        ids = record["metadata"]["ids"]
        if len(ids) != valid.shape[1] or len(ids) != len(set(ids)):
            raise ValueError("AJILE electrode IDs are missing or duplicate.")
        for channel, electrode_id in enumerate(ids):
            n = int(statistics["count"][channel])
            if not n:
                continue
            label = json.dumps(
                [record["subject"], electrode_id], separators=(",", ":")
            )
            chunk_mean = statistics["mean"][channel]
            chunk_m2 = statistics["std"][channel] ** 2 * n
            if label not in pooled:
                pooled[label] = [n, chunk_mean.copy(), chunk_m2.copy()]
                continue
            count, mean, m2 = pooled[label]
            total = count + n
            delta = chunk_mean - mean
            mean += delta * n / total
            m2 += chunk_m2 + delta**2 * count * n / total
            pooled[label] = [total, mean, m2]
    if not pooled:
        raise ValueError("Training recordings have no eligible electrodes.")
    labels = sorted(pooled)
    counts = np.array([pooled[label][0] for label in labels], dtype=np.int64)
    means = np.stack([pooled[label][1] for label in labels])
    stds = np.stack(
        [np.sqrt(pooled[label][2] / pooled[label][0]) for label in labels]
    )
    if means.shape[1] != frequencies:
        raise ValueError("Training STFT frequency dimensions differ.")
    _write_npz(
        complete,
        {
            "identity": np.array(key),
            "keys": np.array(labels),
            "count": counts,
            "mean": means,
            "std": stds,
        },
    )


def main() -> None:
    """Prepare recording or training-dataset statistics from an audit.

    Command arguments are local paths. Output is the resumable cache tree
    documented above, with one complete file per recording and small
    electrode partials during preparation.
    """
    import argparse

    from preprocessors import STFTPreprocessor

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--feature-cache", required=True, type=Path)
    parser.add_argument("--preprocessor", required=True, type=Path)
    parser.add_argument(
        "--normalizing", choices=("recording_zscore", "dataset_zscore")
    )
    args = parser.parse_args()
    cfg = OmegaConf.load(args.preprocessor)
    if args.normalizing is not None:
        cfg.normalizing = args.normalizing
    cfg.feature_cache = str(args.feature_cache.resolve())
    extractor = STFTPreprocessor(cfg)
    inventory = json.loads((args.audit / "inventory.json").read_text())
    if inventory["policy"] != POLICY:
        raise ValueError("AJILE audit preprocessing revision differs.")
    splits = json.loads((args.audit / "splits.json").read_text())
    partition = {
        session: name
        for name, sessions in splits.items()
        for session in sessions
    }
    records = inventory["sessions"]
    if set(partition) != {record["session"] for record in records}:
        raise ValueError("AJILE split sessions do not match the audit.")
    for record in records:
        source = args.source_root / record["source"]
        saved = json.loads(
            (
                args.audit / "sessions" / record["session"] / "source.json"
            ).read_text()
        )
        if (
            source.stat().st_size != saved["size"]
            or source.stat().st_mtime_ns != saved["mtime_ns"]
        ):
            raise ValueError(f"Audited source changed: {source.resolve()}")
        if cfg.normalizing == "recording_zscore":
            valid = eligibility(args.audit, record)
            recording_statistics(
                args.source_root,
                record,
                partition[record["session"]],
                valid,
                extractor,
            )
            print(
                f"Prepared recording statistics: {record['session']}",
                flush=True,
            )
    if cfg.normalizing == "dataset_zscore":
        training = [
            record
            for record in records
            if partition[record["session"]] == "train"
        ]
        dataset_statistics(args.source_root, args.audit, training, extractor)
        print("Prepared training-only dataset statistics", flush=True)


if __name__ == "__main__":
    main()
