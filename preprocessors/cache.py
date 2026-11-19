"""Content-addressed, finite STFT tensors under a configured feature root.

Each .npy file contains one float32 (frames, frequencies) array. Cache keys
include waveform bytes, STFT configuration, and an explicit implementation
revision; masks and model settings never enter reusable feature caches.
"""

import hashlib
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from omegaconf import OmegaConf

from util.experiments.storage import atomic_write, digest

STFT_CACHE_REVISION = "brainbert-stft-v1"


def cached_stft(
    wav: np.ndarray,
    cfg: object,
    extract: Callable,
    normalization_identity: str | None = None,
) -> torch.Tensor:
    """Read or atomically produce a finite (frames, frequencies) STFT tensor."""
    settings = OmegaConf.to_container(cfg, resolve=True)
    root = Path(settings.pop("feature_cache")).expanduser().resolve()
    raw = np.ascontiguousarray(wav)
    if raw.ndim != 1 or raw.size == 0 or not np.isfinite(raw).all():
        raise ValueError(
            "STFT caching requires a finite, nonempty 1D waveform."
        )
    key_data = {
        "revision": STFT_CACHE_REVISION,
        "settings": settings,
        "waveform": hashlib.sha256(raw.tobytes()).hexdigest(),
        "dtype": str(raw.dtype),
        "shape": list(raw.shape),
    }
    if normalization_identity is not None:
        key_data["normalization_identity"] = normalization_identity
    key = digest(key_data)
    path = root / "stft" / key[:2] / f"{key}.npy"

    def audit(candidate: Path) -> None:
        values = np.load(candidate, allow_pickle=False)
        if (
            values.ndim != 2
            or 0 in values.shape
            or not np.isfinite(values).all()
        ):
            raise ValueError(
                f"Invalid cached STFT tensor: {candidate.resolve()}"
            )

    if not path.exists():
        values = extract().detach().cpu().numpy()

        def write(candidate: Path) -> None:
            with candidate.open("wb") as stream:
                np.save(stream, values, allow_pickle=False)

        atomic_write(path, write, audit)
    audit(path)
    return torch.from_numpy(np.load(path, allow_pickle=False))


def raw_stft_path(
    root: Path, namespace: str, source_id: str,
    position: int, cfg: object
) -> Path:
    """Address raw STFT by source identity and sample position."""
    settings = OmegaConf.to_container(cfg, resolve=True)
    settings.pop("feature_cache", None)
    settings.pop("normalizing", None)
    key = digest({
        "revision": STFT_CACHE_REVISION,
        "settings": settings,
        "source": source_id,
    })
    return root / namespace / key / f"{position}.npy"


def cached_raw_stft(
    root: Path,
    namespace: str,
    source_id: str,
    position: int,
    cfg: object,
    extract: Callable,
) -> torch.Tensor:
    """Load or create a finite raw STFT, shape (frames, bins)."""
    path = raw_stft_path(
        root, namespace, source_id, position, cfg
    )
    try:
        values = np.load(path, allow_pickle=False)
    except FileNotFoundError:
        values = extract().detach().cpu().numpy()
        if (
            values.ndim != 2
            or values.shape[1] != cfg.freq_channel_cutoff
            or not np.isfinite(values).all()
        ):
            raise ValueError(
                "Raw STFT must be finite and have the configured bins."
            )

        def write(candidate: Path) -> None:
            with candidate.open("wb") as stream:
                np.save(stream, values, allow_pickle=False)
        atomic_write(path, write, lambda candidate: np.load(
            candidate, allow_pickle=False
        ))
        values = np.load(path, allow_pickle=False)
    if (
        values.ndim != 2
        or values.shape[1] != cfg.freq_channel_cutoff
        or not np.isfinite(values).all()
    ):
        raise ValueError(f"Invalid raw STFT cache: {path.resolve()}")
    return torch.from_numpy(values)


def cached_raw_positions(
    root: Path,
    namespace: str,
    source_id: str,
    cfg: object,
) -> set[int]:
    """List cached sample positions within one raw STFT source.

    Parameters
    ----------
    root : pathlib.Path
        Shared feature-cache root.
    namespace : str
        Dataset cache namespace.
    source_id : str
        Audited source-content identity.
    cfg : object
        STFT settings used to address the cache.

    Returns
    -------
    set[int]
        Nonnegative sample positions with cache files.
    """
    folder = raw_stft_path(
        root, namespace, source_id, 0, cfg
    ).parent
    if not folder.exists():
        return set()
    positions = set()
    for path in folder.glob("*.npy"):
        try:
            position = int(path.stem)
        except ValueError as exc:
            raise ValueError(
                f"Invalid raw STFT cache filename: {path.resolve()}"
            ) from exc
        if position < 0:
            raise ValueError(
                f"Raw STFT cache position must be nonnegative: "
                f"{path.resolve()}"
            )
        positions.add(position)
    return positions
