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
    wav: np.ndarray, cfg: object, extract: Callable
) -> torch.Tensor:
    """Read or atomically produce a finite (frames, frequencies) STFT tensor."""
    settings = OmegaConf.to_container(cfg, resolve=True)
    root = Path(settings.pop("feature_cache")).expanduser().resolve()
    raw = np.ascontiguousarray(wav)
    if raw.ndim != 1 or raw.size == 0 or not np.isfinite(raw).all():
        raise ValueError(
            "STFT caching requires a finite, nonempty 1D waveform."
        )
    key = digest(
        {
            "revision": STFT_CACHE_REVISION,
            "settings": settings,
            "waveform": hashlib.sha256(raw.tobytes()).hexdigest(),
            "dtype": str(raw.dtype),
            "shape": list(raw.shape),
        }
    )
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
