"""AJILE NWB metadata and bounded single-electrode resampling.

NWB data are (native samples, electrodes). Windows are recording-aligned
five-second spans. Released notch filtering and group-median referencing are
preserved. Metadata uses ElectricalSeries electrode-table mappings.
"""

from pathlib import Path
import h5py
import numpy as np
from scipy.signal import resample_poly

RATE = 500
WINDOW = 2500
UP = 512
DOWN = 125
OUTPUT = 10240
HALO = 125
BLOCK_WINDOWS = 32
POLICY = "ajile-released-reference-5s-2048-v1"
REASONS = {"bad_electrode": 1, "blocklist": 2, "nonfinite": 4, "constant": 8}


def metadata(path: Path) -> dict:
    """Read recording, mapped electrode, and interval metadata without signals."""
    with h5py.File(path, "r") as f:
        s = f["acquisition/ElectricalSeries"]
        rate = float(s["starting_time"].attrs["rate"])
        if rate != RATE or len(s["data"].shape) != 2:
            raise ValueError(f"Expected 500 Hz 2D data: {path.resolve()}")
        table = f["general/extracellular_ephys/electrodes"]
        mapping = s["electrodes"][:].astype(int)
        if (
            len(mapping) != s["data"].shape[1]
            or len(set(mapping)) != len(mapping)
            or np.any(mapping < 0)
            or np.any(mapping >= len(table["id"]))
        ):
            raise ValueError(f"Invalid electrode mapping: {path.resolve()}")
        epochs = f["intervals/epochs"]
        groups = table["group_name"].asstr()[:][mapping].tolist()
        if any(not group for group in groups):
            raise ValueError(f"Missing reference group: {path.resolve()}")
        return {
            "samples": int(s["data"].shape[0]),
            "channels": len(mapping),
            "start": float(s["starting_time"][()]),
            "conversion": float(s["data"].attrs["conversion"]),
            "offset": float(s["data"].attrs.get("offset", 0)),
            "channel_conversion": (
                s["channel_conversion"][:].tolist()
                if "channel_conversion" in s
                else None
            ),
            "ids": table["id"][:][mapping].tolist(),
            "groups": groups,
            "good": table["good"][:][mapping].astype(bool).tolist(),
            "epochs": [
                list(row)
                for row in zip(
                    epochs["start_time"][:].tolist(),
                    epochs["stop_time"][:].tolist(),
                    epochs["labels"].asstr()[:].tolist(),
                )
            ],
        }


def annotations(meta: dict) -> tuple:
    """Return blocked-window flags and overlapping activity sets per window."""
    count = meta["samples"] // WINDOW
    blocked = np.zeros(count, dtype=bool)
    activities = [set() for _ in range(count)]
    for start, stop, label in meta["epochs"]:
        if not np.isfinite([start, stop]).all() or stop < start:
            raise ValueError("Invalid AJILE interval boundaries.")
        lo = max(0, int(np.floor((start - meta["start"]) / 5)))
        hi = min(count, int(np.ceil((stop - meta["start"]) / 5)))
        if label.startswith("Blocklist"):
            blocked[lo:hi] = True
        else:
            tags = {x.strip() for x in label.split(",") if x.strip()}
            for index in range(lo, hi):
                activities[index].update(tags)
    return blocked, activities


def waveform(
    path: Path, meta: dict, index: int, channel: int, valid: np.ndarray
) -> np.ndarray:
    """Resample one electrode with context confined to its valid window run.

    Returns
    -------
    numpy.ndarray
        Finite float32 waveform of shape (10240,). Halo alignment preserves
        equivalence to resampling the complete contiguous eligible span.
    """
    if not valid[index, channel]:
        raise ValueError("Requested an excluded AJILE electrode window.")
    start = index * WINDOW
    stop = start + WINDOW
    left = HALO if index > 0 and valid[index - 1, channel] else 0
    right = HALO if index + 1 < len(valid) and valid[index + 1, channel] else 0
    with h5py.File(path, "r") as f:
        values = f["acquisition/ElectricalSeries/data"][
            start - left : stop + right, channel
        ].astype(np.float64)
    factor = meta["conversion"]
    if meta["channel_conversion"] is not None:
        factor *= meta["channel_conversion"][channel]
    values = values * factor + meta["offset"]
    output = resample_poly(values, UP, DOWN)
    offset = left * UP // DOWN
    output = output[offset : offset + OUTPUT].astype("float32")
    if output.shape != (OUTPUT,) or not np.isfinite(output).all():
        raise ValueError(
            "AJILE resampling did not produce 10240 finite samples."
        )
    return output
