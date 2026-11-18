"""Durable JSON and checkpoint writes for shared fits and feature caches.

Writers stage and audit a sibling temporary file. Existing files are backed up
before replacement; backups are removed only after the installed file passes
its audit. JSON documents must contain finite, JSON-compatible values.
"""

import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Callable

BUFFER_BYTES = 1024 * 1024


def digest(value: object) -> str:
    """Hash finite JSON-compatible scientific data with stable SHA256."""
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def file_digest(path: Path) -> str:
    """Hash file bytes without loading a complete recording into memory."""
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(BUFFER_BYTES), b""):
            result.update(chunk)
    return result.hexdigest()


def atomic_write(path: Path, writer: Callable, audit: Callable) -> None:
    """Write, audit, and replace a file while preserving its prior version.

    Parameters
    ----------
    path : pathlib.Path
        Destination file, on the same filesystem as its staging files.
    writer, audit : Callable
        Functions accepting a Path; audit must raise on invalid output.
    """
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    staging = path.with_name(f".{path.name}.{token}.tmp")
    backup = path.with_name(f".{path.name}.{token}.bak")
    installed = False
    try:
        if path.exists():
            os.link(path, backup)
        writer(staging)
        with staging.open("rb") as stream:
            os.fsync(stream.fileno())
        audit(staging)
        os.replace(staging, path)
        installed = True
        audit(path)
        descriptor = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except BaseException:
        if installed and backup.exists():
            os.replace(backup, path)
        elif installed:
            path.unlink()
        raise
    finally:
        staging.unlink(missing_ok=True)
    backup.unlink(missing_ok=True)


def write_json(path: Path, value: object) -> None:
    """Persist finite JSON and verify exact round-trip contents."""
    text = json.dumps(value, sort_keys=True, indent=2, allow_nan=False)

    def audit(candidate: Path) -> None:
        if json.loads(candidate.read_text()) != value:
            raise ValueError(f"JSON audit failed: {candidate.resolve()}")

    atomic_write(path, lambda p: p.write_text(text + "\n"), audit)


def export_model_checkpoint(source: Path, destination: Path) -> None:
    """Export selected CPU weights without optimiser-dependent pickle objects.

    Parameters
    ----------
    source : pathlib.Path
        Final validation-selected checkpoint from a completed fit.
    destination : pathlib.Path
        Shared derived encoder artifact, outside the immutable fit results.
        Existing exports must match the source checksum and are reused.
    """
    import torch
    from omegaconf import OmegaConf

    source, destination = Path(source).resolve(), Path(destination).resolve()
    checksum = file_digest(source)
    if destination.exists():
        existing = torch.load(
            destination, map_location="cpu", weights_only=True
        )
        if existing.get("source_sha256") != checksum:
            raise ValueError("Existing encoder export has a different source.")
        return
    state = torch.load(source, map_location="cpu", weights_only=False)
    configuration = state["model_cfg"]
    if OmegaConf.is_config(configuration):
        configuration = OmegaConf.to_container(configuration, resolve=True)
    weights = state["model"]
    if not weights or any(
        not torch.isfinite(t).all() for t in weights.values()
    ):
        raise ValueError("Encoder export requires nonempty finite weights.")
    result = {"model": weights, "model_cfg": configuration,
              "completed_steps": state["completed_steps"],
              "source_sha256": checksum}

    def audit(path: Path) -> None:
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved["source_sha256"] != checksum:
            raise ValueError("Encoder export source checksum differs.")
        if set(saved["model"]) != set(weights) or any(
            not torch.equal(saved["model"][key], value)
            for key, value in weights.items()
        ):
            raise ValueError("Encoder export differs from selected weights.")

    atomic_write(destination, lambda path: torch.save(result, path), audit)


def publish_training_completion(checkpoints: Path, total_steps: int) -> None:
    """Publish final checkpoint readiness independently of test diagnostics.

    Parameters
    ----------
    checkpoints : pathlib.Path
        Fit checkpoint directory containing saved last and selected states.
    total_steps : int
        Configured optimiser-step ceiling. Early stopping is read from the
        last checkpoint. An unfinished training run is rejected.

    Notes
    -----
    Writes training_complete.json beside the checkpoint directory, with
    final training progress and the selected checkpoint's checksum. It
    does not assert that testing or report generation has completed.
    """
    import torch

    checkpoints = Path(checkpoints).resolve()
    last = torch.load(
        checkpoints / "checkpoint_last.pth", map_location="cpu",
        weights_only=False,
    )
    completed = int(last["completed_steps"])
    stopped = bool(last.get("stopped_early", False))
    if completed != total_steps and not stopped:
        raise ValueError(
            f"Training is unfinished: {completed}/{total_steps} steps."
        )
    best_path = checkpoints / "checkpoint_best.pth"
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    selected = int(best["completed_steps"])
    if selected != last["best_step"] or not 0 < selected <= completed:
        raise ValueError("Selected checkpoint does not match final selection.")
    record = {
        "fit_id": checkpoints.parent.name,
        "completed_steps": completed,
        "total_steps": total_steps,
        "stopped_early": stopped,
        "selected_step": selected,
        "checkpoint": "checkpoints/checkpoint_best.pth",
        "checkpoint_sha256": file_digest(best_path),
    }
    destination = checkpoints.parent / "training_complete.json"
    if destination.exists():
        if json.loads(destination.read_text()) != record:
            raise ValueError("Published final checkpoint selection differs.")
        return
    write_json(destination, record)
