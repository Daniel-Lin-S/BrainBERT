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
