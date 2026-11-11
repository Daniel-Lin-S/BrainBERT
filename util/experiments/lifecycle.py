"""Shared fit identity, locking, completion, and campaign references.

``fits/<hash>/`` contains scientific.json, provenance.json, checkpoints/,
results.json, and complete.json (artifact checksums, written last).
``campaigns/<name>/fits/<hash>.json`` references that shared fit without copying
results. Failed attempts move to quarantine/<hash>/<invocation>/ on retry.
Small logs live at logs/<campaign>/<invocation>/{status,training,errors}/.
Scientific identity excludes placement, device, workers, and logging options.
"""

import fcntl
import json
import logging
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

from omegaconf import OmegaConf

from .storage import digest, file_digest, write_json

SCIENTIFIC_RUNNER = {
    "lr",
    "optim",
    "train_batch_size",
    "valid_batch_size",
    "shuffle",
    "total_steps",
    "grad_clip",
    "scheduler",
    "log_step",
    "checkpoint_step",
}
OPERATIONAL_RUNNER = {
    "device",
    "multi_gpu",
    "num_workers",
    "output_tb",
    "artifact_dir",
    "quiet",
    "gpu_memory_fraction",
    "cpu_threads",
}
DATA_PLACEMENT = {
    "data",
    "cached_features",
    "feature_cache",
    "pinned_test",
    "audit",
}


def scientific_config(cfg: object, dataset: object, splits: dict) -> dict:
    """Collect scientific settings and verified source/split content."""
    runner = OmegaConf.to_container(cfg.exp.runner, resolve=True)
    unknown = set(runner) - SCIENTIFIC_RUNNER - OPERATIONAL_RUNNER
    if unknown:
        raise ValueError(f"Unclassified runner identity fields: {unknown}")
    data = OmegaConf.to_container(cfg.data, resolve=True)
    model = OmegaConf.to_container(cfg.model, resolve=True)
    if "upstream_ckpt" in model:
        model["upstream_checkpoint_sha256"] = file_digest(
            Path(model.pop("upstream_ckpt")).expanduser().resolve()
        )
    return {
        "protocol": OmegaConf.to_container(cfg.protocol, resolve=True),
        "model": model,
        "task": OmegaConf.to_container(cfg.task, resolve=True),
        "criterion": OmegaConf.to_container(cfg.criterion, resolve=True),
        "preprocessor": {
            k: v
            for k, v in OmegaConf.to_container(
                cfg.preprocessor, resolve=True
            ).items()
            if k != "feature_cache"
        },
        "data": {k: v for k, v in data.items() if k not in DATA_PLACEMENT},
        "inventory": dataset.inventory,
        "splits": splits,
        "runner": {k: v for k, v in runner.items() if k in SCIENTIFIC_RUNNER},
    }


class FitRun:
    """Own a shared fit lock and the small records for one invocation."""

    def __init__(
        self,
        artifact_root: str,
        log_root: str,
        campaign: str,
        scientific: dict,
    ) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", campaign):
            raise ValueError(
                "Campaign must contain only letters, digits, _ or -."
            )
        self.root = Path(artifact_root).expanduser().resolve()
        self.fit_id = digest(scientific)
        self.scientific = scientific
        self.path = self.root / "fits" / self.fit_id
        self.invocation = (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            + "-"
            + uuid.uuid4().hex[:8]
        )
        self.logs = Path(log_root).resolve() / campaign / self.invocation
        self.reference = (
            self.root / "campaigns" / campaign / "fits" / f"{self.fit_id}.json"
        )
        self.lock = None
        self.handler = None
        self.reused = False

    def __enter__(self) -> "FitRun":
        lock_path = self.root / "locks" / f"{self.fit_id}.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = lock_path.open("a")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.lock.close()
            raise RuntimeError(f"Fit is already running: {self.fit_id}")
        try:
            self._prepare()
        except BaseException:
            self.lock.close()
            raise
        return self

    def _prepare(self) -> None:
        complete = self.path / "complete.json"
        if complete.exists():
            record = json.loads(complete.read_text())
            if record["fit_id"] != self.fit_id:
                raise ValueError("Completed fit identity does not match.")
            required = {
                "scientific.json",
                "results.json",
                "checkpoints/checkpoint_best.pth",
                "checkpoints/checkpoint_last.pth",
            }
            additional = set(record["artifacts"]) - required
            if not required.issubset(record["artifacts"]) or any(
                not name.startswith("monitoring/") for name in additional
            ):
                raise ValueError("Completion record has missing artifacts.")
            if (
                json.loads((self.path / "scientific.json").read_text())
                != self.scientific
            ):
                raise ValueError("Completed scientific configuration differs.")
            for name, expected in record["artifacts"].items():
                candidate = (self.path / name).resolve()
                if not candidate.is_relative_to(self.path.resolve()):
                    raise ValueError("Completion artifact escapes fit root.")
                if file_digest(self.path / name) != expected:
                    raise ValueError(
                        f"Completed artifact checksum failed: {name}"
                    )
            self.reused = True
        elif self.path.exists():
            quarantine = (
                self.root / "quarantine" / self.fit_id / self.invocation
            )
            quarantine.parent.mkdir(parents=True, exist_ok=True)
            self.path.rename(quarantine)
        if not self.reused:
            write_json(self.path / "scientific.json", self.scientific)
        for folder in ("status", "training", "errors"):
            (self.logs / folder).mkdir(parents=True, exist_ok=True)
        self.handler = logging.FileHandler(self.logs / "training" / "run.log")
        self.handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        logging.getLogger().addHandler(self.handler)
        logging.getLogger().setLevel(logging.INFO)
        self.status("reused" if self.reused else "running")
        write_json(self.reference, {"fit_id": self.fit_id})

    def status(self, phase: str, **details: object) -> None:
        """Replace the compact status document for this invocation."""
        write_json(
            self.logs / "status" / "status.json",
            {
                "fit_id": self.fit_id,
                "phase": phase,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                **details,
            },
        )

    def complete(self, metrics: dict) -> None:
        """Commit metrics and checkpoint checksums before completion."""
        if self.reused or (self.path / "complete.json").exists():
            raise RuntimeError("Completed fits are immutable.")
        names = [
            "checkpoints/checkpoint_best.pth",
            "checkpoints/checkpoint_last.pth",
            "scientific.json",
        ]
        monitor_root = self.path / "monitoring"
        if monitor_root.exists():
            names.extend(
                str(path.relative_to(self.path))
                for path in sorted(monitor_root.rglob("*"))
                if path.is_file()
            )
        checksums = {name: file_digest(self.path / name) for name in names}
        write_json(self.path / "results.json", metrics)
        checksums["results.json"] = file_digest(self.path / "results.json")
        write_json(
            self.path / "complete.json",
            {
                "fit_id": self.fit_id,
                "artifacts": checksums,
            },
        )
        self.status("complete")

    def __exit__(self, kind: object, error: object, traceback: object) -> None:
        try:
            self._record_failure(kind, error)
        finally:
            if self.handler is not None:
                logging.getLogger().removeHandler(self.handler)
                self.handler.close()
            self.lock.close()

    def _record_failure(self, kind: object, error: object) -> None:
        if error is not None:
            write_json(
                self.logs / "errors" / "failure.json",
                {
                    "type": kind.__name__,
                    "message": str(error),
                },
            )
            self.status("failed")
