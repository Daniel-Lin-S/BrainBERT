"""Shared fit identity, locking, completion, and campaign references.

``fits/<hash>/`` contains scientific.json, provenance.json, checkpoints/,
results.json, and complete.json (artifact checksums, written last).
``campaigns/<name>/fits/<hash>.json`` references that shared fit without copying
results. Valid incomplete checkpoints resume; invalid attempts move to
quarantine/<hash>/<invocation>/ on retry.
Small logs live at logs/<campaign>/<invocation>/{status,training,errors}/.
Scientific identity excludes placement, device, workers, and logging options.
"""

import copy
import fcntl
import json
import logging
import math
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

from omegaconf import OmegaConf

from .storage import digest, file_digest, write_json

REFERENCE_BATCH_SIZE = 256

SCIENTIFIC_RUNNER = {
    "early_stopping_patience", "seed", "parent_checkpoint_sha256",
    "align_validation_steps",
    "lr",
    "optim",
    "train_batch_size",
    "valid_batch_size",
    "shuffle",
    "total_steps",
    "epochs",
    "epoch_end_validation",
    "grad_clip",
    "scheduler",
    "log_step",
    "checkpoint_step",
}
OPERATIONAL_RUNNER = {
    "parent_checkpoint", "retain_epoch_checkpoints",
    "fixed_validation", "fixed_training_baseline", "fixed_diagnostics",
    "test_masking", "test_evaluations",
    "test_diagnostics",
    "reports",
    "device",
    "multi_gpu",
    "microbatch_size",
    "adaptive_microbatch",
    "num_workers",
    "output_tb",
    "artifact_dir",
    "quiet",
    "gpu_memory_fraction",
    "cpu_threads",
    "train_monitor_step",
    "resume_from_ckpt",
    "resume_best_ckpt",
    "monitor_root",
}
STATUS_FIELDS = {
    "completed_sessions", "total_sessions", "detail",
    "completed_steps", "total_steps", "epoch",
    "completed_epochs", "total_epochs", "at_optimizer_step",
    "completed_batches", "total_batches",
}
DATA_PLACEMENT = {
    "data",
    "cached_features",
    "feature_cache",
    "pinned_test",
    "pinned_evaluation",
    "audit",
}


def resolve_batch_scaled_lr(cfg: object) -> None:
    """Resolve the default rate from the optimizer batch before fit hashing.

    Parameters
    ----------
    cfg : DictConfig
        Experiment configuration. Explicit runner learning rates take priority.
    """
    exp = cfg.exp
    if "base_lr" not in exp or exp.runner.lr is not None:
        return
    base_lr = exp.base_lr
    batch = exp.runner.train_batch_size
    if (not isinstance(base_lr, (int, float)) or isinstance(base_lr, bool)
            or not math.isfinite(base_lr) or base_lr <= 0):
        raise ValueError("Experiment base_lr must be finite and positive.")
    if type(batch) is not int or batch <= 0:
        raise ValueError("Experiment train_batch_size must be positive.")
    lr = base_lr * batch / REFERENCE_BATCH_SIZE
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError("Effective experiment learning rate is invalid.")
    exp.runner.lr = lr


def scientific_config(cfg: object, dataset: object, splits: dict) -> dict:
    """Collect scientific settings and verified source/split content."""
    resolve_batch_scaled_lr(cfg)
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
    result = {
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
        "runner": {
            k: v for k, v in runner.items()
            if k in SCIENTIFIC_RUNNER and k != "epochs"
        },
    }
    if hasattr(dataset, "selection_manifest"):
        result["selection"] = digest(
            dataset.selection_manifest["indices"]
        )
    return result


def scientific_default_variants(scientific: dict) -> list[dict]:
    """Return configurations equivalent under established defaults.

    Parameters
    ----------
    scientific : dict
        Resolved scientific settings; ramp-up decay defaults to False.

    Returns
    -------
    list of dict
        The original settings and an equivalent legacy form, if applicable.
    """
    variants = [scientific]
    scheduler = scientific.get("runner", {}).get("scheduler", {})
    if scheduler.get("name") != "ramp_up":
        return variants
    if "decay" in scheduler and scheduler["decay"] is not False:
        return variants
    alternate = copy.deepcopy(scientific)
    options = alternate["runner"]["scheduler"]
    if "decay" in options:
        del options["decay"]
    else:
        options["decay"] = False
    variants.append(alternate)
    return variants


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
        if not self.path.exists():
            for alternate in scientific_default_variants(scientific)[1:]:
                alternate_id = digest(alternate)
                if (self.root / "fits" / alternate_id).exists():
                    self.fit_id = alternate_id
                    self.path = self.root / "fits" / alternate_id
                    break
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
        self.resumed = False

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
            saved = json.loads((self.path / "scientific.json").read_text())
            if (digest(saved) != self.fit_id or saved not in
                    scientific_default_variants(self.scientific)):
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
            self.resumed = self._has_valid_recovery()
            if not self.resumed:
                quarantine = (
                    self.root / "quarantine" / self.fit_id / self.invocation
                )
                quarantine.parent.mkdir(parents=True, exist_ok=True)
                self.path.rename(quarantine)
        if not self.reused and not self.resumed:
            write_json(self.path / "scientific.json", self.scientific)
        for folder in ("status", "training", "errors"):
            (self.logs / folder).mkdir(parents=True, exist_ok=True)
        self.handler = logging.FileHandler(self.logs / "training" / "run.log")
        self.handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        logging.getLogger().addHandler(self.handler)
        logging.getLogger().setLevel(logging.INFO)
        phase = "reused" if self.reused else (
            "resuming" if self.resumed else "running"
        )
        self.status(phase)
        write_json(self.reference, {"fit_id": self.fit_id})

    def _has_valid_recovery(self) -> bool:
        """Accept only matching last/best checkpoints for this fit."""
        import torch

        scientific_path = self.path / "scientific.json"
        last_path = self.path / "checkpoints/checkpoint_last.pth"
        best_path = self.path / "checkpoints/checkpoint_best.pth"
        if not all(
            path.is_file()
            for path in (scientific_path, last_path, best_path)
        ):
            return False
        saved = json.loads(scientific_path.read_text())
        if (digest(saved) != self.fit_id or saved not in
                scientific_default_variants(self.scientific)):
            return False
        if self.scientific["runner"].get("shuffle", False):
            return False
        try:
            last = torch.load(
                last_path, map_location="cpu", weights_only=False
            )
            best = torch.load(
                best_path, map_location="cpu", weights_only=False
            )
            step = last["completed_steps"]
            best_step = last["best_step"]
            loss = last["best_validation_loss"]
            previous = best_path.with_name(
                "checkpoint_best.previous.pth"
            )
            if best["completed_steps"] != best_step and previous.exists():
                saved = torch.load(
                    previous, map_location="cpu", weights_only=False
                )
                if saved["completed_steps"] == best_step:
                    previous.replace(best_path)
                    best = saved
            if best["completed_steps"] == best_step:
                previous.unlink(missing_ok=True)
            required = (
                "optim", "scheduler", "model", "python_rng",
                "numpy_rng", "torch_rng",
            )
            return (
                isinstance(step, int) and 0 < step <=
                self.scientific["runner"]["total_steps"]
                and isinstance(best_step, int)
                and 0 < best_step <= step
                and best["completed_steps"] == best_step
                and math.isfinite(float(loss))
                and all(key in last for key in required)
                and "model" in best
            )
        except (OSError, ValueError, KeyError, TypeError, RuntimeError):
            return False

    def status(self, phase: str, **details: object) -> None:
        """Replace lifecycle progress without saving model metrics."""
        unknown = set(details) - STATUS_FIELDS
        if unknown:
            raise ValueError(
                f"Lifecycle status cannot contain metric fields: "
                f"{sorted(unknown)}"
            )
        path = self.logs / "status" / "status.json"
        previous = json.loads(path.read_text()) if path.exists() else {}
        stages = previous.get("stages", {})
        if details:
            stage = stages.get(phase, {})
            stage.update(details)
            stages[phase] = stage
        write_json(
            path,
            {
                "fit_id": self.fit_id,
                "phase": phase,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "stages": stages,
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
