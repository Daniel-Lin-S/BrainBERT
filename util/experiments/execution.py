"""Execute the existing task/runner pipeline inside a shared fit lifecycle.

Managed configurations supply local storage roots, an explicit scientific
protocol, and a pinned test JSON list of manifest-relative sample names.
Source checksums and actual split memberships are saved in scientific.json;
absolute source placement and software versions are provenance only.
"""

import json
import logging
import math
import os
import subprocess
import sys
import warnings
from pathlib import Path

import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Subset

from .lifecycle import OPERATIONAL_RUNNER, FitRun, scientific_config
from .storage import file_digest, write_json

log = logging.getLogger(__name__)


def source_inventory(dataset: object) -> list:
    """Identify all manifest sources independently of storage location.

    Parameters
    ----------
    dataset : object
        Existing dataset exposing ordered ``files`` and ``root_dir``.

    Returns
    -------
    list
        One sample-name and content-checksum dictionary per source file.
    """
    if hasattr(dataset, "inventory"):
        return dataset.inventory
    if not hasattr(dataset, "files") or not dataset.files:
        raise ValueError(
            "This dataset needs an explicit sample identity adapter before "
            "managed fit reuse; data-loading design is separate from storage."
        )
    if len(set(dataset.files)) != len(dataset.files):
        raise ValueError("Dataset manifest contains duplicate sample names.")
    root = Path(dataset.root_dir).resolve()
    inventory = []
    for name in dataset.files:
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            raise ValueError(
                f"Manifest sample escapes the source root: {name}"
            )
        inventory.append({"sample": name, "sha256": file_digest(path)})
    return inventory


def configure_budgeted(cfg: object, task: object) -> None:
    """Freeze optional group sampling and derive training steps."""
    sampling = cfg.task.get("sampling")
    if sampling is None:
        return
    if sampling.mode != "temperature":
        raise ValueError(f"Unknown sampling mode: {sampling.mode}")
    required = (
        "records", "partitions", "ends", "inventory",
        "cached_local_indices", "cache_settings",
        "configure_selection", "group_id",
        "selection_namespace", "selection_revision",
    )
    missing = [
        name for name in required
        if not hasattr(task.dataset, name)
    ]
    if missing:
        raise ValueError(
            f"Temperature sampling needs subgroup membership: "
            f"{missing}"
        )
    from util.experiments.sampling import select_or_load

    if hasattr(task.dataset, "validate_budgeted_preprocessing"):
        task.dataset.validate_budgeted_preprocessing()
    budget = int(sampling.sample_budget)
    if budget <= 0:
        raise ValueError("sample_budget must be positive.")
    batch_size = int(cfg.exp.runner.train_batch_size)
    if batch_size <= 0:
        raise ValueError("train_batch_size must be positive.")
    steps = math.ceil(budget / batch_size)
    if cfg.exp.runner.shuffle:
        raise ValueError(
            "Budgeted sampling requires runner.shuffle=False."
        )
    if (
        "total_steps" in cfg.exp.runner
        or "total_steps" in cfg.exp.runner.scheduler
    ):
        warnings.warn(
            "total_steps is ignored when sample_budget is set.",
            stacklevel=2,
        )
    with open_dict(cfg):
        cfg.task.sampling.alpha = float(sampling.alpha)
        cfg.exp.runner.total_steps = steps
        cfg.exp.runner.scheduler.total_steps = steps
    settings = OmegaConf.to_container(sampling, resolve=True)
    settings["valid_batch_size"] = int(cfg.exp.runner.valid_batch_size)
    selected = select_or_load(
        Path(cfg.storage.artifacts).resolve(), task.dataset, settings
    )
    task.dataset.configure_selection(selected)
    task.train_set = Subset(task.dataset, selected["indices"]["train"])
    task.valid_set = Subset(
        task.dataset, selected["indices"]["validation"]
    )
    task.test_set = Subset(task.dataset, selected["indices"]["test"])
    task.dataset.selection_manifest = selected


def execute(cfg: object) -> dict:
    """Run the existing task with shared artifact guards.

    Parameters
    ----------
    cfg : DictConfig
        Existing scientific groups plus shared storage/runtime settings,
        campaign name, protocol revision, and pinned test membership.

    Returns
    -------
    dict
        Finite test metrics, loaded from a completed fit when available.
    """
    import tasks
    from runner import Runner

    OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    for field in ("campaign", "protocol", "data.pinned_test"):
        if OmegaConf.select(cfg, field) is None:
            raise ValueError(f"Managed execution requires explicit {field}.")
    if cfg.exp.runner.checkpoint_step <= 0:
        raise ValueError("Managed fits require a positive checkpoint_step.")
    unknown = set(cfg.runtime) - (OPERATIONAL_RUNNER - {"artifact_dir"})
    if unknown:
        raise ValueError(
            f"Machine runtime cannot override scientific settings: {unknown}"
        )
    with open_dict(cfg):
        for key, value in cfg.runtime.items():
            cfg.exp.runner[key] = value
    if not 0 < cfg.exp.runner.gpu_memory_fraction <= 1:
        raise ValueError("gpu_memory_fraction must lie in (0, 1].")
    if cfg.exp.runner.cpu_threads <= 0:
        raise ValueError("cpu_threads must be positive.")
    torch.set_num_threads(cfg.exp.runner.cpu_threads)
    if hasattr(os, "nice") and os.getpriority(os.PRIO_PROCESS, 0) < 10:
        os.nice(10 - os.getpriority(os.PRIO_PROCESS, 0))
    with open_dict(cfg):
        cfg.preprocessor.feature_cache = cfg.storage.feature_cache
    task = tasks.setup_task(cfg.task)
    task.load_datasets(cfg.data, cfg.preprocessor)
    task.dataset.inventory = source_inventory(task.dataset)
    if hasattr(task.dataset, "partitions"):
        splits = task.dataset.partitions
        task.dataset.verify_test(Path(cfg.data.pinned_test).resolve())
    else:
        splits = {
            name: [task.dataset.files[i] for i in subset.indices]
            for name, subset in [
                ("train", task.train_set),
                ("validation", task.valid_set),
                ("test", task.test_set),
            ]
        }
        pinned_path = Path(cfg.data.pinned_test).expanduser().resolve()
        pinned = json.loads(pinned_path.read_text())
        if (
            not isinstance(pinned, list)
            or not pinned
            or len(pinned) != len(set(pinned))
            or sorted(pinned) != sorted(splits["test"])
        ):
            raise ValueError(
                f"Test split differs from pinned samples: {pinned_path}"
            )
    configure_budgeted(cfg, task)
    scientific = scientific_config(cfg, task.dataset, splits)
    with FitRun(
        cfg.storage.artifacts, cfg.storage.logs, cfg.campaign, scientific
    ) as run:
        if run.reused:
            log.info("Reusing completed fit %s", run.fit_id)
            return json.loads((run.path / "results.json").read_text())
        write_json(
            run.path / "provenance.json",
            {
                "python": sys.version,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "source_root": str(Path(task.dataset.root_dir).resolve()),
                "git_revision": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], text=True
                ).strip(),
                "invocation": OmegaConf.to_container(cfg, resolve=True),
            },
        )
        with open_dict(cfg):
            cfg.exp.runner.artifact_dir = str(run.path / "checkpoints")
        runner = None
        try:
            if (
                hasattr(task.dataset, "selection_manifest")
                and hasattr(task.dataset, "prepare_selected")
            ):
                task.dataset.prepare_selected(
                    task.dataset.selection_manifest
                )
            model = task.build_model(cfg.model)
            criterion = task.build_criterion(cfg.criterion)
            runner = Runner(cfg.exp.runner, task, model, criterion)
            runner.status_callback = run.status
            best = runner.train()
            metrics = runner.test(best)
            if str(cfg.exp.runner.device).startswith("cuda"):
                metrics["peak_cuda_allocated_bytes"] = (
                    torch.cuda.max_memory_allocated()
                )
            if runner.monitor is not None:
                runner.monitor.close()
            run.complete(metrics)
            return metrics
        finally:
            if runner is not None:
                runner.progress.close()
                if runner.monitor is not None:
                    runner.monitor.close()
                if runner.logger is not None:
                    runner.logger.close()
