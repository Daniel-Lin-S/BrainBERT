"""Execute the existing task/runner pipeline inside a shared fit lifecycle.

Managed configurations supply local storage roots, an explicit scientific
protocol, and a pinned test JSON list of manifest-relative sample names.
Source checksums and actual split memberships are saved in scientific.json;
absolute source placement and software versions are provenance only.
"""

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf, open_dict

from .lifecycle import FitRun, OPERATIONAL_RUNNER, scientific_config
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
            run.complete(metrics)
            return metrics
        finally:
            if runner is not None:
                runner.progress.close()
                if runner.logger is not None:
                    runner.logger.close()
