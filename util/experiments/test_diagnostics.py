"""Replay a completed fit's fixed test set into campaign diagnostic tables.

The saved invocation supplies scientific settings and local data placement.
Only evaluation runs: no optimizer, checkpoint, original result, or event file
is written. CSV products use validation's monitor writers. A separate repair
invocation records lifecycle counts; a diagnostic manifest permits reuse.
"""

import argparse
import json
import math
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
from omegaconf import OmegaConf, open_dict

from .execution import configure_budgeted
from .lifecycle import scientific_config
from .monitoring import TrainingMonitor, diagnostic_names, rank_positions
from .report_settings import report_settings
from .reporting import (
    read_table,
    render,
    verify_fit,
    render_test_evaluation as render_evaluation,
)
from util.mask_utils import replacement_thresholds, test_masked_dataset
from .storage import atomic_write, file_digest, write_json

REPLAY_RTOL = 1e-5
REPLAY_ATOL = 1e-6
DIAGNOSTIC_REVISION = "validation-parity-v1"
GPU_MEMORY_FRACTION = 0.05


def repair(
    artifact_root: Path,
    campaign: str,
    device: str = "cuda:0",
    workers: int = 12,
    smoke_batches: int | None = None,
    diagnostics_config: Path | None = None,
    report_config: Path | None = None,
    evaluation_config: Path | None = None,
    evaluation_name: str | None = None,
    devices: list[str] | None = None,
    interpolation: bool = False,
    training_content_config: Path | None = None,
) -> None:
    """Replay pinned test batches with bounded resources and atomic outputs.

    Parameters
    ----------
    artifact_root : Path
        Shared fit storage root; campaign determines report placement.
    campaign : str
        Campaign with exactly one completed fit reference.
    device : str, optional
        Explicit CUDA device, default cuda:0.
    workers : int, optional
        Loader workers, default 12; external CPU/RAM caps remain required.
    smoke_batches : int or None, optional
        Audit this many batches without publishing, default full evaluation.
    """
    import tasks
    from criterions.pretrain_masked_criterion import CONTENT_THRESHOLD

    artifact_root = artifact_root.resolve()
    campaign_root = artifact_root / "campaigns" / campaign
    references = list((campaign_root / "fits").glob("*.json"))
    if len(references) != 1:
        raise ValueError("Diagnostic replay requires one campaign fit.")
    fit_id = json.loads(references[0].read_text())["fit_id"]
    fit = artifact_root / "fits" / fit_id
    verify_fit(fit)
    cfg = OmegaConf.create(
        json.loads((fit / "provenance.json").read_text())["invocation"]
    )
    fixed = (
        OmegaConf.load(diagnostics_config).get("fixed_masks")
        if diagnostics_config else None
    )
    diagnostics = diagnostic_names(
        OmegaConf.load(diagnostics_config).get("test_diagnostics")
        if diagnostics_config
        else cfg.exp.runner.get("test_diagnostics")
    )
    settings = report_settings(
        OmegaConf.load(report_config)
        if report_config
        else cfg.exp.runner.get("reports")
    )
    training_content = (
        OmegaConf.load(training_content_config)
        if training_content_config else None
    )
    if not training_content:
        training_content = None
    if interpolation and not fixed:
        raise ValueError(
            "Interpolation requires an explicit fixed-mask profile."
        )
    if (not diagnostics and evaluation_config is None and not fixed
            and training_content is None):
        render(artifact_root, campaign, settings=settings)
        return
    evaluation = (
        OmegaConf.load(evaluation_config) if evaluation_config else None
    )
    if evaluation is not None and "test_evaluations" in evaluation:
        if evaluation_name not in evaluation.test_evaluations:
            raise ValueError("Select a configured --evaluation-name.")
        evaluation = OmegaConf.create(
            {
                "name": evaluation_name,
                "test_masking": evaluation.test_evaluations[evaluation_name],
            }
        )
    if evaluation is not None:
        replacement_thresholds(evaluation.test_masking)
        if Path(evaluation.name).name != evaluation.name:
            raise ValueError("Evaluation name must be one path component.")
        evaluation_root = campaign_root / "evaluations" / evaluation.name
    reports = campaign_root / "reports"
    manifest_path = reports / "tables/test/diagnostic_manifest.csv"
    checkpoint = fit / "checkpoints/checkpoint_best.pth"
    checkpoint_hash = file_digest(checkpoint)
    if evaluation is not None and (evaluation_root / "complete.json").exists():
        saved = json.loads((evaluation_root / "complete.json").read_text())
        if (
            saved["policy"] != OmegaConf.to_container(evaluation.test_masking)
            or saved["checkpoint_sha256"] != checkpoint_hash
        ):
            raise ValueError(
                "Completed evaluation has different policy/checkpoint."
            )
        for name, checksum in saved["files"].items():
            if file_digest(evaluation_root / name) != checksum:
                raise ValueError(f"Evaluation checksum differs: {name}")
        render_evaluation(
            evaluation_root, fit, reports, evaluation.name, settings
        )
        return
    if (
        not fixed and evaluation is None and manifest_path.exists()
        and smoke_batches is None
    ):
        saved = read_table(manifest_path).iloc[0]
        if (
            saved.fit_id != fit_id
            or saved.checkpoint_sha256 != checkpoint_hash
            or saved.revision != DIAGNOSTIC_REVISION
        ):
            raise ValueError("Existing diagnostic provenance differs.")
        for name, column, diagnostic in (
            ("representation/rank", "rank_sha256", "representation_rank"),
            (
                "frequency/reconstruction",
                "frequency_sha256",
                "frequency_errors",
            ),
        ):
            if diagnostic not in diagnostics:
                continue
            if (
                file_digest(reports / f"tables/test/{name}.csv")
                != saved[column]
            ):
                raise ValueError("Diagnostic output checksum mismatch.")
        render(
            artifact_root, campaign, diagnostics_only=True, settings=settings
        )
        return
    if evaluation is None and not reports.joinpath("output_index.csv").exists():
        raise ValueError("Generate available saved-data reports before replay.")
    cpu_only = interpolation and evaluation is None
    if not cpu_only and (
        not torch.cuda.is_available() or not device.startswith("cuda")
    ):
        raise RuntimeError("Test replay requires an available CUDA device.")
    torch.set_num_threads(2)
    selected_devices = devices or [device]
    if len(set(selected_devices)) != len(selected_devices):
        raise ValueError("Diagnostic devices must be distinct.")
    for selected_device in ([] if cpu_only else selected_devices):
        if not selected_device.startswith("cuda:"):
            raise ValueError("Diagnostic devices must be explicit CUDA IDs.")
        torch.cuda.set_per_process_memory_fraction(
            GPU_MEMORY_FRACTION, torch.device(selected_device)
        )
    device = selected_devices[0]
    invocation = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    invocation += "-diagnostics-" + uuid.uuid4().hex[:8]
    logs = Path(cfg.storage.logs).resolve() / campaign / invocation
    status_path = logs / "status/status.json"

    stages = {}

    def progress(
        done: int, total: int, stage: str = "test_diagnostics"
    ) -> None:
        stages[stage] = {
            "completed_batches": done, "total_batches": total,
        }
        write_json(
            status_path,
            {
                "fit_id": fit_id,
                "phase": stage,
                "stages": stages,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
        )

    try:
        with open_dict(cfg):
            cfg.preprocessor.feature_cache = cfg.storage.feature_cache
        task = tasks.setup_task(cfg.task)
        task.load_datasets(cfg.data, cfg.preprocessor)
        configure_budgeted(cfg, task)
        current = scientific_config(cfg, task.dataset, task.dataset.partitions)
        if current != json.loads((fit / "scientific.json").read_text()):
            raise ValueError(
                "Replay scientific settings or membership changed."
            )
        task.dataset.verify_test(Path(cfg.data.pinned_test))
        task.dataset.prepare_selected(task.dataset.selection_manifest)
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if training_content is not None:
            from .fixed_mask_diagnostics import (
                training_content_counts, plot_training_content,
            )
            from .storage import digest

            result = training_content_counts(
                task, cfg.task, OmegaConf.to_container(training_content),
                campaign_root / "diagnostics",
                {"fit_id": fit_id, "selection_sha256": digest(
                    task.dataset.selection_manifest,
                ), "task": OmegaConf.to_container(cfg.task, resolve=True)},
                workers, cfg.exp.runner.valid_batch_size, progress,
                smoke_batches,
            )
            if smoke_batches is None:
                plot_training_content(
                    result, task.dataset.extracter.frequency_axis(), reports,
                    fit, state["completed_steps"], settings,
                )
        if not cpu_only:
            model = task.build_model(cfg.model)
            task.load_model_weights(model, state["model"], False)
            model.to(device).eval()
        if fixed:
            from .fixed_mask_diagnostics import evaluate_fixed, plot_fixed
            from .storage import digest

            fixed_options = OmegaConf.to_container(fixed, resolve=True)
            fixed_identity = {
                "checkpoint_sha256": checkpoint_hash,
                "selection_sha256": digest(task.dataset.selection_manifest),
                "checkpoint_step": state["completed_steps"],
            }
            if interpolation:
                from .fixed_mask_diagnostics import (
                    evaluate_interpolation, plot_interpolation,
                )

                root = campaign_root / "diagnostics" / "fixed_masks"
                result = evaluate_interpolation(
                    task, fixed_options, root, fixed_identity, workers,
                    progress, cfg.exp.runner.valid_batch_size, smoke_batches,
                    alpha=cfg.criterion.alpha,
                )
                if smoke_batches is None:
                    plot_interpolation(
                        result, root, task.dataset.extracter.frequency_axis(),
                        reports, fit, state["completed_steps"], settings,
                    )
                write_json(status_path, {
                    "phase": "complete", "stages": stages, "fit_id": fit_id,
                })
                return
            replacements = fixed_options.pop("test_replacements")
            if (not replacements or len(set(replacements)) != len(replacements)
                    or set(replacements) - {"zero", "random"}):
                raise ValueError(
                    "Expected distinct zero/random fixed-test replacement policies."
                )
            for replacement in replacements:
                def policy_progress(done, total, stage):
                    progress(done, total, f"{replacement}/{stage}")

                policy_options = {**fixed_options, "replacement": replacement}
                outputs = evaluate_fixed(
                    task, model, cfg, policy_options,
                    campaign_root / "diagnostics" / "fixed_masks" / replacement,
                    {
                        "checkpoint_sha256": checkpoint_hash,
                        "selection_sha256": digest(
                            task.dataset.selection_manifest
                        ),
                        "checkpoint_step": state["completed_steps"],
                    },
                    device, workers, policy_progress, smoke_batches,
                    devices=selected_devices,
                    publish=lambda ready: plot_fixed(
                        ready, task.dataset.extracter.frequency_axis(),
                        float(fixed.frame_hop_ms), reports, fit,
                        state["completed_steps"], settings,
                    ),
                )
                if smoke_batches is None:
                    plot_fixed(
                        outputs, task.dataset.extracter.frequency_axis(),
                        float(fixed.frame_hop_ms), reports, fit,
                        state["completed_steps"], settings,
                    )
            write_json(status_path, {
                "phase": "complete", "stages": stages, "fit_id": fit_id,
            })
            print(f"Fixed-mask diagnostics finished: {reports}", flush=True)
            return
        criterion = task.build_criterion(cfg.criterion)
        subset = test_masked_dataset(
            task.test_set, evaluation.test_masking if evaluation else None
        )
        if smoke_batches is not None:
            subset = torch.utils.data.Subset(
                subset,
                range(
                    min(
                        len(subset),
                        smoke_batches * cfg.exp.runner.valid_batch_size,
                    )
                ),
            )
        positions = rank_positions(len(subset))
        loader = task.get_batch_iterator(
            subset,
            cfg.exp.runner.valid_batch_size,
            shuffle=False,
            num_workers=workers,
        )
        started = time.monotonic()
        output = task.get_valid_outs(
            model,
            loader,
            criterion,
            device,
            rank_indices=positions
            if "representation_rank" in diagnostics
            else None,
            collect_frequency="frequency_errors" in diagnostics,
            progress_callback=progress,
        )
        if smoke_batches is not None:
            print(
                f"Smoke: {len(loader)} batches in "
                f"{time.monotonic() - started:.1f}s; peak CUDA bytes "
                f"{torch.cuda.max_memory_allocated()}",
                flush=True,
            )
            return
        if evaluation is not None:
            if evaluation_root.exists():
                raise ValueError(
                    "Incomplete evaluation exists; inspect before retrying."
                )
            stats = output["monitor_accumulator"]
            if stats.sums.reshape(3, 3, -1)[:, 2, 0].sum() != 0:
                raise ValueError(
                    "Zero/random evaluation contains unchanged targets."
                )
            with_monitor = TrainingMonitor(
                evaluation_root / "monitoring",
                task.dataset.extracter.frequency_axis(),
                cfg.criterion.alpha,
                CONTENT_THRESHOLD,
                1,
                cfg.preprocessor.normalizing,
                tensorboard=False,
            )
            try:
                with_monitor.record(
                    "test",
                    state["completed_steps"],
                    stats,
                    checkpoint_step=state["completed_steps"],
                    plot_images=False,
                )
                if "representation_rank" in diagnostics:
                    with_monitor.record_rank(
                        state["completed_steps"],
                        output["rank_embeddings"],
                        phase="test",
                    )
                if "frequency_errors" in diagnostics:
                    with_monitor.record_frequency(
                        state["completed_steps"],
                        stats,
                        phase="test",
                    )
            finally:
                with_monitor.close()
            scalar = {
                k: v
                for k, v in output.items()
                if k not in {"monitor_accumulator", "rank_embeddings"}
            }
            write_json(evaluation_root / "results.json", scalar)
            write_json(
                evaluation_root / "complete.json",
                {
                    "fit_id": fit_id,
                    "checkpoint_sha256": checkpoint_hash,
                    "checkpoint_step": state["completed_steps"],
                    "selection_digest": current["selection"],
                    "policy": OmegaConf.to_container(evaluation.test_masking),
                    "windows": len(subset),
                    "batches": len(loader),
                    "files": {
                        str(p.relative_to(evaluation_root)): file_digest(p)
                        for p in evaluation_root.rglob("*")
                        if p.is_file()
                    },
                },
            )
            render_evaluation(
                evaluation_root, fit, reports, evaluation.name, settings
            )
            write_json(
                status_path,
                {
                    "fit_id": fit_id,
                    "phase": "complete",
                    "stages": {
                        "test_diagnostics": {
                            "completed_batches": len(loader),
                            "total_batches": len(loader),
                        }
                    },
                },
            )
            print(f"Test evaluation published: {evaluation_root}", flush=True)
            return
        expected = json.loads((fit / "results.json").read_text())
        for key in expected.keys() & output.keys():
            if not math.isclose(
                output[key],
                expected[key],
                rel_tol=REPLAY_RTOL,
                abs_tol=REPLAY_ATOL,
            ):
                raise ValueError(f"Replayed test metric differs: {key}")
        original = []
        for path in (fit / "monitoring").rglob("metrics.csv"):
            table = read_table(path)
            original.extend(table[table.phase == "test"].to_dict("records"))
        if len(original) != 1:
            raise ValueError("Expected one saved test aggregate.")
        stats = output["monitor_accumulator"]
        if int(stats.sums[:, 0].sum()) != int(original[0]["count"]):
            raise ValueError("Replayed selected-cell count changed.")
        if int(stats.sums[:, 2].sum()) != int(original[0]["content_count"]):
            raise ValueError("Replayed content-cell count changed.")
        with tempfile.TemporaryDirectory(
            prefix="brainbert-test-diagnostics-"
        ) as tmp:
            monitor = TrainingMonitor(
                Path(tmp) / "monitoring",
                task.dataset.extracter.frequency_axis(),
                cfg.criterion.alpha,
                CONTENT_THRESHOLD,
                1,
                cfg.preprocessor.normalizing,
                tensorboard=False,
            )
            try:
                if "representation_rank" in diagnostics:
                    monitor.record_rank(
                        state["completed_steps"],
                        output["rank_embeddings"],
                        phase="test",
                    )
                if "frequency_errors" in diagnostics:
                    monitor.record_frequency(
                        state["completed_steps"], stats, phase="test"
                    )
            finally:
                monitor.close()
            outputs = {}
            for source, destination, label in (
                ("test_rank.csv", "representation/rank.csv", "rank"),
                (
                    "test_frequency.csv.gz",
                    "frequency/reconstruction.csv",
                    "frequency",
                ),
            ):
                if not (Path(tmp) / "monitoring" / source).exists():
                    continue
                frame = read_table(Path(tmp) / "monitoring" / source)
                target = reports / "tables/test" / destination
                atomic_write(
                    target,
                    lambda p: frame.to_csv(p, index=False),
                    lambda p: pd.read_csv(p),
                )
                outputs[label + "_sha256"] = file_digest(target)
        manifest = pd.DataFrame(
            [
                {
                    "fit_id": fit_id,
                    "checkpoint_sha256": checkpoint_hash,
                    "checkpoint_step": state["completed_steps"],
                    "revision": DIAGNOSTIC_REVISION,
                    "windows": len(subset),
                    "rank_windows": (
                        len(positions)
                        if "representation_rank" in diagnostics
                        else None
                    ),
                    "batches": len(loader),
                    "diagnostics": ",".join(sorted(diagnostics)),
                    "selection_digest": current["selection"],
                    **outputs,
                }
            ]
        )
        atomic_write(
            manifest_path,
            lambda p: manifest.to_csv(p, index=False),
            lambda p: pd.read_csv(p),
        )
        render(
            artifact_root, campaign, diagnostics_only=True, settings=settings
        )
        write_json(
            status_path,
            {
                "fit_id": fit_id,
                "phase": "complete",
                "stages": {
                    "test_diagnostics": {
                        "completed_batches": len(loader),
                        "total_batches": len(loader),
                    }
                },
            },
        )
        print(f"Test diagnostics published: {reports}", flush=True)
    except Exception as error:
        write_json(logs / "errors/failure.json", {"error": str(error)})
        write_json(status_path, {"fit_id": fit_id, "phase": "failed"})
        raise


def main() -> None:
    """Repair diagnostics separately from saved-result publication."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--interpolation", action="store_true")
    parser.add_argument("--devices", nargs="+")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--smoke-batches", type=int)
    parser.add_argument("--diagnostics-config", type=Path)
    parser.add_argument("--report-config", type=Path)
    parser.add_argument("--training-content-config", type=Path)
    parser.add_argument("--evaluation-config", type=Path)
    parser.add_argument("--evaluation-name")
    args = parser.parse_args()
    repair(
        args.artifact_root,
        args.campaign,
        args.device,
        args.workers,
        args.smoke_batches,
        args.diagnostics_config,
        args.report_config,
        args.evaluation_config,
        args.evaluation_name,
        args.devices,
        args.interpolation,
        args.training_content_config,
    )


if __name__ == "__main__":
    main()
