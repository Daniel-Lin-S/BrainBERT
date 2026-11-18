"""Train task models and evaluate the lowest-validation-loss snapshot.

Checkpoints in the run directory contain ``model`` parameter tensors,
``model_cfg`` architecture settings, ``optim`` optimizer state, and
``scheduler`` learning-rate state. ``checkpoint_last.pth`` stores the latest
saved update; ``checkpoint_best.pth`` stores the lowest observed validation
loss. A negative checkpoint interval disables disk saves, not model selection.
"""

import copy
import json
import logging
import os
import random
import warnings
from pathlib import Path

from omegaconf import open_dict

import numpy as np
import torch
import torch_optimizer as torch_optim
from tensorboardX import SummaryWriter
from torch.utils.data import Subset
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from schedulers import build_scheduler
from util.experiments.monitoring import (
    Accumulator, TrainingMonitor, diagnostic_names, rank_positions,
)
from util.experiments.storage import (
    atomic_write, publish_training_completion, write_json,
)

STATUS_STEP_INTERVAL = 10

log = logging.getLogger(__name__)


def restore_cuda_rng(states: list[torch.Tensor]) -> None:
    """Restore checkpoint random streams to the currently visible GPUs.

    Parameters
    ----------
    states : list of torch.Tensor
        One one-dimensional byte tensor per checkpoint CUDA device, ordered
        by logical device index. CPU RNG restoration must precede this call.

    Notes
    -----
    Newly visible devices start from the restored CPU generator's seed.
    Matching logical devices retain their saved streams. Changing device
    count cannot reproduce the original dropout draws exactly.
    """
    if not states:
        raise ValueError("Checkpoint CUDA random-state list is empty.")
    devices = torch.cuda.device_count()
    if len(states) != devices:
        warnings.warn(
            f"Checkpoint has {len(states)} CUDA random streams, but "
            f"{devices} devices are visible; restoring matching streams "
            "and seeding new devices. Stochastic replay will differ.",
            RuntimeWarning, stacklevel=2,
        )
        torch.cuda.manual_seed_all(torch.initial_seed())
    torch.cuda.set_rng_state_all([state.cpu() for state in states[:devices]])


class Runner:
    def __init__(self, cfg, task, model, criterion):
        self.cfg = cfg
        self.model = model
        self.task = task
        self.evaluator = None
        self.status_callback = None
        self.completed_steps = 0
        self.microbatch_ready_checks = 0
        self.microbatch_last_check = -1
        self.stale_epochs = 0
        self.epoch_best_at_start = float("inf")
        self.stopped_early = False
        self.pending_fixed_validation = False
        self.device = torch.device(cfg.device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.criterion = criterion
        self.exp_dir = str(
            Path(cfg.get("artifact_dir", os.getcwd())).resolve()
        )
        Path(self.exp_dir).mkdir(parents=True, exist_ok=True)
        fraction = cfg.get("gpu_memory_fraction")
        if fraction is not None and str(self.device).startswith("cuda"):
            if not 0 < fraction <= 1:
                raise ValueError("gpu_memory_fraction must lie in (0, 1].")
            devices = (
                range(torch.cuda.device_count())
                if cfg.multi_gpu
                else [
                    (
                        torch.device(self.device).index
                        if torch.device(self.device).index is not None
                        else torch.cuda.current_device()
                    )
                ]
            )
            for device in devices:
                torch.cuda.set_per_process_memory_fraction(fraction, device)
            torch.cuda.reset_peak_memory_stats()
        self.output_tb = cfg.get("output_tb", True)
        self.logger = None
        self.monitor = None
        self.rank_indices = None
        self.best_step = None
        self.best_validation_loss = None
        if self.output_tb and task.cfg.name != "spec_pretrain":
            self.logger = SummaryWriter(
                str(Path(self.exp_dir) / "tensorboard")
            )

        if cfg.multi_gpu:
            self.model = torch.nn.DataParallel(self.model)
            log.info(f"Use {torch.cuda.device_count()} GPUs")
        assert not (cfg.device == "cpu" and cfg.multi_gpu)
        self.model.to(self.device)
        self.optim = self._init_optim(self.cfg)
        self.scheduler = build_scheduler(self.cfg.scheduler, self.optim)
        total_steps = self.cfg.total_steps
        self.progress = tqdm(
            total=total_steps,
            dynamic_ncols=True,
            desc="overall",
            disable=cfg.get("quiet", False),
        )

        if cfg.get("resume_from_ckpt") is not None:
            self.restore_managed_checkpoint()
        elif cfg.get("parent_checkpoint"):
            state = self.load_from_ckpt(cfg.parent_checkpoint)
            self.completed_steps = state["completed_steps"]
            self.progress.n = self.completed_steps
            self.best_step = self.completed_steps
            self.best_validation_loss = state["best_validation_loss"]
            self.resumed_best_state = state["model"]
            random.setstate(state["python_rng"])
            np.random.set_state(state["numpy_rng"])
            torch.set_rng_state(state["torch_rng"].cpu())
            if torch.cuda.is_available():
                restore_cuda_rng(state["cuda_rng"])
        elif "start_from_ckpt" in cfg:
            self.load_from_ckpt(cfg.start_from_ckpt)

        if task.cfg.name == "spec_pretrain":
            from criterions.pretrain_masked_criterion import CONTENT_THRESHOLD
            from preprocessors import STFTPreprocessor

            extractor = getattr(task.dataset, "extracter", None)
            if extractor is None:
                extractor = STFTPreprocessor(task.dataset.preprocessor_cfg)
            self.criterion.frequency_hz = extractor.frequency_axis()
            base = Path(self.exp_dir)
            root = base.parent if base.name == "checkpoints" else base
            monitor_root = Path(
                cfg.get("monitor_root", root / "monitoring")
            )
            self.monitor = TrainingMonitor(
                monitor_root,
                extractor.frequency_axis(),
                criterion.cfg.alpha,
                CONTENT_THRESHOLD,
                optimizer_groups=len(self.optim.param_groups),
                normalization_mode=extractor.cfg.normalizing,
                tensorboard=self.output_tb,
                tensorboard_root=root / "monitoring/tensorboard",
                purge_step=(
                    self.completed_steps + 1
                    if cfg.get("resume_from_ckpt") is not None else None
                ),
            )

    def load_from_ckpt(self, ckpt_path):
        init_state = torch.load(
            ckpt_path, map_location=self.device, weights_only=False
        )
        self.task.load_model_weights(
            self.model, init_state["model"], self.cfg.multi_gpu
        )
        self.optim.load_state_dict(init_state["optim"])
        scheduler_state = init_state["scheduler"]
        if scheduler_state is None:
            raise ValueError(
                "Checkpoint has no scheduler state; exact scheduler "
                "restoration is unavailable for this legacy checkpoint."
            )
        self.scheduler.load_state_dict(scheduler_state)
        return init_state

    def restore_managed_checkpoint(self):
        """Restore optimizer progress and the selected best snapshot."""
        last = self.load_from_ckpt(self.cfg.resume_from_ckpt)
        best = torch.load(
            self.cfg.resume_best_ckpt,
            map_location=self.device, weights_only=False,
        )
        self.pending_fixed_validation = last.get(
            "pending_fixed_validation", False,
        )
        self.stale_epochs = last.get("stale_epochs", 0)
        self.epoch_best_at_start = last.get(
            "epoch_best_at_start", last["best_validation_loss"],
        )
        self.stopped_early = last.get("stopped_early", False)
        self.completed_steps = last["completed_steps"]
        self.progress.n = self.completed_steps
        self.best_step = last["best_step"]
        self.best_validation_loss = last["best_validation_loss"]
        self.resumed_best_state = best["model"]
        random.setstate(last["python_rng"])
        np.random.set_state(last["numpy_rng"])
        torch.set_rng_state(last["torch_rng"].cpu())
        if torch.cuda.is_available() and "cuda_rng" in last:
            restore_cuda_rng(last["cuda_rng"])

    def _init_optim(self, args):
        if args.optim == "SGD":
            optim = torch.optim.SGD(
                self.model.parameters(), lr=args.lr, momentum=0.9
            )
        elif args.optim == "Adam":
            optim = torch.optim.Adam(
                self.model.parameters(), lr=args.lr, weight_decay=0.01
            )
        elif args.optim == "AdamW":
            optim = torch.optim.AdamW(self.model.parameters(), lr=args.lr)
        elif args.optim == "AdamW_finetune":
            model = self.model.module if self.cfg.multi_gpu else self.model
            linear_out_params = list(model.linear_out.parameters())
            ignored_params = list(map(id, linear_out_params))
            base_params = filter(
                lambda p: id(p) not in ignored_params, self.model.parameters()
            )

            optim = torch.optim.AdamW(
                [
                    {"params": base_params},
                    {"params": linear_out_params, "lr": args.lr},
                ],
                lr=args.lr * 0.1,
            )
        elif args.optim == "LAMB":
            optim = torch_optim.Lamb(self.model.parameters(), lr=args.lr)
        else:
            raise ValueError(f"Unknown optimizer: {args.optim!r}")
        return optim

    def output_logs(self, train_logging_outs, val_logging_outs):
        global_step = self.progress.n
        train_logging_outs["lr"] = self.scheduler.get_lr()
        standard_metrics = ["lr", "loss", "grad_norm"]
        all_standard_metrics = {}

        def add_prefix(prefix, outs):
            for k, v in outs.items():
                if k in standard_metrics:
                    all_standard_metrics[f"{prefix}_{k}"] = v

        add_prefix("train", train_logging_outs)
        add_prefix("val", val_logging_outs)

        log.info(all_standard_metrics)

        if self.logger is not None:
            for k, v in all_standard_metrics.items():
                self.logger.add_scalar(k, v, global_step=global_step)
        self.task.output_logs(
            train_logging_outs, val_logging_outs, self.logger, global_step
        )

    def report_progress(
        self, phase: str, completed: int, total: int,
    ) -> None:
        """Persist optimizer and current stage progress for managed fits."""
        if self.status_callback is None:
            return
        epoch = getattr(self, "epoch_index", 0)
        completed_epochs = (
            min(self.completed_steps // self.steps_per_epoch, self.cfg.epochs)
            if self.cfg.get("epochs") is not None
            and hasattr(self, "steps_per_epoch") else None
        )
        details = {
            "epoch": epoch,
            "completed_epochs": completed_epochs,
            "total_epochs": self.cfg.get("epochs"),
        }
        if phase == "training":
            details.update(
                completed_steps=self.completed_steps,
                total_steps=self.cfg.total_steps,
            )
        else:
            details.update(
                at_optimizer_step=self.completed_steps,
                completed_batches=completed,
                total_batches=total,
            )
        self.status_callback(phase, **details)

    def get_valid_outs(self):
        valid_loader = self.get_batch_iterator(
            self.task.valid_set,
            self.cfg.valid_batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
        )
        progress = lambda done, total: self.report_progress(
            "validation", done, total
        )
        if self.monitor is not None:
            return self.task.get_valid_outs(
                self.model, valid_loader, self.criterion, self.device,
                rank_indices=self.rank_indices, collect_frequency=True,
                progress_callback=progress,
            )
        if self.task.cfg.name == "spec_pretrain":
            return self.task.get_valid_outs(
                self.model, valid_loader, self.criterion, self.device,
                progress_callback=progress,
            )
        valid_logging_outs = self.task.get_valid_outs(
            self.model, valid_loader, self.criterion, self.device
        )
        return valid_logging_outs

    def save_checkpoint_last(self, states, best_val=False):
        cwd = self.exp_dir
        if best_val:
            save_path = os.path.join(cwd, "checkpoint_best.pth")
        else:
            save_path = os.path.join(cwd, "checkpoint_last.pth")
        log.debug(f"Saving checkpoint to {save_path}")

        def audit(path: Path) -> None:
            saved = torch.load(path, map_location="cpu", weights_only=False)
            if set(saved) != set(states):
                raise ValueError(f"Checkpoint keys differ: {path.resolve()}")
            for name, tensor in saved["model"].items():
                if not torch.isfinite(tensor).all():
                    raise ValueError(f"Nonfinite checkpoint parameter: {name}")

        atomic_write(Path(save_path), lambda p: torch.save(states, p), audit)
        log.debug(f"Saved checkpoint to {save_path}")

    def save_checkpoints(self, best_val=False):
        all_states = {}
        all_states = self.task.save_model_weights(
            self.model, all_states, self.cfg.multi_gpu
        )
        all_states["pending_fixed_validation"] = self.pending_fixed_validation
        all_states["stale_epochs"] = self.stale_epochs
        all_states["epoch_best_at_start"] = self.epoch_best_at_start
        all_states["stopped_early"] = self.stopped_early
        all_states["completed_steps"] = self.completed_steps
        all_states["microbatch_size"] = self.cfg.get("microbatch_size")
        all_states["best_step"] = self.best_step
        all_states["best_validation_loss"] = (
            self.best_validation_loss
        )
        all_states["python_rng"] = random.getstate()
        all_states["numpy_rng"] = np.random.get_state()
        all_states["torch_rng"] = torch.get_rng_state()
        if torch.cuda.is_available():
            all_states["cuda_rng"] = torch.cuda.get_rng_state_all()
        all_states["optim"] = self.optim.state_dict()
        all_states["scheduler"] = self.scheduler.get_state_dict()
        if self.cfg.multi_gpu:
            all_states["model_cfg"] = self.model.module.cfg
        else:
            all_states["model_cfg"] = self.model.cfg
        if not best_val:
            self.save_checkpoint_last(all_states)
            return
        best_path = Path(self.exp_dir) / "checkpoint_best.pth"
        previous = Path(self.exp_dir) / "checkpoint_best.previous.pth"
        if previous.exists():
            raise RuntimeError(
                f"Unresolved best-checkpoint backup: {previous.resolve()}"
            )
        if best_path.exists():
            os.link(best_path, previous)
        try:
            self.save_checkpoint_last(all_states, best_val=True)
            self.save_checkpoint_last(all_states)
        except BaseException:
            if previous.exists():
                os.replace(previous, best_path)
            else:
                best_path.unlink(missing_ok=True)
            raise
        previous.unlink(missing_ok=True)

    def maybe_relax_microbatch(self) -> None:
        """Increase forward rows after sustained measured CUDA headroom.

        The configured optimizer batch remains fixed. Profiled target peaks
        include model and optimizer allocations. Reclaimable allocator cache
        contributes to available memory, bounded by the per-process cap.
        Changes occur between optimizer updates and are recorded in logs and
        subsequent checkpoints.
        """
        options = self.cfg.get("adaptive_microbatch")
        current = self.cfg.get("microbatch_size")
        if options is None or current is None or self.device.type != "cuda":
            return
        interval = options["check_interval"]
        required = options["consecutive_checks"]
        margin = options["margin_bytes"]
        if interval <= 0 or required <= 0 or margin < 0:
            raise ValueError("Adaptive microbatch policy has invalid limits.")
        if (self.completed_steps <= 0
                or self.completed_steps % interval
                or self.completed_steps == self.microbatch_last_check):
            return
        self.microbatch_last_check = self.completed_steps
        target = min(current * 2, self.cfg.train_batch_size)
        peak = options["profile_peaks_bytes"][str(target)]
        if peak <= 0:
            raise ValueError("Adaptive target requires a positive profile peak.")
        devices = (
            range(torch.cuda.device_count()) if self.cfg.multi_gpu
            else [self.device.index]
        )
        available = []
        for device in devices:
            free, total = torch.cuda.mem_get_info(device)
            reusable = torch.cuda.memory_reserved(device)
            cap = int(total * self.cfg.gpu_memory_fraction)
            available.append(min(free + reusable, cap))
        if not available:
            raise ValueError("Adaptive microbatch policy found no CUDA devices.")
        if min(available) < peak + margin:
            self.microbatch_ready_checks = 0
            return
        self.microbatch_ready_checks += 1
        if self.microbatch_ready_checks < required:
            return
        self.cfg.microbatch_size = (
            None if target == self.cfg.train_batch_size else target
        )
        self.microbatch_ready_checks = 0
        log.info(
            "Optimizer step %s: forward batch %s -> %s after %s memory "
            "checks; optimizer batch %s; available bytes %s",
            self.completed_steps, current, target, required,
            self.cfg.train_batch_size, available,
        )

    def run_epoch(
        self, train_loader, total_loss, best_state,
        epoch_complete=True,
    ):
        best_model, best_val = best_state
        for batch_index, batch in enumerate(train_loader):
            if self.progress.n >= self.progress.total:
                break
            self.maybe_relax_microbatch()
            self.model.train()
            learning_rates = [group["lr"] for group in self.optim.param_groups]
            logging_out = self.task.train_step(
                batch,
                self.model,
                self.criterion,
                self.optim,
                self.scheduler,
                self.device,
                self.cfg.grad_clip,
                self.cfg.get("microbatch_size"),
            )
            self.completed_steps += 1
            if (
                self.completed_steps == 1
                or self.completed_steps % STATUS_STEP_INTERVAL == 0
                or self.completed_steps == self.cfg.total_steps
            ):
                self.report_progress(
                    "training", self.completed_steps,
                    self.cfg.total_steps,
                )
            if self.monitor is not None:
                self.monitor.train.add(logging_out, self.cfg.grad_clip)
                self.monitor.epoch_train.add(
                    logging_out, self.cfg.grad_clip
                )
            total_loss.append(logging_out["loss"])
            validation_step = (
                self.completed_steps if self.cfg.get("align_validation_steps")
                else self.progress.n
            )
            log_step = (
                validation_step % self.cfg.log_step == 0
                or self.progress.n == self.progress.total - 1
            )

            ckpt_step = False
            if self.cfg.checkpoint_step > -1:
                ckpt_step = (
                    validation_step % self.cfg.checkpoint_step == 0
                    or self.progress.n == self.progress.total - 1
                )

            epoch_end = (
                self.cfg.get("epoch_end_validation", False)
                and (
                    batch_index == len(train_loader) - 1
                    or self.progress.n == self.progress.total - 1
                )
            )
            ckpt_step = ckpt_step or epoch_end
            monitor_step = self.cfg.get("train_monitor_step")
            if (
                self.monitor is not None
                and monitor_step is not None
                and self.completed_steps > 1
                and (self.completed_steps - 1) % monitor_step == 0
                and not (ckpt_step or log_step)
            ):
                self.monitor.record(
                    "train", self.completed_steps, self.monitor.train,
                    learning_rates=learning_rates, plot_images=False,
                )
                self.monitor.train = Accumulator()
            valid_logging_outs = {}
            saved_best = False
            if ckpt_step or log_step:
                self.model.eval()
                valid_logging_outs = self.get_valid_outs()
                if self.monitor is not None:
                    stats = valid_logging_outs.pop("monitor_accumulator")
                    rank = valid_logging_outs.pop(
                        "rank_embeddings", None
                    )
                    self.monitor.record(
                        "train",
                        self.completed_steps,
                        self.monitor.train,
                        learning_rates=learning_rates,
                        plot_images=epoch_end or self.completed_steps == 1,
                    )
                    self.monitor.train = Accumulator()
                    self.monitor.record(
                        "validation",
                        self.completed_steps,
                        stats,
                        checkpoint_step=self.completed_steps,
                        plot_images=epoch_end or self.completed_steps == 1,
                    )
                    self.monitor.record_frequency(
                        self.completed_steps, stats,
                        plot_image=epoch_end,
                    )
                    if rank is not None:
                        self.monitor.record_rank(
                            self.completed_steps, rank
                        )
                    if epoch_end:
                        self.monitor.record_epoch(
                            self.epoch_index, self.completed_steps,
                            stats,
                            epoch_complete
                            and batch_index == len(train_loader) - 1,
                        )
                val_loss = valid_logging_outs["loss"]
                if not np.isfinite(val_loss):
                    raise ValueError("Validation loss must be finite.")
                if val_loss < best_val["loss"]:
                    self.best_step = self.completed_steps
                    best_val = valid_logging_outs.copy()
                    self.best_validation_loss = float(val_loss)
                    best_model = copy.deepcopy(self.model)
                    if self.cfg.checkpoint_step > -1:
                        self.save_checkpoints(best_val=True)
                        saved_best = True
                if self.status_callback is not None:
                    self.status_callback(
                        "training",
                        completed_steps=self.completed_steps,
                        total_steps=self.cfg.total_steps,
                        epoch=self.epoch_index,
                        total_epochs=self.cfg.get("epochs"),
                        completed_epochs=min(
                            self.completed_steps // self.steps_per_epoch,
                            self.cfg.epochs,
                        ) if self.cfg.get("epochs") is not None else None,
                    )
            if log_step:
                logging_out["loss"] = np.mean(total_loss)
                self.output_logs(logging_out, valid_logging_outs)
                total_loss = []
            if (
                ckpt_step and self.cfg.checkpoint_step > -1
                and not saved_best
            ):
                self.save_checkpoints()
            if self.progress.disable:
                self.progress.n += 1
            else:
                self.progress.update(1)
        return total_loss, (best_model, best_val)

    def save_epoch_snapshot(self) -> None:
        """Retain weight-only snapshots of completed epochs."""
        if not self.cfg.get("retain_epoch_checkpoints", False):
            return
        if self.completed_steps % self.steps_per_epoch:
            raise ValueError("Epoch snapshots require a completed data pass.")
        epoch = self.completed_steps // self.steps_per_epoch
        destination = Path(self.exp_dir) / "epochs" / f"epoch_{epoch:03d}.pth"
        if destination.exists():
            saved = torch.load(destination, map_location="cpu",
                               weights_only=False)
            if saved["completed_steps"] != self.completed_steps:
                raise ValueError("Existing epoch snapshot has different steps.")
            return
        state = self.task.save_model_weights(
            self.model, {}, self.cfg.multi_gpu,
        )
        state.update(epoch=epoch, completed_steps=self.completed_steps)
        state["model_cfg"] = (
            self.model.module.cfg if self.cfg.multi_gpu else self.model.cfg
        )
        atomic_write(destination, lambda p: torch.save(state, p),
                     lambda p: torch.load(p, map_location="cpu",
                                          weights_only=False))

    def run_fixed_validation(self) -> None:
        """Save shared-reference fixed-mask validation statistics per epoch.

        Conditions use the complete pinned validation set, original content
        coefficient and frozen training means. Diagnostic objectives use the
        unweighted reference; regular validation selects each model using its
        own optimised objective. No diagnostic updates the stopping state.
        """
        options = self.cfg.get("fixed_validation")
        if not options:
            return
        from util.experiments.fixed_mask_diagnostics import (
            balanced_bins, evaluate_conditions,
        )
        from util.mask_utils import test_masked_dataset
        from omegaconf import OmegaConf
        from util.experiments.storage import digest

        dataset = test_masked_dataset(self.task.valid_set, targets_only=True)
        if len(dataset) != options.batches * self.cfg.valid_batch_size:
            raise ValueError(
                "Fixed validation differs from pinned batch budget."
            )
        baseline = json.loads(
            Path(self.cfg.fixed_training_baseline).read_text()
        )
        expected = digest(self.task.dataset.selection_manifest)
        if baseline["identity"]["selection_sha256"] != expected:
            raise ValueError("Frozen training means use a different selection.")
        mean = torch.tensor(baseline["mean"], dtype=torch.float64)
        durations = list(options.durations_frames)
        conditions = [("spectral", "spectral", 0)] + [
            (f"{kind}_{d}", kind, d) for d in durations
            for kind in ("temporal", "joint")
        ]
        root = Path(self.exp_dir).parent / "diagnostics/validation_fixed"
        root = root / f"epoch_{self.epoch_index:03d}"
        identity = {
            "step": self.completed_steps,
            "selection_sha256": expected,
            "configuration": OmegaConf.to_container(options),
        }
        destination = root / "results.json"
        if destination.exists():
            if json.loads(destination.read_text())["identity"] != identity:
                raise ValueError("Fixed validation identity changed.")
            return
        self.model.eval()
        result = evaluate_conditions(
            self.task, dataset, self.model, [str(self.device)], conditions,
            balanced_bins(len(dataset), len(mean), options.seed), mean,
            durations, options.seed, self.criterion.cfg.alpha,
            self.cfg.valid_batch_size, self.cfg.num_workers, root, identity,
            lambda done, total, name: self.report_progress(
                f"validation_fixed_{name}", done, total,
            ), None,
        )
        write_json(destination, {"identity": identity, "conditions": result})
        (root / "test_partial.json").unlink(missing_ok=True)

    def scheduler_step(self):
        pass

    def train(self):
        if self.monitor is not None:
            sources = []
            for subset in (self.task.train_set, self.task.valid_set):
                while isinstance(subset, Subset):
                    subset = subset.dataset
                sources.append(subset)
            if sources[0] is not sources[1]:
                raise ValueError(
                    "Training and validation monitors require the same "
                    "source dataset and STFT preprocessing settings."
                )
            windows = len(self.task.valid_set)
            if windows < 2:
                warnings.warn(
                    "Validation embedding rank requires at least two windows."
                )
            else:
                self.rank_indices = rank_positions(windows)
                write_json(
                    self.monitor.root / "rank_subset.json",
                    {"validation_windows": windows,
                     "positions": self.rank_indices.tolist()},
                )
        train_loader = self.get_batch_iterator(
            self.task.train_set,
            self.cfg.train_batch_size,
            shuffle=self.cfg.shuffle,
            num_workers=self.cfg.num_workers,
            persistent_workers=self.cfg.num_workers > 0,
        )

        self.steps_per_epoch = len(train_loader)
        if len(train_loader) == 0:
            raise ValueError(
                "Training loader must contain at least one batch."
            )
        if self.cfg.total_steps <= 0 or self.cfg.log_step <= 0:
            raise ValueError("total_steps and log_step must be positive.")
        if self.cfg.checkpoint_step == 0:
            raise ValueError("checkpoint_step must be positive or negative.")
        monitor_step = self.cfg.get("train_monitor_step")
        if monitor_step is not None and (
            not isinstance(monitor_step, int) or monitor_step <= 0
        ):
            raise ValueError("train_monitor_step must be a positive integer.")
        total_loss = []
        best_val = {"loss": float("inf")}
        best_model = None
        if self.completed_steps:
            best_val = {"loss": self.best_validation_loss}
            best_model = copy.deepcopy(self.model)
            self.task.load_model_weights(
                best_model, self.resumed_best_state,
                self.cfg.multi_gpu,
            )
            del self.resumed_best_state
            self.monitor.start_step = self.completed_steps + 1
        best_state = (best_model, best_val)
        with logging_redirect_tqdm():
            if self.cfg.checkpoint_step > -1 and not self.completed_steps:
                self.save_checkpoints()
            self.epoch_index = self.completed_steps // self.steps_per_epoch
            skipped = self.completed_steps % self.steps_per_epoch
            if (self.cfg.get("parent_checkpoint")
                    and not self.cfg.get("resume_from_ckpt")):
                initial = self.get_valid_outs()["loss"]
                self.best_validation_loss = float(initial)
                best_state = (copy.deepcopy(self.model), {"loss": initial})
                self.save_checkpoints(best_val=True)
                self.save_epoch_snapshot()
            if self.cfg.get("artifact_dir") and (
                self.completed_steps == self.cfg.total_steps
                or self.stopped_early
            ):
                publish_training_completion(
                    Path(self.exp_dir), self.cfg.total_steps,
                )
            if self.pending_fixed_validation:
                self.run_fixed_validation()
                self.pending_fixed_validation = False
                self.save_checkpoints()
            while (self.progress.n < self.progress.total
                   and not self.stopped_early):
                if not skipped:
                    self.epoch_best_at_start = (
                        self.best_validation_loss
                        if self.best_validation_loss is not None
                        else float("inf")
                    )
                self.epoch_index += 1
                remaining = train_loader
                if skipped:
                    offset = skipped * self.cfg.train_batch_size
                    indices = range(offset, len(self.task.train_set))
                    remaining = self.get_batch_iterator(
                        Subset(self.task.train_set, indices),
                        self.cfg.train_batch_size,
                        shuffle=False,
                        num_workers=self.cfg.num_workers,
                        persistent_workers=self.cfg.num_workers > 0,
                    )
                total_loss, best_state = self.run_epoch(
                    remaining, total_loss, best_state,
                    epoch_complete=not bool(skipped),
                )
                skipped = 0
                patience = self.cfg.get("early_stopping_patience")
                if patience is not None:
                    if type(patience) is not int or patience < 1:
                        raise ValueError(
                            "Early-stop patience must be positive."
                        )
                    improved = (
                        self.best_validation_loss < self.epoch_best_at_start
                    )
                    self.stale_epochs = 0 if improved else self.stale_epochs + 1
                    self.stopped_early = self.stale_epochs >= patience
                self.save_epoch_snapshot()
                self.pending_fixed_validation = bool(
                    self.cfg.get("fixed_validation")
                )
                if self.cfg.checkpoint_step > 0:
                    self.save_checkpoints()
                if self.cfg.get("artifact_dir") and (
                    self.completed_steps == self.cfg.total_steps
                    or self.stopped_early
                ):
                    publish_training_completion(
                        Path(self.exp_dir), self.cfg.total_steps,
                    )
                self.run_fixed_validation()
                self.pending_fixed_validation = False
                if self.cfg.checkpoint_step > 0:
                    self.save_checkpoints()
            self.progress.close()
        return best_state[0]

    def test(self, best_model: torch.nn.Module) -> dict:
        """Evaluate explicitly named test policies with separate monitor outputs."""
        policies = self.cfg.get("test_evaluations")
        if not policies:
            return self._test_once(best_model)
        original_cfg, original_monitor = self.cfg, self.monitor
        if original_monitor is None:
            raise ValueError(
                "Named test evaluations require lifecycle monitoring."
            )
        metadata = json.loads(
            (original_monitor.root / "metadata.json").read_text()
        )
        results = {}
        try:
            for index, (name, policy) in enumerate(policies.items()):
                if Path(name).name != name:
                    raise ValueError(
                        "Test evaluation name must be one path component."
                    )
                self.cfg = copy.deepcopy(original_cfg)
                with open_dict(self.cfg):
                    self.cfg.test_masking = policy
                root = original_monitor.root / "test_evaluations" / name
                self.monitor = TrainingMonitor(
                    root / "monitoring", original_monitor.frequencies,
                    original_monitor.alpha, metadata["content_threshold"],
                    original_monitor.lr_groups,
                    metadata["stft_normalization_mode"], tensorboard=False,
                )
                try:
                    results[name] = self._test_once(
                        best_model, index, len(policies)
                    )
                    write_json(root / "policy.json", dict(policy))
                finally:
                    self.monitor.close()
        finally:
            self.cfg, self.monitor = original_cfg, original_monitor
        return {"test_evaluations": results}

    def _test_once(
        self, best_model: torch.nn.Module, policy_index: int = 0,
        policy_count: int = 1,
    ) -> dict:
        """Evaluate the validation-selected model on the held-out loader.

        Parameters
        ----------
        best_model : torch.nn.Module
            Snapshot returned by training at the lowest validation loss.

        Returns
        -------
        dict
            Test metrics produced by the task criterion.
        """
        if best_model is None:
            raise ValueError("Testing requires a validation-selected model.")
        best_model.eval()
        from util.mask_utils import test_masked_dataset

        test_set = test_masked_dataset(
            self.task.test_set, self.cfg.get("test_masking")
        )
        test_loader = self.get_batch_iterator(
            test_set,
            self.cfg.valid_batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            persistent_workers=self.cfg.num_workers > 0,
        )

        diagnostics = diagnostic_names(self.cfg.get("test_diagnostics"))
        if self.task.cfg.name == "spec_pretrain":
            test_outs = self.task.get_valid_outs(
                best_model, test_loader, self.criterion, self.device,
                rank_indices=(
                    rank_positions(len(self.task.test_set))
                    if "representation_rank" in diagnostics else None
                ),
                collect_frequency="frequency_errors" in diagnostics,
                progress_callback=lambda done, total:
                self.report_progress(
                    "test", policy_index * total + done, policy_count * total
                ),
            )
        else:
            self.report_progress("test", 0, len(test_loader))
            test_outs = self.task.get_valid_outs(
                best_model, test_loader, self.criterion, self.device
            )
            self.report_progress("test", len(test_loader), len(test_loader))
        if self.monitor is not None:
            if "representation_rank" in diagnostics:
                self.monitor.record_rank(
                    self.best_step, test_outs.pop("rank_embeddings"),
                    phase="test",
                )
            if "frequency_errors" in diagnostics:
                self.monitor.record_frequency(
                    self.best_step, test_outs["monitor_accumulator"],
                    phase="test",
                )
            self.monitor.record(
                "test",
                self.completed_steps,
                test_outs.pop("monitor_accumulator"),
                checkpoint_step=self.best_step,
            )
        log.info(f"test_results {test_outs}")
        return test_outs

    def get_batch_iterator(self, dataset, batch_size, **kwargs):
        return self.task.get_batch_iterator(dataset, batch_size, **kwargs)
