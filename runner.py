"""Train task models and evaluate the lowest-validation-loss snapshot.

Checkpoints in the run directory contain ``model`` parameter tensors,
``model_cfg`` architecture settings, ``optim`` optimizer state, and
``scheduler`` learning-rate state. ``checkpoint_last.pth`` stores the latest
saved update; ``checkpoint_best.pth`` stores the lowest observed validation
loss. A negative checkpoint interval disables disk saves, not model selection.
"""

import copy
from pathlib import Path
from util.experiments.storage import atomic_write
from tqdm.contrib.logging import logging_redirect_tqdm
import numpy as np
import os
from tqdm import tqdm
import torch
import logging
from tensorboardX import SummaryWriter
from schedulers import build_scheduler
import torch_optimizer as torch_optim

log = logging.getLogger(__name__)


class Runner:
    def __init__(self, cfg, task, model, criterion):
        self.cfg = cfg
        self.model = model
        self.task = task
        self.evaluator = None
        self.status_callback = None
        self.completed_steps = 0
        self.device = cfg.device
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
        if self.output_tb:
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

        if "start_from_ckpt" in cfg:
            self.load_from_ckpt()

    def load_from_ckpt(self):
        ckpt_path = self.cfg.start_from_ckpt
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

    def get_valid_outs(self):
        valid_loader = self.get_batch_iterator(
            self.task.valid_set,
            self.cfg.valid_batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
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
        all_states["completed_steps"] = self.completed_steps
        all_states["optim"] = self.optim.state_dict()
        all_states["scheduler"] = self.scheduler.get_state_dict()
        if self.cfg.multi_gpu:
            all_states["model_cfg"] = self.model.module.cfg
        else:
            all_states["model_cfg"] = self.model.cfg
        self.save_checkpoint_last(all_states)
        if best_val:
            self.save_checkpoint_last(all_states, best_val)

    def run_epoch(self, train_loader, total_loss, best_state):
        best_model, best_val = best_state
        for batch in train_loader:
            if self.progress.n >= self.progress.total:
                break
            self.model.train()
            logging_out = self.task.train_step(
                batch,
                self.model,
                self.criterion,
                self.optim,
                self.scheduler,
                self.device,
                self.cfg.grad_clip,
            )
            self.completed_steps += 1
            total_loss.append(logging_out["loss"])
            log_step = (
                self.progress.n % self.cfg.log_step == 0
                or self.progress.n == self.progress.total - 1
            )

            ckpt_step = False
            if self.cfg.checkpoint_step > -1:
                ckpt_step = (
                    self.progress.n % self.cfg.checkpoint_step == 0
                    or self.progress.n == self.progress.total - 1
                )

            valid_logging_outs = {}
            saved_best = False
            if ckpt_step or log_step:
                self.model.eval()
                valid_logging_outs = self.get_valid_outs()
                val_loss = valid_logging_outs["loss"]
                if not np.isfinite(val_loss):
                    raise ValueError("Validation loss must be finite.")
                if val_loss < best_val["loss"]:
                    best_val = valid_logging_outs.copy()
                    best_model = copy.deepcopy(self.model)
                    if self.cfg.checkpoint_step > -1:
                        self.save_checkpoints(best_val=True)
                        saved_best = True
                if self.status_callback is not None:
                    self.status_callback(
                        "running",
                        completed_steps=self.completed_steps,
                        validation=valid_logging_outs,
                        best_validation=best_val,
                    )
            if log_step:
                logging_out["loss"] = np.mean(total_loss)
                self.output_logs(logging_out, valid_logging_outs)
                total_loss = []
            if ckpt_step and not saved_best:
                self.save_checkpoints()
            if self.progress.disable:
                self.progress.n += 1
            else:
                self.progress.update(1)
        return total_loss, (best_model, best_val)

    def scheduler_step(self):
        pass

    def train(self):
        train_loader = self.get_batch_iterator(
            self.task.train_set,
            self.cfg.train_batch_size,
            shuffle=self.cfg.shuffle,
            num_workers=self.cfg.num_workers,
            persistent_workers=self.cfg.num_workers > 0,
        )

        if len(train_loader) == 0:
            raise ValueError(
                "Training loader must contain at least one batch."
            )
        if self.cfg.total_steps <= 0 or self.cfg.log_step <= 0:
            raise ValueError("total_steps and log_step must be positive.")
        if self.cfg.checkpoint_step == 0:
            raise ValueError("checkpoint_step must be positive or negative.")
        total_loss = []
        best_val = {"loss": float("inf")}
        best_model = None
        best_state = (best_model, best_val)
        with logging_redirect_tqdm():
            if self.cfg.checkpoint_step > -1:
                self.save_checkpoints()
            while self.progress.n < self.progress.total:
                total_loss, best_state = self.run_epoch(
                    train_loader, total_loss, best_state
                )
            self.progress.close()
        return best_state[0]

    def test(self, best_model: torch.nn.Module) -> dict:
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
        test_loader = self.get_batch_iterator(
            self.task.test_set,
            self.cfg.valid_batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            persistent_workers=self.cfg.num_workers > 0,
        )

        test_outs = self.task.get_valid_outs(
            best_model, test_loader, self.criterion, self.device
        )
        log.info(f"test_results {test_outs}")
        return test_outs

    def get_batch_iterator(self, dataset, batch_size, **kwargs):
        return self.task.get_batch_iterator(dataset, batch_size, **kwargs)
