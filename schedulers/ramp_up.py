"""Warm up learning rate linearly, with optional post-warmup decay.

The scheduler configuration supplies total_steps, warmup (a fraction), and
optional decay (False by default). When enabled, StepLR multiplies the rate
by 0.99 after each integer interval, with at most 100 intervals over the
remaining training budget. Learning rates are per optimiser update.
Checkpoint state uses the existing BaseScheduler serialization contract;
restoring a checkpoint preserves its stored schedule.
"""
from __future__ import annotations

import math
from typing import Any

from torch.optim import Optimizer
from torch.optim.lr_scheduler import StepLR
from warmup_scheduler import GradualWarmupScheduler

from .base_scheduler import BaseScheduler

DECAY_INTERVALS = 100
DECAY_GAMMA = 0.99


class RampUp(BaseScheduler):
    """Build linear warmup followed by a constant or decaying rate.

    Parameters
    ----------
    cfg : object
        Configuration with integer total_steps and finite warmup fraction.
        Optional decay is a boolean, default False. Warmup must span at
        least one update and end before the total training budget.
    optim : Optimizer
        Optimiser whose parameter-group learning rates are scheduled.
    """

    def __init__(self, cfg: Any, optim: Optimizer) -> None:
        super().__init__()
        self.cfg = cfg
        decay = getattr(cfg, "decay", False)
        if type(decay) is not bool:
            raise ValueError(f"decay must be a boolean; got {decay!r}.")
        total = cfg.total_steps
        fraction = cfg.warmup
        if type(total) is not int or total < 2:
            raise ValueError(
                f"total_steps must be an integer >= 2; got {total!r}."
            )
        if (not isinstance(fraction, (int, float))
                or isinstance(fraction, bool) or not math.isfinite(fraction)
                or not 0 < fraction < 1):
            raise ValueError(
                f"warmup must be a finite fraction in (0, 1); got {fraction!r}."
            )
        warmup = int(fraction * total)
        if warmup < 1:
            raise ValueError(
                "warmup * total_steps must span at least one update; "
                f"got {fraction} * {total}."
            )
        after = None
        if decay:
            interval = math.ceil((total - warmup) / DECAY_INTERVALS)
            after = StepLR(optim, step_size=interval, gamma=DECAY_GAMMA)
        self.scheduler = GradualWarmupScheduler(
            optim, multiplier=1, total_epoch=warmup, after_scheduler=after,
        )

    def step(self, loss: float | None = None) -> None:
        """Advance one optimiser update; loss is unused by this schedule."""
        self.scheduler.step()

    def get_lr(self) -> float:
        """Return the actual first-group rate, including delegated decay."""
        return float(self.scheduler.optimizer.param_groups[0]["lr"])

    def load_state_dict(self, init_state: dict[str, Any]) -> None:
        """Restore saved scheduling progress against the current optimiser.

        Parameters
        ----------
        init_state : dict
            Existing warmup scheduler state, including its optional nested
            StepLR. Stored schedules take precedence over new-run settings.
        """
        super().load_state_dict(init_state)
        after = self.scheduler.after_scheduler
        if after is not None:
            after.optimizer = self.scheduler.optimizer
