"""Scientific diagnostics for existing masked-spectrogram passes.

``batch_statistics`` consumes (batch, frames, frequencies) tensors and returns
small detached sufficient statistics. ``TrainingMonitor`` writes fit-local
CSV tables and TensorBoard events. metrics.csv holds pooled/scalar metrics;
mask_groups.csv holds nine conditional groups and strategy/action marginals;
frequency_coverage.csv holds counts and valid-cell exposure per frequency.
Rows identify phase, update, interval, and evaluated checkpoint. Missing
conditional means are empty CSV fields, never fabricated zeros. Metadata
records normalization and categorical semantics; no spectrograms are saved.
"""

import csv
import os
from pathlib import Path

import numpy as np
import torch
from tensorboardX import SummaryWriter

from .storage import write_json

STRATEGIES = ("temporal_only", "spectral_only", "overlap")
ACTIONS = ("zero", "random", "unchanged")
SUM_FIELDS = (
    "count",
    "absolute_error_sum",
    "content_count",
    "content_error_sum",
    "zero_baseline_error_sum",
    "target_sum",
    "target_square_sum",
    "prediction_sum",
    "prediction_square_sum",
)
ORIGINAL_METRICS = ("loss", "l1_loss", "content_aware_loss")
GROUPS = len(STRATEGIES) * len(ACTIONS)
CONTEXT_FIELDS = {
    "phase",
    "step",
    "interval_start",
    "interval_end",
    "checkpoint_step",
}


@torch.no_grad()
def batch_statistics(
    output: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    padding: torch.Tensor,
    provenance: dict,
    threshold: float,
) -> dict:
    """Reduce valid selected cells without retaining an autograd graph.

    Parameters
    ----------
    output, target, mask : torch.Tensor
        Prediction, target, and original selection of shape (B, T, F).
    padding : torch.Tensor
        Boolean padding indicator of shape (B, T).
    provenance : dict
        uint8 selection and final-action arrays of shape (B, T, F).
    threshold : float
        Original absolute-target content threshold.

    Returns
    -------
    dict
        Float64 (9, 9) sufficient statistics, int64 (3, F) selection counts,
        and int64 (F,) valid-cell exposures, on the CPU.
    """
    selection = provenance["selection"].to(output.device)
    action = provenance["action"].to(output.device)
    if any(x.shape != target.shape for x in (output, mask, selection, action)):
        raise ValueError(
            "Prediction, target, masks and provenance must match."
        )
    if padding.shape != target.shape[:2]:
        raise ValueError("Padding must have shape (batch, frames).")
    valid = (~padding.bool()).unsqueeze(-1).expand_as(mask)
    selected = mask.bool() & valid
    if not torch.equal((selection != 0) & valid, selected):
        raise ValueError("Mask provenance differs from original selection.")
    if ((selection > 3) | (action > 2)).any():
        raise ValueError("Invalid selection or replacement action code.")
    y = target.detach()[selected].double()
    p = output.detach()[selected].double()
    if not torch.isfinite(y).all() or not torch.isfinite(p).all():
        raise ValueError("Selected targets and predictions must be finite.")
    if not y.numel():
        raise ValueError("Monitoring requires selected, non-padding cells.")
    action_index = torch.tensor([2, 0, 1], device=output.device)
    indices = (selection[selected].long() - 1) * len(ACTIONS)
    indices += action_index[action[selected].long()]
    error = (y - p).abs()
    content = y.abs() > threshold
    values = torch.stack(
        (
            torch.ones_like(y),
            error,
            content.double(),
            error * content,
            y.abs(),
            y,
            y.square(),
            p,
            p.square(),
        ),
        dim=1,
    )
    sums = torch.zeros(
        (GROUPS, len(SUM_FIELDS)), dtype=torch.float64, device=output.device
    )
    sums.index_add_(0, indices, values)
    coverage = torch.stack(
        [
            ((selection == code) & valid).sum(dim=(0, 1))
            for code in range(1, len(STRATEGIES) + 1)
        ]
    )
    return {
        "sums": sums.cpu().numpy(),
        "coverage": coverage.cpu().numpy(),
        "exposure": valid.sum(dim=(0, 1)).cpu().numpy(),
    }


class Accumulator:
    """Pool additive statistics across batches, keeping scalar loss intact."""

    def __init__(self) -> None:
        self.sums = np.zeros((GROUPS, len(SUM_FIELDS)), dtype=np.float64)
        self.coverage = None
        self.exposure = None
        self.batches = 0
        self.original = dict.fromkeys(ORIGINAL_METRICS, 0.0)
        self.gradient_sum = 0.0
        self.gradient_max = 0.0
        self.clipped = 0

    def add(self, outs: dict, grad_clip: float | None = None) -> None:
        """Add one batch; grad_clip is optional and defaults to disabled."""
        stats = outs["monitor_statistics"]
        self.sums += stats["sums"]
        if self.coverage is None:
            self.coverage = stats["coverage"].copy()
            self.exposure = stats["exposure"].copy()
        else:
            if self.coverage.shape != stats["coverage"].shape:
                raise ValueError("Frequency count changed within a pass.")
            self.coverage += stats["coverage"]
            self.exposure += stats["exposure"]
        self.batches += 1
        for key in ORIGINAL_METRICS:
            self.original[key] += outs[key]
        if "grad_norm" in outs:
            norm = float(outs["grad_norm"])
            if not np.isfinite(norm):
                raise ValueError("Gradient norm must be finite.")
            self.gradient_sum += norm
            self.gradient_max = max(self.gradient_max, norm)
            self.clipped += bool(grad_clip and norm > grad_clip)


def summarize(sums: np.ndarray, alpha: float) -> dict:
    """Convert one group's nine sums into conditional population metrics."""
    if not np.isfinite(sums).all():
        raise ValueError("Monitor statistics must be finite.")
    result = dict(zip(SUM_FIELDS, sums.tolist()))
    n, _, nc = sums[:3]
    result["count"] = int(n)
    result["content_count"] = int(nc)
    result["l1"] = sums[1] / n if n else None
    result["content_l1"] = sums[3] / nc if nc else None
    result["weighted_content_l1"] = alpha * sums[3] / nc if nc else None
    result["diagnostic_objective"] = (
        result["l1"] + result["weighted_content_l1"] if n and nc else None
    )
    result["zero_baseline_l1"] = sums[4] / n if n else None
    for name, i in (("target", 5), ("prediction", 7)):
        mean = sums[i] / n if n else None
        result[f"{name}_mean"] = mean
        # Roundoff can make the population variance slightly negative.
        variance = max(0.0, sums[i + 1] / n - mean**2) if n else None
        result[f"{name}_std"] = np.sqrt(variance) if n else None
    return result


class TrainingMonitor:
    """Write compact diagnostics underneath one new fit's artifact root."""

    def __init__(
        self,
        root: Path,
        frequencies: np.ndarray,
        alpha: float,
        threshold: float,
        optimizer_groups: int,
        tensorboard: bool = True,
    ) -> None:
        """Create writers; tensorboard is optional and defaults to True.

        Parameters
        ----------
        root : Path
            New monitoring directory, never a completed fit's directory.
        frequencies : numpy.ndarray
            Actual STFT bin centers in Hz, shape (F,).
        alpha, threshold : float
            Existing content weighting and absolute-target threshold.
        optimizer_groups : int
            Number of optimizer groups for stable learning-rate columns.
        tensorboard : bool, optional
            Enable event output, default True; CSV is always written.
        """
        self.root = root.resolve()
        self.frequencies = np.asarray(frequencies, dtype=float)
        if (
            self.frequencies.ndim != 1
            or not self.frequencies.size
            or not np.isfinite(self.frequencies).all()
            or np.any(np.diff(self.frequencies) <= 0)
        ):
            raise ValueError("Expected finite, increasing STFT bin centers.")
        if optimizer_groups <= 0:
            raise ValueError(
                "Expected at least one optimizer parameter group."
            )
        self.lr_groups = optimizer_groups
        self.alpha = alpha
        self.root.mkdir(parents=True, exist_ok=False)
        self.streams = {}
        self.writers = {}
        self.train = Accumulator()
        self.start_step = 1
        self.cumulative_coverage = np.zeros(
            (len(STRATEGIES), len(frequencies)), dtype=np.int64
        )
        self.cumulative_exposure = np.zeros(len(frequencies), dtype=np.int64)
        self.writer = None
        self.closed = False
        write_json(
            self.root / "metadata.json",
            {
                "schema_version": 1,
                "frequency_hz": self.frequencies.tolist(),
                "strategies": list(STRATEGIES),
                "actions": list(ACTIONS),
                "alpha": alpha,
                "content_threshold": threshold,
                "units": "preprocessed spectrogram magnitude",
                "normalization": "sum of cell errors / own selected-cell count",
                "content_normalization": (
                    "own count with " "abs(target) > threshold"
                ),
                "original_objective": "unchanged mean of original batch losses",
                "missing": "empty CSV field; omitted TensorBoard scalar",
                "action_semantics": (
                    "last actual write; " "no-op preserves prior action"
                ),
                "random_semantics": "copied source may already be corrupted",
                "variance": "population variance from pooled float64 moments",
                "step": (
                    "completed optimizer updates; "
                    "evaluation uses checkpoint_step"
                ),
                "train_interval": "predictions collected while weights change",
            },
        )
        if tensorboard:
            self.writer = SummaryWriter(str(self.root / "tensorboard"))

    def _row(self, table: str, row: dict, tag: str | None = None) -> None:
        """Append a row and mirror numeric measurements to TensorBoard."""
        if table not in self.writers:
            stream = (self.root / f"{table}.csv").open("x", newline="")
            self.streams[table] = stream
            self.writers[table] = csv.DictWriter(stream, fieldnames=list(row))
            self.writers[table].writeheader()
        for value in row.values():
            if isinstance(value, (float, np.floating)) and not np.isfinite(
                value
            ):
                raise ValueError(
                    "CSV monitor values must be finite or missing."
                )
        self.writers[table].writerow(row)
        if self.writer is not None and tag is not None:
            for key, value in row.items():
                if key not in CONTEXT_FIELDS and isinstance(
                    value, (int, float, np.number)
                ):
                    self.writer.add_scalar(
                        f"{row['phase']}/{tag}/{key}", value, row["step"]
                    )

    def record(
        self,
        phase: str,
        step: int,
        accumulator: Accumulator,
        checkpoint_step: int | None = None,
        learning_rates: list | None = None,
    ) -> None:
        """Write one training interval or evaluation pass.

        Parameters
        ----------
        phase : str
            One of train, validation, or test.
        step : int
            Completed optimizer updates.
        accumulator : Accumulator
            Pooled sufficient statistics for this interval or pass.
        checkpoint_step : int or None, optional
            Evaluated snapshot update, default None for training intervals.
        learning_rates : list or None, optional
            Last applied rate per group, default None for evaluation.
        """
        if phase not in {"train", "validation", "test"}:
            raise ValueError(f"Unknown monitoring phase: {phase!r}")
        if phase == "train" and (
            learning_rates is None or len(learning_rates) != self.lr_groups
        ):
            raise ValueError("Training requires one learning rate per group.")
        if not accumulator.batches:
            raise ValueError("Cannot record an empty monitoring interval.")
        if accumulator.coverage.shape[1] != len(self.frequencies):
            raise ValueError(
                "STFT frequency axis differs from batch features."
            )
        if np.any(accumulator.exposure <= 0):
            raise ValueError(
                "Every frequency needs positive valid-cell exposure."
            )
        context = {
            "phase": phase,
            "step": step,
            "interval_start": self.start_step if phase == "train" else step,
            "interval_end": step,
            "checkpoint_step": checkpoint_step,
        }
        overall = summarize(accumulator.sums.sum(axis=0), self.alpha)
        overall.update(
            {
                f"original_{k}": v / accumulator.batches
                for k, v in accumulator.original.items()
            }
        )
        overall["batches"] = accumulator.batches
        overall.update(
            {
                "gradient_norm_mean": (
                    accumulator.gradient_sum / accumulator.batches
                    if phase == "train"
                    else None
                ),
                "gradient_norm_max": (
                    accumulator.gradient_max if phase == "train" else None
                ),
                "clipping_fraction": (
                    accumulator.clipped / accumulator.batches
                    if phase == "train"
                    else None
                ),
            }
        )
        # Group count stays fixed for a run; evaluations use empty LR fields.
        for i in range(self.lr_groups):
            overall[f"lr_group_{i}_last_used"] = (
                learning_rates[i] if learning_rates is not None else None
            )
        self._row("metrics", {**context, **overall}, "overall")
        grid = accumulator.sums.reshape(
            len(STRATEGIES), len(ACTIONS), len(SUM_FIELDS)
        )
        for i, strategy in enumerate(STRATEGIES):
            for j, action in enumerate(ACTIONS):
                self._group(context, strategy, action, grid[i, j])
            self._group(context, strategy, "all", grid[i].sum(axis=0))
        for j, action in enumerate(ACTIONS):
            self._group(context, "all", action, grid[:, j].sum(axis=0))
        self._coverage(
            context, "interval", accumulator.coverage, accumulator.exposure
        )
        if phase == "train":
            self.cumulative_coverage += accumulator.coverage
            self.cumulative_exposure += accumulator.exposure
            self._coverage(
                context,
                "cumulative",
                self.cumulative_coverage,
                self.cumulative_exposure,
            )
            self.start_step = step + 1
        if self.writer is not None:
            self._heatmap(context, grid)
        self.flush()

    def _group(
        self, context: dict, strategy: str, action: str, sums: np.ndarray
    ) -> None:
        """Write a conditional group without rescaling its denominator."""
        self._row(
            "mask_groups",
            {
                **context,
                "strategy": strategy,
                "action": action,
                **summarize(sums, self.alpha),
            },
            f"masks/{strategy}/{action}",
        )

    def _coverage(
        self,
        context: dict,
        scope: str,
        counts: np.ndarray,
        exposure: np.ndarray,
    ) -> None:
        """Save frequency counts and rates; plot the actual Hz axis."""
        series = dict(zip(STRATEGIES, counts))
        series.update(
            {
                "temporal": counts[0] + counts[2],
                "spectral": counts[1] + counts[2],
                "union": counts.sum(axis=0),
            }
        )
        for name, selected in series.items():
            for i, hz in enumerate(self.frequencies):
                self._row(
                    "frequency_coverage",
                    {
                        **context,
                        "scope": scope,
                        "strategy": name,
                        "frequency_hz": hz,
                        "selected_cells": int(selected[i]),
                        "valid_cells": int(exposure[i]),
                        "selection_rate": (
                            selected[i] / exposure[i] if exposure[i] else None
                        ),
                    },
                )
        if self.writer is not None:
            self._frequency_plot(context, scope, series, exposure)

    def _frequency_plot(
        self, context: dict, scope: str, series: dict, exposure: np.ndarray
    ) -> None:
        """Render count/rate bars without expanding counts into samples."""
        import matplotlib.pyplot as plt
        import seaborn as sns

        with sns.axes_style("whitegrid"), sns.plotting_context("talk"):
            fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
            width = (
                np.min(np.diff(self.frequencies))
                if len(self.frequencies) > 1
                else 1.0
            ) * 0.8
            bottom = np.zeros_like(exposure, dtype=float)
            for name, color in zip(
                STRATEGIES, sns.color_palette("colorblind", 3)
            ):
                values = series[name]
                axes[0].bar(
                    self.frequencies,
                    values,
                    width,
                    bottom=bottom,
                    label=name,
                    color=color,
                )
                rates = np.divide(
                    values,
                    exposure,
                    out=np.zeros_like(bottom),
                    where=exposure > 0,
                )
                base = np.divide(
                    bottom,
                    exposure,
                    out=np.zeros_like(bottom),
                    where=exposure > 0,
                )
                axes[1].bar(
                    self.frequencies, rates, width, bottom=base, color=color
                )
                bottom += values
            axes[0].set_ylabel("Selected cells")
            axes[1].set_ylabel("Selection rate")
            axes[1].set_xlabel("Frequency (Hz)")
            axes[1].set_ylim(0, 1)
            axes[0].legend(
                frameon=False,
                loc="lower left",
                bbox_to_anchor=(0, 1),
                ncol=3,
                fontsize=12,
            )
            fig.tight_layout()
            self.writer.add_figure(
                f"{context['phase']}/coverage/{scope}", fig, context["step"]
            )
            plt.close(fig)

    def _heatmap(self, context: dict, grid: np.ndarray) -> None:
        """Plot conditional L1 with unobserved groups left blank."""
        import matplotlib.pyplot as plt
        import seaborn as sns

        values = np.full(grid.shape[:2], np.nan)
        np.divide(
            grid[:, :, 1], grid[:, :, 0], out=values, where=grid[:, :, 0] > 0
        )
        if not np.isfinite(values).any():
            return
        with sns.plotting_context("talk"):
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(
                values,
                mask=~np.isfinite(values),
                annot=True,
                fmt=".3g",
                cmap="viridis",
                ax=ax,
                xticklabels=ACTIONS,
                yticklabels=STRATEGIES,
                cbar_kws={"label": "Conditional L1"},
            )
            ax.tick_params(axis="y", rotation=0)
            fig.tight_layout()
            self.writer.add_figure(
                f"{context['phase']}/masks/l1", fig, context["step"]
            )
            plt.close(fig)

    def flush(self) -> None:
        """Durably flush CSV records and queued TensorBoard events."""
        for stream in self.streams.values():
            stream.flush()
            os.fsync(stream.fileno())
        if self.writer is not None:
            self.writer.flush()

    def close(self) -> None:
        """Close writers before completion checksums, also on failure."""
        if self.closed:
            return
        self.flush()
        for stream in self.streams.values():
            stream.close()
        if self.writer is not None:
            self.writer.close()
        self.closed = True
