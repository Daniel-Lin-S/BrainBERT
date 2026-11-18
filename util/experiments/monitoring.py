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
from collections.abc import Mapping, Sequence
import gzip
import os
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from tensorboardX import SummaryWriter
from threadpoolctl import threadpool_limits

from .storage import write_json

STRATEGIES = ("temporal_only", "spectral_only", "overlap")
ACTIONS = ("zero", "random", "unchanged")
SUM_FIELDS = (
    "count",
    "absolute_error_sum",
    "content_count",
    "content_error_sum",
    "zero_baseline_error_sum",
    "content_zero_baseline_error_sum",
    "target_sum",
    "target_square_sum",
    "prediction_sum",
    "prediction_square_sum",
)
TENSORBOARD_GROUP_FIELDS = frozenset({
    "count", "content_count", "l1", "content_l1",
    "relative_l1", "relative_content_l1",
})
TENSORBOARD_OVERALL_EXCLUDED = frozenset(SUM_FIELDS) - {
    "count", "content_count",
}
ORIGINAL_METRICS = (
    "loss",
    "l1_loss",
    "content_l1",
    "content_aware_loss",
    "zero_baseline_l1",
    "content_zero_baseline_l1",
)
DIAGNOSTICS = frozenset({"representation_rank", "frequency_errors"})
RANK_SAMPLE_WINDOWS = 2048
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
    include_frequency: bool = False,
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
    include_frequency : bool, optional
        Collect (strategy, frequency, statistic) sums, default False.

    Returns
    -------
    dict
        Float64 (9, 10) sufficient statistics, int64 (3, F) counts,
        and int64 (F,) valid-cell exposures, on the CPU. Optional
        frequency sums have shape (3, F, 6).
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
            y.abs() * content,
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
    result = {
        "sums": sums.cpu().numpy(),
        "coverage": coverage.cpu().numpy(),
        "exposure": valid.sum(dim=(0, 1)).cpu().numpy(),
    }
    if include_frequency:
        bins = selected.nonzero(as_tuple=True)[2]
        frequencies = target.shape[2]
        group = (selection[selected].long() - 1) * frequencies + bins
        values = torch.stack(
            (torch.ones_like(y), error, content.double(),
             error * content, y.abs(), y.abs() * content), dim=1
        )
        frequency = torch.zeros(
            (len(STRATEGIES) * frequencies, 6),
            dtype=torch.float64, device=output.device,
        )
        frequency.index_add_(0, group, values)
        result["frequency"] = frequency.reshape(
            len(STRATEGIES), frequencies, 6
        ).cpu().numpy()
    return result


class Accumulator:
    """Pool additive statistics across batches, keeping scalar loss intact."""

    def __init__(self) -> None:
        self.sums = np.zeros((GROUPS, len(SUM_FIELDS)), dtype=np.float64)
        self.coverage = None
        self.exposure = None
        self.frequency = None
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
        if "frequency" in stats:
            if self.frequency is None:
                self.frequency = stats["frequency"].copy()
            else:
                self.frequency += stats["frequency"]
        self.batches += 1
        for key in ORIGINAL_METRICS:
            self.original[key] += outs[key]
        for key in (
            "spectral_content_l1", "unweighted_total_loss",
            "spectral_total_loss",
        ):
            if key in outs:
                self.original[key] = self.original.get(key, 0.) + outs[key]
        if "grad_norm" in outs:
            norm = float(outs["grad_norm"])
            if not np.isfinite(norm):
                raise ValueError("Gradient norm must be finite.")
            self.gradient_sum += norm
            self.gradient_max = max(self.gradient_max, norm)
            self.clipped += bool(grad_clip and norm > grad_clip)


def summarize(sums: np.ndarray, alpha: float) -> dict:
    """Convert one group's ten sums into conditional population metrics."""
    if not np.isfinite(sums).all():
        raise ValueError("Monitor statistics must be finite.")
    result = dict(zip(SUM_FIELDS, sums.tolist()))
    n, _, nc = sums[:3]
    result["count"] = int(n)
    result["content_count"] = int(nc)
    result["l1"] = sums[1] / n if n else None
    result["content_l1"] = sums[3] / nc if nc else None
    result["weighted_content_l1"] = alpha * sums[3] / nc if nc else None
    result["zero_baseline_l1"] = sums[4] / n if n else None
    result["content_zero_baseline_l1"] = sums[5] / nc if nc else None
    result["relative_l1"] = sums[1] / sums[4] if sums[4] > 0 else None
    result["relative_content_l1"] = sums[3] / sums[5] if sums[5] > 0 else None
    for name, i in (("target", 6), ("prediction", 8)):
        mean = sums[i] / n if n else None
        result[f"{name}_mean"] = mean
        # Roundoff can make the population variance slightly negative.
        variance = max(0.0, sums[i + 1] / n - mean**2) if n else None
        result[f"{name}_std"] = np.sqrt(variance) if n else None
    return result


def summarize_accumulator(
    accumulator: Accumulator, alpha: float, training: bool,
) -> dict:
    """Summarize original and pooled losses over one nonempty pass."""
    if not accumulator.batches:
        raise ValueError("Cannot summarize an empty monitoring pass.")
    result = summarize(accumulator.sums.sum(axis=0), alpha)
    result.update({
        f"original_{name}": value / accumulator.batches
        for name, value in accumulator.original.items()
    })
    baseline = accumulator.original["zero_baseline_l1"]
    content_base = accumulator.original["content_zero_baseline_l1"]
    result["original_relative_l1"] = (
        accumulator.original["l1_loss"] / baseline
        if baseline > 0 else None
    )
    result["original_relative_content_l1"] = (
        accumulator.original["content_l1"] / content_base
        if content_base > 0 else None
    )
    result["batches"] = accumulator.batches
    result["gradient_norm_mean"] = (
        accumulator.gradient_sum / accumulator.batches
        if training else None
    )
    result["gradient_norm_max"] = (
        accumulator.gradient_max if training else None
    )
    result["clipping_fraction"] = (
        accumulator.clipped / accumulator.batches if training else None
    )
    return result


def diagnostic_names(value: object) -> frozenset:
    """Resolve explicit diagnostic names; missing or empty means disabled."""
    if value is None or value == "" or (
        isinstance(value, (Mapping, Sequence)) and len(value) == 0
    ):
        return frozenset()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("test_diagnostics must be a list of diagnostic names.")
    names = frozenset(value)
    if names - DIAGNOSTICS:
        raise ValueError(
            f"Unknown test diagnostics: {sorted(names - DIAGNOSTICS)}"
        )
    return names


def rank_positions(windows: int) -> np.ndarray:
    """Select the same bounded, evenly spaced rank subset for any split."""
    if windows < 2:
        raise ValueError("Embedding rank requires at least two windows.")
    return np.unique(np.linspace(
        0, windows - 1, min(RANK_SAMPLE_WINDOWS, windows), dtype=int,
    ))


def embedding_rank(embeddings: np.ndarray) -> dict:
    """Measure diversity of centered validation window embeddings (N, D)."""
    values = np.asarray(embeddings, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("Effective rank requires at least two windows.")
    if not np.isfinite(values).all():
        raise ValueError("Validation embeddings must be finite.")
    centered = values - values.mean(axis=0)
    with threadpool_limits(limits=1):
        covariance = centered.T @ centered / (len(values) - 1)
        eigenvalues = np.clip(np.linalg.eigvalsh(covariance), 0, None)[::-1]
    total = eigenvalues.sum()
    maximum = min(len(values) - 1, values.shape[1])
    if total == 0:
        warnings.warn("Validation embeddings have zero centered variance.")
        effective = 0.0
        pca95 = 0
    else:
        probability = eigenvalues[eigenvalues > 0] / total
        effective = float(np.exp(-np.sum(probability * np.log(probability))))
        pca95 = int(np.searchsorted(
            np.cumsum(eigenvalues), 0.95 * total
        ) + 1)
    return {
        "windows": len(values),
        "embedding_dim": values.shape[1],
        "effective_rank": effective,
        "effective_rank_fraction": effective / maximum,
        "pca95_components": pca95,
    }


class TrainingMonitor:
    """Write compact diagnostics underneath one new fit's artifact root."""

    def __init__(
        self,
        root: Path,
        frequencies: np.ndarray,
        alpha: float,
        threshold: float,
        optimizer_groups: int,
        normalization_mode: str,
        tensorboard: bool = True,
        tensorboard_root: Path | None = None,
        purge_step: int | None = None,
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
        normalization_mode : str
            STFT normalization applied by the shared source dataset.
        tensorboard : bool, optional
            Enable event output, default True; CSV is always written.
        tensorboard_root : Path or None, optional
            Shared event directory across resumes; default root/tensorboard.
        purge_step : int or None, optional
            First unsaved update to hide after recovery, default None.
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
        if not isinstance(normalization_mode, str) or not normalization_mode:
            raise ValueError(
                "Monitoring requires an explicit STFT normalization mode."
            )
        self.lr_groups = optimizer_groups
        self.alpha = alpha
        self.root.mkdir(parents=True, exist_ok=False)
        self.streams = {}
        self.writers = {}
        self.train = Accumulator()
        self.epoch_train = Accumulator()
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
                "schema_version": 2,
                "frequency_hz": self.frequencies.tolist(),
                "strategies": list(STRATEGIES),
                "actions": list(ACTIONS),
                "alpha": alpha,
                "content_threshold": threshold,
                "units": "preprocessed spectrogram magnitude",
                "stft_normalization_mode": normalization_mode,
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
            event_root = (
                self.root / "tensorboard"
                if tensorboard_root is None else tensorboard_root.resolve()
            )
            self.writer = SummaryWriter(
                str(event_root), purge_step=purge_step,
                filename_suffix=f".{time.time_ns()}.{os.getpid()}",
            )

    def _row(self, table: str, row: dict, tag: str | None = None) -> None:
        """Append a row and mirror numeric measurements to TensorBoard."""
        if table not in self.writers:
            stream = (
                gzip.open(self.root / f"{table}.csv.gz", "xt", newline="")
                if table.endswith("_frequency")
                else (self.root / f"{table}.csv").open("x", newline="")
            )
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
            allowed = (
                TENSORBOARD_GROUP_FIELDS if tag.startswith("masks/")
                else set(row) - TENSORBOARD_OVERALL_EXCLUDED
            )
            for key, value in row.items():
                if (
                    key in allowed
                    and key not in CONTEXT_FIELDS
                    and isinstance(value, (int, float, np.number))
                ):
                    metric = (
                        "total_loss"
                        if tag == "overall" and key == "original_loss"
                        else key
                    )
                    self.writer.add_scalar(
                        f"{row['phase']}/{tag}/{metric}",
                        value, row["step"],
                    )

    def record(
        self,
        phase: str,
        step: int,
        accumulator: Accumulator,
        checkpoint_step: int | None = None,
        learning_rates: list | None = None,
        plot_images: bool = True,
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
        overall = summarize_accumulator(
            accumulator, self.alpha, phase == "train"
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
            context, "interval", accumulator.coverage, accumulator.exposure,
            plot_images=plot_images,
        )
        if phase == "train":
            self.cumulative_coverage += accumulator.coverage
            self.cumulative_exposure += accumulator.exposure
            self._coverage(
                context,
                "cumulative",
                self.cumulative_coverage,
                self.cumulative_exposure,
                plot_images=plot_images,
            )
            self.start_step = step + 1
        if self.writer is not None and plot_images:
            self._heatmap(context, grid)
        self.flush()

    def record_rank(
        self, step: int, embeddings: np.ndarray, phase: str = "validation"
    ) -> None:
        """Save centered window-embedding rank at one validation update."""
        summary = embedding_rank(embeddings)
        self._row(f"{phase}_rank", {"step": step, **summary})
        if self.writer is not None:
            for name in (
                "effective_rank", "effective_rank_fraction",
                "pca95_components",
            ):
                self.writer.add_scalar(
                    f"{phase}/rank/{name}", summary[name], step
                )
        self.flush()

    def record_frequency(
        self, step: int, stats: Accumulator, plot_image: bool = False,
        phase: str = "validation",
    ) -> None:
        """Save validation reconstruction by STFT bin and mask strategy."""
        if stats.frequency is None:
            raise ValueError("Validation frequency statistics are missing.")
        pooled = stats.frequency.sum(axis=0)
        groups = [("all", pooled)] + list(zip(
            STRATEGIES, stats.frequency
        ))
        for strategy, bins in groups:
            for frequency_hz, values in zip(self.frequencies, bins):
                count, error, content, content_error, baseline, content_base = (
                    values
                )
                self._row(f"{phase}_frequency", {
                    "step": step,
                    "frequency_hz": frequency_hz,
                    "strategy": strategy,
                    "selected_cells": int(count),
                    "content_cells": int(content),
                    "l1": error / count if count else None,
                    "content_l1": (
                        content_error / content if content else None
                    ),
                    "zero_baseline_l1": (
                        baseline / count if count else None
                    ),
                    "relative_l1": (
                        error / baseline if baseline > 0 else None
                    ),
                    "relative_content_l1": (
                        content_error / content_base
                        if content_base > 0 else None
                    ),
                })
        if self.writer is not None:
            selected = pooled[:, 0] > 0
            ratios = pooled[:, 4] > 0
            if np.any(selected):
                self.writer.add_histogram(
                    f"{phase}/frequency/l1",
                    pooled[selected, 1] / pooled[selected, 0], step,
                )
            if np.any(ratios):
                self.writer.add_histogram(
                    f"{phase}/frequency/relative_l1",
                    pooled[ratios, 1] / pooled[ratios, 4], step,
                )
        self.flush()

    def record_epoch(
        self, epoch: int, step: int, validation: Accumulator,
        complete: bool,
    ) -> None:
        """Save online training and final-snapshot validation summaries."""
        for phase, stats in (
            ("train_online", self.epoch_train),
            ("validation", validation),
        ):
            if not stats.batches:
                raise ValueError(f"Epoch {epoch} {phase} has no batches.")
            context = {
                "epoch": epoch, "step": step, "phase": phase,
                "complete": complete,
            }
            summary = summarize_accumulator(
                stats, self.alpha, phase == "train_online"
            )
            self._row("epoch_metrics", {**context, **summary})
            grid = stats.sums.reshape(
                len(STRATEGIES), len(ACTIONS), len(SUM_FIELDS)
            )
            for i, strategy in enumerate(STRATEGIES):
                for j, action in enumerate(ACTIONS):
                    self._group(
                        context, strategy, action, grid[i, j],
                        table="epoch_mask_groups",
                    )
                self._group(
                    context, strategy, "all", grid[i].sum(axis=0),
                    table="epoch_mask_groups",
                )
            for j, action in enumerate(ACTIONS):
                self._group(
                    context, "all", action, grid[:, j].sum(axis=0),
                    table="epoch_mask_groups",
                )
            self._coverage(
                context, "epoch", stats.coverage, stats.exposure,
                plot_images=False, table="epoch_frequency_coverage",
            )
            if self.writer is not None:
                self.writer.add_scalar(
                    f"epoch/{phase}/original_loss",
                    summary["original_loss"], epoch,
                )
                self.writer.add_scalar(
                    f"epoch/{phase}/l1", summary["l1"], epoch,
                )
        self.epoch_train = Accumulator()
        self.flush()

    def _group(
        self, context: dict, strategy: str, action: str, sums: np.ndarray,
        table: str = "mask_groups",
    ) -> None:
        """Write a conditional group without rescaling its denominator."""
        self._row(
            table,
            {
                **context,
                "strategy": strategy,
                "action": action,
                **summarize(sums, self.alpha),
            },
            f"masks/{strategy}/{action}"
            if table == "mask_groups"
            and (strategy == "all") != (action == "all")
            else None,
        )

    def _coverage(
        self,
        context: dict,
        scope: str,
        counts: np.ndarray,
        exposure: np.ndarray,
        plot_images: bool = True,
        table: str = "frequency_coverage",
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
                    table,
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
        if self.writer is not None and plot_images:
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
