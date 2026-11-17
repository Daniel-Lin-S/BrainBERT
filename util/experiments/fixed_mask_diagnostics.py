"""Fixed test masks and sufficient statistics for paired reconstruction.

Inputs are normalized targets (batch, time, frequency) and padding (batch,
 time). Outputs contain batch objective means and per-frequency error sums;
no predictions or spectrogram tensors are persisted. Training and validation
mask generators are never modified.
"""

import warnings
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from criterions.pretrain_masked_criterion import CONTENT_THRESHOLD

MASK_TYPES = ("temporal", "spectral", "joint")
SUM_COLUMNS = (
    "count", "error_sum", "baseline_sum", "content_count",
    "content_error_sum", "content_baseline_sum",
)


def fixed_masks(
    padding: torch.Tensor,
    frequencies: int,
    indices: torch.Tensor,
    duration: int,
    maximum_duration: int,
    bins: torch.Tensor,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct paired temporal and spectral masks.

    Parameters
    ----------
    padding : torch.Tensor
        Boolean padded-frame indicators, shape (B, T).
    frequencies : int
        Number of spectral bins.
    indices : torch.Tensor
        Stable benchmark positions, shape (B,).
    duration, maximum_duration : int
        Current and largest gap widths in frames, both positive.
    bins : torch.Tensor
        Balanced frequency assignments for the complete benchmark, shape (N,).
    seed : int
        Fixed temporal-position seed.

    Returns
    -------
    tuple of torch.Tensor
        Boolean temporal and spectral masks, each shape (B, T, F).
    """
    if not 0 < duration <= maximum_duration:
        raise ValueError("Expected 0 < duration <= maximum_duration.")
    valid = ~padding.bool()
    lengths = valid.sum(1)
    if (lengths < maximum_duration).any():
        raise ValueError("A test window is shorter than the largest gap.")
    frames = torch.arange(padding.shape[1], device=padding.device)
    if not torch.equal(valid, frames[None, :] < lengths[:, None]):
        raise ValueError("Expected contiguous valid frames before padding.")
    starts = []
    for index, length in zip(indices.tolist(), lengths.tolist()):
        generator = torch.Generator().manual_seed(seed + index)
        start = torch.randint(
            length - maximum_duration + 1, (1,), generator=generator
        ).item()
        starts.append(start + (maximum_duration - duration) // 2)
    starts = torch.tensor(starts, device=padding.device)
    temporal = (
        (frames[None, :] >= starts[:, None])
        & (frames[None, :] < starts[:, None] + duration)
    )[:, :, None].expand(-1, -1, frequencies)
    selected = bins[indices.cpu()].to(padding.device)
    axis = torch.arange(frequencies, device=padding.device)
    spectral = valid[:, :, None] & (
        axis[None, None, :] == selected[:, None, None]
    )
    return temporal, spectral


def balanced_bins(size: int, frequencies: int, seed: int) -> torch.Tensor:
    """Return deterministic balanced bin assignments of shape (size,)."""
    if size < frequencies or frequencies < 1:
        raise ValueError("Benchmark must include at least one window per bin.")
    generator = torch.Generator().manual_seed(seed)
    return (torch.arange(size) % frequencies)[
        torch.randperm(size, generator=generator)
    ]


def reconstruction_statistics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    selected: torch.Tensor,
    scored: torch.Tensor,
    frequency_mean: torch.Tensor,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduce actual objectives and frequency-specific baseline errors.

    Parameters
    ----------
    prediction, target : torch.Tensor
        Normalized spectra, shape (B, T, F).
    selected, scored : torch.Tensor
        Objective mask and frequency diagnostic mask, shape (B, T, F).
    frequency_mean : torch.Tensor
        Frozen training-target mean, shape (F,).
    alpha : float
        Content-loss coefficient from the fitted criterion.

    Returns
    -------
    tuple of torch.Tensor
        Six batch scalars: L1, content L1, total loss and their zero-predictor
        counterparts; frequency sums of shape (F, 6), ordered SUM_COLUMNS.
    """
    if prediction.shape != target.shape or selected.shape != target.shape:
        raise ValueError("Prediction, target and objective mask must match.")
    if scored.shape != target.shape or frequency_mean.shape != target.shape[2:]:
        raise ValueError("Diagnostic mask or frequency mean shape differs.")
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError("Prediction and target must be finite.")
    if not torch.isfinite(frequency_mean).all():
        raise ValueError("Training frequency means must be finite.")
    error = (prediction - target).abs().double()
    content = target.abs() > CONTENT_THRESHOLD
    content_selected = selected & content
    if not selected.any() or not content_selected.any():
        raise ValueError("Objective requires masked and content target cells.")
    l1 = error[selected].mean()
    content_l1 = error[content_selected].mean()
    zero = target.abs().double()
    baseline_l1 = zero[selected].mean()
    baseline_content = zero[content_selected].mean()
    objective = torch.stack((
        l1, content_l1, l1 + alpha * content_l1,
        baseline_l1, baseline_content,
        baseline_l1 + alpha * baseline_content,
    ))
    baseline = (target.double() - frequency_mean.double()).abs()
    content_scored = scored & content
    sums = torch.stack([
        scored.sum((0, 1)), (error * scored).sum((0, 1)),
        (baseline * scored).sum((0, 1)), content_scored.sum((0, 1)),
        (error * content_scored).sum((0, 1)),
        (baseline * content_scored).sum((0, 1)),
    ], dim=1)
    return objective.cpu(), sums.cpu()


def checked_ratio(numerator: float, denominator: float) -> float | None:
    """Return an explicit missing value for an undefined diagnostic ratio."""
    if denominator <= 0:
        warnings.warn("Constant prediction L1 is zero; ratio is undefined.")
        return None
    return numerator / denominator


@torch.inference_mode()
def evaluate_fixed(
    task: Any, model: torch.nn.Module, cfg: Any,
    options: Mapping, root: Path, identity: dict, device: str,
    workers: int, progress: Callable, smoke_batches: int | None = None,
) -> dict:
    """Replay fixed test conditions and persist small resumable summaries.

    Parameters
    ----------
    task, model, cfg : object
        Prepared task, selected model and saved scientific configuration.
    options : mapping
        Explicit durations, seed, bandwidth and test batch limit.
    root : pathlib.Path
        Dedicated diagnostic storage, separate from completed fit results.
    identity : dict
        Checkpoint and selection provenance for safe reuse.
    device : str
        Evaluation device.
    workers : int
        Bounded data loader worker count.
    progress : callable
        Receives completed and total batch counts and stage name.
    smoke_batches : int or None, optional
        Bounded audit without publishing, default None.

    Returns
    -------
    dict
        Completed condition summaries and training baseline provenance.
    """
    import json
    import math
    from .storage import write_json

    durations = list(options["durations_frames"])
    if not durations or any(type(d) is not int or d <= 0 for d in durations):
        raise ValueError("Expected positive integer durations_frames.")
    if len(set(durations)) != len(durations):
        raise ValueError("Temporal durations must be unique.")
    if options["spectral_bins"] != 1 or options["replacement"] != "zero":
        raise ValueError("Fixed diagnostics require one bin and zero masking.")
    if options["joint_scoring"] != "intersection":
        raise ValueError("Expected joint_scoring: intersection.")
    size = cfg.exp.runner.valid_batch_size
    test_batches = math.ceil(len(task.test_set) / size)
    if test_batches != options["batches"]:
        raise ValueError(
            f"Expected {options['batches']} test batches, got {test_batches}."
        )
    identity = {**identity, "configuration": dict(options)}
    baseline_path = root / "training_baseline.json"
    if baseline_path.exists() and smoke_batches is None:
        baseline = json.loads(baseline_path.read_text())
        if baseline["identity"] != identity:
            raise ValueError("Saved training baseline provenance differs.")
        mean = torch.tensor(baseline["mean"], dtype=torch.float64)
    else:
        loader = task.get_batch_iterator(
            task.train_set, size, shuffle=False, num_workers=workers
        )
        sums, count = None, 0
        for batch_index, batch in enumerate(loader):
            target = batch["target"].double()
            valid = ~batch["attn_mask"].bool()
            if not torch.isfinite(target).all():
                raise ValueError("Training baseline targets must be finite.")
            value = (target * valid[:, :, None]).sum((0, 1))
            sums = value if sums is None else sums + value
            count += int(valid.sum())
            progress(batch_index + 1, len(loader), "training_baseline")
            if smoke_batches and batch_index + 1 >= smoke_batches:
                break
        if count == 0:
            raise ValueError("Training baseline has no valid target frames.")
        mean = sums / count
        baseline = {
            "identity": identity, "mean": mean.tolist(),
            "frame_count": count,
            "source": "normalized training targets; all valid frames",
        }
        if smoke_batches is None:
            write_json(baseline_path, baseline)
    frequencies = len(mean)
    bins = balanced_bins(len(task.test_set), frequencies, options["seed"])
    conditions = [("spectral", 0)] + [
        (kind, duration) for duration in durations
        for kind in ("temporal", "joint")
    ]
    outputs = {}
    for kind, duration in conditions:
        name = f"{kind}_{duration}" if duration else kind
        path = root / "conditions" / f"{name}.json"
        if path.exists() and smoke_batches is None:
            saved = json.loads(path.read_text())
            if saved["identity"] != identity or saved["baseline"] != baseline:
                raise ValueError(f"Saved condition provenance differs: {name}")
            outputs[name] = saved
            progress(saved["batches"], saved["batches"], name)
            continue
        loader = task.get_batch_iterator(
            task.test_set, size, shuffle=False, num_workers=workers
        )
        objectives = torch.zeros(6, dtype=torch.float64)
        frequency = torch.zeros(frequencies, 6, dtype=torch.float64)
        offset = 0
        for batch_index, batch in enumerate(loader):
            target = batch["target"].to(device)
            padding = batch["attn_mask"].to(device).bool()
            indices = torch.arange(offset, offset + len(target))
            temporal, spectral = fixed_masks(
                padding, frequencies, indices, duration or min(durations),
                max(durations), bins, options["seed"],
            )
            if kind == "temporal":
                selected = scored = temporal
            elif kind == "spectral":
                selected = scored = spectral
            else:
                selected, scored = temporal | spectral, temporal & spectral
            prediction, _ = model(target.masked_fill(selected, 0), padding)
            objective, sums = reconstruction_statistics(
                prediction, target, selected, scored, mean.to(device),
                cfg.criterion.alpha,
            )
            objectives += objective
            frequency += sums
            offset += len(target)
            progress(batch_index + 1, len(loader), name)
            if smoke_batches and batch_index + 1 >= smoke_batches:
                break
        completed = batch_index + 1
        saved = {
            "identity": identity, "baseline": baseline,
            "mask": kind, "duration_frames": duration,
            "batches": completed, "windows": offset,
            "objective_means": (objectives / completed).tolist(),
            "frequency_sums": frequency.tolist(),
        }
        if smoke_batches is None:
            write_json(path, saved)
        outputs[name] = saved
        print(f"Completed fixed-mask condition: {name}", flush=True)
    return outputs



def plot_fixed(
    outputs: dict, frequency_axis: Sequence[float], frame_hop_ms: float,
    reports: Path, fit: Path, step: int, settings: Mapping,
) -> None:
    """Publish four fixed-mask figures and their numerical source tables.

    Parameters
    ----------
    outputs : dict
        Condition summaries returned by evaluate_fixed.
    frequency_axis : sequence of float
        Physical STFT centres in Hz, shape (F,).
    frame_hop_ms : float
        Frame spacing in milliseconds.
    reports, fit : pathlib.Path
        Campaign report directory and immutable shared fit directory.
    step : int
        Validation-selected checkpoint update.
    settings : mapping
        Explicit report formats, splits and families.
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    from .reporting import Report

    report = Report(reports, fit, step, [], settings)
    metrics = ("l1", "content_l1", "total_loss")
    rows, frequency_rows = [], []
    for name, result in outputs.items():
        values = result["objective_means"]
        row = {
            "condition": name, "mask": result["mask"],
            "duration_frames": result["duration_frames"],
            "duration_ms": result["duration_frames"] * frame_hop_ms,
        }
        for index, metric in enumerate(metrics):
            row[metric] = values[index]
            row[f"constant_prediction_{metric}"] = values[index + 3]
            row[f"relative_{metric}"] = checked_ratio(
                values[index], values[index + 3]
            )
        rows.append(row)
        for hz, sums in zip(frequency_axis, result["frequency_sums"]):
            frequency_rows.append({
                "condition": name, "frequency_hz": hz,
                **dict(zip(SUM_COLUMNS, sums)),
                "relative_l1": checked_ratio(sums[1], sums[2]),
                "relative_content_l1": checked_ratio(sums[4], sums[5]),
            })
    overall = pd.DataFrame(rows)
    frequencies = pd.DataFrame(frequency_rows)
    prefix = "test/fixed_masks"
    report.table(
        f"{prefix}/duration/objective", overall,
        "Test fixed-mask objectives and previous zero-predictor baselines",
    )
    report.table(
        f"{prefix}/frequency/reconstruction", frequencies,
        "Test frequency errors relative to frozen training means",
    )
    styles = {
        "temporal": ("#0072B2", "-", "o"),
        "spectral": ("#009E73", "--", "s"),
        "joint": ("#D55E00", "-.", "^"),
    }
    titles = ("L1", "Content L1", "Total loss")
    fig, axes = plt.subplots(1, 3, figsize=(23, 7))
    for ax, metric, title in zip(axes, metrics, titles):
        for kind in ("temporal", "joint"):
            subset = overall[overall["mask"] == kind].sort_values(
                "duration_ms"
            )
            color, style, marker = styles[kind]
            ax.plot(
                subset.duration_ms, subset[f"relative_{metric}"],
                color=color, ls=style, marker=marker, lw=2.5,
                label=kind.capitalize(),
            )
        spectral = overall[overall["mask"] == "spectral"].iloc[0]
        ax.axhline(
            spectral[f"relative_{metric}"], color=styles["spectral"][0],
            ls="--", lw=2.5, label="Spectral",
        )
        ax.axhline(1, color="0.5", ls=":", label="Constant prediction")
        ticks = overall[overall["mask"] == "temporal"]
        ax.set_xticks(ticks.duration_ms, [
            f"{row.duration_ms:.0f} ms\n({row.duration_frames} frames)"
            for row in ticks.itertuples()
        ])
        ax.set_xlabel("Temporal mask duration")
        ax.set_ylabel("Loss / constant prediction loss")
        ax.set_title(title)
    report.figure(
        f"{prefix}/frequency/duration_comparison", fig,
        "Mask duration and relative reconstruction loss; "
        f"validation-selected checkpoint {step}",
    )
    for duration in sorted(overall.duration_frames.unique()):
        if not duration:
            continue
        fig, ax = plt.subplots(figsize=(13, 8))
        for kind in MASK_TYPES:
            name = "spectral" if kind == "spectral" else f"{kind}_{duration}"
            subset = frequencies[frequencies.condition == name]
            color, style, marker = styles[kind]
            label = "Joint intersection" if kind == "joint" else kind.title()
            ax.plot(
                subset.frequency_hz, subset.relative_l1,
                color=color, ls=style, lw=2.5, label=label,
            )
        ax.axhline(1, color="0.5", ls=":", label="Constant prediction L1")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("L1 / constant prediction L1")
        report.figure(
            f"{prefix}/frequency/duration_{duration}", fig,
            f"Reconstruction: {duration} frames "
            f"({duration * frame_hop_ms:.0f} ms); "
            f"validation-selected checkpoint {step}",
        )
    print(f"Fixed-mask reports saved: {reports.resolve()}", flush=True)
