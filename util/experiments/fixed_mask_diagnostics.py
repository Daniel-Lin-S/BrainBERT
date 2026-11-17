"""Reconstruction diagnostics and training content-count estimates.

Inputs are normalised targets (B, T, F), padding (B, T), and saved condition
summaries. Condition JSONs retain six batch objective means (model/constant
L1, content L1, total_loss) and frequency sums ordered by SUM_COLUMNS.
Interpolation JSON retains per-frequency all/content error sums and three
batch objective means per duration. Training-content JSON records sampled
window identity and selected/content counts (F,) from the original recipe.

Reports group raw and constant-normalised frequency figures by metric,
duration comparisons separately, and interpolation figures by metric.
Tables retain counts, three losses and matched baseline ratios. No target
or prediction tensors are persisted; completed scientific results and the
training/validation mask generators are never modified.
"""

import warnings
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from criterions.pretrain_masked_criterion import CONTENT_THRESHOLD

PARTIAL_SAVE_BATCHES = 16
LOSS_METRICS = ("l1", "content_l1", "total_loss")
LOSS_LABELS = ("L1", "Content L1", "Total loss")
BASELINE_LABELS = (
    "Constant prediction L1", "Constant prediction L1 (content)",
    "Constant prediction total_loss",
)
MASK_STYLES = {
    "temporal": ("#0072B2", "-"),
    "spectral": ("#009E73", "--"),
    "joint": ("#D55E00", "-."),
}

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
    publish: Callable | None = None,
    devices: Sequence[str] | None = None,
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

    publish : callable or None, optional
        Publish ready summaries after each condition, default None.

    devices : sequence of str or None, optional
        Independent inference devices; default uses device alone.

    Returns
    -------
    dict
        Completed condition summaries and training baseline provenance.
    """
    import json
    import math
    from .storage import write_json
    from util.mask_utils import test_masked_dataset

    training_targets = test_masked_dataset(task.train_set, targets_only=True)
    test_targets = test_masked_dataset(task.test_set, targets_only=True)
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
        baseline = training_baseline(
            task, training_targets, size, workers, root, identity,
            progress, smoke_batches,
        )
        mean = torch.tensor(baseline["mean"], dtype=torch.float64)
        if smoke_batches is None:
            write_json(baseline_path, baseline)
            (root / "training_baseline_partial.json").unlink(missing_ok=True)
    frequencies = len(mean)
    bins = balanced_bins(len(task.test_set), frequencies, options["seed"])
    conditions = [("spectral", 0)] + [
        (kind, duration) for duration in durations
        for kind in ("temporal", "joint")
    ]
    outputs, pending = {}, []
    for kind, duration in conditions:
        name = f"{kind}_{duration}" if duration else kind
        path = root / "conditions" / f"{name}.json"
        if path.exists() and smoke_batches is None:
            saved = json.loads(path.read_text())
            if saved["identity"] != identity or saved["baseline"] != baseline:
                raise ValueError(f"Saved condition provenance differs: {name}")
            outputs[name] = saved
            progress(saved["batches"], saved["batches"], name)
        else:
            pending.append((name, kind, duration))
    if pending:
        evaluated = evaluate_conditions(
            task, test_targets, model, devices or [device], pending,
            bins, mean, durations, options["seed"], cfg.criterion.alpha,
            size, workers, root, identity, progress, smoke_batches,
        )
        for name, kind, duration in pending:
            result = evaluated[name]
            saved = {
                "identity": identity, "baseline": baseline,
                "mask": kind, "duration_frames": duration, **result,
            }
            if smoke_batches is None:
                write_json(root / "conditions" / f"{name}.json", saved)
            outputs[name] = saved
            if publish is not None and smoke_batches is None:
                publish(outputs)
    if smoke_batches is None:
        (root / "test_partial.json").unlink(missing_ok=True)
    return outputs


@torch.inference_mode()
def training_baseline(
    task: Any, dataset: Any, size: int, workers: int, root: Path,
    identity: dict, progress: Callable, smoke_batches: int | None,
) -> dict:
    """Stream training targets once, retaining resumable frequency sums.

    Dataset entries contain targets (T, F); the result contains a frozen
    mean (F,), valid-frame count and the exact selection identity. Partial
    sums are checkpointed every PARTIAL_SAVE_BATCHES batches. Smoke runs
    never read or write scientific baseline artifacts.
    """
    import json
    import math
    from .storage import write_json

    partial_path = root / "training_baseline_partial.json"
    offset, count, sums = 0, 0, None
    if partial_path.exists() and smoke_batches is None:
        partial = json.loads(partial_path.read_text())
        if partial["identity"] != identity:
            raise ValueError("Partial training baseline provenance differs.")
        offset, count = partial["windows"], partial["count"]
        sums = torch.tensor(partial["sums"], dtype=torch.float64)
    total = math.ceil(len(dataset) / size)
    subset = torch.utils.data.Subset(dataset, range(offset, len(dataset)))
    loader = task.get_batch_iterator(
        subset, size, shuffle=False, num_workers=workers,
    )
    completed = offset // size
    for batch in loader:
        target = batch["target"].double()
        valid = ~batch["attn_mask"].bool()
        if not torch.isfinite(target).all():
            raise ValueError("Training baseline targets must be finite.")
        value = (target * valid[:, :, None]).sum((0, 1))
        sums = value if sums is None else sums + value
        count += int(valid.sum())
        offset += len(target)
        completed += 1
        progress(completed, total, "training_baseline")
        if smoke_batches is None and completed % PARTIAL_SAVE_BATCHES == 0:
            write_json(partial_path, {
                "identity": identity, "windows": offset,
                "count": count, "sums": sums.tolist(),
            })
        if smoke_batches and completed >= smoke_batches:
            break
    if count == 0:
        raise ValueError("Training baseline has no valid target frames.")
    return {
        "identity": identity, "mean": (sums / count).tolist(),
        "frame_count": count,
        "source": "normalized training targets; all valid frames",
    }


@torch.inference_mode()
def condition_batch(
    model: torch.nn.Module, device: str, conditions: list,
    batch: dict, indices: torch.Tensor, bins: torch.Tensor,
    mean: torch.Tensor, durations: list[int], seed: int, alpha: float,
) -> dict:
    """Evaluate assigned conditions using one device-local target transfer.

    Returns objective sums (6,) and frequency sums (F, 6) for each condition.
    Each worker owns a model replica; no gradient communication is needed.
    """
    target = batch["target"].to(device)
    padding = batch["attn_mask"].to(device).bool()
    mean = mean.to(device)
    results = {}
    for name, kind, duration in conditions:
        temporal, spectral = fixed_masks(
            padding, len(mean), indices, duration or min(durations),
            max(durations), bins, seed,
        )
        if kind == "temporal":
            selected = scored = temporal
        elif kind == "spectral":
            selected = scored = spectral
        else:
            selected, scored = temporal | spectral, temporal & spectral
        prediction, _ = model(target.masked_fill(selected, 0), padding)
        results[name] = reconstruction_statistics(
            prediction, target, selected, scored, mean, alpha,
        )
    return results


@torch.inference_mode()
def evaluate_conditions(
    task: Any, dataset: Any, model: torch.nn.Module,
    devices: Sequence[str], conditions: list, bins: torch.Tensor,
    mean: torch.Tensor, durations: list[int], seed: int, alpha: float,
    size: int, workers: int, root: Path, identity: dict,
    progress: Callable, smoke_batches: int | None,
) -> dict:
    """Share one test stream among independent device-owned conditions.

    Partial artifacts contain only aggregate objective and frequency sums;
    no predictions or targets are saved. Batch ordering and reduction order
    match serial evaluation, independent of device count.
    """
    import copy
    import json
    import math
    from concurrent.futures import ThreadPoolExecutor
    from .storage import write_json

    if not devices or len(set(devices)) != len(devices):
        raise ValueError("Expected a nonempty list of distinct devices.")
    selected_devices = list(devices[:len(conditions)])
    replicas = [model.to(selected_devices[0])]
    replicas.extend(
        copy.deepcopy(model).to(device).eval()
        for device in selected_devices[1:]
    )
    groups = [conditions[i::len(replicas)] for i in range(len(replicas))]
    sums = {
        name: (torch.zeros(6, dtype=torch.float64),
               torch.zeros(len(mean), 6, dtype=torch.float64))
        for name, _, _ in conditions
    }
    partial_path = root / "test_partial.json"
    offset, completed = 0, 0
    if partial_path.exists() and smoke_batches is None:
        saved = json.loads(partial_path.read_text())
        if saved["identity"] != identity:
            raise ValueError("Partial fixed-mask test provenance differs.")
        if set(sums) <= set(saved["sums"]):
            offset, completed = saved["windows"], saved["batches"]
            sums = {
                name: tuple(torch.tensor(x, dtype=torch.float64) for x in
                            saved["sums"][name]) for name in sums
            }
    subset = torch.utils.data.Subset(dataset, range(offset, len(dataset)))
    loader = task.get_batch_iterator(
        subset, size, shuffle=False, num_workers=workers,
    )
    total = math.ceil(len(dataset) / size)
    # Start loader processes before starting inference threads.
    iterator = iter(loader)
    with ThreadPoolExecutor(max_workers=len(replicas)) as executor:
        for batch in iterator:
            indices = torch.arange(offset, offset + len(batch["target"]))
            futures = [executor.submit(
                condition_batch, replica, device, group, batch, indices,
                bins, mean, durations, seed, alpha,
            ) for replica, device, group in zip(
                replicas, selected_devices, groups,
            )]
            for future in futures:
                for name, (objective, frequency) in future.result().items():
                    sums[name][0].add_(objective)
                    sums[name][1].add_(frequency)
                    progress(completed + 1, total, name)
            offset += len(batch["target"])
            completed += 1
            if smoke_batches is None and completed % PARTIAL_SAVE_BATCHES == 0:
                write_json(partial_path, {
                    "identity": identity, "windows": offset,
                    "batches": completed,
                    "sums": {name: [x.tolist() for x in values]
                             for name, values in sums.items()},
                })
            if smoke_batches and completed >= smoke_batches:
                break
    if completed == 0:
        raise ValueError("Fixed-mask test benchmark contains no batches.")
    return {
        name: {
            "batches": completed, "windows": offset,
            "objective_means": (values[0] / completed).tolist(),
            "frequency_sums": values[1].tolist(),
        } for name, values in sums.items()
    }



def frequency_losses(sums: Any, alpha: float) -> dict:
    """Derive cell-pooled losses from an (F, 6) sufficient-statistic array.

    Content L1 divides content errors by content counts, not all counts.
    Frequency total_loss applies L1 + alpha * content_L1 within each bin;
    it is not an additive decomposition of the batch-averaged objective.
    Undefined baseline ratios are warned about and represented as NaN.

    Parameters
    ----------
    sums : array-like
        Frequency accumulators (F, 6), ordered by SUM_COLUMNS.
    alpha : float
        Fitted nonnegative content-loss coefficient.

    Returns
    -------
    dict
        Model, constant and relative losses, each shaped (F, 3).
    """
    import numpy as np

    values = np.asarray(sums, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(SUM_COLUMNS):
        raise ValueError("Expected frequency statistics with shape (F, 6).")
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Frequency statistics must be nonempty and finite.")
    if not np.isfinite(alpha) or alpha < 0 or (values < 0).any():
        raise ValueError("Loss coefficient and statistics must be nonnegative.")
    if (values[:, [0, 3]] <= 0).any():
        raise ValueError("Every frequency needs selected and content cells.")
    if (values[:, 3] > values[:, 0]).any():
        raise ValueError("Content count cannot exceed selected-cell count.")
    losses = {}
    for name, columns in (("model", (1, 4)), ("constant", (2, 5))):
        l1 = values[:, columns[0]] / values[:, 0]
        content = values[:, columns[1]] / values[:, 3]
        losses[name] = np.column_stack((l1, content, l1 + alpha * content))
    baseline = losses["constant"]
    if (baseline <= 0).any():
        warnings.warn("Constant prediction loss is zero; ratio is undefined.")
    losses["relative"] = np.divide(
        losses["model"], baseline,
        out=np.full_like(baseline, np.nan), where=baseline > 0,
    )
    return losses


def fixed_loss_tables(
    outputs: dict, frequency_axis: Sequence[float], frame_hop_ms: float,
    alpha: float,
) -> tuple:
    """Return objective and frequency tables from matching condition JSONs.

    Rows identify the condition and physical frequency, counts, three losses,
    their matched constant-predictor losses, and ratios. Overall rows retain
    the saved batch reduction; joint frequency rows score the intersection.
    """
    import numpy as np
    import pandas as pd

    if not outputs:
        raise ValueError("Fixed-mask reporting requires saved conditions.")
    axis = np.asarray(frequency_axis, dtype=float)
    if axis.ndim != 1 or not len(axis) or not np.isfinite(axis).all():
        raise ValueError("Expected nonempty finite frequency centres (F,).")
    identity = next(iter(outputs.values()))["identity"]
    rows, frequency_rows = [], []
    for name, result in outputs.items():
        if result["identity"] != identity:
            raise ValueError("Fixed-mask conditions have different identities.")
        values = np.asarray(result["objective_means"], dtype=float)
        if values.shape != (6,) or not np.isfinite(values).all():
            raise ValueError("Expected six finite saved objective means.")
        if not np.allclose(
            values[[2, 5]], values[[0, 3]] + alpha * values[[1, 4]],
        ):
            raise ValueError("Saved objectives disagree with criterion alpha.")
        row = {
            "condition": name, "mask": result["mask"],
            "duration_frames": result["duration_frames"],
            "duration_ms": result["duration_frames"] * frame_hop_ms,
        }
        for index, metric in enumerate(LOSS_METRICS):
            row[metric] = values[index]
            row[f"constant_prediction_{metric}"] = values[index + 3]
            row[f"relative_{metric}"] = checked_ratio(
                values[index], values[index + 3],
            )
        rows.append(row)
        sums = np.asarray(result["frequency_sums"], dtype=float)
        if sums.shape != (len(axis), len(SUM_COLUMNS)):
            raise ValueError("Frequency centres and statistic rows differ.")
        losses = frequency_losses(sums, alpha)
        for i, hz in enumerate(axis):
            row = {
                "condition": name, "frequency_hz": hz,
                **dict(zip(SUM_COLUMNS, sums[i])),
            }
            for j, metric in enumerate(LOSS_METRICS):
                row[metric] = losses["model"][i, j]
                row[f"constant_prediction_{metric}"] = losses["constant"][i, j]
                row[f"relative_{metric}"] = losses["relative"][i, j]
            frequency_rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(frequency_rows)


def plot_fixed(
    outputs: dict, frequency_axis: Sequence[float], frame_hop_ms: float,
    reports: Path, fit: Path, step: int, settings: Mapping,
) -> None:
    """Publish raw and constant-normalised test losses from saved summaries.

    Frequency figures are grouped by metric, with raw/normalised pairs for
    each duration. Duration figures retain saved batch objective means.
    Tables contain all three losses, counts, and matched baselines. No model
    evaluation or change to completed diagnostic results occurs here.
    """
    import json
    import pandas as pd
    from .reporting import Report

    alpha = json.loads((fit / "scientific.json").read_text())["criterion"][
        "alpha"
    ]
    overall, frequencies = fixed_loss_tables(
        outputs, frequency_axis, frame_hop_ms, alpha,
    )
    report = Report(reports, fit, step, [], settings)
    index = reports / "output_index.csv"
    if index.exists():
        report.entries = pd.read_csv(index).to_dict("records")
    report.table(
        "test/fixed_masks/duration/objective", overall,
        "Batch-averaged test objectives; joint masks score their union",
    )
    report.table(
        "test/fixed_masks/frequency/reconstruction", frequencies,
        "Cell-pooled frequency losses; joint masks score their intersection; "
        "total_loss = L1 + alpha * content_L1 within each frequency",
    )
    expected = next(iter(outputs.values()))["identity"][
        "configuration"
    ]["durations_frames"]
    durations = [d for d in expected if all(
        name in outputs for name in ("spectral", f"temporal_{d}", f"joint_{d}")
    )]
    root = reports.parent / "diagnostics/fixed_masks"
    interpolation_path = root / "interpolation_metrics.json"
    if not interpolation_path.exists():
        interpolation_path = root / "interpolation.json"
    interpolation = (
        json.loads(interpolation_path.read_text())
        if interpolation_path.exists() else None
    )
    if interpolation is not None and interpolation["identity"] != next(
        iter(outputs.values())
    )["identity"]:
        raise ValueError("Interpolation and model provenance differ.")
    if (interpolation is not None and "criterion_alpha" in interpolation
            and interpolation["criterion_alpha"] != alpha):
        raise ValueError("Interpolation criterion coefficient differs.")
    for normalised in (False, True):
        for duration in durations:
            plot_frequency_losses(
                report, frequencies, duration, frame_hop_ms, normalised,
                interpolation, alpha,
            )
        if set(durations) == set(expected):
            plot_duration_losses(report, overall, durations, normalised)
    if set(durations) == set(expected):
        retire_fixed_figures(report, durations)
    print(f"Fixed-mask reports saved: {reports.resolve()}", flush=True)


def plot_frequency_losses(
    report: Any, frame: Any, duration: int, hop_ms: float,
    normalised: bool, interpolation: dict | None, alpha: float,
) -> None:
    """Draw three losses across frequency for one paired mask duration."""
    import matplotlib.pyplot as plt
    import numpy as np

    temporal = frame[frame.condition == f"temporal_{duration}"]
    for metric, title in zip(LOSS_METRICS, LOSS_LABELS):
        fig, ax = plt.subplots(figsize=(13, 8))
        for kind, (color, style) in MASK_STYLES.items():
            name = "spectral" if kind == "spectral" else f"{kind}_{duration}"
            subset = frame[frame.condition == name]
            label = "Joint intersection" if kind == "joint" else kind.title()
            column = f"relative_{metric}" if normalised else metric
            ax.plot(subset.frequency_hz, subset[column], color=color,
                    ls=style, lw=2.5, label=label)
        baseline_label = BASELINE_LABELS[LOSS_METRICS.index(metric)]
        if normalised:
            ax.axhline(1, color="0.55", ls=":", label=baseline_label)
        else:
            ax.plot(
                temporal.frequency_hz,
                temporal[f"constant_prediction_{metric}"],
                color="0.55", ls=":", lw=2.5,
                label=f"{baseline_label}\n(temporal)",
            )
            if interpolation is not None and (
                metric == "l1" or "content_error_sums" in interpolation
            ):
                l1 = np.asarray(interpolation["error_sums"][str(duration)])
                l1 = l1 / temporal["count"].to_numpy()
                value = l1
                if metric != "l1":
                    content = np.asarray(
                        interpolation["content_error_sums"][str(duration)]
                    ) / temporal.content_count.to_numpy()
                    value = content if metric == "content_l1" else (
                        l1 + alpha * content
                    )
                ax.plot(temporal.frequency_hz, value, color="0.25",
                        ls="--", lw=2.5, label="Temporal interpolation")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel(
            f"{title} / constant prediction loss" if normalised
            else f"{title} (normalised targets)"
        )
        suffix = "_normalised" if normalised else ""
        report.figure(
            f"test/fixed_masks/frequency/{metric}/duration_{duration}{suffix}",
            fig, f"{title}: {duration} frames ({duration * hop_ms:.0f} ms); "
            f"validation-selected checkpoint {report.best}",
        )


def plot_duration_losses(
    report: Any, frame: Any, durations: Sequence[int], normalised: bool,
) -> None:
    """Plot the recorded batch-mean objectives, with joint union scoring."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(23, 7))
    for ax, metric, title in zip(axes, LOSS_METRICS, LOSS_LABELS):
        column = f"relative_{metric}" if normalised else metric
        for kind in ("temporal", "joint"):
            subset = frame[frame["mask"] == kind].sort_values("duration_ms")
            color, style = MASK_STYLES[kind]
            ax.plot(subset.duration_ms, subset[column], color=color,
                    ls=style, marker="o", lw=2.5, label=kind.title())
        ax.axhline(frame[frame["mask"] == "spectral"].iloc[0][column],
                   color=MASK_STYLES["spectral"][0], ls="--", lw=2.5,
                   label="Spectral")
        if normalised:
            ax.axhline(1, color="0.55", ls=":", label="Constant prediction")
        ticks = frame[frame["mask"] == "temporal"].sort_values("duration_ms")
        ax.set_xticks(ticks.duration_ms, [
            f"{row.duration_ms:.0f} ms\n({row.duration_frames} frames)"
            for row in ticks.itertuples()
        ])
        ax.set_xlabel("Temporal mask duration")
        ax.set_ylabel("Loss / constant prediction loss" if normalised
                      else "Loss (normalised targets)")
        ax.set_title(title)
    name = "constant_normalised" if normalised else "raw"
    report.figure(
        f"test/fixed_masks/duration/{name}", fig,
        "Mask duration and reconstruction loss; "
        f"validation-selected checkpoint {report.best}",
    )


def retire_fixed_figures(report: Any, durations: Sequence[int]) -> None:
    """Remove superseded derived figures only after replacements are audited.

    Completed diagnostic JSONs are never modified. Legacy adaptive figures
    remain archived separately from the active fixed-mask report tree.
    """
    import pandas as pd
    from .storage import atomic_write, file_digest

    prefix = "figures/test/fixed_masks"
    replacements = [f"{prefix}/duration/{name}.png"
                    for name in ("raw", "constant_normalised")]
    replacements.extend(
        f"{prefix}/frequency/{metric}/duration_{d}{suffix}.png"
        for metric in LOSS_METRICS for d in durations
        for suffix in ("", "_normalised")
    )
    if not all((report.root / name).exists() for name in replacements):
        return
    removed = set()
    folders = ("frequency", "l1_by_frequency", "normalised_l1_by_frequency")
    names = ["duration_comparison", *[f"duration_{d}" for d in durations]]
    for folder in folders:
        for name in names:
            relative = f"{prefix}/{folder}/{name}.png"
            (report.root / relative).unlink(missing_ok=True)
            removed.add(relative)
        directory = report.root / prefix / folder
        if directory.exists() and not any(directory.iterdir()):
            directory.rmdir()
    archive = report.root.parent / "diagnostics/fixed_masks/previous_reports"
    for pattern in ("frequency_reconstruction*.png", "frequency_coverage*.png"):
        for old in (report.root / "figures/test").glob(pattern):
            archive.mkdir(parents=True, exist_ok=True)
            destination = archive / old.name
            if destination.exists():
                if file_digest(old) != file_digest(destination):
                    raise ValueError("Archived frequency figure differs.")
                old.unlink()
            else:
                old.rename(destination)
            removed.add(str(old.relative_to(report.root)))
    report.entries = [e for e in report.entries if e["output"] not in removed]
    frame = pd.DataFrame(report.entries)
    atomic_write(
        report.root / "output_index.csv",
        lambda p: frame.to_csv(p, index=False), lambda p: pd.read_csv(p),
    )


def linear_temporal_errors(
    target: torch.Tensor, padding: torch.Tensor, mask: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """Sum linear interpolation errors over a single temporal gap.

    Parameters
    ----------
    target : torch.Tensor
        Normalized target magnitudes, shape (B, T, F).
    padding : torch.Tensor
        Padded-frame indicators, shape (B, T).
    mask : torch.Tensor
        Contiguous temporal gap shared across frequencies, shape (B, T, F).

    Returns
    -------
    tuple
        Absolute-error sums of shape (F, 2) for all/content cells, and the
        edge-gap window count. Where
        a gap touches a window edge, use the single available endpoint.
    """
    if target.ndim != 3 or mask.shape != target.shape:
        raise ValueError("Expected target and mask with shape (B, T, F).")
    if padding.shape != target.shape[:2]:
        raise ValueError("Padding must match target batch and time dimensions.")
    if not torch.isfinite(target).all():
        raise ValueError("Interpolation targets must be finite.")
    temporal = mask[:, :, 0].bool()
    if not torch.equal(mask.bool(), temporal[:, :, None].expand_as(mask)):
        raise ValueError("Temporal masks must match across all frequencies.")
    lengths = (~padding.bool()).sum(dim=1)
    counts = temporal.sum(dim=1)
    if (counts == 0).any():
        raise ValueError("Interpolation requires one nonempty mask per window.")
    starts = temporal.int().argmax(dim=1)
    ends = starts + counts
    frames = torch.arange(target.shape[1], device=target.device)
    expected = (frames[None, :] >= starts[:, None]) & (
        frames[None, :] < ends[:, None]
    )
    if not torch.equal(temporal, expected) or (ends > lengths).any():
        raise ValueError("Expected one contiguous mask within valid frames.")
    has_left, has_right = starts > 0, ends < lengths
    if not (has_left | has_right).all():
        raise ValueError("Interpolation needs at least one unmasked endpoint.")
    rows = torch.arange(target.shape[0], device=target.device)
    left_idx = torch.where(has_left, starts - 1, ends)
    right_idx = torch.where(has_right, ends, starts - 1)
    left = target[rows, left_idx]
    right = target[rows, right_idx]
    fraction = (frames[None, :] - starts[:, None] + 1).to(target.dtype)
    fraction = fraction / (counts[:, None] + 1)
    estimate = left[:, None, :] + fraction[:, :, None] * (
        right - left
    )[:, None, :]
    residual = (estimate - target).abs().double() * mask
    error = torch.stack((
        residual.sum((0, 1)),
        (residual * (target.abs() > CONTENT_THRESHOLD)).sum((0, 1)),
    ), dim=1)
    if not torch.isfinite(error).all():
        raise ValueError("Interpolation errors must be finite.")
    return error, int((~(has_left & has_right)).sum())


@torch.inference_mode()
def evaluate_interpolation(
    task: Any, options: Mapping, root: Path, identity: dict,
    workers: int, progress: Callable, batch_size: int,
    smoke_batches: int | None = None, *, alpha: float,
) -> dict:
    """Evaluate interpolation on the original paired temporal masks.

    The existing condition JSON supplies exact BrainBERT error and counts;
    a single pass computes interpolation error sums for all durations.
    The resulting JSON records sums, edge counts and provenance without
    persisting targets. Partial aggregate sums allow interrupted replay.

    Parameters
    ----------
    task : object
        Prepared task with the pinned test dataset.
    options : mapping
        Explicit fixed durations, test batch budget, and mask seed.
    root : pathlib.Path
        Directory containing immutable condition summaries.
    identity : dict
        Checkpoint and test-selection provenance.
    workers, batch_size : int
        Loader concurrency and original evaluation batch size.
    progress : callable
        Lifecycle callback receiving completed/total batches and stage.
    smoke_batches : int or None, optional
        Stop without publishing after this many batches; default None.
    alpha : float
        Required fitted coefficient for content L1.

    Returns
    -------
    dict
        Per-duration frequency errors (F,) for all/content cells and three
        batch-averaged objectives, with the matching evaluation identity.
    """
    import json
    import math
    from .storage import write_json
    from util.mask_utils import test_masked_dataset

    identity = {**identity, "configuration": dict(options)}
    durations = list(options["durations_frames"])
    conditions = {}
    for duration in durations:
        path = root / "conditions" / f"temporal_{duration}.json"
        saved = json.loads(path.read_text())
        if saved["identity"] != identity:
            raise ValueError("Saved temporal condition provenance differs.")
        conditions[str(duration)] = saved
    dataset = test_masked_dataset(task.test_set, targets_only=True)
    total = math.ceil(len(dataset) / batch_size)
    if total != options["batches"]:
        raise ValueError("Test batch count differs from saved diagnostic.")
    frequency_count = len(conditions[str(durations[0])]["frequency_sums"])
    bins = balanced_bins(len(dataset), frequency_count, options["seed"])
    result_path = root / "interpolation_metrics.json"
    partial_path = root / "interpolation_metrics_partial.json"
    if result_path.exists() and smoke_batches is None:
        result = json.loads(result_path.read_text())
        if (result["identity"] != identity
                or result["criterion_alpha"] != alpha):
            raise ValueError("Saved interpolation provenance differs.")
        return result
    sums = {str(d): torch.zeros(frequency_count, 2, dtype=torch.float64)
            for d in durations}
    objectives = {str(d): torch.zeros(3, dtype=torch.float64)
                  for d in durations}
    edges = {str(d): 0 for d in durations}
    offset, completed = 0, 0
    if partial_path.exists() and smoke_batches is None:
        state = json.loads(partial_path.read_text())
        if (state["identity"] != identity
                or state["criterion_alpha"] != alpha):
            raise ValueError("Partial interpolation provenance differs.")
        offset, completed = state["windows"], state["batches"]
        sums = {k: torch.tensor(v, dtype=torch.float64)
                for k, v in state["error_sums"].items()}
        objectives = {k: torch.tensor(v, dtype=torch.float64)
                      for k, v in state["objective_sums"].items()}
        edges = state["edge_windows"]
    subset = torch.utils.data.Subset(dataset, range(offset, len(dataset)))
    loader = task.get_batch_iterator(
        subset, batch_size, shuffle=False, num_workers=workers,
    )
    for batch in loader:
        target, padding = batch["target"], batch["attn_mask"].bool()
        indices = torch.arange(offset, offset + len(target))
        for duration in durations:
            mask, _ = fixed_masks(
                padding, frequency_count, indices, duration, max(durations),
                bins, options["seed"],
            )
            error, edge_count = linear_temporal_errors(target, padding, mask)
            content_count = (mask & (
                target.abs() > CONTENT_THRESHOLD
            )).sum()
            if content_count <= 0:
                raise ValueError("Interpolation batch has no content cells.")
            l1 = error[:, 0].sum() / mask.sum()
            content_l1 = error[:, 1].sum() / content_count
            objectives[str(duration)] += torch.stack((
                l1, content_l1, l1 + alpha * content_l1,
            ))
            sums[str(duration)] += error
            edges[str(duration)] += edge_count
        offset += len(target)
        completed += 1
        progress(completed, total, "temporal_interpolation")
        if smoke_batches is None and completed % PARTIAL_SAVE_BATCHES == 0:
            write_json(partial_path, {
                "identity": identity, "windows": offset,
                "batches": completed,
                "error_sums": {k: v.tolist() for k, v in sums.items()},
                "objective_sums": {k: v.tolist()
                                   for k, v in objectives.items()},
                "criterion_alpha": alpha, "edge_windows": edges,
            })
        if smoke_batches and completed >= smoke_batches:
            break
    if completed == 0:
        raise ValueError("Interpolation test dataset has no batches.")
    result = {
        "identity": identity, "windows": offset, "batches": completed,
        "method": "linear between adjacent visible frames; edge hold",
        "error_sums": {k: v[:, 0].tolist() for k, v in sums.items()},
        "content_error_sums": {k: v[:, 1].tolist() for k, v in sums.items()},
        "objective_means": {k: (v / completed).tolist()
                            for k, v in objectives.items()},
        "criterion_alpha": alpha, "edge_windows": edges,
    }
    if smoke_batches is None:
        for duration in durations:
            expected = conditions[str(duration)]
            if (
                offset != expected["windows"]
                or completed != expected["batches"]
            ):
                raise ValueError("Interpolation and model membership differ.")
            counts = [row[0] for row in expected["frequency_sums"]]
            if any(count != offset * duration for count in counts):
                raise ValueError("Temporal condition mask counts differ.")
        old_path = root / "interpolation.json"
        if old_path.exists():
            previous = json.loads(old_path.read_text())
            if previous["identity"] != identity:
                raise ValueError("Previous interpolation provenance differs.")
            for duration in durations:
                if not torch.allclose(
                    torch.tensor(previous["error_sums"][str(duration)],
                                 dtype=torch.float64),
                    torch.tensor(result["error_sums"][str(duration)],
                                 dtype=torch.float64),
                    rtol=1e-10, atol=1e-8,
                ):
                    raise ValueError("Replayed interpolation L1 changed.")
        write_json(result_path, result)
        partial_path.unlink(missing_ok=True)
    return result


def plot_interpolation(
    result: dict, root: Path, axis: Sequence[float], reports: Path,
    fit: Path, step: int, settings: Mapping,
) -> None:
    """Publish paired model/interpolation losses and standalone baselines.

    Each metric has duration-specific frequency figures for relative model
    loss and absolute baseline loss. Overall comparisons use the exact saved
    batch means; frequency comparisons pool cells within each bin.
    """
    import json
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from .reporting import Report

    report = Report(reports, fit, step, [], settings)
    index = reports / "output_index.csv"
    if index.exists():
        report.entries = pd.read_csv(index).to_dict("records")
    options = result["identity"]["configuration"]
    alpha = result["criterion_alpha"]
    rows, overall_rows, outputs = [], [], {}
    for path in sorted((root / "conditions").glob("*.json")):
        saved = json.loads(path.read_text())
        if saved["identity"] != result["identity"]:
            raise ValueError("Interpolation and model provenance differ.")
        outputs[path.stem] = saved
    for duration in options["durations_frames"]:
        saved = outputs[f"temporal_{duration}"]
        model = frequency_losses(saved["frequency_sums"], alpha)
        baseline_sums = np.asarray(saved["frequency_sums"], dtype=float)
        baseline_sums[:, 1] = result["error_sums"][str(duration)]
        baseline_sums[:, 4] = result["content_error_sums"][str(duration)]
        interpolation = frequency_losses(baseline_sums, alpha)["model"]
        ms = duration * options["frame_hop_ms"]
        for metric_index, (metric, title) in enumerate(zip(
            LOSS_METRICS, LOSS_LABELS,
        )):
            numerator = model["model"][:, metric_index]
            denominator = interpolation[:, metric_index]
            ratio = [checked_ratio(a, b)
                     for a, b in zip(numerator, denominator)]
            for i, hz in enumerate(axis):
                rows.append({
                    "duration_frames": duration, "frequency_hz": hz,
                    "metric": metric, "model_loss": numerator[i],
                    "interpolation_loss": denominator[i],
                    "constant_prediction_loss": model["constant"][
                        i, metric_index
                    ],
                    "model_over_interpolation": ratio[i],
                    "count": baseline_sums[i, 0],
                    "content_count": baseline_sums[i, 3],
                })
            fig, ax = plt.subplots(figsize=(13, 8))
            ax.plot(axis, ratio, color="#0072B2", lw=2.5, label="Model")
            ax.axhline(1, color="0.5", ls="--", lw=2.5,
                       label="Temporal interpolation")
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel(f"{title} / interpolation loss")
            prefix = f"test/fixed_masks/interpolation/{metric}"
            report.figure(
                f"{prefix}/duration_{duration}", fig,
                f"Temporal {title}: {duration} frames ({ms:.0f} ms); "
                f"validation-selected checkpoint {step}",
            )
            fig, ax = plt.subplots(figsize=(13, 8))
            ax.plot(axis, denominator, color="0.25", lw=2.5,
                    label="Temporal interpolation")
            ax.plot(axis, model["constant"][:, metric_index],
                    color="0.55", ls="--", lw=2.5,
                    label=BASELINE_LABELS[metric_index])
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel(f"{title} (normalised targets)")
            report.figure(
                f"{prefix}/baseline_{duration}", fig,
                f"Temporal baseline {title}: {duration} frames ({ms:.0f} ms)",
            )
            overall_rows.append({
                "duration_frames": duration, "duration_ms": ms,
                "metric": metric,
                "model_loss": saved["objective_means"][metric_index],
                "interpolation_loss": result["objective_means"][
                    str(duration)
                ][metric_index],
                "constant_prediction_loss": saved["objective_means"][
                    metric_index + 3
                ],
            })
    report.table(
        "test/fixed_masks/interpolation/frequency/reconstruction",
        pd.DataFrame(rows), "Cell-pooled losses on paired temporal masks",
    )
    overall = pd.DataFrame(overall_rows)
    overall["model_over_interpolation"] = [
        checked_ratio(a, b) for a, b in zip(
            overall.model_loss, overall.interpolation_loss,
        )
    ]
    report.table(
        "test/fixed_masks/interpolation/duration/objective", overall,
        "Paired temporal batch-averaged objectives",
    )
    plot_interpolation_duration(report, overall)
    removed = set()
    for duration in options["durations_frames"]:
        for name in ("duration", "baseline"):
            relative = (
                "figures/test/fixed_masks/interpolation/frequency/"
                f"{name}_{duration}.png"
            )
            replacement = reports / relative.replace("/frequency/", "/l1/")
            if replacement.exists():
                (reports / relative).unlink(missing_ok=True)
                removed.add(relative)
    from .storage import atomic_write

    entries = pd.DataFrame([e for e in report.entries
                            if e["output"] not in removed])
    atomic_write(index, lambda p: entries.to_csv(p, index=False),
                 lambda p: pd.read_csv(p))
    previous = reports / "figures/test/fixed_masks/interpolation/frequency"
    if previous.exists() and not any(previous.iterdir()):
        previous.rmdir()
    plot_fixed(outputs, axis, options["frame_hop_ms"], reports, fit, step,
               settings)
    print(f"Interpolation reports saved: {reports.resolve()}", flush=True)


def plot_interpolation_duration(report: Any, frame: Any) -> None:
    """Compare temporal batch objectives across durations for all metrics."""
    import matplotlib.pyplot as plt

    for normalised in (False, True):
        fig, axes = plt.subplots(1, 3, figsize=(23, 7))
        for ax, metric, title in zip(axes, LOSS_METRICS, LOSS_LABELS):
            subset = frame[frame.metric == metric].sort_values("duration_ms")
            if normalised:
                ax.plot(subset.duration_ms, subset.model_over_interpolation,
                        color="#0072B2", lw=2.5, marker="o", label="Model")
                ax.axhline(1, color="0.25", ls="--",
                           label="Temporal interpolation")
            else:
                for column, color, style, label in (
                    ("model_loss", "#0072B2", "-", "Model"),
                    ("interpolation_loss", "0.25", "--",
                     "Temporal interpolation"),
                    ("constant_prediction_loss", "0.55", ":",
                     "Constant prediction"),
                ):
                    ax.plot(subset.duration_ms, subset[column], color=color,
                            ls=style, marker="o", lw=2.5, label=label)
            ax.set_title(title)
            ax.set_xticks(subset.duration_ms, [
                f"{row.duration_ms:.0f} ms\n({row.duration_frames} frames)"
                for row in subset.itertuples()
            ])
            ax.set_xlabel("Temporal mask duration")
            ax.set_ylabel("Loss / interpolation loss" if normalised
                          else "Loss (normalised targets)")
        name = "normalised" if normalised else "raw"
        report.figure(
            f"test/fixed_masks/interpolation/duration/{name}", fig,
            "Temporal reconstruction and local interpolation; "
            f"validation-selected checkpoint {report.best}",
        )


def training_content_counts(
    task: Any, task_cfg: Any, options: Mapping, root: Path, identity: dict,
    workers: int, batch_size: int, progress: Callable,
    smoke_batches: int | None = None,
) -> dict:
    """Count selected content cells in a reproducible training-window sample.

    Targets have shape (B, T, F). Sampling is uniform without replacement
    within the saved training selection, preserving its subgroup mixture in
    expectation. Original training masks are replayed with a per-window seed,
    independent of loader concurrency. Counts describe this diagnostic draw,
    not the exact historical random masks. No predictions are computed.

    Parameters
    ----------
    task, task_cfg : object
        Prepared task and original training masking configuration.
    options : mapping
        Explicit sample_windows, selection_seed and mask_seed.
    root : pathlib.Path
        Campaign diagnostic directory, outside immutable fit storage.
    identity : dict
        Saved training selection and scientific fit identity.
    workers, batch_size : int
        Bounded loader concurrency and batch size.
    progress : callable
        Lifecycle callback receiving completed/total batches and stage.
    smoke_batches : int or None, optional
        Stop after this many batches without saving; default None.

    Returns
    -------
    dict
        Sample provenance and selected/content counts of shape (F,).
    """
    import json
    import math
    import random
    import numpy as np
    from .storage import digest, write_json
    from util.mask_utils import mask_inputs, test_masked_dataset

    if set(options) != {"sample_windows", "selection_seed", "mask_seed"}:
        raise ValueError("Training content options require sample_windows, "
                         "selection_seed and mask_seed.")
    if any(type(v) is not int or v < 0 for v in options.values()):
        raise ValueError("Training content options must be nonnegative ints.")
    size = options["sample_windows"]
    if not 0 < size <= len(task.train_set):
        raise ValueError("Training sample must fit within saved selection.")
    indices = np.random.default_rng(options["selection_seed"]).choice(
        len(task.train_set), size=size, replace=False,
    )
    indices.sort()
    provenance = {
        **identity, "configuration": dict(options),
        "sample_indices_sha256": digest(indices.tolist()),
        "content_threshold": CONTENT_THRESHOLD,
    }
    destination = root / "training_content.json"
    if destination.exists() and smoke_batches is None:
        saved = json.loads(destination.read_text())
        if saved["identity"] != provenance:
            raise ValueError("Saved training content sample differs.")
        return saved
    dataset = torch.utils.data.Subset(
        test_masked_dataset(task.train_set, targets_only=True), indices,
    )
    loader = task.get_batch_iterator(
        dataset, batch_size, shuffle=False, num_workers=workers,
    )
    count = content_count = None
    offset = 0
    total = math.ceil(size / batch_size)
    random_state = random.getstate()
    try:
        for batch_index, batch in enumerate(loader):
            target = batch["target"]
            valid = ~batch["attn_mask"].bool()
            if not torch.isfinite(target).all():
                raise ValueError("Training content targets must be finite.")
            if count is None:
                count = torch.zeros(target.shape[2], dtype=torch.int64)
                content_count = torch.zeros_like(count)
            for row in range(len(target)):
                random.seed(options["mask_seed"] + int(indices[offset]))
                values = target[row, valid[row]]
                _, selected = mask_inputs(values, task_cfg)
                selected = selected.bool()
                count += selected.sum(0)
                content_count += (
                    selected & (values.abs() > CONTENT_THRESHOLD)
                ).sum(0)
                offset += 1
            progress(batch_index + 1, total, "training_content_sample")
            if smoke_batches and batch_index + 1 >= smoke_batches:
                break
    finally:
        random.setstate(random_state)
    if count is None or (count <= 0).any() or content_count.sum() == 0:
        raise ValueError("Training sample lacks selected/content cells.")
    result = {
        "identity": provenance, "windows": offset,
        "training_windows": len(task.train_set),
        "selected_count": count.tolist(),
        "content_count": content_count.tolist(),
        "source": "sampled training targets with original training masks",
        "historical_masks": False,
    }
    if smoke_batches is None:
        write_json(destination, result)
    return result


def plot_training_content(
    result: dict, axis: Sequence[float], reports: Path, fit: Path,
    step: int, settings: Mapping,
) -> None:
    """Plot selected training content counts and fractions by frequency.

    The table contains raw selected/content counts, content prevalence among
    selected cells and each bin's share of selected content cells. No loss
    contributions are inferred from these counts.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from .reporting import Report

    count = np.asarray(result["selected_count"], dtype=float)
    content = np.asarray(result["content_count"], dtype=float)
    if count.shape != (len(axis),) or content.shape != count.shape:
        raise ValueError("Frequency centres and training counts differ.")
    if (count <= 0).any() or not np.isfinite(count + content).all():
        raise ValueError("Training counts must be finite with positive totals.")
    if (content < 0).any() or (content > count).any() or content.sum() <= 0:
        raise ValueError("Training content counts must lie within totals.")
    frame = pd.DataFrame({
        "frequency_hz": axis, "selected_count": count.astype(int),
        "content_count": content.astype(int),
        "content_fraction": content / count,
        "share_of_content_cells": content / content.sum(),
        "sample_windows": result["windows"],
        "historical_masks": False,
    })
    report = Report(reports, fit, step, [], settings)
    index = reports / "output_index.csv"
    if index.exists():
        report.entries = pd.read_csv(index).to_dict("records")
    report.table("training/content_distribution/counts", frame,
                 "Training subset counts under a reproducible original-mask "
                 "draw; not historical masks or reconstruction errors")
    if len(axis) < 2 or (np.diff(axis) <= 0).any():
        raise ValueError("Histogram needs increasing frequency centres.")
    width = float(np.min(np.diff(axis))) * 0.85
    fig, axes = plt.subplots(1, 2, figsize=(20, 7))
    for ax, values, ylabel in zip(
        axes, (content, 100 * content / count),
        ("Selected content cells", "Content / selected cells (%)"),
    ):
        ax.bar(axis, values, width=width, color="#0072B2")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel(ylabel)
        ax.set_xlim(min(axis) - width, max(axis) + width)
        ax.ticklabel_format(axis="y", style="plain")
    report.figure(
        "training/content_distribution/counts", fig,
        f"Training content cells: {result['windows']:,}-window estimate; "
        f"|target| > {CONTENT_THRESHOLD:g}",
    )
    print(f"Training content reports saved: {reports.resolve()}", flush=True)
