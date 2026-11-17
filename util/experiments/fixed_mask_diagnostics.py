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

PARTIAL_SAVE_BATCHES = 16

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
    index_path = reports / "output_index.csv"
    if index_path.exists():
        report.entries = pd.read_csv(index_path).to_dict("records")
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
    complete_durations = [
        d for d in sorted(overall.duration_frames.unique()) if d
        and f"temporal_{d}" in outputs and f"joint_{d}" in outputs
    ]
    expected = next(iter(outputs.values()))["identity"][
        "configuration"
    ]["durations_frames"]
    if set(complete_durations) == set(expected):
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
    for duration in complete_durations:
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
    if set(complete_durations) == set(expected):
        from .storage import atomic_write, file_digest

        ready = reports / "figures/test/fixed_masks/frequency"
        required = [ready / "duration_comparison.png"] + [
            ready / f"duration_{duration}.png" for duration in expected
        ]
        if all(path.exists() for path in required):
            archive = reports.parent / (
                "diagnostics/fixed_masks/previous_reports"
            )
            removed = set()
            for pattern in (
                "frequency_reconstruction*.png", "frequency_coverage*.png",
            ):
                for old in (reports / "figures/test").glob(pattern):
                    archive.mkdir(parents=True, exist_ok=True)
                    destination = archive / old.name
                    if destination.exists():
                        if file_digest(old) != file_digest(destination):
                            raise ValueError(
                                "Archived frequency figure differs."
                            )
                        old.unlink()
                    else:
                        old.rename(destination)
                    removed.add(str(old.relative_to(reports)))
            index = pd.read_csv(index_path)
            index = index[~index.output.isin(removed)]
            atomic_write(
                index_path, lambda p: index.to_csv(p, index=False),
                lambda p: pd.read_csv(p),
            )
    print(f"Fixed-mask reports saved: {reports.resolve()}", flush=True)



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
        Absolute-error sums of shape (F,), and edge-gap window count. Where
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
    error = ((estimate - target).abs().double() * mask).sum((0, 1))
    if not torch.isfinite(error).all():
        raise ValueError("Interpolation errors must be finite.")
    return error, int((~(has_left & has_right)).sum())


@torch.inference_mode()
def evaluate_interpolation(
    task: Any, options: Mapping, root: Path, identity: dict,
    workers: int, progress: Callable, batch_size: int,
    smoke_batches: int | None = None,
) -> dict:
    """Evaluate interpolation on the original paired temporal masks.

    The existing condition JSON supplies exact BrainBERT error and counts;
    a single pass computes interpolation error sums for all durations.
    The resulting JSON records sums, edge counts and provenance without
    persisting targets. Partial aggregate sums allow interrupted replay.
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
    result_path = root / "interpolation.json"
    partial_path = root / "interpolation_partial.json"
    if result_path.exists() and smoke_batches is None:
        result = json.loads(result_path.read_text())
        if result["identity"] != identity:
            raise ValueError("Saved interpolation provenance differs.")
        return result
    sums = {str(d): torch.zeros(frequency_count, dtype=torch.float64)
            for d in durations}
    edges = {str(d): 0 for d in durations}
    offset, completed = 0, 0
    if partial_path.exists() and smoke_batches is None:
        state = json.loads(partial_path.read_text())
        if state["identity"] != identity:
            raise ValueError("Partial interpolation provenance differs.")
        offset, completed = state["windows"], state["batches"]
        sums = {k: torch.tensor(v, dtype=torch.float64)
                for k, v in state["error_sums"].items()}
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
                "edge_windows": edges,
            })
        if smoke_batches and completed >= smoke_batches:
            break
    if completed == 0:
        raise ValueError("Interpolation test dataset has no batches.")
    result = {
        "identity": identity, "windows": offset, "batches": completed,
        "method": "linear between adjacent visible frames; edge hold",
        "error_sums": {k: v.tolist() for k, v in sums.items()},
        "edge_windows": edges,
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
        write_json(result_path, result)
        partial_path.unlink(missing_ok=True)
    return result


def plot_interpolation(
    result: dict, root: Path, axis: Sequence[float], reports: Path,
    fit: Path, step: int, settings: Mapping,
) -> None:
    """Publish paired relative L1 curves and frequency-level source table."""
    import json
    import matplotlib.pyplot as plt
    import pandas as pd
    from .reporting import Report

    report = Report(reports, fit, step, [], settings)
    index = reports / "output_index.csv"
    if index.exists():
        report.entries = pd.read_csv(index).to_dict("records")
    rows = []
    for duration in result["identity"]["configuration"]["durations_frames"]:
        saved = json.loads(
            (root / "conditions" / f"temporal_{duration}.json").read_text()
        )
        frame_rows = []
        for hz, model, baseline in zip(
            axis, saved["frequency_sums"], result["error_sums"][str(duration)]
        ):
            ratio = model[1] / baseline if baseline > 0 else None
            if ratio is None:
                warnings.warn("Interpolation L1 is zero; ratio is undefined.")
            frame_rows.append({
                "duration_frames": duration, "frequency_hz": hz,
                "count": model[0], "model_error_sum": model[1],
                "interpolation_error_sum": baseline,
                "interpolation_l1": baseline / model[0],
                "model_l1": model[1] / model[0],
                "constant_prediction_l1": model[2] / model[0],
                "relative_l1": ratio,
            })
        frame = pd.DataFrame(frame_rows)
        rows.extend(frame_rows)
        fig, ax = plt.subplots(figsize=(13, 8))
        ax.plot(
            frame.frequency_hz, frame.relative_l1,
            color="#0072B2", lw=2.5, label="BrainBERT",
        )
        ax.axhline(1, color="0.5", ls="--", lw=2.5,
                   label="Temporal interpolation")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("L1 / interpolation L1")
        ms = duration * result["identity"]["configuration"]["frame_hop_ms"]
        report.figure(
            f"test/fixed_masks/interpolation/frequency/duration_{duration}",
            fig, f"Temporal reconstruction: {duration} frames ({ms:.0f} ms); "
            f"validation-selected checkpoint {step}",
        )
        fig, ax = plt.subplots(figsize=(13, 8))
        ax.plot(
            frame.frequency_hz, frame.model_l1,
            color="#0072B2", lw=2.5, label="BrainBERT",
        )
        ax.plot(
            frame.frequency_hz, frame.interpolation_l1,
            color="0.25", ls="--", lw=2.5,
            label="Temporal interpolation",
        )
        ax.plot(
            frame.frequency_hz, frame.constant_prediction_l1,
            color="0.55", ls=":", lw=2.5,
            label="Constant prediction L1",
        )
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("L1 (normalized targets)")
        report.figure(
            f"test/fixed_masks/l1_by_frequency/duration_{duration}",
            fig, f"Temporal reconstruction L1: {duration} frames "
            f"({ms:.0f} ms); validation-selected checkpoint {step}",
        )
        fig, ax = plt.subplots(figsize=(13, 8))
        ax.plot(
            frame.frequency_hz, frame.interpolation_l1,
            color="0.25", lw=2.5, label="Temporal interpolation",
        )
        ax.plot(
            frame.frequency_hz, frame.constant_prediction_l1,
            color="0.55", ls="--", lw=2.5, label="Constant prediction L1",
        )
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("L1 (normalized targets)")
        report.figure(
            f"test/fixed_masks/interpolation/frequency/baseline_{duration}",
            fig, f"Temporal baselines: {duration} frames ({ms:.0f} ms)",
        )
    report.table(
        "test/fixed_masks/interpolation/frequency/reconstruction",
        pd.DataFrame(rows),
        "Temporal reconstruction relative to local interpolation",
    )
    print(f"Interpolation reports saved: {reports.resolve()}", flush=True)
