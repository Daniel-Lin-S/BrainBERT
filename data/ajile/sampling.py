"""Prepare AJILE selected raw STFTs and session normalization moments.

Input is a frozen selection manifest and audited source recordings.
Output is raw STFT files and per-session moment tables in the local cache.
"""

from pathlib import Path

import numpy as np

from util.experiments.sampling import (
    allocate, load_pinned_evaluation, select_group, select_or_load
)
from util.experiments.storage import digest

DEFAULT_ALPHA = 0.0
DEFAULT_SEED = 0
DEFAULT_EVAL_BATCHES = 128
DEFAULT_EVAL_BATCH_SIZE = 256
PREPROCESS_NICE = 10


def select_evaluation_group(
    dataset: object,
    record: dict,
    quota: int,
    cached: set[int],
    rng: np.random.Generator,
) -> list[int]:
    """Balance electrodes, prioritizing cached eligible windows."""
    from data.ajile.normalization import eligibility

    valid = eligibility(dataset.audit, record)
    positions = np.flatnonzero(valid)
    channels = record["metadata"]["channels"]
    groups = [
        np.flatnonzero(positions % channels == channel)
        for channel in range(channels)
    ]
    active = [group for group in groups if len(group)]
    if quota > len(positions):
        raise ValueError("Evaluation budget exceeds eligible examples.")
    quotas = allocate([len(group) for group in active], quota, 0)
    excess = 0
    for index, group in enumerate(active):
        if quotas[index] > len(group):
            excess += quotas[index] - len(group)
            quotas[index] = len(group)
    while excess:
        candidates = [
            index for index, group in enumerate(active)
            if quotas[index] < len(group)
        ]
        if not candidates:
            raise ValueError("Evaluation quota has no remaining capacity.")
        for index in candidates:
            if not excess:
                break
            quotas[index] += 1
            excess -= 1
    cache_by_group = []
    for group in active:
        selected = set()
        for index in cached:
            position = np.searchsorted(group, index)
            if position < len(group) and group[position] == index:
                selected.add(int(position))
        cache_by_group.append(selected)
    chosen = []
    for group, group_quota, group_cache in zip(
        active, quotas, cache_by_group
    ):
        relative = select_group(
            len(group), group_quota, group_cache, rng, False
        )
        chosen.extend(int(group[index]) for index in relative)
    return sorted(chosen)


def _prepare_session(
    dataset: object, session: int, indices: list[int]
) -> tuple[str, dict]:
    """Cache selected spectra and fit moments for one session."""
    from data.ajile.normalization import eligibility
    from util.experiments.storage import atomic_write

    root = Path(dataset.extracter.cfg.feature_cache).resolve()
    frequencies = len(dataset.extracter.frequency_axis())
    record = dataset.records[session]
    indices.sort()
    selected_id = digest({
        "source": record["content_id"],
        "indices": indices,
        "preprocessor": dataset.cache_settings(),
    })
    path = root / "ajile_selected_stats" / (
        selected_id + ".npz"
    )
    dataset.active = session
    dataset.valid = eligibility(dataset.audit, record)
    dataset.positions = np.flatnonzero(dataset.valid)
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            means = saved["mean"].copy()
            stds = saved["std"].copy()
        for index in indices:
            dataset.raw_spectrum(index)
    else:
        channels = record["metadata"]["channels"]
        count = np.zeros(channels, dtype=np.int64)
        means = np.full(
            (channels, frequencies), np.nan, dtype=np.float64
        )
        m2 = np.zeros((channels, frequencies), dtype=np.float64)
        offset = dataset.ends[session - 1] if session else 0
        for index in indices:
            local = index - offset
            position = int(dataset.positions[local])
            channel = position % channels
            values = dataset.raw_spectrum(index).double().numpy()
            n = values.shape[0]
            chunk_mean = values.mean(axis=0)
            chunk_m2 = ((values - chunk_mean) ** 2).sum(axis=0)
            if count[channel] == 0:
                means[channel] = chunk_mean
            else:
                delta = chunk_mean - means[channel]
                total = count[channel] + n
                means[channel] += delta * n / total
                m2[channel] += (
                    chunk_m2
                    + delta**2 * count[channel] * n / total
                )
            if count[channel] == 0:
                m2[channel] = chunk_m2
            count[channel] += n
        stds = np.full_like(means, np.nan)
        active = count > 0
        stds[active] = np.sqrt(
            m2[active] / count[active, None]
        )
        if (
            not np.isfinite(means[active]).all()
            or not np.isfinite(stds[active]).all()
            or np.any(stds[active] <= 0)
        ):
            raise ValueError(
                f"Selected AJILE moments are invalid: "
                f"{record['session']}"
            )
        def write(candidate: Path) -> None:
            with candidate.open("wb") as stream:
                np.savez_compressed(
                    stream, mean=means, std=stds, count=count
                )
        atomic_write(
            path, write,
            lambda candidate: np.load(
                candidate, allow_pickle=False
            ),
        )
    channels = record["metadata"]["channels"]
    offset = dataset.ends[session - 1] if session else 0
    active = np.unique(
        dataset.positions[np.asarray(indices) - offset] % channels
    )
    expected = (channels, frequencies)
    if means.shape != expected or stds.shape != expected:
        raise ValueError(
            f"Expected selected AJILE moments shape {expected}, "
            f"got {means.shape} and {stds.shape}."
        )
    if (
        not np.isfinite(means[active]).all()
        or not np.isfinite(stds[active]).all()
        or np.any(stds[active] <= 0)
    ):
        raise ValueError(
            f"Selected AJILE moments are invalid: {record['session']}"
        )
    return record["session"], {"mean": means, "std": stds}


def prepare_selected(
    dataset: object, selection: dict, workers: int = 1,
    progress_callback=None,
) -> dict:
    """Cache selected raw STFTs and fit per-session frequency moments.

    Parameters
    ----------
    dataset : object
        AJILE dataset with source paths and spectral settings.
    selection : dict
        Frozen manifest of global indices by phase.
    workers : int, optional
        Independent session workers, default 1.
    progress_callback : callable, optional
        Receives completed and total session counts, default None.

    Returns
    -------
    dict
        Session-keyed mean and standard deviation arrays, each shaped
        (electrodes, frequencies).
    """
    from bisect import bisect_right

    chosen = set(sum(selection["indices"].values(), []))
    by_session = {}
    for index in chosen:
        session = bisect_right(dataset.ends, index)
        by_session.setdefault(session, []).append(index)
    moments = {}
    if workers <= 0:
        raise ValueError("workers must be positive.")
    tasks = [(dataset, session, sorted(indices))
             for session, indices in sorted(by_session.items())]
    if progress_callback is not None:
        progress_callback(0, len(tasks))
    if workers == 1:
        results = (_prepare_session(*task) for task in tasks)
        for name, values in results:
            moments[name] = values
            if progress_callback is not None:
                progress_callback(len(moments), len(tasks))
            print(f"Prepared AJILE session {name}.", flush=True)
    else:
        from concurrent.futures import (
            ProcessPoolExecutor, as_completed,
        )

        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(_prepare_session, *task) for task in tasks
            ]
            for future in as_completed(futures):
                name, values = future.result()
                moments[name] = values
                if progress_callback is not None:
                    progress_callback(len(moments), len(tasks))
                print(
                    f"Prepared AJILE session {name}: "
                    f"{len(moments)}/{len(tasks)} sessions.",
                    flush=True,
                )
    dataset.active = None
    dataset.valid = None
    dataset.positions = None
    dataset.selected_statistics = moments
    return moments


def main() -> None:
    """Freeze AJILE selections and prepare their reusable STFT caches."""
    import argparse
    import os

    from omegaconf import OmegaConf

    from datasets.ajile_dataset import AJILE12

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--feature-cache", required=True, type=Path)
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--preprocessor", required=True, type=Path)
    parser.add_argument("--sample-budget", type=int)
    parser.add_argument("--pinned-evaluation", type=Path)
    parser.add_argument(
        "--validation-batches", type=int,
        default=DEFAULT_EVAL_BATCHES,
    )
    parser.add_argument(
        "--test-batches", type=int, default=DEFAULT_EVAL_BATCHES
    )
    parser.add_argument(
        "--valid-batch-size", type=int,
        default=DEFAULT_EVAL_BATCH_SIZE,
    )
    parser.add_argument(
        "--alpha", type=float, default=DEFAULT_ALPHA
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if hasattr(os, "nice"):
        os.nice(PREPROCESS_NICE)
    preprocessor = OmegaConf.load(args.preprocessor)
    preprocessor.feature_cache = str(args.feature_cache.resolve())
    data_cfg = OmegaConf.create({
        "data": str(args.source_root.resolve()),
        "audit": str(args.audit.resolve()),
    })
    dataset = AJILE12(data_cfg, preprocessor_cfg=preprocessor)
    if args.sample_budget is None and args.pinned_evaluation is None:
        raise ValueError(
            "Full-data preparation requires --pinned-evaluation."
        )
    settings = {
        "mode": "temperature",
        "sample_budget": args.sample_budget,
        "alpha": args.alpha,
        "seed": args.seed,
        "validation_batches": args.validation_batches,
        "test_batches": args.test_batches,
        "valid_batch_size": args.valid_batch_size,
    }
    pinned = (
        load_pinned_evaluation(args.pinned_evaluation, dataset)
        if args.pinned_evaluation is not None else None
    )
    if args.sample_budget is None:
        selected = {"indices": {"train": [], **pinned["indices"]}}
    else:
        selected = select_or_load(args.artifact_root, dataset, settings)
        if pinned is not None:
            for phase in ("validation", "test"):
                if selected["indices"][phase] != pinned["indices"][phase]:
                    raise ValueError(
                        f"Selected {phase} windows differ from pin."
                    )
    dataset.configure_selection(selected)
    if args.max_examples is not None:
        if args.max_examples <= 0:
            raise ValueError("max-examples must be positive.")
        indices = sorted(set(sum(selected["indices"].values(), [])))
        for index in indices[:args.max_examples]:
            dataset.raw_spectrum(index)
        print(
            f"Prepared {min(args.max_examples, len(indices))} "
            "selected raw STFTs for profiling.",
            flush=True,
        )
        return
    prepare_selected(dataset, selected, workers=args.workers)
    print(
        "Prepared selected AJILE raw STFTs and normalization moments.",
        flush=True,
    )


if __name__ == "__main__":
    main()
