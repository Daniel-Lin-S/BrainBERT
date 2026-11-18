"""Reusable budgeted group sampling and frozen selection manifests.

Input is dataset-provided group membership, cache membership, and sampling
settings. Output is a JSON manifest of global train, validation, and test
indices under a dataset namespace in the artifact root.
"""

import fcntl
import json
import math
import warnings
from pathlib import Path

import numpy as np

from util.experiments.storage import digest, write_json


def allocate(counts: list[int], budget: int, alpha: float) -> list[int]:
    """Allocate exact group quotas by largest-remainder rounding.

    Parameters
    ----------
    counts : list[int]
        Positive eligible-example counts, one per subgroup.
    budget : int
        Total examples to draw.
    alpha : float
        Subgroup count exponent in [0, 1].

    Returns
    -------
    list[int]
        Per-group integer quotas summing to the budget.
    """
    if not counts or any(count <= 0 for count in counts):
        raise ValueError("Every sampling group needs eligible examples.")
    if budget <= 0 or not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("Expected positive budget and alpha in [0, 1].")
    weights = np.power(np.asarray(counts, dtype=float), alpha)
    targets = budget * weights / weights.sum()
    quotas = np.floor(targets).astype(np.int64)
    order = np.argsort(-(targets - quotas), kind="stable")
    quotas[order[:budget - int(quotas.sum())]] += 1
    return quotas.tolist()


def select_group(
    eligible: int,
    quota: int,
    cached: set[int],
    rng: np.random.Generator,
    allow_repeat: bool,
) -> list[int]:
    """Choose cached examples first and then uncached examples.

    Parameters
    ----------
    eligible : int
        Number of eligible examples in the group.
    quota : int
        Draws requested from the group.
    cached : set[int]
        Local eligible indices with compatible raw STFT caches.
    rng : numpy.random.Generator
        Sampling generator.
    allow_repeat : bool
        Whether draws may repeat after every eligible example is used.

    Returns
    -------
    list[int]
        Local eligible indices of length `quota`.
    """
    if eligible <= 0 or quota < 0:
        raise ValueError("Expected positive eligibility and valid quota.")
    if any(index < 0 or index >= eligible for index in cached):
        raise ValueError("Cached example index is outside eligibility.")
    if not allow_repeat and quota > eligible:
        raise ValueError("Evaluation budget exceeds eligible examples.")
    old = sorted(cached)
    rng.shuffle(old)
    chosen = old[:min(quota, len(old))]
    unique_target = min(quota, eligible)
    if len(chosen) < unique_target:
        remaining = np.setdiff1d(
            np.arange(eligible), np.asarray(old, dtype=np.int64),
            assume_unique=True,
        )
        added = rng.choice(
            remaining, size=unique_target - len(chosen), replace=False
        )
        chosen.extend(int(index) for index in added)
    if quota > eligible:
        chosen.extend(
            int(index)
            for index in rng.integers(eligible, size=quota - eligible)
        )
    return chosen


def selection_path(
    root: Path, namespace: str, identity: dict
) -> Path:
    """Locate a manifest shared by models with the same sampling protocol."""
    return root / namespace / "selections" / f"{digest(identity)}.json"


def load_pinned_evaluation(path: Path, dataset: object) -> dict:
    """Verify fixed evaluation windows against audited source splits.

    Parameters
    ----------
    path : pathlib.Path
        JSON with source content IDs and validation/test global indices.
    dataset : object
        Dataset exposing ordered records, cumulative ends and partitions.

    Returns
    -------
    dict
        Verified pinned evaluation membership.
    """
    from bisect import bisect_right

    pinned = json.loads(path.expanduser().resolve().read_text())
    expected = {
        record["session"]: record["content_id"]
        for record in dataset.records
    }
    if pinned.get("content_ids") != expected:
        raise ValueError("Pinned evaluation source identities differ.")
    if set(pinned.get("indices", {})) != {"validation", "test"}:
        raise ValueError("Pinned evaluation needs validation and test lists.")
    for phase in ("validation", "test"):
        indices = pinned["indices"][phase]
        if not indices or len(indices) != len(set(indices)):
            raise ValueError(f"Pinned {phase} windows are empty or repeated.")
        allowed = set(dataset.partitions[phase])
        for index in indices:
            if not isinstance(index, int) or not 0 <= index < len(dataset):
                raise ValueError(f"Pinned {phase} index is out of range.")
            record = dataset.records[bisect_right(dataset.ends, index)]
            if record["session"] not in allowed:
                raise ValueError(f"Pinned {phase} window is outside split.")
    return pinned


def nested_training_groups(dataset: object, parent: dict) -> dict:
    """Validate a parent manifest and group its global training indices.

    Parameters
    ----------
    dataset : object
        Audited dataset with ordered records, cumulative ends and splits.
    parent : dict
        Frozen identity and train/validation/test global index lists.

    Returns
    -------
    dict
        Session positions mapped to unique local training indices.
    """
    from bisect import bisect_right

    indices = parent["indices"]["train"]
    budget = parent["identity"]["sampling"]["sample_budget"]
    if len(indices) != budget or len(set(indices)) != budget:
        raise ValueError("Nested training requires an exact unique budget.")
    groups = {}
    for index in indices:
        if type(index) is not int or not 0 <= index < dataset.ends[-1]:
            raise ValueError("Nested training index is outside eligibility.")
        position = bisect_right(dataset.ends, index)
        record = dataset.records[position]
        if dataset.group_id(record) not in dataset.partitions["train"]:
            raise ValueError("Nested training index is outside train split.")
        offset = dataset.ends[position - 1] if position else 0
        groups.setdefault(position, set()).add(index - offset)
    return groups


def extend_group(
    eligible: int, quota: int, retained: set[int], cached: set[int],
    rng: np.random.Generator,
) -> list[int]:
    """Retain parent windows and fill from cached, then fresh windows.

    Parameters
    ----------
    eligible : int
        Number of eligible local windows.
    quota : int
        Target unique window count, no smaller than the parent count.
    retained, cached : set[int]
        Parent membership and compatible cache membership, respectively.
    rng : numpy.random.Generator
        Seeded generator used for additional draws.

    Returns
    -------
    list[int]
        Unique local indices of length quota containing all parent windows.
    """
    if not len(retained) <= quota <= eligible:
        raise ValueError(
            f"Nested quota must be between {len(retained)} and "
            f"{eligible}, got {quota}."
        )
    remaining = np.setdiff1d(
        np.arange(eligible), sorted(retained), assume_unique=True
    )
    cached_remaining = cached - retained
    mapped = set(np.searchsorted(remaining, sorted(cached_remaining)))
    added = select_group(
        len(remaining), quota - len(retained), mapped, rng, False
    ) if quota > len(retained) else []
    return sorted(retained) + remaining[added].tolist()


def select_or_load(
    root: Path, dataset: object, settings: dict,
    parent: dict | None = None,
) -> dict:
    """Freeze cache-aware indices across models and cache growth.

    Parameters
    ----------
    root : pathlib.Path
        Local artifact root for shared selection manifests.
    dataset : object
        Dataset exposing subgroup records, partitions, and cache membership.
    settings : dict
        Scientific sampling budget, alpha, seed, and evaluation batch counts.
    parent : dict, optional
        Compatible smaller frozen selection to retain, default None.

    Returns
    -------
    dict
        Selection identity and global train, validation, and test indices.
    """
    settings = {**settings, "alpha": float(settings["alpha"])}
    identity = {
        "revision": dataset.selection_revision,
        "source": dataset.inventory,
        "partitions": dataset.partitions,
        "sampling": settings,
        "preprocessor": dataset.cache_settings(),
    }
    if parent is not None:
        previous = parent["identity"]
        expected = {**identity, "sampling": {
            **settings,
            "sample_budget": previous["sampling"]["sample_budget"],
        }}
        if previous != expected:
            raise ValueError("Nested parent source or sampling differs.")
        if settings["sample_budget"] <= previous["sampling"]["sample_budget"]:
            raise ValueError("Nested budget must exceed the parent budget.")
        nested_training_groups(dataset, parent)
    path = selection_path(root, dataset.selection_namespace, identity)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _select_or_load_unlocked(
            path, dataset, settings, identity, parent
        )


def _select_or_load_unlocked(
    path: Path, dataset: object, settings: dict, identity: dict,
    parent: dict | None = None,
) -> dict:
    """Read or write one selection while its manifest lock is held."""
    if path.exists():
        saved = json.loads(path.read_text())
        if saved["identity"] != identity:
            raise ValueError("Sampling selection manifest identity differs.")
        if parent is not None:
            nested_training_groups(dataset, saved)
            if not set(parent["indices"]["train"]).issubset(
                saved["indices"]["train"]
            ):
                raise ValueError("Existing selection omits parent windows.")
            for phase in ("validation", "test"):
                if saved["indices"][phase] != parent["indices"][phase]:
                    raise ValueError("Existing evaluation differs from parent.")
        return saved
    retained = (
        nested_training_groups(dataset, parent) if parent is not None else {}
    )
    rng = np.random.default_rng(settings["seed"])
    result = {}
    for phase in ("train", "validation", "test"):
        if parent is not None and phase != "train":
            result[phase] = list(parent["indices"][phase])
            continue
        records = [
            (position, record)
            for position, record in enumerate(dataset.records)
            if dataset.group_id(record) in dataset.partitions[phase]
        ]
        budget = (
            settings["sample_budget"]
            if phase == "train"
            else settings[f"{phase}_batches"] * settings["valid_batch_size"]
        )
        alpha = settings["alpha"] if phase == "train" else 0
        quotas = allocate(
            [record["eligible"] for _, record in records], budget, alpha
        )
        selected = []
        for (position, record), quota in zip(records, quotas):
            cached = dataset.cached_local_indices(record)
            if parent is not None:
                local = extend_group(
                    record["eligible"], quota, retained.get(position, set()),
                    cached, rng,
                )
            elif phase == "train":
                local = select_group(
                    record["eligible"], quota, cached, rng, True
                )
            elif hasattr(dataset, "select_evaluation_group"):
                local = dataset.select_evaluation_group(
                    record, quota, cached, rng
                )
            else:
                local = select_group(
                    record["eligible"], quota, cached, rng, False
                )
            if phase == "train" and quota > record["eligible"]:
                warnings.warn(
                    f"Group {dataset.group_id(record)} repeats examples "
                    "because its quota exceeds eligibility.",
                    stacklevel=2,
                )
            offset = dataset.ends[position - 1] if position else 0
            selected.extend(offset + index for index in local)
        if phase == "train":
            rng.shuffle(selected)
        result[phase] = selected
    saved = {"identity": identity, "indices": result}
    if parent is not None:
        saved["parent_sha256"] = digest(parent)
    write_json(path, saved)
    return saved
