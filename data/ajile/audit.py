"""Resumable full AJILE audit; invoke with --config pointing to local YAML.

Output sessions/<session>/blocks/*.npz stores uint8 reason bitmasks
(windows, electrodes), raw-content hashes and native block start. Session
summary.json contains exclusive/independent reasons, electrode/group/activity
counts and unique versus electrode-weighted durations. inventory.json and
splits.json are written only after all sessions finish. Logs hold status only.
"""

import argparse
from collections import Counter
import fcntl
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np
from omegaconf import OmegaConf
from util.experiments.storage import atomic_write, digest, write_json
from .core import BLOCK_WINDOWS, POLICY, RATE, REASONS, WINDOW
from .core import annotations, metadata


def save_block(path: Path, reasons: np.ndarray, raw_hash: str) -> None:
    """Commit an audited eligibility block without overwriting completed blocks."""

    def write(candidate: Path) -> None:
        with candidate.open("wb") as stream:
            np.savez_compressed(stream, reasons=reasons, sha256=raw_hash)

    def audit(candidate: Path) -> None:
        with np.load(candidate, allow_pickle=False) as saved:
            if not np.array_equal(saved["reasons"], reasons):
                raise ValueError("Eligibility block failed its write audit.")

    atomic_write(path, write, audit)


def summarize(meta: dict, reasons: np.ndarray, activities: list) -> dict:
    """Reduce eligibility to reconciled counts; activity memberships overlap."""
    valid = reasons == 0
    by_window = valid.sum(axis=1)
    combinations, memberships, groups = Counter(), Counter(), Counter()
    for count, tags in zip(by_window, activities):
        combinations[" | ".join(sorted(tags)) or "unlabeled"] += int(count)
        for tag in tags or {"unlabeled"}:
            memberships[tag] += int(count)
    for group, count in zip(meta["groups"], valid.sum(axis=0)):
        groups[group] += int(count)
    remaining = np.ones_like(valid)
    exclusive = {}
    for name, flag in REASONS.items():
        matched = (reasons & flag) != 0
        exclusive[name] = int((remaining & matched).sum())
        remaining &= ~matched
    return {
        "eligible": int(valid.sum()),
        "candidate": int(valid.size),
        "unique_eligible_seconds": int(np.any(valid, axis=1).sum()) * 5,
        "electrode_seconds": int(valid.sum()) * 5,
        "trailing_native_samples": meta["samples"] % WINDOW,
        "independent_rejections": {
            name: int(((reasons & flag) != 0).sum())
            for name, flag in REASONS.items()
        },
        "exclusive_rejections": exclusive,
        "by_electrode": dict(
            zip(map(str, meta["ids"]), map(int, valid.sum(axis=0)))
        ),
        "by_group": dict(groups),
        "by_activity_combination": dict(combinations),
        "by_activity_membership": dict(memberships),
    }


def progress_counts(paths: list[Path], completed: set[str]) -> dict:
    """Count finished and remaining recordings and fully finished subjects.

    Parameters
    ----------
    paths : list of pathlib.Path
        Full recording inventory; parent directory identifies each subject.
    completed : set of str
        Recording stems with durable session summaries.

    Returns
    -------
    dict
        Recording/subject totals and per-subject recording counts. Remaining
        counts include the active recording and its unfinished subject.
    """
    if not paths:
        raise ValueError("Progress requires a nonempty recording inventory.")
    totals = Counter(path.parent.name for path in paths)
    done = Counter(
        path.parent.name for path in paths if path.stem in completed
    )
    subjects_done = sum(done[name] == count for name, count in totals.items())
    recordings_done = sum(done.values())
    return {
        "corpus": {
            "recordings": {
                "total": len(paths),
                "completed": recordings_done,
                "remaining": len(paths) - recordings_done,
            },
            "subjects": {
                "total": len(totals),
                "completed": subjects_done,
                "remaining": len(totals) - subjects_done,
            },
        },
        "subjects": {
            name: {
                "recordings": {
                    "total": count,
                    "completed": done[name],
                    "remaining": count - done[name],
                }
            }
            for name, count in sorted(totals.items())
        },
    }


def recording_progress(
    path: Path, samples: int, scanned: int, eligible: int
) -> dict:
    """Describe one recording's scan in seconds and electrode-window examples.

    Parameters
    ----------
    path : pathlib.Path
        Recording source; parent directory identifies the subject.
    samples, scanned : int
        Total and processed native time points, converted to seconds here.
    eligible : int
        Accepted single-electrode windows in the processed portion.

    Returns
    -------
    dict
        Separate duration_seconds and training_examples groups.
    """
    if samples <= 0 or not 0 <= scanned <= samples or eligible < 0:
        raise ValueError("Invalid recording duration or progress counts.")
    return {
        "id": path.stem,
        "subject": path.parent.name,
        "duration_seconds": {
            "total": samples / RATE,
            "scanned": scanned / RATE,
            "remaining": (samples - scanned) / RATE,
        },
        "training_examples": {
            "eligible": eligible,
            "unit": "single_electrode_window",
            "window_duration_seconds": WINDOW / RATE,
        },
    }


def scan(
    path: Path,
    root: Path,
    output: Path,
    status: Path,
    progress: dict | None = None,
) -> dict:
    """Scan bounded blocks and resume durable outputs.

    Parameters
    ----------
    path, root, output, status : pathlib.Path
        Source recording, source root, artifact root and status destination.
    progress : dict, optional
        Corpus progress counters; default None omits corpus-wide fields.

    Returns
    -------
    dict
        Completed recording summary with eligibility and source identity.
    """
    meta = metadata(path)
    blocked, activities = annotations(meta)
    folder = output / "sessions" / path.stem
    fingerprint = {
        "source": str(path.relative_to(root)),
        "size": path.stat().st_size,
        "mtime_ns": path.stat().st_mtime_ns,
        "metadata": meta,
        "policy": POLICY,
    }
    manifest = folder / "source.json"
    if manifest.exists():
        if json.loads(manifest.read_text()) != fingerprint:
            raise ValueError(f"Source changed since audit: {path.resolve()}")
    else:
        write_json(manifest, fingerprint)
    completed = folder / "summary.json"
    if completed.exists():
        return json.loads(completed.read_text())
    blocks, hashes = [], []
    with h5py.File(path, "r") as f:
        source = f["acquisition/ElectricalSeries/data"]
        for first in range(0, meta["samples"], BLOCK_WINDOWS * WINDOW):
            block_path = folder / "blocks" / f"{first:012d}.npz"
            if not block_path.exists():
                raw = source[first : first + BLOCK_WINDOWS * WINDOW, :]
                raw_hash = hashlib.sha256(raw.tobytes()).hexdigest()
                n = len(raw) // WINDOW
                windows = raw[: n * WINDOW].reshape(
                    n, WINDOW, meta["channels"]
                )
                reasons = np.zeros((n, meta["channels"]), dtype=np.uint8)
                reasons[:, ~np.asarray(meta["good"])] |= REASONS[
                    "bad_electrode"
                ]
                offset = first // WINDOW
                reasons[blocked[offset : offset + n], :] |= REASONS[
                    "blocklist"
                ]
                reasons[~np.isfinite(windows).all(axis=1)] |= REASONS[
                    "nonfinite"
                ]
                reasons[(windows == windows[:, :1, :]).all(axis=1)] |= REASONS[
                    "constant"
                ]
                save_block(block_path, reasons, raw_hash)
            with np.load(block_path, allow_pickle=False) as saved:
                blocks.append(saved["reasons"])
                hashes.append(str(saved["sha256"]))
            write_json(
                status,
                {
                    **(progress if progress is not None else {}),
                    "phase": "scanning",
                    "current_recording": recording_progress(
                        path,
                        meta["samples"],
                        min(first + BLOCK_WINDOWS * WINDOW, meta["samples"]),
                        sum(int((b == 0).sum()) for b in blocks),
                    ),
                },
            )
    if path.stat().st_mtime_ns != fingerprint["mtime_ns"]:
        raise ValueError("Source changed during scan.")
    reasons = np.concatenate(blocks)
    summary = summarize(meta, reasons, activities)
    record = {
        "session": path.stem,
        "subject": path.parent.name,
        "source": str(path.relative_to(root)),
        "metadata": meta,
        "content_id": digest({"metadata": meta, "blocks": hashes}),
        **summary,
    }
    write_json(folder / "summary.json", record)
    return record


def assign(records: list) -> dict:
    """Optimize absolute 98/1/1 count deviation with session disjointness."""
    from scipy.optimize import Bounds, LinearConstraint, milp

    n = len(records)
    weights = np.array([r["eligible"] for r in records], dtype=float)
    total = weights.sum()
    if total <= 0:
        raise ValueError("No eligible AJILE examples.")
    weights /= total
    rows, lower, upper = [], [], []
    for i in range(n):
        row = np.zeros(2 * n + 3)
        row[i] = row[n + i] = 1
        rows.append(row)
        lower.append(0)
        upper.append(1)
    for subject in sorted({r["subject"] for r in records}):
        indices = [
            i
            for i, r in enumerate(records)
            if r["subject"] == subject and r["eligible"] > 0
        ]
        if not indices:
            raise ValueError(f"No eligible sessions for {subject}.")
        row = np.zeros(2 * n + 3)
        row[indices] = 1
        row[np.array(indices) + n] = 1
        rows.append(row)
        lower.append(0)
        upper.append(len(indices) - 1)
    for partition in range(2):
        row = np.zeros(2 * n + 3)
        row[partition * n : (partition + 1) * n] = weights
        rows.append(row)
        lower.append(1 / total)
        upper.append(np.inf)
    for k, (a, target) in enumerate(
        [
            (np.r_[weights, np.zeros(n)], 0.01),
            (np.r_[np.zeros(n), weights], 0.01),
            (np.r_[weights, weights], 0.02),
        ]
    ):
        for sign in (1, -1):
            row = np.r_[sign * a, np.zeros(3)]
            row[2 * n + k] = -1
            rows.append(row)
            lower.append(-np.inf)
            upper.append(sign * target)
    objective = np.r_[np.zeros(2 * n), np.ones(3)]
    bounds = Bounds(np.zeros(2 * n + 3), np.r_[np.ones(2 * n), [np.inf] * 3])
    constraint = LinearConstraint(np.array(rows), lower, upper)
    result = milp(
        objective,
        integrality=np.r_[np.ones(2 * n), np.zeros(3)],
        bounds=bounds,
        constraints=constraint,
        options={"mip_rel_gap": 0},
    )
    if not result.success:
        raise RuntimeError(f"Session assignment failed: {result.message}")
    # Stable secondary ordering within the optimal scientific objective.
    rows.append(objective)
    lower.append(-np.inf)
    upper.append(result.fun + 1e-10)
    tie = np.r_[np.arange(1, 2 * n + 1), np.zeros(3)]
    result = milp(
        tie,
        integrality=np.r_[np.ones(2 * n), np.zeros(3)],
        bounds=bounds,
        constraints=LinearConstraint(np.array(rows), lower, upper),
        options={"mip_rel_gap": 0},
    )
    if not result.success:
        raise RuntimeError("Deterministic session tie-break failed.")
    partitions = {"train": [], "validation": [], "test": []}
    for i, record in enumerate(records):
        name = (
            "validation"
            if result.x[i] > 0.5
            else "test" if result.x[n + i] > 0.5 else "train"
        )
        partitions[name].append(record["session"])
    return partitions


def main() -> None:
    """Run the locally configured full audit with one kernel-held writer lock."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    root, output, logs = map(
        lambda x: Path(x).resolve(), (cfg.source, cfg.output, cfg.logs)
    )
    output.mkdir(parents=True, exist_ok=True)
    status = logs / "status" / "status.json"
    with (output / "audit.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            paths = sorted(root.glob("sub-*/*.nwb"))
            if not paths:
                raise ValueError("No AJILE recordings found.")
            completed = {
                path.stem
                for path in paths
                if (output / "sessions" / path.stem / "summary.json").exists()
            }
            records = []
            for path in paths:
                progress = progress_counts(paths, completed)
                records.append(scan(path, root, output, status, progress))
                completed.add(path.stem)
            progress = progress_counts(paths, completed)
            write_json(status, {**progress, "phase": "assigning_splits"})
            partitions = assign(records)
            inventory = {
                "policy": POLICY,
                "sessions": records,
                "total_eligible": sum(r["eligible"] for r in records),
                "by_subject": {
                    subject: sum(
                        r["eligible"]
                        for r in records
                        if r["subject"] == subject
                    )
                    for subject in sorted({r["subject"] for r in records})
                },
            }
            for field in (
                "by_group",
                "by_activity_combination",
                "by_activity_membership",
                "independent_rejections",
                "exclusive_rejections",
            ):
                totals = Counter()
                for record in records:
                    totals.update(record[field])
                inventory[field] = dict(totals)
            inventory["unique_eligible_seconds"] = sum(
                r["unique_eligible_seconds"] for r in records
            )
            inventory["electrode_seconds"] = sum(
                r["electrode_seconds"] for r in records
            )
            write_json(output / "inventory.json", inventory)
            write_json(output / "splits.json", partitions)
            test = [r for r in records if r["session"] in partitions["test"]]
            write_json(
                output / "pinned_test.json",
                {
                    "sessions": partitions["test"],
                    "content_ids": {
                        r["session"]: r["content_id"] for r in test
                    },
                },
            )
            write_json(
                status,
                {
                    **progress,
                    "phase": "complete",
                    "corpus": {
                        **progress["corpus"],
                        "training_examples": {
                            "eligible": inventory["total_eligible"],
                            "unit": "single_electrode_window",
                            "window_duration_seconds": WINDOW / RATE,
                            "by_split": {
                                name: sum(
                                    r["eligible"]
                                    for r in records
                                    if r["session"] in sessions
                                )
                                for name, sessions in partitions.items()
                            },
                        },
                    },
                },
            )
        except BaseException as error:
            write_json(
                logs / "errors" / "failure.json",
                {"type": type(error).__name__, "message": str(error)},
            )
            previous = (
                json.loads(status.read_text()) if status.exists() else {}
            )
            write_json(
                status, {**previous, "phase": "failed", "error": str(error)}
            )
            raise


if __name__ == "__main__":
    main()
