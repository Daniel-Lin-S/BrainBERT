"""Synthetic NWB audit/resume and session-split regression checks."""

from pathlib import Path
import h5py
import numpy as np
from data.ajile.audit import scan, assign


def test_audit_resume(tmp_path: Path) -> None:
    """Verify independent bad-channel and blocklist exclusions and tails."""
    path = tmp_path / "sub-01" / "session.nwb"
    path.parent.mkdir()
    with h5py.File(path, "w") as f:
        s = f.create_group("acquisition/ElectricalSeries")
        s.create_dataset(
            "data", data=np.random.default_rng(1).normal(size=(7501, 2))
        ).attrs["conversion"] = 1e-6
        s.create_dataset("electrodes", data=[1, 0])
        s.create_dataset("starting_time", data=0.0).attrs["rate"] = 500.0
        e = f.create_group("general/extracellular_ephys/electrodes")
        e.create_dataset("id", data=[11, 22])
        e.create_dataset("good", data=[False, True])
        e.create_dataset(
            "group_name", data=["GRID", "GRID"], dtype=h5py.string_dtype()
        )
        ep = f.create_group("intervals/epochs")
        ep.create_dataset("start_time", data=[5.0])
        ep.create_dataset("stop_time", data=[10.0])
        ep.create_dataset(
            "labels",
            data=["Blocklist (Data break)"],
            dtype=h5py.string_dtype(),
        )
    report = scan(path, tmp_path, tmp_path / "audit", tmp_path / "status.json")
    assert report["eligible"] == 2
    assert report["metadata"]["ids"] == [22, 11]
    assert report["trailing_native_samples"] == 1
    assert sum(report["exclusive_rejections"].values()) + 2 == 6
    assert (
        scan(path, tmp_path, tmp_path / "audit", tmp_path / "status.json")
        == report
    )


def test_session_assignment() -> None:
    """Keep sessions disjoint and preserve per-subject training coverage."""
    records = [
        {"session": str(i), "subject": str(i // 3), "eligible": count}
        for i, count in enumerate([980, 10, 10, 980, 10, 10])
    ]
    first = assign(records)
    assert first == assign(records)
    assert len(set(sum(first.values(), []))) == 6
    assert first["train"] == ["0", "3"]


def test_progress_uses_seconds_and_separate_levels() -> None:
    """Keep recording time separate from corpus counts and electrode windows."""
    from data.ajile.audit import progress_counts, recording_progress

    paths = [Path('/subject/a.nwb'), Path('/subject/b.nwb')]
    progress = progress_counts(paths, {'a'})
    assert progress['corpus']['recordings']['remaining'] == 1
    assert progress['subjects']['subject']['recordings']['completed'] == 1
    recording = recording_progress(paths[1], 5001, 2500, 80)
    assert recording['duration_seconds'] == {
        'total': 10.002, 'scanned': 5.0, 'remaining': 5.002,
    }
    assert recording['training_examples']['eligible'] == 80
    assert recording['training_examples']['window_duration_seconds'] == 5
