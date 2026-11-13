"""Single-electrode AJILE examples from a completed quality audit.

The audit provides source identities and compact eligibility bitmasks.
Each item is one electrode's waveform and the existing BrainBERT masked STFT
batch dictionary. No model input contains a channel axis. Source placement
and audit paths are local settings, not scientific identity.
"""

import json
from bisect import bisect_right
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from data.ajile.core import POLICY, waveform
from data.ajile.normalization import (
    dataset_statistics,
    eligibility,
    recording_statistics,
)
from datasets import register_dataset
from datasets.masked_tf_dataset import prepare_example
from preprocessors import STFTPreprocessor


class SessionIndices:
    """Compact sequence of dataset indices for whole-session partitions."""

    def __init__(self, spans: list) -> None:
        self.spans = spans
        self.ends = np.cumsum([end - start for start, end in spans]).tolist()

    def __len__(self) -> int:
        return self.ends[-1] if self.ends else 0

    def __getitem__(self, index: int) -> int:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        part = bisect_right(self.ends, index)
        previous = self.ends[part - 1] if part else 0
        return self.spans[part][0] + index - previous


@register_dataset(name="ajile12")
class AJILE12(Dataset):
    """Load an eligible electrode window without materializing waveforms."""

    def __init__(
        self,
        cfg: object,
        task_cfg: object = None,
        preprocessor_cfg: object = None,
    ) -> None:
        self.root_dir = str(Path(cfg.data).resolve())
        self.audit = Path(cfg.audit).resolve()
        inventory = json.loads((self.audit / "inventory.json").read_text())
        if inventory["policy"] != POLICY:
            raise ValueError("AJILE audit preprocessing revision differs.")
        self.records = inventory["sessions"]
        self.partitions = json.loads((self.audit / "splits.json").read_text())
        self.inventory = [
            {
                "session": r["session"],
                "content_id": r["content_id"],
                "policy": POLICY,
            }
            for r in self.records
        ]
        self.ends = np.cumsum([r["eligible"] for r in self.records]).tolist()
        self.task_cfg = task_cfg
        self.extracter = STFTPreprocessor(preprocessor_cfg)
        self.active = None
        self.valid = None
        self.positions = None
        self.statistics = None
        self.dataset_statistics = None
        self.key_to_index = None
        self.session_partition = {
            session: name
            for name, sessions in self.partitions.items()
            for session in sessions
        }
        assigned = sum(self.partitions.values(), [])
        if len(assigned) != len(set(assigned)) or set(assigned) != {
            r["session"] for r in self.records
        }:
            raise ValueError(
                "AJILE session partitions overlap or are incomplete."
            )
        for record in self.records:
            path = Path(self.root_dir) / record["source"]
            saved = json.loads(
                (
                    self.audit / "sessions" / record["session"] / "source.json"
                ).read_text()
            )
            if (
                path.stat().st_size != saved["size"]
                or path.stat().st_mtime_ns != saved["mtime_ns"]
            ):
                raise ValueError(f"Audited source changed: {path.resolve()}")

    def __len__(self) -> int:
        return self.ends[-1]

    def partition_indices(self, name: str) -> SessionIndices:
        """Return a compact index sequence, keeping all electrodes together."""
        selected = set(self.partitions[name])
        spans = [
            (self.ends[i - 1] if i else 0, self.ends[i])
            for i, r in enumerate(self.records)
            if r["session"] in selected
        ]
        result = SessionIndices(spans)
        if not len(result):
            raise ValueError(
                f"AJILE {name} partition has no eligible examples."
            )
        return result

    def prepare_statistics(self) -> dict:
        """Fit electrode/frequency moments on training recordings only."""
        if self.extracter.cfg.normalizing != "dataset_zscore":
            raise ValueError("Expected dataset_zscore for shared statistics.")
        if self.dataset_statistics is None:
            training = set(self.partitions["train"])
            records = [r for r in self.records if r["session"] in training]
            self.dataset_statistics = dataset_statistics(
                Path(self.root_dir), self.audit, records, self.extracter
            )
            self.key_to_index = {
                key: index
                for index, key in enumerate(self.dataset_statistics["keys"])
            }
        return self.dataset_statistics

    def get_input_dim(self) -> int:
        return self.extracter.cfg.freq_channel_cutoff

    def verify_test(self, path: Path) -> None:
        """Verify pinned sessions and their source identities."""
        expected = {
            "sessions": self.partitions["test"],
            "content_ids": {
                r["session"]: r["content_id"]
                for r in self.records
                if r["session"] in self.partitions["test"]
            },
        }
        if json.loads(path.read_text()) != expected:
            raise ValueError("AJILE pinned test membership/content differs.")

    def __getitem__(self, index: int) -> dict:
        if not 0 <= index < len(self):
            raise IndexError(index)
        session = bisect_right(self.ends, index)
        record = self.records[session]
        if self.active != session:
            self.valid = eligibility(self.audit, record)
            self.positions = np.flatnonzero(self.valid)
            if len(self.positions) != record["eligible"]:
                raise ValueError("AJILE eligibility index count differs.")
            self.statistics = None
            if self.extracter.cfg.normalizing == "recording_zscore":
                self.statistics = recording_statistics(
                    Path(self.root_dir),
                    record,
                    self.session_partition[record["session"]],
                    self.valid,
                    self.extracter,
                )
            self.active = session
        local = index - (self.ends[session - 1] if session else 0)
        window, channel = divmod(
            int(self.positions[local]), record["metadata"]["channels"]
        )
        wav = waveform(
            Path(self.root_dir) / record["source"],
            record["metadata"],
            window,
            channel,
            self.valid,
        )
        statistics = None
        if self.extracter.cfg.normalizing == "dataset_zscore":
            shared = self.prepare_statistics()
            label = json.dumps(
                [record["subject"], record["metadata"]["ids"][channel]],
                separators=(",", ":"),
            )
            if label not in self.key_to_index:
                raise ValueError(
                    "Evaluation electrode has no training STFT statistics: "
                    f"{label}"
                )
            row = self.key_to_index[label]
            statistics = {
                "mean": shared["mean"][row],
                "std": shared["std"][row],
                "identity": f"{shared['identity']}:electrode:{label}",
            }
        elif self.statistics is not None:
            statistics = {
                "mean": self.statistics["mean"][channel],
                "std": self.statistics["std"][channel],
                "identity": (
                    f"{self.statistics['identity']}:electrode:{channel}"
                ),
            }
        return prepare_example(
            wav, self.extracter, self.task_cfg, statistics=statistics
        )
