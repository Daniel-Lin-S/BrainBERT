"""Single-electrode AJILE examples from a completed quality audit.

The audit provides source identities and compact eligibility bitmasks.
Each item is one electrode's waveform and the existing BrainBERT masked STFT
batch dictionary. No model input contains a channel axis. Source placement
and audit paths are local settings, not scientific identity.
"""

from bisect import bisect_right
import json
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset
from datasets import register_dataset
from datasets.masked_tf_dataset import prepare_example
from preprocessors import STFTPreprocessor
from data.ajile.core import POLICY, waveform


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
    """Load one eligible electrode/window without materializing waveform files."""

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

    def get_input_dim(self) -> int:
        return self.extracter.cfg.freq_channel_cutoff

    def verify_test(self, path: Path) -> None:
        """Verify pinned sessions and their complete source-content identities."""
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
            blocks = []
            for path in sorted(
                (self.audit / "sessions" / record["session"] / "blocks").glob(
                    "*.npz"
                )
            ):
                with np.load(path, allow_pickle=False) as saved:
                    blocks.append(saved["reasons"])
            self.valid = np.concatenate(blocks) == 0
            self.positions = np.flatnonzero(self.valid)
            if len(self.positions) != record["eligible"]:
                raise ValueError("AJILE eligibility index count differs.")
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
        return prepare_example(wav, self.extracter, self.task_cfg)
