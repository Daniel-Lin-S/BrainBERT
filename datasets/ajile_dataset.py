"""Single-electrode AJILE examples from a completed quality audit.

The audit provides source identities and compact eligibility bitmasks.
Each item is one electrode's waveform and the existing BrainBERT masked STFT
batch dictionary. No model input contains a channel axis. Source placement
and audit paths are local settings, not scientific identity.
"""

import json
import random
from bisect import bisect_right
from pathlib import Path

import numpy as np
import torch
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
from preprocessors.cache import (
    cached_raw_positions, cached_raw_stft
)
from util.experiments.storage import digest


RAW_STFT_NAMESPACE = "ajile_raw_stft"
SELECTION_NAMESPACE = "ajile12"
SELECTION_REVISION = "ajile-budgeted-v2"


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

    selection_namespace = SELECTION_NAMESPACE
    selection_revision = SELECTION_REVISION
    raw_stft_namespace = RAW_STFT_NAMESPACE

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
        self.selection = None
        self.selected_train = False
        self.selected_statistics = {}
        self.eval_indices = set()
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


    def group_id(self, record: dict) -> str:
        """Return the audited session identity of one record."""
        return record["session"]

    def cache_settings(self) -> dict:
        """Return spectral settings without local cache placement."""
        settings = dict(self.extracter.cfg)
        settings.pop("feature_cache", None)
        return settings

    def cached_local_indices(self, record: dict) -> set[int]:
        """Find eligible examples with raw STFT cache entries."""
        root = Path(self.extracter.cfg.feature_cache).resolve()
        cached_positions = cached_raw_positions(
            root, RAW_STFT_NAMESPACE, record["content_id"],
            self.extracter.cfg,
        )
        if not cached_positions:
            return set()
        valid = eligibility(self.audit, record)
        lookup = {
            int(position): index
            for index, position in enumerate(np.flatnonzero(valid))
        }
        invalid = cached_positions - lookup.keys()
        if invalid:
            raise ValueError(
                f"AJILE cache has {len(invalid)} ineligible positions "
                f"for session {record['session']}."
            )
        return {lookup[position] for position in cached_positions}

    def select_evaluation_group(
        self, record: dict, quota: int, cached: set[int], rng: object
    ) -> list[int]:
        """Balance selected eligible examples across electrodes."""
        from data.ajile.sampling import select_evaluation_group

        return select_evaluation_group(
            self, record, quota, cached, rng
        )

    def validate_budgeted_preprocessing(self) -> None:
        """Require selected recording moments for AJILE budgeted fits."""
        if self.extracter.cfg.normalizing != "recording_zscore":
            raise ValueError(
                "AJILE budgeted fitting requires recording_zscore."
            )

    def prepare_selected(
        self, selection: dict, progress_callback=None
    ) -> dict:
        """Prepare selected raw spectra and AJILE recording moments."""
        from data.ajile.sampling import prepare_selected

        return prepare_selected(
            self, selection, progress_callback=progress_callback
        )

    def configure_selection(self, selection: dict) -> None:
        """Attach frozen indices and evaluation mask membership."""
        self.selection = selection["indices"]
        self.selected_train = bool(self.selection["train"])
        self.eval_indices = set(
            self.selection["validation"] + self.selection["test"]
        )

    def raw_spectrum(
        self, index: int, wav: np.ndarray = None
    ) -> torch.Tensor:
        """Load or extract one raw STFT, shape (frames, frequencies)."""
        session = bisect_right(self.ends, index)
        record = self.records[session]
        valid = (self.valid if self.active == session
                 else eligibility(self.audit, record))
        positions = (self.positions if self.active == session
                     else np.flatnonzero(valid))
        local = index - (self.ends[session - 1] if session else 0)
        position = int(positions[local])
        window, channel = divmod(
            position, record["metadata"]["channels"]
        )

        def extract() -> torch.Tensor:
            source = Path(self.root_dir) / record["source"]
            signal = wav if wav is not None else waveform(
                source, record["metadata"], window, channel, valid
            )
            return self.extracter.unnormalized(signal[:-1])

        return cached_raw_stft(
            Path(self.extracter.cfg.feature_cache).resolve(),
            RAW_STFT_NAMESPACE, record["content_id"],
            position, self.extracter.cfg, extract,
        )

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
            full_train = (
                not self.selected_train
                and record["session"] in self.partitions["train"]
            )
            if (
                self.extracter.cfg.normalizing == "recording_zscore"
                and (self.selection is None or full_train)
            ):
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
        if self.selection is not None and (
            self.selected_train or index in self.eval_indices
        ):
            raw = self.raw_spectrum(index, wav)
            moments = self.selected_statistics.get(record["session"])
            if moments is None:
                raise ValueError(
                    f"Missing selected AJILE moments: {record['session']}"
                )
            values = (
                raw.double().numpy() - moments["mean"][channel]
            ) / moments["std"][channel]
            if not np.isfinite(values).all():
                raise ValueError("Selected AJILE STFT is nonfinite.")
            feature = torch.from_numpy(values.astype(np.float32))
            if index not in self.eval_indices:
                return prepare_example(
                    wav, self.extracter, self.task_cfg, features=feature
                )
            state = random.getstate()
            random.seed(digest({
                "index": index,
                "source": record["content_id"],
            }))
            try:
                return prepare_example(
                    wav, self.extracter, self.task_cfg, features=feature
                )
            finally:
                random.setstate(state)
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
