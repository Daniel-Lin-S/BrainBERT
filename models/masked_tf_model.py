import json

import numpy as np
import torch
import torch.nn as nn

from models import register_model
from models.base_model import BaseModel
from models.spec_prediction_head import SpecPredictionHead
from models.transformer_encoder_input import TransformerEncoderInput


@register_model("masked_tf_model")
class MaskedTFModel(BaseModel):
    def __init__(self):
        super(MaskedTFModel, self).__init__()

    def forward(
        self,
        input_specs,
        src_key_mask,
        intermediate_rep=False,
        rep_from_layer=-1,
    ):
        input_specs, pos_enc = self.input_encoding(input_specs)
        input_specs = input_specs.transpose(
            0, 1
        )  # nn.Transformer wants [seq, batch, dim]
        if rep_from_layer == -1:
            output_specs = self.transformer(
                input_specs, src_key_padding_mask=src_key_mask
            )
        else:
            raise NotImplementedError
        output_specs = output_specs.transpose(0, 1)  # [batch, seq, dim]
        if intermediate_rep:
            return output_specs
        output_specs = self.spec_prediction_head(output_specs)
        return output_specs, pos_enc

    def set_stft_statistics(self, statistics: dict) -> None:
        """Persist training-fitted electrode/frequency moments in state_dict.

        Parameters
        ----------
        statistics : dict
            Sorted subject/electrode keys, float64 mean/std arrays of shape
            (electrodes, frequencies), and frame counts (electrodes,).
        """
        keys = [str(key) for key in statistics["keys"]]
        mean = np.asarray(statistics["mean"], dtype=np.float64)
        std = np.asarray(statistics["std"], dtype=np.float64)
        count = np.asarray(statistics["count"], dtype=np.int64)
        if (
            not keys
            or len(keys) != len(set(keys))
            or mean.shape != (len(keys), self.cfg.input_dim)
            or std.shape != mean.shape
            or count.shape != (len(keys),)
            or not np.isfinite(mean).all()
            or not np.isfinite(std).all()
            or np.any(std <= 0)
            or np.any(count <= 0)
        ):
            raise ValueError(
                "Expected finite, positive training STFT moments for "
                "every electrode and frequency."
            )
        labels = json.dumps(keys, separators=(",", ":")).encode("utf-8")
        fields = {
            "stft_mean": torch.from_numpy(mean.copy()),
            "stft_std": torch.from_numpy(std.copy()),
            "stft_count": torch.from_numpy(count.copy()),
            "stft_keys_utf8": torch.tensor(list(labels), dtype=torch.uint8),
        }
        for name, value in fields.items():
            if name in self._buffers:
                raise ValueError("STFT statistics are already attached.")
            self.register_buffer(name, value, persistent=True)

    def normalize_stft(
        self, spectrum: torch.Tensor, subject: str, electrode_id: int
    ) -> torch.Tensor:
        """Apply checkpoint STFT moments to raw (frames, frequencies) data."""
        if "stft_keys_utf8" not in self._buffers:
            raise ValueError("Model has no fitted STFT statistics.")
        labels = json.loads(
            bytes(self.stft_keys_utf8.cpu().tolist()).decode("utf-8")
        )
        key = json.dumps([subject, electrode_id], separators=(",", ":"))
        if key not in labels:
            raise ValueError(f"Electrode has no training statistics: {key}")
        row = labels.index(key)
        if spectrum.ndim != 2 or spectrum.shape[1] != self.cfg.input_dim:
            raise ValueError(
                "Expected raw STFT shape (frames, input_dim), got "
                f"{tuple(spectrum.shape)}."
            )
        result = (
            (spectrum.double() - self.stft_mean[row].to(spectrum.device))
            / self.stft_std[row].to(spectrum.device)
        ).to(spectrum.dtype)
        if not torch.isfinite(result).all():
            raise ValueError("Normalized STFT must be finite.")
        return result

    def load_weights(self, states: dict) -> None:
        """Restore persisted STFT moments with the model parameters."""
        names = ("stft_mean", "stft_std", "stft_count", "stft_keys_utf8")
        present = [name in states for name in names]
        if any(present) and not all(present):
            raise ValueError("Checkpoint has incomplete STFT statistics.")
        for name in names:
            if name in states and name in self._buffers:
                if not torch.equal(
                    states[name].cpu(), self._buffers[name].cpu()
                ):
                    raise ValueError(
                        f"Checkpoint STFT statistics differ: {name}"
                    )
            elif name in states:
                self.register_buffer(
                    name, torch.empty_like(states[name]), persistent=True
                )
            elif name in self._buffers:
                raise ValueError(
                    "Checkpoint lacks required dataset STFT statistics."
                )
        super().load_weights(states)

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            if module.bias is not None:
                module.bias.data.zero_()
        if isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def build_model(self, cfg):
        self.cfg = cfg
        hidden_dim = self.cfg.hidden_dim
        self.input_encoding = TransformerEncoderInput(cfg)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=self.cfg.nhead,
            dim_feedforward=self.cfg.layer_dim_feedforward,
            activation=self.cfg.layer_activation,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=self.cfg.encoder_num_layers
        )
        self.spec_prediction_head = SpecPredictionHead(cfg)
        self.apply(self.init_weights)
