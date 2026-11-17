"""Compute masked L1 and content-aware reconstruction losses.

Batches contain spectra ``masked_input`` and ``target`` and boolean
``mask_label`` arrays of shape (batch, frames, frequencies), an ``attn_mask``
of shape (batch, frames), and a list of raw ``wavs``. Returns a scalar tensor
loss and a dictionary of scalar metrics and first-example diagnostic images.
Empty masked or content-aware target sets raise ValueError.
"""

import torch

from criterions import register_criterion
from util.experiments.monitoring import batch_statistics

from .base_criterion import BaseCriterion

CONTENT_THRESHOLD = 1.0



def spectral_content_loss(
    errors: torch.Tensor, content: torch.Tensor,
    frequencies: torch.Tensor, strength: float, exponent: float,
) -> torch.Tensor:
    """Return a normalised weighted mean over selected content cells.

    Parameters
    ----------
    errors : torch.Tensor
        Absolute reconstruction errors, shape (B, T, F).
    content : torch.Tensor
        Selected content-cell mask, shape (B, T, F).
    frequencies : torch.Tensor
        Physical frequency centres, shape (F,).
    strength, exponent : float
        Nonnegative spectral emphasis and positive power.
    """
    import math

    if (errors.shape != content.shape or errors.ndim != 3
            or frequencies.shape != errors.shape[-1:]):
        raise ValueError("Expected errors/mask (B,T,F) and frequencies (F,).")
    if (not math.isfinite(strength) or strength < 0
            or not math.isfinite(exponent) or exponent <= 0):
        raise ValueError("Spectral strength must be >= 0 and power > 0.")
    if (not torch.isfinite(frequencies).all() or frequencies.max() <= 0
            or (frequencies < 0).any()):
        raise ValueError("Spectral centres must be finite, nonnegative Hz.")
    if not content.any() or not torch.isfinite(errors).all():
        raise ValueError("Content loss needs finite errors and content cells.")
    weights = 1 + strength * (frequencies / frequencies.max()).pow(exponent)
    weights = weights.expand_as(errors).masked_select(content)
    return (errors.masked_select(content) * weights).sum() / weights.sum()


@register_criterion("pretrain_masked_criterion")
class PretrainMaskedCriterion(BaseCriterion):
    def __init__(self):
        super(PretrainMaskedCriterion, self).__init__()
        pass

    def build_criterion(self, cfg):
        self.cfg = cfg

    def forward(
        self, model, batch, device, embedding_rows=None,
        collect_frequency=False,
    ):
        """Compute the unchanged masked-cell objective and diagnostics.

        Parameters
        ----------
        model : torch.nn.Module
            Model predicting spectra of shape (B, T, F).
        batch : dict
            Masked inputs, targets and masks of shape (B, T, F).
        device : str
            Training or evaluation device.
        embedding_rows : sequence of int, optional
            Batch rows to pool into (N, D) final embeddings, default None.
        collect_frequency : bool, optional
            Collect validation frequency sums, default False.

        Returns
        -------
        tuple
            Original scalar loss and detached diagnostic values.
        """
        # x = batch[:,:10] #potentially don't move to device if dataparallel
        pad_mask = batch["attn_mask"].to(device)
        masked_input = batch["masked_input"].to(
            device
        )  # potentially don't move to device if dataparallel
        mask = batch["mask_label"].bool().to(device)
        result = model.forward(
            masked_input, pad_mask, return_hidden=True
        ) if embedding_rows else model.forward(masked_input, pad_mask)
        if embedding_rows:
            output, pos_enc, hidden = result
        else:
            output, pos_enc = result
        labels = batch["target"].to(device)
        true_activity = labels.masked_select(mask)
        predicted = output.masked_select(mask)
        if true_activity.numel() == 0:
            raise ValueError(
                "Pre-training requires at least one masked target."
            )
        l1 = torch.mean(torch.abs(true_activity - predicted))
        non_zero_idxs = torch.abs(true_activity) > CONTENT_THRESHOLD
        if not non_zero_idxs.any():
            raise ValueError(
                "Content-aware loss requires a masked target with absolute "
                f"value greater than {CONTENT_THRESHOLD}."
            )
        non_zero = torch.mean(
            torch.abs(true_activity[non_zero_idxs] - predicted[non_zero_idxs])
        )
        comparison = {}
        objective_content = non_zero
        spectral = self.cfg.get("content_spectral_weighting")
        if spectral is not None:
            if not hasattr(self, "frequency_hz"):
                raise ValueError("Spectral loss requires physical frequencies.")
            weighted = spectral_content_loss(
                (output - labels).abs(),
                mask & (labels.abs() > CONTENT_THRESHOLD),
                torch.as_tensor(self.frequency_hz, device=labels.device,
                                dtype=labels.dtype),
                spectral.strength, spectral.exponent,
            )
            comparison = {
                "spectral_content_l1": weighted.item(),
                "unweighted_total_loss": (
                    l1 + self.cfg.alpha * non_zero
                ).item(),
                "spectral_total_loss": (l1 + self.cfg.alpha * weighted).item(),
            }
            if spectral.apply:
                objective_content = weighted
        content_aware_loss = self.cfg.alpha * objective_content
        loss = l1 + content_aware_loss
        if not torch.isfinite(loss):
            raise ValueError("Pre-training loss must be finite.")
        output_log_spec = output[0].detach().cpu()
        content_l1 = non_zero
        wav = batch["wavs"][0]
        images = {
            "input_spectrogram": masked_input[0].detach().cpu(),
            "ground_truth": labels[0].detach().cpu(),
            "pred_spectrogram": output_log_spec,
            "pos_enc": pos_enc[0].detach().cpu(),
        }
        if wav is not None:
            images["wav"] = wav
        logging_output = {
            **comparison,
            "loss": loss.item(),
            "images": images,
            "l1_loss": l1.item(),
            "content_l1": content_l1.item(),
            "zero_baseline_l1": true_activity.abs().mean().item(),
            "content_zero_baseline_l1": (
                true_activity[non_zero_idxs].abs().mean().item()
            ),
            "content_aware_loss": content_aware_loss.item(),
        }
        if embedding_rows:
            sampled_hidden = hidden[embedding_rows].detach()
            sampled_padding = pad_mask[embedding_rows]
            valid = (~sampled_padding).unsqueeze(-1)
            lengths = valid.sum(dim=1)
            if (lengths <= 0).any():
                raise ValueError("Embedding pooling needs valid frames.")
            logging_output["window_embeddings"] = (
                (sampled_hidden * valid).sum(dim=1) / lengths
            )
        if "mask_provenance" in batch:
            logging_output["monitor_statistics"] = batch_statistics(
                output,
                labels,
                mask,
                pad_mask,
                batch["mask_provenance"],
                CONTENT_THRESHOLD,
                include_frequency=collect_frequency,
            )
        return loss, logging_output
