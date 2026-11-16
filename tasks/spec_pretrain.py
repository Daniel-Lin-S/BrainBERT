"""Build masked-STFT pre-training batches and validation summaries.

Input batches contain one-electrode spectrograms and mask provenance.
Validation returns original scalar losses and compact scientific statistics;
optional window embeddings support checkpoint-level rank monitoring.
"""
import numpy as np
import torch

import models
from tasks import register_task
from tasks.base_task import BaseTask
from tasks.batch_utils import spec_collator
from util.experiments.monitoring import Accumulator
from util.tensorboard_utils import (
    plot_tensorboard_line,
    plot_tensorboard_spectrogram,
)


@register_task(name="spec_pretrain")
class SpecPretrain(BaseTask):
    def __init__(self, cfg):
        super(SpecPretrain, self).__init__(cfg)

    @classmethod
    def setup_task(cls, cfg):
        return cls(cfg)

    def get_valid_outs(
        self, model, valid_loader, criterion, device,
        rank_indices=None, collect_frequency=False,
    ):
        """Evaluate one pass and optionally collect bounded embeddings.

        Parameters
        ----------
        model : torch.nn.Module
            Pre-training model producing (B, T, F) predictions.
        valid_loader : torch.utils.data.DataLoader
            Fixed validation batches.
        criterion : torch.nn.Module
            Original masked reconstruction criterion.
        device : str
            Inference device.
        rank_indices : sequence of int, optional
            Loader positions to pool into (N, D) embeddings, default None.
        collect_frequency : bool, optional
            Collect frequency sufficient statistics, default False.

        Returns
        -------
        dict
            Batch-mean original losses and optional monitor statistics.
        """
        if len(valid_loader) == 0:
            raise ValueError(
                "Evaluation loader must contain at least one batch."
            )
        model.eval()
        all_outs = {
            "loss": 0,
            "content_aware_loss": 0,
            "l1_loss": 0,
            "content_l1": 0,
            "zero_baseline_l1": 0,
            "content_zero_baseline_l1": 0,
        }
        monitor = Accumulator()
        embeddings = []
        offset = 0
        with torch.no_grad():
            for batch in valid_loader:
                size = batch["target"].shape[0]
                selected = [] if rank_indices is None else [
                    index - offset for index in rank_indices
                    if offset <= index < offset + size
                ]
                _, valid_outs = criterion(
                    model, batch, device,
                    embedding_rows=selected,
                    collect_frequency=collect_frequency,
                )
                if selected:
                    embeddings.append(
                        valid_outs.pop("window_embeddings")
                        .cpu().numpy()
                    )
                offset += size
                if "monitor_statistics" in valid_outs:
                    monitor.add(valid_outs)
                all_outs["loss"] += valid_outs["loss"]
                all_outs["content_aware_loss"] += valid_outs[
                    "content_aware_loss"
                ]
                all_outs["l1_loss"] += valid_outs["l1_loss"]
                all_outs["content_l1"] += valid_outs["content_l1"]
                all_outs["zero_baseline_l1"] += valid_outs["zero_baseline_l1"]
                all_outs["content_zero_baseline_l1"] += valid_outs[
                    "content_zero_baseline_l1"
                ]
        for key in all_outs:
            all_outs[key] /= len(valid_loader)
        if rank_indices is not None:
            if len(rank_indices) != sum(len(x) for x in embeddings):
                raise ValueError("Validation embedding subset is incomplete.")
            all_outs["rank_embeddings"] = (
                np.concatenate(embeddings) if embeddings else None
            )
        if monitor.batches:
            if monitor.batches != len(valid_loader):
                raise ValueError(
                    "Evaluation has inconsistent mask provenance."
                )
            all_outs["monitor_accumulator"] = monitor
        return all_outs

    def build_model(self, cfg):
        assert hasattr(self, "dataset")
        input_dim = self.dataset.get_input_dim()
        assert input_dim == cfg.input_dim
        model = models.build_model(cfg)
        extractor = getattr(self.dataset, "extracter", None)
        if (
            extractor is not None
            and extractor.cfg.normalizing == "dataset_zscore"
        ):
            if not hasattr(self.dataset, "prepare_statistics"):
                raise ValueError(
                    "Dataset-level STFT requires explicit training-electrode "
                    "membership and a statistics adapter."
                )
            if not hasattr(model, "set_stft_statistics"):
                raise ValueError(
                    "Model cannot persist fitted STFT statistics."
                )
            model.set_stft_statistics(self.dataset.prepare_statistics())
        return model

    def get_batch_iterator(self, dataset, batch_size, shuffle=True, **kwargs):
        return super().get_batch_iterator(
            dataset,
            batch_size,
            shuffle=shuffle,
            collate_fn=spec_collator,
            **kwargs,
        )

    def output_logs(
        self, train_logging_outs, val_logging_outs, writer, global_step
    ):
        if writer is None:
            return
        for k in train_logging_outs["images"]:
            image = train_logging_outs["images"][k]
            if k == "wav":
                tb_image = plot_tensorboard_line(image, title="wav")
            else:
                tb_image = plot_tensorboard_spectrogram(image)
            if writer is not None:
                writer.add_image(k, tb_image, global_step)

        if writer is not None:
            loss_metrics = ["l1_loss", "content_aware_loss", "content_l1"]
            all_loss_metrics = {}

            def add_prefix(prefix, outs):
                for k, v in outs.items():
                    if k in loss_metrics:
                        all_loss_metrics[f"{prefix}_{k}"] = v

            add_prefix("train", train_logging_outs)
            add_prefix("val", val_logging_outs)
            for k, v in all_loss_metrics.items():
                writer.add_scalar(k, v, global_step=global_step)
