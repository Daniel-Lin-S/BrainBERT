from models import register_model
import torch.nn as nn
import torch
from models.base_model import BaseModel

POOL_HALF_WIDTH = 5


@register_model("finetune_model")
class FinetuneModel(BaseModel):
    def __init__(self):
        super(FinetuneModel, self).__init__()

    def forward(self, inputs, pad_mask):
        if self.frozen_upstream:
            self.upstream.eval()
            with torch.no_grad():
                outputs = self.upstream(
                    inputs, pad_mask, intermediate_rep=True
                )
        else:
            outputs = self.upstream(inputs, pad_mask, intermediate_rep=True)
        middle = int(outputs.shape[1] / 2)
        if outputs.shape[1] == 0:
            raise ValueError("Fine-tuning requires at least one time frame.")
        start = max(0, middle - POOL_HALF_WIDTH)
        stop = min(outputs.shape[1], middle + POOL_HALF_WIDTH)
        outputs = outputs[:, start:stop].mean(dim=1)
        out = self.linear_out(outputs)
        return out

    def build_model(self, cfg, upstream_model):
        self.cfg = cfg
        self.upstream = upstream_model
        self.upstream_cfg = self.upstream.cfg
        hidden_dim = self.upstream_cfg.hidden_dim
        self.linear_out = nn.Linear(
            in_features=hidden_dim, out_features=1
        )  # TODO hardcode out_features
        self.frozen_upstream = cfg.frozen_upstream
