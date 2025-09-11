from torch import nn
from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP


class sp_Qwen3MLP(Qwen3MLP):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj