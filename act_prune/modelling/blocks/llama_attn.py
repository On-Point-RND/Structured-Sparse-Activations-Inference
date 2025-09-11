from torch import nn
from transformers.models.llama.modeling_llama import (
    LlamaAttention, 
    eager_attention_forward  
)
from transformers.models.llama.configuration_llama import LlamaConfig

class LlamaAttention_act_sp(LlamaAttention):

    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__(config, layer_idx)

    def forward(self, hidden_states, position_embeddings, attention_mask, past_key_value = None, cache_position = None, **kwargs):
        attn_output, attn_weights = super().forward(hidden_states, position_embeddings, attention_mask, past_key_value, cache_position, **kwargs)
        return attn_output, attn_weights
    
    @classmethod
    def from_original(cls, orig_SelfAttn, sparsity_type=None):
        sp_SelfAttn = cls(orig_SelfAttn.config, orig_SelfAttn.layer_idx)
        sp_SelfAttn.q_proj = orig_SelfAttn.q_proj
        sp_SelfAttn.k_proj = orig_SelfAttn.k_proj
        sp_SelfAttn.v_proj = orig_SelfAttn.v_proj
        sp_SelfAttn.o_proj = orig_SelfAttn.o_proj

        return sp_SelfAttn