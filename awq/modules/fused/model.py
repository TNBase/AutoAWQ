import torch
import torch.nn as nn
from awq.modules.fused.block import MPTBlock
from transformers.modeling_outputs import BaseModelOutputWithPast

class MPTModel(nn.Module):
    def __init__(self, vocab_size, blocks, wte, norm_f):
        super().__init__()
        self.vocab_size = vocab_size
        self.wte = wte
        self.blocks: list[MPTBlock] = nn.ModuleList(blocks)
        self.norm_f = norm_f
        self.attn_uses_sequence_id = False
        self.prefix_lm = False

    @torch.inference_mode()
    def forward(self, input_ids, attn_bias=None, attention_mask=None, is_causal=None, *args, **kwargs):
        _bsz, seqlen = input_ids.shape
        h = self.wte(input_ids)

        mask = None
        if seqlen > 1:
            mask = torch.full(
                (1, 1, seqlen, seqlen), float("-inf"), device=input_ids.device
            )
            mask = torch.triu(mask, diagonal=self.blocks[0].attn.start_pos + 1).type_as(h)

        for layer in self.blocks:
            h, _, past_key_value = layer(h, None, attention_mask=mask, is_causal=is_causal)
        h = self.norm_f(h)

        return BaseModelOutputWithPast(last_hidden_state=h, past_key_values=past_key_value, hidden_states=(), attentions=())