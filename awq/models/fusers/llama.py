import torch
from typing import List, Tuple
from awq.quantize.qmodule import WQLinear
from awq.utils.utils import set_module_name
from awq.modules import QuantLlamaMLP, FTLlamaRMSNorm, QuantLlamaAttention, QuantLlamaAttentionFused
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaRMSNorm, LlamaMLP
from ..inference_models.llama import LlamaAttentionFused

class LlamaFuser:
    def __init__(self, model):
        self.model = model

        self.attention_modules: List[Tuple[str, LlamaAttention]] = [
            (name, module) for name, module in self.model.named_modules()
            if any(isinstance(module, layer_type) for layer_type in [LlamaAttention, LlamaAttentionFused])
        ]

        self.rmsnorm_modules: List[Tuple[str, LlamaRMSNorm]] = [
            (name, module) for name, module in self.model.named_modules()
            if isinstance(module, LlamaRMSNorm)
        ]
        
        self.mlp_modules: List[Tuple[str, LlamaMLP]] = [
            (name, module) for name, module in self.model.named_modules()
            if isinstance(module, LlamaMLP)
        ]
    
    def fuse_attention(self):
        for name, module in self.attention_modules:
            qkv_layer: WQLinear = self._fuse_qkv(module)
            
            attn = QuantLlamaAttention(
                module.hidden_size,
                module.num_heads,
                qkv_layer,
                module.o_proj,
                next(iter(qkv_layer.state_dict().values())).device,
            )


            set_module_name(self.model, name, attn)
    
    def _fuse_qkv(self, module: LlamaAttention):
        """Turn a separate q, k, and v projection into a single qkv projection.""""
        # get qkv and bias
        q_proj, k_proj, v_proj = module.q_proj, module.k_proj, module.v_proj
        bias = torch.cat([q_proj.bias, k_proj.bias, v_proj.bias], dim=0) if q_proj.bias is not None else None

        # create module
        qkv_layer = WQLinear(
            q_proj.w_bit, 
            q_proj.group_size, 
            q_proj.in_features, 
            q_proj.out_features + k_proj.out_features + v_proj.out_features, 
            q_proj.bias is not None,
            next(iter(module.state_dict().values())).device
        )

        # replace buffers with real weights
        qkv_layer.qweight = torch.cat([q_proj.qweight, k_proj.qweight, v_proj.qweight], dim=1)
        qkv_layer.qzeros = torch.cat([q_proj.qzeros, k_proj.qzeros, v_proj.qzeros], dim=1)
        qkv_layer.scales = torch.cat([q_proj.scales, k_proj.scales, v_proj.scales], dim=1)
        qkv_layer.bias = bias

        return qkv_layer

    def fuse_rmsnorm(self):
        for name, module in self.rmsnorm_modules:
            norm = FTLlamaRMSNorm(module.weight, module.variance_epsilon)
            set_module_name(self.model, name, norm)

    def fuse_mlp(self):
        for name, module in self.mlp_modules:
            mlp = QuantLlamaMLP(module.gate_proj, module.down_proj, module.up_proj)
            set_module_name(self.model, name, mlp)
