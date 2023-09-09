from typing import List, Tuple
from awq.utils.utils import set_module_name
from awq.modules.fused_mlp import QuantMPTMLP
from transformers.models.mpt.modeling_mpt import MptMLP

class MptFuser:
    def __init__(self, model):
        self.model = model

        self.mlp_modules: List[Tuple[str, MptMLP]] = [
            (name, module) for name, module in self.model.named_modules()
            if isinstance(module, MptMLP)
        ]
    
    def fuse_attention(self):
        pass

    def fuse_layernorm(self):
        pass

    def fuse_mlp(self):
        for name, module in self.mlp_modules:
            mlp = QuantMPTMLP(module.up_proj, module.act, module.down_proj)
            set_module_name(self.model, name, mlp)