# GAU_IMPLEMENTATION_FILE   # DO NOT CHANGE OR REMOVE THE MAKK HERE, KEEP IT ALWAYS THE FIRST LINE TO ALLOW PARSER DETECT THE GAU IMPLEMENTATION FILES #

import torch    
import torch.nn as nn
import torch.nn.functional as F

from model_discovery.model.utils.modules import GAUBase,gau_test,UnitDecl # DO NOT CHANGE THIS IMPORT STATEMENT #


from typing import Any, Dict, Optional, Tuple, Union

import torch.nn.functional as F

from transformers.utils import logging


logger = logging.get_logger(__name__)


# YOU CAN IMPORT MORE MODULES HERE #

# YOU CAN DEFINE MORE CLASSES OR FUNCTIONS HERE #




class RotaryEmbedding(GAUBase): 
    def __init__(self,embed_dim: int, block_loc: tuple, kwarg_all: dict, device=None,dtype=None, 
                 dim=None, max_position_embeddings=16,base=10000, scaling_factor=1.0, **kwargs):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)

        self.scaling_factor = scaling_factor
        self.dim = dim if dim is not None else embed_dim//4
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)


    @torch.no_grad()
    def _forward(self, X, input, position_ids, **Z):
        # x: [bs, num_attention_heads, seq_len, head_size]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = input.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        Z['cos'] = cos.to(**self.factory_kwargs)
        Z['sin'] = sin.to(**self.factory_kwargs)
        return X,Z


@gau_test
def test_rotaryembedding(device=None,dtype=None):
    embed_dim=128
    block_loc=(0,6)
    kwarg_all={}
    rotaryembedding = RotaryEmbedding(embed_dim, block_loc, kwarg_all, device=device, dtype=dtype)
    x = torch.randn(1,100,128).to(device=device,dtype=dtype)
    y = rotaryembedding(x)
    assert y.shape==(1,100,128)



CHILDREN_DECLARATIONS = [
]

SPEC = {
    "unitname": "RotaryEmbedding",
    "inputs": ['X'],
    "outputs": ['Y'],
    "document": '''
RotaryEmbedding
''',
}
ARGS = {
    'dim':None,
    'max_position_embeddings':16,
    'base':10000,
    'scaling_factor':1.0,
}
CHILDREN = []
DESC='''
''' 
