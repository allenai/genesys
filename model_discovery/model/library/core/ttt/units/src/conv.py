# GAU_IMPLEMENTATION_FILE   # DO NOT CHANGE OR REMOVE THE MAKK HERE, KEEP IT ALWAYS THE FIRST LINE TO ALLOW PARSER DETECT THE GAU IMPLEMENTATION FILES #

import torch    
import torch.nn as nn
import torch.nn.functional as F

from model_discovery.model.utils.modules import GAUBase,gau_test,UnitDecl # DO NOT CHANGE THIS IMPORT STATEMENT #


from typing import Any, Dict, Optional, Tuple, Union

import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils._pytree import tree_map

from transformers.utils import logging
from transformers.activations import ACT2FN
# from transformers.utils.import_utils import is_causal_conv1d_available # CAUSE EARLY INIT CUDA

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except:
    causal_conv1d_update, causal_conv1d_fn = None, None

logger = logging.get_logger(__name__)


# YOU CAN IMPORT MORE MODULES HERE #

# YOU CAN DEFINE MORE CLASSES OR FUNCTIONS HERE #






class Conv(GAUBase): 
    def __init__(self,embed_dim: int, block_loc: tuple, kwarg_all: dict, device=None,dtype=None, 
                 conv_kernel=4, rms_norm_eps=1e-6, **kwargs):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)

        kwarg_all['eps'] = rms_norm_eps
        self.norm = RMSNorm(embed_dim=self.embed_dim, block_loc=self.block_loc, kwarg_all=self.kwarg_all, **self.factory_kwargs, **self.kwarg_all)
        self.conv = nn.Conv1d(
            embed_dim,
            embed_dim,
            bias=True,
            kernel_size=conv_kernel,
            groups=embed_dim,
            padding=conv_kernel - 1,
            **self.factory_kwargs
        )

    def __call__(self, X, **Z):
        hidden_states = X
        seq_len = hidden_states.shape[1]
        hidden_states = self.norm(hidden_states, **Z)[0]

        # [B, C, L]
        hidden_states = hidden_states.transpose(1, 2)

        if causal_conv1d_fn is None:
            hidden_states = self.conv(hidden_states)[..., :seq_len]
        else:
            conv_weights = self.conv.weight.view(self.conv.weight.size(0), self.conv.weight.size(2))
            hidden_states = causal_conv1d_fn(hidden_states, conv_weights, self.conv.bias, activation=None)

        # [B, L, C]
        hidden_states = hidden_states.transpose(1, 2)

        return hidden_states




@gau_test
def test_conv(device=None,dtype=None):
    embed_dim=128
    block_loc=(0,6)
    kwarg_all={}
    conv = Conv(embed_dim, block_loc, kwarg_all, device=device, dtype=dtype)
    x = torch.randn(1,100,128).to(device=device,dtype=dtype)
    y = conv(x)
    assert y.shape==(1,100,128)



CHILDREN_DECLARATIONS = [
    UnitDecl(
        unitname="RMSNorm",
        requirements="", 
        inputs=['X'],
        outputs=['Y'],
    ),
]

SPEC = {
    "unitname": "Conv",
    "inputs": ['X'],
    "outputs": ['Y'],
    "document": '''
Conv
''',
}
ARGS = {
    'conv_kernel':4,
    'rms_norm_eps':1e-6,
}
CHILDREN = ['RMSNorm']
DESC='''
''' 
