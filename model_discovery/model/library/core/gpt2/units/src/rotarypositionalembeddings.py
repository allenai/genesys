# GAU_IMPLEMENTATION_FILE   # DO NOT CHANGE OR REMOVE THE MAKK HERE, KEEP IT ALWAYS THE FIRST LINE TO ALLOW PARSER DETECT THE GAU IMPLEMENTATION FILES #

import torch    
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

from model_discovery.model.utils.modules import GAUBase,gau_test,UnitDecl # DO NOT CHANGE THIS IMPORT STATEMENT #

from typing import Optional




class RotaryPositionalEmbeddings(GAUBase): # DO NOT CHANGE THIS CLASS NAME #
    """
    This class implements Rotary Positional Embeddings (RoPE)
    proposed in https://arxiv.org/abs/2104.09864.

    Reference implementation (used for correctness verfication)
    can be found here:
    https://github.com/meta-llama/llama/blob/main/llama/model.py#L80

    In this implementation we cache the embeddings for each position upto
    ``max_seq_len`` by computing this during init.

    Args:
        dim (int): Embedding dimension. This is usually set to the dim of each
            head in the attention module computed as ````embed_dim`` // ``num_heads````
        max_seq_len (int): Maximum expected sequence length for the
            model, if exceeded the cached freqs will be recomputed
        base (int): The base for the geometric progression used to compute
            the rotation angles
    """

    def __init__(
        self,
        embed_dim: int,
        block_loc: tuple,
        kwarg_all: dict,
        device=None,
        dtype=None,
        rotary_emb_base: int = 10_000,
        rotary_emb_dim: int = None,
        max_seq_len: int = 4096,
        **kwargs
    ) -> None:
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)
        self.dim = rotary_emb_dim
        self.base = rotary_emb_base
        self.max_seq_len = max_seq_len
        self._rope_init()

    # We need to explicitly define reset_parameters for FSDP initialization, see
    # https://github.com/pytorch/pytorch/blob/797d4fbdf423dd9320ebe383fb57ffb1135c4a99/torch/distributed/fsdp/_init_utils.py#L885
    def reset_parameters(self):
        self._rope_init()

    def _rope_init(self):
        theta = 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2,**self.factory_kwargs)[: (self.dim // 2)].float() / self.dim)
        )
        self.register_buffer("theta", theta, persistent=False)
        self.build_rope_cache(self.max_seq_len)

    def build_rope_cache(self, max_seq_len: int = 4096) -> None:
        # Create position indexes `[0, 1, ..., max_seq_len - 1]`
        seq_idx = torch.arange(
            max_seq_len, dtype=self.theta.dtype, device=self.theta.device
        )

        # Outer product of theta and position index; output tensor has
        # a shape of [max_seq_len, dim // 2]
        idx_theta = torch.einsum("i, j -> ij", seq_idx, self.theta).float()

        # cache includes both the cos and sin components and so the output shape is
        # [max_seq_len, dim // 2, 2]
        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
        self.register_buffer("cache", cache, persistent=False)

    def _forward(self, X: Tensor, input_emb: Tensor, input_pos: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            x (Tensor): input tensor with shape
                [b, s, n_h, h_d]
            input_pos (Optional[Tensor]): Optional tensor which contains the position ids
                of each token. During training, this is used to indicate the positions
                of each token relative to its sample when packed, shape [b, s].
                During inference, this indicates the position of the current token.
                If none, assume the index of the token is its position id. Default is None.

        Returns:
            Tensor: output tensor with RoPE applied

        Notation used for tensor shapes:
            - b: batch size
            - s: sequence length
            - n_h: num heads
            - h_d: head dim

        TODO: The implementation below can be made more efficient
        for inference.
        """
        # input tensor has shape [b, s, n_h, h_d]
        seq_len = input_emb.size(1)

        # extract the values based on whether input_pos is set or not
        rope_cache = (
            self.cache[:seq_len] if input_pos is None else self.cache[input_pos]
        )

        # reshape input; the last dimension is used for computing the output.
        # Cast to float to match the reference implementation
        # tensor has shape [b, s, n_h, h_d // 2, 2]
        xshaped = input_emb.float().reshape(*input_emb.shape[:-1], -1, 2)

        # reshape the cache for broadcasting
        # tensor has shape [b, s, 1, h_d // 2, 2] if packed samples,
        # otherwise has shape [1, s, 1, h_d // 2, 2]
        rope_cache = rope_cache.view(-1, xshaped.size(1), 1, xshaped.size(3), 2)

        # tensor has shape [b, s, n_h, h_d // 2, 2]
        x_out = torch.stack(
            [
                xshaped[..., 0] * rope_cache[..., 0]
                - xshaped[..., 1] * rope_cache[..., 1],
                xshaped[..., 1] * rope_cache[..., 0]
                + xshaped[..., 0] * rope_cache[..., 1],
            ],
            -1,
        )

        # tensor has shape [b, s, n_h, h_d]
        x_out = x_out.flatten(3)
        output_emb = x_out.type_as(input_emb)
        return X,{"output_emb":output_emb}



@gau_test
def test_rotarypositionalembeddings(device=None,dtype=None):    
    embed_dim=128
    block_loc=(0,6)
    kwarg_all={}
    rotarypositionalembeddings = RotaryPositionalEmbeddings(embed_dim, block_loc, kwarg_all, device=device, dtype=dtype,**kwarg_all)
    input_emb = torch.randn(1,100,128).to(device=device,dtype=dtype)
    input_pos = torch.arange(128).to(device=device,dtype=dtype)
    X=torch.randn(1,100,128).to(device=device,dtype=dtype)
    Z={
        "input_emb":input_emb,
        "input_pos":input_pos,
    }
    _,Z_= rotarypositionalembeddings(X,**Z)
    output_emb=Z_["output_emb"]
    assert output_emb.shape==(1,100,128)


CHILDREN_DECLARATIONS = []


SPEC ={
    "unitname": "RotaryPositionalEmbeddings",
    "inputs": ['input_emb','*input_pos'],
    "outputs": ['output_emb'],
    "document": '''
This class implements Rotary Positional Embeddings (RoPE)
proposed in https://arxiv.org/abs/2104.09864.

Reference implementation (used for correctness verfication)
can be found here:
https://github.com/meta-llama/llama/blob/main/llama/model.py#L80

In this implementation we cache the embeddings for each position upto
``max_seq_len`` by computing this during init.

Args:
    dim (int): Embedding dimension. This is usually set to the dim of each
        head in the attention module computed as ````embed_dim`` // ``num_heads````
    max_seq_len (int): Maximum expected sequence length for the
        model, if exceeded the cached freqs will be recomputed
    base (int): The base for the geometric progression used to compute
        the rotation angles
''',
}
ARGS = {
    "rotary_emb_base":10_000,
    "max_seq_len":4096,
}
CHILDREN = []
DESC='''
'''