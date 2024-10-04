# GAU_IMPLEMENTATION_FILE   # DO NOT CHANGE OR REMOVE THE MAKK HERE, KEEP IT ALWAYS THE FIRST LINE TO ALLOW PARSER DETECT THE GAU IMPLEMENTATION FILES #

import torch    
import torch.nn as nn
import torch.nn.functional as F

from model_discovery.model.utils.modules import GAUBase,gau_test,UnitDecl # DO NOT CHANGE THIS IMPORT STATEMENT #


from einops import rearrange, repeat




class SSDMinimalDiscrete(GAUBase):
    """
    SSDMinimalDiscrete (State Space Discrete Minimal) implements a discrete-time state space model.

    This class provides an efficient implementation of the SSM algorithm, particularly
    suited for processing sequential data in chunks. It uses a minimal discrete-time
    formulation that is both memory-efficient and computationally effective.

    Args:
        embed_dim (int): The embedding dimension of the input.
        block_loc (tuple): The location of the block within the larger model structure.
        kwarg_all (dict): Additional keyword arguments.
        device (torch.device, optional): The device to run the module on.
        dtype (torch.dtype, optional): The data type of the module's parameters.

    Inputs:
        X (torch.Tensor): The input tensor of shape (batch, length, n_heads, d_head).
        A (torch.Tensor): The state transition tensor of shape (batch, length, n_heads).
        B (torch.Tensor): The input-to-state tensor of shape (batch, length, n_heads, d_state).
        C (torch.Tensor): The state-to-output tensor of shape (batch, length, n_heads, d_state).
        dt (torch.Tensor): The time step tensor of shape (batch, length, n_heads).
        chunk_size (int): The size of chunks for processing the sequence.

    Outputs:
        Y (torch.Tensor): The output tensor of shape (batch, length, n_heads, d_head).

    The class implements the forward pass of the SSM algorithm, including:
    1. Intra-chunk computations (diagonal blocks)
    2. Inter-chunk state propagation
    3. State-to-output conversion

    This implementation is designed to be efficient for long sequences by processing
    the input in chunks, which allows for better parallelization and memory usage.
    """
    def __init__(self, embed_dim: int, block_loc: tuple, kwarg_all: dict, device=None, dtype=None, **kwargs):
        self.factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)

    def _forward(self, X, x, A, B, C, dt, chunk_size):
        y, _ = self.ssd_minimal_discrete(x*dt.unsqueeze(-1), A*dt, B, C, chunk_size)
        Z_={
            'y':y,
        }
        return X, Z_

    def segsum(self, x):
        """More stable segment sum calculation."""
        T = x.size(-1)
        x = repeat(x, "... d -> ... d e", e=T)
        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=-1)
        x = x.masked_fill(~mask, 0)
        x_segsum = torch.cumsum(x, dim=-2)
        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=0)
        x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
        return x_segsum

    def ssd_minimal_discrete(self, X, A, B, C, block_len, initial_states=None):
        """
        Arguments:
            X: (batch, length, n_heads, d_head)
            A: (batch, length, n_heads)
            B: (batch, length, n_heads, d_state)
            C: (batch, length, n_heads, d_state)
        Return:
            Y: (batch, length, n_heads, d_head)
        """
        assert X.dtype == A.dtype == B.dtype == C.dtype
        # assert X.shape[1] % block_len == 0

        # Rearrange into blocks/chunks
        X, A, B, C = [rearrange(x, "b (c l) ... -> b c l ...", l=block_len) for x in (X, A, B, C)]

        A = rearrange(A, "b c l h -> b h c l")
        A_cumsum = torch.cumsum(A, dim=-1)

        # 1. Compute the output for each intra-chunk (diagonal blocks)
        L = torch.exp(self.segsum(A))
        Y_diag  = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", C, B, L, X)

        # 2. Compute the state for each intra-chunk
        # (right term of low-rank factorization of off-diagonal blocks; B terms)
        decay_states = torch.exp((A_cumsum[:, :, :, -1:] - A_cumsum))
        states = torch.einsum("bclhn,bhcl,bclhp->bchpn", B, decay_states, X)

        # 3. Compute the inter-chunk SSM recurrence; produces correct SSM states at chunk boundaries
        # (middle term of factorization of off-diag blocks; A terms)
        if initial_states is None:
            initial_states = torch.zeros_like(states[:, :1])
        states = torch.cat([initial_states, states], dim=1)
        decay_chunk = torch.exp(self.segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0))))
        new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
        states, final_state = new_states[:, :-1], new_states[:, -1]

        # 4. Compute state -> output conversion per chunk
        # (left term of low-rank factorization of off-diagonal blocks; C terms)
        state_decay_out = torch.exp(A_cumsum)
        Y_off = torch.einsum('bclhn,bchpn,bhcl->bclhp', C, states, state_decay_out)

        # Add output of intra-chunk and inter-chunk terms (diagonal and off-diagonal blocks)
        Y = rearrange(Y_diag+Y_off, "b c l h p -> b (c l) h p")
        return Y, final_state


@gau_test
def test_ssdminimaldiscrete(device=None, dtype=None):
    embed_dim = 128
    block_loc = (0, 6)
    kwarg_all = {}
    chunk_size = 16
    batch_size = 2
    seq_len = 32
    n_heads = 4
    d_head = 32
    d_state = 16

    ssd = SSDMinimalDiscrete(embed_dim, block_loc, kwarg_all, device=device, dtype=dtype)

    # Create input tensors
    X = torch.randn(batch_size, seq_len, n_heads, d_head, device=device, dtype=dtype)
    A = torch.randn(batch_size, seq_len, n_heads, device=device, dtype=dtype)
    B = torch.randn(batch_size, seq_len, n_heads, d_state, device=device, dtype=dtype)
    C = torch.randn(batch_size, seq_len, n_heads, d_state, device=device, dtype=dtype)
    dt = torch.rand(batch_size, seq_len, n_heads, device=device, dtype=dtype)

    # Create input dictionary
    Z = {
        'x': X,
        'A': A,
        'B': B,
        'C': C,
        'dt': dt,
        'chunk_size': chunk_size
    }

    # Run the forward pass
    _, Z_ = ssd(X, **Z)

    # Check output shape
    assert Z_['y'].shape == (batch_size, seq_len, n_heads, d_head), f"Expected output shape {(batch_size, seq_len, n_heads, d_head)}, but got {Z_['y'].shape}"

    # Check output dtype
    assert Z_['y'].dtype == dtype, f"Expected output dtype {dtype}, but got {Z_['y'].dtype}"

    # Check output device
    assert Z_['y'].device == device, f"Expected output device {device}, but got {Z_['y'].device}"

    print("SSDMinimalDiscrete test passed successfully!")


CHILDREN_DECLARATIONS = []

SPEC = {
    "unitname": "SSDMinimalDiscrete",
    "inputs": ['X','A','B','C','dt','chunk_size'],
    "outputs": ['Y'],
    "document": 
    """
    SSDMinimalDiscrete (State Space Discrete Minimal) implements a discrete-time state space model.

    This class provides an efficient implementation of the SSM algorithm, particularly
    suited for processing sequential data in chunks. It uses a minimal discrete-time
    formulation that is both memory-efficient and computationally effective.

    Args:
        embed_dim (int): The embedding dimension of the input.
        block_loc (tuple): The location of the block within the larger model structure.
        kwarg_all (dict): Additional keyword arguments.
        device (torch.device, optional): The device to run the module on.
        dtype (torch.dtype, optional): The data type of the module's parameters.

    Inputs:
        X (torch.Tensor): The input tensor of shape (batch, length, n_heads, d_head).
        A (torch.Tensor): The state transition tensor of shape (batch, length, n_heads).
        B (torch.Tensor): The input-to-state tensor of shape (batch, length, n_heads, d_state).
        C (torch.Tensor): The state-to-output tensor of shape (batch, length, n_heads, d_state).
        dt (torch.Tensor): The time step tensor of shape (batch, length, n_heads).
        chunk_size (int): The size of chunks for processing the sequence.

    Outputs:
        Y (torch.Tensor): The output tensor of shape (batch, length, n_heads, d_head).

    The class implements the forward pass of the SSM algorithm, including:
    1. Intra-chunk computations (diagonal blocks)
    2. Inter-chunk state propagation
    3. State-to-output conversion

    This implementation is designed to be efficient for long sequences by processing
    the input in chunks, which allows for better parallelization and memory usage.
""",
}
ARGS = {}
CHILDREN = []
DESC="""
"""


