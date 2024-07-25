# gab.py

import torch
import torch.nn as nn

from model_discovery.model.utils.modules import GABBase # DO NOT CHANGE THIS IMPORT STATEMENT #



##### WIP, TERRIBLE, SUPER SLOW !!! #####

# YOU CAN IMPORT MORE MODULES HERE #

# YOU CAN DEFINE MORE CLASSES OR FUNCTIONS HERE #


import math
from einops import rearrange, repeat


from torchtune.modules import RotaryPositionalEmbeddings



class SimpleRetention(nn.Module):
    def __init__(self, hidden_size, gamma, head_size=None, double_v_dim=False, device=None,dtype=None):
        """
        Simple retention mechanism based on the paper
        "Retentive Network: A Successor to Transformer for Large Language Models"[https://arxiv.org/pdf/2307.08621.pdf]
        """
        super(SimpleRetention, self).__init__()

        self.hidden_size = hidden_size
        if head_size is None:
            head_size = hidden_size
        self.head_size = head_size

        self.v_dim = head_size * 2 if double_v_dim else head_size
        self.gamma = gamma

        self.W_Q = nn.Parameter(torch.randn(hidden_size, head_size, device=device,dtype=dtype) / hidden_size)
        self.W_K = nn.Parameter(torch.randn(hidden_size, head_size, device=device,dtype=dtype) / hidden_size)
        self.W_V = nn.Parameter(torch.randn(hidden_size, self.v_dim, device=device,dtype=dtype) / hidden_size)
        
        self.xpos = RotaryPositionalEmbeddings(head_size).to(device=device,dtype=dtype)

    def forward(self, X):
        """
        Parallel (default) representation of the retention mechanism.
        X: (batch_size, sequence_length, hidden_size)
        """
        sequence_length = X.shape[1]
        D = self._get_D(sequence_length).to(self.W_Q.device).to(dtype=X.dtype)

        Q = (X @ self.W_Q)
        K = (X @ self.W_K)

        Q = rearrange(Q, '... (h d) -> ... h d', d=self.head_size)
        K = rearrange(K, '... (h d) -> ... h d', d=self.head_size)
        Q = self.xpos(Q)
        K = self.xpos(K)
        Q = rearrange(Q, '... h d -> ... (h d)')
        K = rearrange(K, '... h d -> ... (h d)')

        V = X @ self.W_V
        ret = (Q @ K.permute(0, 2, 1)) * D.unsqueeze(0)
        
        return ret @ V
        
    def forward_recurrent(self, x_n, s_n_1, n):
        """
        Recurrent representation of the retention mechanism.
        x_n: (batch_size, 1, hidden_size)
        s_n_1: (batch_size, hidden_size, v_dim)
        """

        Q = (x_n @ self.W_Q)
        K = (x_n @ self.W_K)

        Q = rearrange(Q, '... (h d) -> ... h d', d=self.head_size)
        K = rearrange(K, '... (h d) -> ... h d', d=self.head_size)
        Q = self.xpos(Q)
        K = self.xpos(K)
        Q = rearrange(Q, '... h d -> ... (h d)')
        K = rearrange(K, '... h d -> ... (h d)')

        V = x_n @ self.W_V

        # K: (batch_size, 1, hidden_size)
        # V: (batch_size, 1, v_dim)
        # s_n = gamma * s_n_1 + K^T @ V

        s_n = self.gamma * s_n_1 + (K.transpose(-1, -2) @ V)
        
        return (Q @ s_n), s_n
    
    def forward_chunkwise(self, x_i, r_i_1, i):
        """
        Chunkwise representation of the retention mechanism.
        x_i: (batch_size, chunk_size, hidden_size)
        r_i_1: (batch_size, hidden_size, v_dim)
        """
        batch, chunk_size, _ = x_i.shape
        D = self._get_D(chunk_size).to(self.W_Q.device).to(dtype=x_i.dtype)

        Q = (x_i @ self.W_Q)
        K = (x_i @ self.W_K)

        Q = rearrange(Q, '... (h d) -> ... h d', d=self.head_size)
        K = rearrange(K, '... (h d) -> ... h d', d=self.head_size)
        Q = self.xpos(Q)
        K = self.xpos(K)
        Q = rearrange(Q, '... h d -> ... (h d)')
        K = rearrange(K, '... h d -> ... (h d)')

        V = x_i @ self.W_V
        
        r_i =(K.transpose(-1, -2) @ (V * D[-1].view(1, chunk_size, 1))) + (self.gamma ** chunk_size) * r_i_1

        inner_chunk = ((Q @ K.transpose(-1, -2)) * D.unsqueeze(0)) @ V
        
        #e[i,j] = gamma ** (i+1)
        e = torch.zeros(batch, chunk_size, 1).to(self.W_Q.device).to(dtype=x_i.dtype)
        
        for _i in range(chunk_size):
            e[:, _i, :] = self.gamma ** (_i + 1)
        
        cross_chunk = (Q @ r_i_1) * e
        
        return inner_chunk + cross_chunk, r_i


    def _get_D(self, sequence_length):
        n = torch.arange(sequence_length).unsqueeze(1)
        m = torch.arange(sequence_length).unsqueeze(0)

        # Broadcast self.gamma ** (n - m) with appropriate masking to set values where n < m to 0
        D = (self.gamma ** (n - m)) * (n >= m).float()  #this results in some NaN when n is much larger than m
        # fill the NaN with 0
        D[D != D] = 0

        return D
    


class MultiScaleRetention(nn.Module):
    def __init__(self, hidden_size, heads, double_v_dim=False, device=None,dtype=None):
        """
        Multi-scale retention mechanism based on the paper
        "Retentive Network: A Successor to Transformer for Large Language Models"[https://arxiv.org/pdf/2307.08621.pdf]
        """
        super(MultiScaleRetention, self).__init__()
        self.hidden_size = hidden_size
        self.v_dim = hidden_size * 2 if double_v_dim else hidden_size
        self.heads = heads
        assert hidden_size % heads == 0, "hidden_size must be divisible by heads"
        self.head_size = hidden_size // heads
        self.head_v_dim = hidden_size * 2 if double_v_dim else hidden_size
        
        self.gammas = (1 - torch.exp(torch.linspace(math.log(1/32), math.log(1/512), heads))).detach().cpu().tolist()

        self.swish = lambda x: x * torch.sigmoid(x)
        self.W_G = nn.Parameter(torch.randn(hidden_size, self.v_dim, device=device,dtype=dtype) / hidden_size)
        self.W_O = nn.Parameter(torch.randn(self.v_dim, hidden_size, device=device,dtype=dtype) / hidden_size)
        self.group_norm = nn.GroupNorm(heads, self.v_dim).to(device=device,dtype=dtype)

        self.retentions = nn.ModuleList([
            SimpleRetention(self.hidden_size, gamma, self.head_size, double_v_dim, device=device,dtype=dtype) for gamma in self.gammas
        ])

    def forward(self, X):
        """
        parallel representation of the multi-scale retention mechanism
        """

        # apply each individual retention mechanism to X
        Y = []
        for i in range(self.heads):
            Y.append(self.retentions[i](X))
        
        Y = torch.cat(Y, dim=2)
        Y_shape = Y.shape
        Y = self.group_norm(Y.reshape(-1, self.v_dim)).reshape(Y_shape)

        return (self.swish(X @ self.W_G) * Y) @ self.W_O
    
    def forward_recurrent(self, x_n, s_n_1s, n):
        """
        recurrent representation of the multi-scale retention mechanism
        x_n: (batch_size, 1, hidden_size)
        s_n_1s: (batch_size, heads, head_size, head_size)

        """
    
        # apply each individual retention mechanism to a slice of X
        Y = []
        s_ns = []
        for i in range(self.heads):
            y, s_n = self.retentions[i].forward_recurrent(
                x_n[:, :, :], s_n_1s[i], n
                )
            Y.append(y)
            s_ns.append(s_n)
        
        Y = torch.cat(Y, dim=2)
        Y_shape = Y.shape
        Y = self.group_norm(Y.reshape(-1, self.v_dim)).reshape(Y_shape)
        
        return (self.swish(x_n @ self.W_G) * Y) @ self.W_O, s_ns

    def forward_chunkwise(self, x_i, r_i_1s, i):
        """
        chunkwise representation of the multi-scale retention mechanism
        x_i: (batch_size, chunk_size, hidden_size)
        r_i_1s: (batch_size, heads, head_size, head_size)
        """
        batch, chunk_size, _ = x_i.shape

        # apply each individual retention mechanism to a slice of X
        Y = []
        r_is = []
        for j in range(self.heads):
            y, r_i = self.retentions[j].forward_chunkwise(
                x_i[:, :, :], r_i_1s[j], i
                )
            Y.append(y)
            r_is.append(r_i)
        
        
        Y = torch.cat(Y, dim=2)
        Y_shape = Y.shape
        Y = self.group_norm(Y.reshape(-1, self.v_dim)).reshape(Y_shape)

        return (self.swish(x_i @ self.W_G) * Y) @ self.W_O, r_is



def pad_to_block_length(X, block_len):
    pad_len = (block_len - X.shape[1] % block_len) % block_len
    if pad_len > 0:
        padding = torch.zeros(X.shape[0], pad_len, *X.shape[2:], dtype=X.dtype, device=X.device)
        X = torch.cat([X, padding], dim=1)
    return X

class GAB(GABBase):
    """Generalized Autoregressive Block
        Input:        X: (batch, seqlen, embed_dim)
        Output:       Y: (batch, seqlen, embed_dim)
        Constraints:  Causal, differentiable, parameter number, complexity, parallelizable
    """
    def __init__(self,embed_dim: int, device=None,dtype=None,heads=8,**kwargs): # YOU CAN ADD MORE ARGUMENTS, BUT YOU HAVE TO HAVE embed_dim, device, dtype AS THE ARGUTMENTS #
        # argv: list of hyperparameters
        factory_kwargs = {"device": device, "dtype": dtype} # remember to pass it to nn layers
        super().__init__(embed_dim) # DO NOT CHANGE THIS LINE #
        
        # COMPLETING THE CODE HERE #
        self.embed_dim = embed_dim
        self.heads = heads
        ffn_size = 4 * embed_dim

        self.retention = MultiScaleRetention(embed_dim, heads, device=device,dtype=dtype)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_size),
            nn.GELU(),
            nn.Linear(ffn_size, embed_dim)
        ).to(device=device,dtype=dtype)
        self.layer_norm_1 = nn.LayerNorm(embed_dim).to(device=device,dtype=dtype)
        self.layer_norm_2 = nn.LayerNorm(embed_dim).to(device=device,dtype=dtype)


    # YOU CAN ADD MORE FUNCTIONS HERE #

    def forward_recurrent(self, x_n, s_n_1s, n):
        """
        X: (batch_size, sequence_length, hidden_size)
        s_n_1s: list of lists of tensors of shape (batch_size, hidden_size // heads, hidden_size // heads)

        """
        o_n, s_n = self.retention.forward_recurrent(self.layer_norm_1(x_n), s_n_1s, n)
        y_n = o_n + x_n
        x_n = self.ffn(self.layer_norm_2(y_n)) + y_n
        
        return x_n, s_n
    
    def forward_chunkwise(self, x_i, r_i_1s, i):
        """
        X: (batch_size, sequence_length, hidden_size)
        r_i_1s: list of lists of tensors of shape (batch_size, hidden_size // heads, hidden_size // heads)

        """
        o_i, r_i = self.retention.forward_chunkwise(self.layer_norm_1(x_i), r_i_1s, i)
        y_i = o_i + x_i
        x_i = self.ffn(self.layer_norm_2(y_i)) + y_i
        return x_i, r_i
    

    def _forward(self,X,mode='recurrent',**kwargs): # type hints are optional but recommended

        # THE CODE HERE MUST BE COMPLETED #

        if mode=='chunkwise':
            chunk_size=32
            batch_size,_seqlen,_=X.shape
            X=pad_to_block_length(X, chunk_size)
            sequence_length=X.shape[1]
            
            r_n_1s = [
                torch.zeros(self.embed_dim // self.heads, self.retention.v_dim // self.heads).unsqueeze(0).repeat(batch_size, 1, 1).to(X.device).to(X.dtype)
                for _ in range(self.heads)
            ]
            Y_chunkwise = []
            for i in range(sequence_length):
                y_i, r_i = self.forward_chunkwise(X[:, i:i+1, :], r_n_1s, i)
                Y_chunkwise.append(y_i)
                r_n_1s = r_i
            
            Y_chunkwise = torch.concat(Y_chunkwise, dim=1)

            X=Y_chunkwise[:,:_seqlen,:]

        elif mode=='recurrent':

            batch_size, sequence_length, hidden_size = X.shape

            s_n_1s = [
                    torch.zeros(hidden_size // self.heads, self.retention.v_dim // self.heads).unsqueeze(0).repeat(batch_size, 1, 1).to(X.device).to(X.dtype)
                    for _ in range(self.heads)
            ]
            Y_recurrent = []
            for i in range(sequence_length):
                y_n, s_ns = self.forward_recurrent(X[:, i:i+1, :], s_n_1s, i)
                Y_recurrent.append(y_n)
                s_n_1s = s_ns

            Y_recurrent = torch.concat(Y_recurrent, dim=1)

        else:
            Y = self.retention(self.layer_norm_1(X)) + X
            X = self.ffn(self.layer_norm_2(Y)) + Y
        return X
    
""" The dictionary of hyperparameters for constructing a GAB layer
    embed_dim, device, dtype should NOT be included in gab_config
"""
gab_config = {
    # THE HYPERPARAMETERS OF ADDITIONAL ARGUMENTS IN GAB CLASS #
    'heads': 8
}
