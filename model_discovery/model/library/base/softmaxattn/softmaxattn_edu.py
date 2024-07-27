import torch
import torch.nn as nn

class LogAttention(nn.Module):
    """
    As proposed by Franz A. Heinsen, March 2024.

    Args:
        is_causal: (optional) bool, if True, compute causal log-attention.

    Input shapes:
        Q:     [..., n_queries, d_key] queries.
        K:     [..., n_context, d_key] keys.
        log_V: [..., n_context, d_val] log-values.

    Output shapes:
        log_attention: [..., n_queries, d_val] log of Softmax mixtures of values.
    """

    def __init__(self, is_causal=True):
        super().__init__()
        self.is_causal = is_causal

    def forward(self, Q, K, log_V, using_prev_context=False):
        Q = Q.unsqueeze(-1)                                          # [..., n_queries, d_key, 1]
        K = K.unsqueeze(-1)                                          # [..., n_context, d_key, 1]
        log_V = log_V.unsqueeze(-2)                                  # [..., n_context, 1, d_val]

        if self.is_causal:
            K = K.to(torch.float32) if self.training else K          # work-around for PyTorch 2.2 cuda issue
            H_S = torch.logcumsumexp(K + log_V, dim=-3).to(Q.dtype)  # [..., n_context, d_key, d_val]
            H_Z = torch.logcumsumexp(K        , dim=-3).to(Q.dtype)  # [..., n_context, d_key, 1]
        else:
            H_S = torch.logsumexp(K + log_V, dim=-3, keepdim=True)   # [..., 1, d_key, d_val]
            H_Z = torch.logsumexp(K        , dim=-3, keepdim=True)   # [..., 1, d_key, 1]

        if using_prev_context:
            H_S = self.prev_H_S.logaddexp(H_S)                       # [..., :, d_key, d_val]
            H_Z = self.prev_H_Z.logaddexp(H_Z)                       # [..., :, d_key, 1]

        self.prev_H_S = H_S[..., -1:, :, :].detach()                 # [..., 1, d_key, d_val]
        self.prev_H_Z = H_Z[..., -1:, :, :].detach()                 # [..., 1, d_key, d_val]

        log_S = torch.logsumexp(Q + H_S, dim=-2)                     # [..., n_queries, d_val]
        log_Z = torch.logsumexp(Q + H_Z, dim=-2)                     # [..., n_queries, 1]
    
        return log_S - log_Z
