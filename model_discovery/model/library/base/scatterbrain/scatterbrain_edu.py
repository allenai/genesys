import math
import torch
import torch.nn as nn

from einops import rearrange

from fast_transformers.local_product import local_dot_product, local_weighted_average

from src.models.attention.feature_maps_sb import softmax_kernel,FeatureMap
from contextlib import contextmanager
from functools import partial
from torch.cuda.amp import autocast

try:
    from apex import amp
    APEX_AVAILABLE = True
except:
    APEX_AVAILABLE = False


def gaussian_orthogonal_random_matrix(nrows, ncols, scaling=0, device=None, dtype=None):
    factory_kwargs = {'device': device, 'dtype': dtype}
    nblocks = int(math.ceil(nrows / ncols))
    # TD [2021-10-28]: Sometimes QR fails on CUDA
    unstructured_blocks = torch.randn((nblocks, ncols, ncols), device='cpu')
    q, r = torch.linalg.qr(unstructured_blocks)
    # To make sure Q is uniform from the Haar distribution https://arxiv.org/pdf/math-ph/0609050.pdf
    q *= rearrange(torch.diagonal(r, dim1=-2, dim2=-1).sign(), 'b c -> b 1 c')
    q = q.to(**factory_kwargs)
    # TD [2021-10-28] Idk why the transpose is necessary. I suspect it isn't.
    # https://github.com/google-research/google-research/blob/ea313c6e96acce6c863de41615c6cf4079b8ca94/performer/fast_attention/jax/fast_attention.py#L362
    q = rearrange(q, 'b c c1 -> b c1 c')
    g_ortho = rearrange(q, 'b c1 c -> (b c1) c')[:nrows]

    if scaling == 0:
        multiplier = torch.randn((nrows, ncols), **factory_kwargs).norm(dim=1)
        return rearrange(multiplier, 'r -> r 1') * g_ortho
    elif scaling == 1:
        return math.sqrt(ncols) * g_ortho
    else:
        raise ValueError(f'Invalid scaling {scaling}')



class SBPerformerFeatures(FeatureMap):
    """Random Fourier Features for the RBF kernel according to [1].
    [1]: "Weighted Sums of Random Kitchen Sinks: Replacing minimization with
         randomization in learning" by A. Rahimi and Benjamin Recht.
    Arguments
    ---------
        query_dims: int, The input query dimensions in order to sample
                          the noise matrix
        n_features: int, The size of the feature map (should be divisible by 2)
                (default: query_dims)
        softmax_temp: float, The temerature for the Gaussian kernel
                      approximation exp(-t * |x-y|^2)
                      (default: 1/sqrt(query_dims))
        orthogonal: bool, When True the random matrix is initialized for
                    orthogonal random features to reduce the approximation
                    variance (default: False)
        redraw: int, Redraw the random matrix every 'redraw' times
                (default: 1)
        deterministic_eval: bool, Only redraw the random matrix during training
                            (default: False)
    """
    def __init__(self, query_dims, n_features=None, ortho_scaling=0, softmax_temp=None,
                 orthogonal=False, cosh=True, redraw=1, deterministic_eval=False, eps=0.0,
                 device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(query_dims)
        self.n_features = n_features or int(query_dims * math.log(query_dims))
        self.ortho_scaling = ortho_scaling
        # TODO: we're not using @orthogonal atm
        self.orthogonal = orthogonal
        self.softmax_temp = 1 / math.sqrt(query_dims) if softmax_temp is None else softmax_temp
        self.cosh = cosh
        # self.redraw = redraw
        # TODO: not redrawing atm, so I'm setting it to an irrational number
        self.redraw = math.pi
        self.deterministic_eval = deterministic_eval
        self.eps = eps  # Stabilizer for softmax kernel

        # Make a buffer for storing the sampled projection_matrix
        self.register_buffer("projection_matrix", torch.zeros(self.query_dims, self.n_features,
                                                              **factory_kwargs))
        self.factory_kwargs = factory_kwargs
        self._calls = -1

    def new_feature_map(self, device):
        # If we are not training skip the generation of a new feature map
        if self.deterministic_eval and not self.training:
            return

        # Only redraw the new feature map every self.redraw times
        self._calls += 1
        if (self._calls % self.redraw) != 0:
            return

        # We use the cosh feature map so the number of rows is halved
        nb_rows = self.n_features if not self.cosh else self.n_features // 2
        projection_matrix = gaussian_orthogonal_random_matrix(nrows=nb_rows,
                                                              ncols=self.query_dims,
                                                              scaling=self.ortho_scaling,
                                                              device=device,
                                                              dtype=self.factory_kwargs['dtype'])
        self.register_buffer("projection_matrix", projection_matrix)

    def forward_queries(self, x, return_log=False):
        return softmax_kernel(x, projection_matrix=self.projection_matrix, is_query=True,
                              softmax_temp=self.softmax_temp, eps=self.eps, cosh=self.cosh,
                              return_log=return_log)

    def forward_keys(self, x, return_log=False):
        return softmax_kernel(x, projection_matrix=self.projection_matrix, is_query=False,
                              softmax_temp=self.softmax_temp, eps=self.eps, cosh=self.cosh,
                              return_log=return_log)
    

@contextmanager
def null_context():
    yield


def linear_attention_normalization(q, k, causal=False):
    if not causal:
        return torch.einsum('...nm,...m->...n', q, k.sum(dim=-2))
    else:
        return torch.einsum('...nm,...nm->...n', q, k.cumsum(dim=-2))


# efficient causal linear attention, created by EPFL
def causal_linear_attention(q, k, v, need_weights=False):
    from fast_transformers.causal_product import causal_dot_product
    autocast_enabled = torch.is_autocast_enabled()
    is_half = isinstance(q, torch.cuda.HalfTensor)
    assert not is_half or APEX_AVAILABLE, 'half tensors can only be used if nvidia apex is available'
    cuda_context = null_context if not autocast_enabled else partial(autocast, enabled = False)
    causal_dot_product_fn = amp.float_function(causal_dot_product) if is_half else causal_dot_product
    with cuda_context():
        if autocast_enabled:
            q, k, v = map(lambda t: t.float(), (q, k, v))
        q_k_v = causal_dot_product_fn(q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1)).squeeze(1)
        if need_weights:
            attn = torch.einsum('...im,...jm', q, k)
            causal_mask = torch.triu(torch.ones(q.shape[-2], k.shape[-2], dtype=torch.bool,
                                                device=k.device), diagonal=1)
            attn.masked_fill_(causal_mask, 0.0)
        else:
            attn = None
    return q_k_v, attn


# non-causal linear attention
def linear_attention(q, k, v, need_weights=False):
    k_v = torch.einsum('...nm,...nd->...md', k, v)
    q_k_v = torch.einsum('...nm,...md->...nd', q, k_v)
    attn = None if not need_weights else torch.einsum('...im,...jm->...ij', q, k)
    return q_k_v, attn


class SBLocalAttention(nn.Module):
    """Implement fast local attention where a query can only attend to
    neighboring keys.
    In this attention module the query Q_i can only attend to a key K_j if
    |i-j| < local_context/2.
    Arguments
    ---------
        local_context: The neighborhood to consider for local attention.
        softmax_temp: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.1)
    """
    def __init__(self, local_context, dim_heads, nb_features=None, ortho_scaling=0,
                 causal=False, softmax_temp=None, attention_dropout=0.0, softmax_eps=0.0,
                 device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.feature_map = SBPerformerFeatures(dim_heads, nb_features, ortho_scaling=ortho_scaling,
                                               softmax_temp=softmax_temp, eps=softmax_eps,
                                               **factory_kwargs)
        self.local_context = local_context
        self.causal = causal
        self.softmax_temp = softmax_temp
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax_eps = softmax_eps

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None, need_weights=False,
                return_attn_unnormalized=False):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            query: (B, T, H, E) The tensor containing the query
            key: (B, S, H, E) The tensor containing the key
            value: (B, S, H, D) The tensor containing the value
            attn_mask: An implementation of BaseMask that encodes where each
                       query can attend to
            key_padding_mask: An implementation of BaseMask that encodes how
                         many queries each sequence in the batch consists of
        """
        # Extract some shapes and compute the temperature
        B, T, H, E = query.shape
        _, S, _, D = value.shape
        softmax_temp = self.softmax_temp or 1 / math.sqrt(E)

        # TODO: check causal
        if attn_mask is None:
            attn_mask_additive_matrix = torch.zeros(T, S, device=query.device)
        else:
            attn_mask_additive_matrix = attn_mask.additive_matrix_finite
        if key_padding_mask is None:
            key_padding_mask_lengths = torch.full(size=(B,), fill_value=S, dtype=torch.long,
                                                  device=key.device)
        else:
            key_padding_mask_lengths = key_padding_mask.lengths

        # Permute the dimensions to BHTE instead of BTHE
        query = rearrange(query, 'b t h e -> b h t e').contiguous()
        key = rearrange(key, 'b s h e -> b h s e').contiguous()
        value = rearrange(value, 'b s h d -> b h s d').contiguous()

        self.feature_map.new_feature_map(query.device)
        q_prime, q_prime_log_scale = self.feature_map.forward_queries(query)
        k_prime, k_prime_log_scale = self.feature_map.forward_keys(key)

        prime_log_scale = q_prime_log_scale + k_prime_log_scale
        m = q_prime.shape[-1]
        if key_padding_mask is not None and not key_padding_mask.all_ones:
            k_prime = k_prime.masked_fill(~rearrange(key_padding_mask.bool_matrix,
                                                     'b s -> b 1 s 1'), 0.0)
        attn_fn = linear_attention if not self.causal else causal_linear_attention
        q_prime_k_prime_1 = linear_attention_normalization(q_prime, k_prime, causal=self.causal)
        q_prime_k_prime_v, attn_prime = attn_fn(q_prime, k_prime, value, need_weights=need_weights)

        QK = softmax_temp * local_dot_product(
            query, key, attn_mask_additive_matrix, key_padding_mask_lengths,
            self.local_context
        )
        dots_prime = local_dot_product(
            q_prime, k_prime, attn_mask_additive_matrix, key_padding_mask_lengths,
            self.local_context
        )
        # local_dot_product fills in -1e24 for invalid locations. We want to set them to zero.
        # dots_prime[dots_prime <= -1e24] = 0.0
        i = rearrange(torch.arange(T, device=query.device), 't -> 1 1 t 1')
        j = torch.arange(self.local_context, device=query.device)
        local_idx = i - self.local_context // 2 + j
        valid_idx_mask = ((local_idx >= 0)
                          & (local_idx < rearrange(key_padding_mask_lengths, 'b -> b 1 1 1')))
        dots_prime.masked_fill_(~valid_idx_mask, 0.0)
        assert torch.all(dots_prime >= 0)

        # Compute the normalization first
        QK_lse = torch.logsumexp(QK, dim=-1, keepdim=True)
        dots_prime_sum = dots_prime.sum(dim=-1, keepdim=True)
        lr_log_normalization = torch.log((rearrange(q_prime_k_prime_1, 'b h s -> b h s 1')
                                          - dots_prime_sum).clamp_min_(1e-24)) + prime_log_scale
        log_normalization = torch.logaddexp(QK_lse, lr_log_normalization)

        prime_scale = torch.exp(prime_log_scale - log_normalization)
        # When we drop out, we want that location in the attn matrix to be zero.
        # So here we dropout just torch.exp(QK) and leave -dots_prime, so that when we add it back
        # to attn_prime it's equivalent to setting that location to zero.
        dots = self.dropout(torch.exp(QK - log_normalization)) - dots_prime * prime_scale

        out_local = local_weighted_average(dots, value)
        out = out_local + q_prime_k_prime_v * prime_scale

        attn = None
        if need_weights:
            attn_local = torch.zeros(B, H, T, S, device=query.device)
            k = torch.arange(S, device=key.device)
            idx = k - i
            local_mask = ((idx >= -(self.local_context // 2))
                          & (idx < (self.local_context + 1) // 2)
                          & (k < rearrange(key_padding_mask_lengths, 'b -> b 1 1 1')))
            attn_local.masked_scatter_(local_mask, dots.masked_select(valid_idx_mask))
            attn = attn_local + attn_prime * prime_scale
            if return_attn_unnormalized:  # For testing purpose
                attn = (attn, attn * torch.exp(log_normalization),
                        attn_prime * torch.exp(prime_log_scale))
        return rearrange(out, 'b h t d -> b t h d'), attn