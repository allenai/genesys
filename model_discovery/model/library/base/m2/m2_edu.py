# Copyright (c) 2023, Dan Fu and Simran Arora.
# Adapted from https://github.com/HazyResearch/safari/blob/main/src/models/sequence/hyena.py

import torch.nn as nn
from einops import rearrange
import opt_einsum as oe

import math

import torch
import torch.nn.functional as F

contract = oe.contract

from src.utils.train import OptimModule


def fftconv_ref(u_variable, k, D_variable, dropout_mask, gelu=True, k_rev=None, flashfft=None):
    # u.shape:   B H L
    seqlen = u_variable.shape[-1]

    if flashfft is not None:
        y = flashfft(u_variable.to(dtype=torch.bfloat16).contiguous(), k)
    else:
        fft_size = 2 * seqlen
        k_f = torch.fft.rfft(k, n=fft_size) / fft_size
        if k_rev is not None:
            k_rev_f = torch.fft.rfft(k_rev, n=fft_size) / fft_size
            k_f = k_f + k_rev_f.conj()
        u_f = torch.fft.rfft(u_variable.to(dtype=k.dtype), n=fft_size)

        if len(u_variable.shape) > 3:
            k_f = k_f.unsqueeze(1)

        y = torch.fft.irfft(u_f * k_f, n=fft_size, norm="forward")[..., :seqlen]

    out = y + u_variable * D_variable

    if gelu:
        out = F.gelu(out)
    if dropout_mask is not None:
        return (out * rearrange(dropout_mask, "b H -> b H 1")).to(dtype=u_variable.dtype)
    else:
        return out.to(dtype=u_variable.dtype)


@torch.jit.script
def mul_sum(q, y):
    return (q * y).sum(dim=1)


class Sin(nn.Module):
    def __init__(self, dim, w=10, w_mod=1, train_freq=True):
        super().__init__()

        init_tensor = torch.ones(1, dim)
        self.freq = (
            nn.Parameter(w * init_tensor)
            if train_freq
            else w * torch.ones(1, dim)
        )
        self.w_mod = w_mod

    def forward(self, x):
        return torch.sin(self.w_mod * self.freq * x)


class PositionalEmbedding(OptimModule):
    def __init__(self, emb_dim: int, seq_len: int, lr_pos_emb: float = 1e-5, **kwargs):
        """Complex exponential positional embeddings for Hyena filters."""
        super().__init__()

        self.seq_len = seq_len
        # The time embedding fed to the filteres is normalized so that t_f = 1
        t = torch.linspace(0, 1, self.seq_len)[None, :, None]  # 1, L, 1

        if emb_dim > 1:
            bands = (emb_dim - 1) // 2
        # To compute the right embeddings we use the "proper" linspace
        t_rescaled = torch.linspace(0, seq_len - 1, seq_len)[None, :, None]
        w = 2 * math.pi * t_rescaled / seq_len  # 1, L, 1

        f = torch.linspace(1e-4, bands - 1, bands)[None, None]
        z = torch.exp(-1j * f * w)
        z = torch.cat([t, z.real, z.imag], dim=-1)
        self.register("z", z, lr=lr_pos_emb)
        self.register("t", t, lr=0.0)

    def forward(self, L):
        return self.z[:, :L], self.t[:, :L]


class ExponentialModulation(OptimModule):
    def __init__(
        self,
        d_model,
        fast_decay_pct=0.3,
        slow_decay_pct=1.5,
        target=1e-2,
        modulation_lr=0.0,
        shift: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.shift = shift
        max_decay = math.log(target) / fast_decay_pct
        min_decay = math.log(target) / slow_decay_pct
        deltas = torch.linspace(min_decay, max_decay, d_model)[None, None]
        self.register("deltas", deltas, lr=modulation_lr)

    def forward(self, t, x):
        decay = torch.exp(-t * self.deltas.abs())
        x = x * (decay + self.shift)
        return x


class HyenaFilter(OptimModule):
    def __init__(
        self,
        d_model,
        emb_dim=3,  # dim of input to MLP, augments with positional encoding
        order=16,  # width of the implicit MLP
        seq_len=1024,
        lr=1e-3,
        lr_pos_emb=1e-5,
        dropout=0.0,
        w=1,  # frequency of periodic activations
        w_mod=1, # non-learnable modification of w
        wd=0,  # weight decay of kernel parameters
        bias=True,
        num_inner_mlps=2,
        linear_mixer=False,
        modulate: bool = True,
        normalized=False,
        bidirectional=False,
        **kwargs,
    ):
        """
        Implicit long filter with modulation.

        Args:
            d_model: number of channels in the input
            emb_dim: dimension of the positional encoding (`emb_dim` - 1) // 2 is the number of bands
            order: width of the FFN
            num_inner_mlps: number of inner linear layers inside filter MLP

        Note:
            filter_dropout is not implemented
        """
        super().__init__()
        
        self.d_model=d_model
        self.emb_dim=emb_dim
        self.seq_len=seq_len
        self.modulate=modulate
        self.use_bias = bias
        self.bidirectional = bidirectional

        self.bias = nn.Parameter(torch.randn(self.d_model))
        self.dropout = nn.Dropout(dropout)

        act = Sin(dim=order, w=w, w_mod=w_mod)
        assert (
            emb_dim % 2 != 0 and emb_dim >= 3
        ), "emb_dim must be odd and greater or equal to 3 (time, sine and cosine)"
        self.pos_emb = PositionalEmbedding(emb_dim, seq_len, lr_pos_emb)

        # uses a variable number of inner linear layers
        if linear_mixer is False:
            self.implicit_filter = nn.Sequential(
                nn.Linear(emb_dim, order),
                act,
            )
            for i in range(num_inner_mlps):
                self.implicit_filter.append(nn.Linear(order, order))
                self.implicit_filter.append(act)
            self.implicit_filter.append(nn.Linear(order, d_model, bias=False))
        else:
            self.implicit_filter = nn.Sequential(
                nn.Linear(emb_dim, d_model, bias=False),
            )

        if self.bidirectional:
            self.implicit_filter_rev = nn.Sequential(
                nn.Linear(emb_dim, order),
                act,
            )
            for i in range(num_inner_mlps):
                self.implicit_filter_rev.append(nn.Linear(order, order))
                self.implicit_filter_rev.append(act)
            self.implicit_filter_rev.append(nn.Linear(order, d_model, bias=False))

        self.modulation = ExponentialModulation(d_model, **kwargs)

        self.normalized = normalized
        for c in self.implicit_filter.children():
            for name, v in c.state_dict().items():
                optim = {"weight_decay": wd, "lr": lr}
                setattr(getattr(c, name), "_optim", optim)
        
        self.flashfft = None

    def filter(self, L, *args, **kwargs):
        z, t = self.pos_emb(L)
        h = self.implicit_filter(z)
        if self.modulate:
            h = self.modulation(t, h)
        if self.normalized:
            h = h / torch.norm(h, dim=-1, p=1, keepdim=True)
        return h
    
    def filter_rev(self, L, *args, **kwargs):
        z, t = self.pos_emb(L)
        h = self.implicit_filter_rev(z)
        if self.modulate:
            h = self.modulation(t, h)
        if self.normalized:
            h = h / torch.norm(h, dim=-1, p=1, keepdim=True)
        return h

    def forward(self, x, L, k_fwd=None, k_rev=None, bias=None, *args, **kwargs):
        if k_fwd is None:
            k_fwd = self.filter(L)
            if self.bidirectional and k_rev is None:
                k_rev = self.filter_rev(L)

        # Ensure compatibility with filters that return a tuple
        k_fwd = k_fwd[0] if type(k_fwd) is tuple else k_fwd
        if bias is None:
            bias = self.bias
        bias = bias if self.use_bias else 0 * bias

        if self.bidirectional:
            k_rev = k_rev[0] if type(k_rev) is tuple else k_rev
            k = F.pad(k_fwd, (0, L)) \
                      + F.pad(k_rev.flip(-1), (L, 0))
        else:
            k = k_fwd

        
        y = fftconv_ref(
            x, 
            k, 
            bias, 
            dropout_mask=None,
            gelu=False,
            flashfft=self.flashfft,
        )

        return y.to(dtype=x.dtype)

class MonarchMixerSequenceMixing(nn.Module):
    def __init__(
        self,
        d_model,
        l_max=128,
        dropout=0.0,
        hyena_kernel_lr=None,
        bidirectional=False,
        hyena_lr_pos_emb=1e-5,
        hyena_w=10,
        hyena_w_mod=1,
        hyena_wd=0.1,
        hyena_emb_dim=3,
        hyena_filter_dropout=0.0,
        hyena_filter_order=16,
        residual_long_conv=False,
        hyena_training_additions=False,
    ):
        super().__init__()

        self.d_model = d_model
        self.l_max = l_max
        self.kernel_lr = hyena_kernel_lr
        self.channels = 1
        self.bidirectional = bidirectional
        self.residual_long_conv = residual_long_conv
        self.NUM_PROJECTIONS = 3

        print('-- Bidirectional:', self.bidirectional)
        print("-- Using Long Conv Residual:", self.residual_long_conv)
        print('-- Hyena w:', hyena_w)
        print('-- Hyena w mod:', hyena_w_mod)
        print(f"-- Hyena filter order: {hyena_filter_order}")
        print(f"-- Hyena filter dropout: {hyena_filter_dropout}")
        print(f"-- Hyena filter wd: {hyena_wd}")
        print(f"-- Hyena filter emb dim: {hyena_emb_dim}")
        print(f"-- Hyena filter lr: {hyena_kernel_lr}")
        print(f"-- Hyena filter lr pos emb: {hyena_lr_pos_emb}")

        self.filter_fn = HyenaFilter(
            self.d_model,
            order=hyena_filter_order,
            seq_len=self.l_max,
            dropout=hyena_filter_dropout,
            bidirectional=self.bidirectional,
            lr=hyena_kernel_lr,
            lr_pos_emb=hyena_lr_pos_emb,
            w=hyena_w,  # frequency of periodic activations
            w_mod=hyena_w_mod,
            wd=hyena_wd,  # weight decay of kernel parameters
            emb_dim=hyena_emb_dim,
        )
        
        if self.residual_long_conv:
            self.filter_fn2 = HyenaFilter(
                self.d_model,
                order=hyena_filter_order,
                seq_len=self.l_max,
                dropout=hyena_filter_dropout,
                bidirectional=self.bidirectional,
                lr=hyena_kernel_lr,
                lr_pos_emb=hyena_lr_pos_emb,
                w=hyena_w,  # frequency of periodic activations
                w_mod=hyena_w_mod,
                wd=hyena_wd,  # weight decay of kernel parameters
                emb_dim=hyena_emb_dim,
            )
        
        # setup projections
        self.in_linear = nn.Linear(d_model, 3 * d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        self.hyena_training_additions = hyena_training_additions
        if self.hyena_training_additions:
            self.act = nn.Identity()
            self.drop = nn.Dropout(dropout)
            self.layernorm = nn.LayerNorm(d_model)
        
        # setup short conv
        total_width = self.d_model * self.NUM_PROJECTIONS
        self.short_filter = nn.Conv1d(
            in_channels=total_width,
            out_channels=total_width,
            kernel_size=3,
            groups=total_width,
            padding=2,
        )


    def forward(self, u, **kwargs):
        # u is B L H
        if self.hyena_training_additions:
            u = self.layernorm(u)
        L = u.size(-2)

        # in projection
        u_orig = u
        u = self.in_linear(u)
        u = rearrange(u, "b l d -> b d l")
        
        # short filter
        uc = self.short_filter(u)[..., :L]

        x1, x2, v = uc.split(self.d_model, dim=1)
        
        v = v * x1
        if self.hyena_training_additions:
            v = self.drop(v)

        k = self.filter_fn.filter(L, device=u.device)
        k = rearrange(k, "c l d -> c d l")[0] # `c` is always 1 by default

        if self.bidirectional:
            k_rev = self.filter_fn.filter_rev(L, device=u.device)
            k_rev = rearrange(k_rev, "c l d -> c d l")[0] # `c` is always 1 by default
        else:
            k_rev = None

        y = self.filter_fn(v, L, k_fwd=k, k_rev=k_rev, bias= self.filter_fn.bias[None, :, None])

        if self.residual_long_conv:
            k2 = self.filter_fn2.filter(L, device=u.device)
            k2 = rearrange(k2, "c l d -> c d l")[0]

            if self.bidirectional:
                k2_rev = self.filter_fn2.filter_rev(L, device=u.device)
                k2_rev = rearrange(k2_rev, "c l d -> c d l")[0] # `c` is always 1 by default
            else:
                k2_rev = None                

            yu = self.filter_fn2(u_orig.transpose(-1, -2), L, k_fwd=k2, k_rev=k2_rev, bias= self.filter_fn2.bias[None, :, None])
        
        # post gating
        y = y * x2

        if self.residual_long_conv:
            y = y + yu

        y = y.transpose(-1, -2)
        if self.hyena_training_additions:
            y = self.drop(self.act(y))
        y = self.out_linear(y)

        return y, None

 