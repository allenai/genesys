# https://alinush.github.io/2020/03/19/multiplying-a-vector-by-a-toeplitz-matrix.html
# https://stackoverflow.com/questions/69809789/is-there-any-way-to-create-a-tensor-with-a-specific-pattern-in-pytorch
# https://github.com/cheerss/CrossFormer/blob/main/models/crossformer.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from .rpe import Rpe

from .helpers import get_activation_fn, get_norm_fn, print_params



class GLU(nn.Module):
    def __init__(self, d1, d2, act_fun, fina_act="None", dropout=0.0, bias=True):
        super().__init__()
        # get local varables
        params = locals()
        # print params
        print_params(**params)
        
        self.l1 = nn.Linear(d1, d2, bias=bias)
        self.l2 = nn.Linear(d1, d2, bias=bias)
        self.l3 = nn.Linear(d2, d1, bias=bias)
        self.act_fun = get_activation_fn(act_fun)
        self.p = dropout
        if self.p > 0.0:
            self.dropout = nn.Dropout(p=dropout)
        self.fina_act = get_activation_fn(fina_act)

    def forward(self, x):
        o1 = self.l1(x)
        weight = self.act_fun(o1)
        if self.p > 0.0:
            weight = self.dropout(weight)
        o2 = self.l2(x)
        output = weight * o2
        output = self.l3(output)
        output = self.fina_act(output)

        return output
    
class Tno(nn.Module):
    def __init__(
        self, 
        h, 
        dim, 
        rpe_dim, 
        causal=False, 
        use_decay=False, 
        use_multi_decay=False, 
        residual=False, 
        act="relu", 
        par_type=1, 
        gamma=0.999,
        bias=True,
        act_type="none",
        layers=3,
        norm_type="simplermsnorm",
    ):
        super().__init__()
        # get local varables
        params = locals()
        # print params
        print_params(**params)
        
        self.h = h
        self.dim = dim
        self.causal = causal
        self.par_type = par_type
        self.zero_value = 0
        self.use_decay = use_decay
        if self.use_decay:
            self.gamma = nn.Parameter(torch.ones(h, 1, dim) * gamma, requires_grad=False)
        self.use_multi_decay = use_multi_decay
        if self.use_multi_decay:
            self.lambda_ = gamma
            self.gamma = nn.Parameter(torch.randn(h, 1, dim))

        self.rpe = Rpe(
            dim=rpe_dim, 
            outdim=h * dim, 
            residual=residual,
            act=act,
            bias=bias, 
            layers=layers,
            norm_type=norm_type,
        )
        
        if self.causal:
            self.forward = self.forward_causal
        else:
            self.forward = self.forward_non_causal
            
        self.act_fun = get_activation_fn(act_type)

    def get_pos(self, n):
        if self.par_type == 1:
            index = torch.arange(1, 1 + n).reshape(n, -1) * 1.0
        elif self.par_type == 2:
            index = torch.arange(1, 1 + n).reshape(n, -1) * 1.0 / n
        elif self.par_type == 3:
            index = torch.exp(torch.arange(1, 1 + n).reshape(n, -1) * 1.0 / n)
        
        return index
        
    def get_zero(self):
        index = torch.zeros(1).reshape(1, -1) * 1.0
        if self.par_type == 3:
            index = torch.exp(index)
            
        return index

    def get_neg(self, n):
        if self.causal:
            index = torch.ones(self.h * n * self.dim).reshape(self.h, n, self.dim) * self.zero_value
        else:
            if self.par_type == 1:
                index = -torch.arange(1, 1 + n).flip(0).reshape(n, -1) * 1.0
            elif self.par_type == 2:
                index = -torch.arange(1, 1 + n).flip(0).reshape(n, -1) * 1.0 / n

        return index
    
    def rpe_transform(self, x):
        # n, 1 -> n, (d * h)
        res = self.rpe(x)
        # n, (d * h) -> h, n, d
        res = rearrange(res, 'n (h d) -> h n d', h=self.h)

        return res
    
    def forward_causal(self, x, dim=-2, normalize=False):
        # x: b, h, n, d
        n = x.shape[dim]
        # a0, a1, ... , a(n-1), a0, a(-(n-1)), ... , a(-1)
        ##### coef
        # 1, d, 1 -> h, 1, d
        zero = self.rpe_transform(self.get_zero().to(x))
        pos = self.rpe_transform(self.get_pos(n - 1).to(x))

        if self.use_decay or self.use_multi_decay:
            coef = torch.arange(1, n).reshape(1, -1, 1).to(x)
            if self.use_decay:
                gamma = self.gamma
            else:
                gamma = torch.sigmoid(self.gamma)
                gamma = self.lambda_ + (1 - self.lambda_) * gamma
            gamma = gamma ** coef
            pos = gamma * pos
        a = torch.cat([zero, pos, zero], dim=1)
        a = self.act_fun(a)

        # x: b, h, n, d
        # a: h, l, d
        output = self.compute(x, a, dim, n)

        if normalize:
            size = list(x.shape[:-1]) + [1]
            ones = torch.ones(size).to(x)
            denorm = self.compute(ones, a, dim, n)
            output = output / denorm

        return output
        
    def forward_non_causal(self, x, dim=-2, normalize=False):
        # x: b, h, n, d
        n = x.shape[dim]
        # a0, a1, ... , a(n-1), a0, a(-(n-1)), ... , a(-1)
        ##### coef
        # 1, d, 1 -> h, 1, d
        zero = self.rpe_transform(self.get_zero().to(x))
        pos = self.rpe_transform(self.get_pos(n - 1).to(x))
        neg_index = self.get_neg(n - 1).to(x)
        if self.causal:
            neg = neg_index
        else:
            neg = self.rpe_transform(neg_index)

        if self.use_decay or self.use_multi_decay:
            coef = torch.arange(1, n).reshape(1, -1, 1).to(x)
            if self.use_decay:
                gamma = self.gamma
            else:
                gamma = torch.sigmoid(self.gamma)
                gamma = self.lambda_ + (1 - self.lambda_) * gamma
            gamma = gamma ** coef
            pos = gamma * pos
            neg = torch.flip(gamma, dims=[1]) * neg
        a = torch.cat([zero, pos, zero, neg], dim=1)
        a = self.act_fun(a)
        # x: b, h, n, d
        # a: h, l, d
        output = self.compute(x, a, dim, n)

        if normalize:
            size = list(x.shape[:-1]) + [1]
            ones = torch.ones(size).to(x)
            denorm = self.compute(ones, a, dim, n)
            output = output / denorm

        return output
    
    def compute(self, x, a, dim, n):
        # x: b, h, n, d
        # a: h, n, d
        y = torch.fft.rfft(x, 2 * n, dim=dim)
        v = torch.fft.rfft(a, 2 * n, dim=dim).unsqueeze(0)
        u = v * y
        output = torch.fft.irfft(u, 2 * n, dim=dim)[:, :, :n, :]

        return output

    def toeplizt_matrix(self, x, dim):
        assert dim == -2
        # shape of x: b, h, n, d
        n = x.shape[dim]
        # c: first col, r: first row
        # 1, d, 1 -> h, 1, d
        zero = self.rpe_transform(self.get_zero().to(x))
        pos = self.rpe_transform(self.get_pos(n - 1).to(x))
        neg_index = self.get_neg(n - 1).to(x)
        if self.causal:
            neg = neg_index
        else:
            neg = self.rpe_transform(neg_index)
                
        if self.use_decay or self.use_multi_decay:
            coef = torch.arange(1, n).reshape(1, -1, 1)
            if self.use_decay:
                gamma = self.gamma
            else:
                gamma = torch.sigmoid(self.gamma)
                gamma = self.lambda_ + (1 - self.lambda_) * gamma
            gamma = gamma ** coef
            pos = gamma * pos
            neg = torch.flip(gamma, dims=[1]) * neg
        zero = self.act_fun(zero)
        pos = self.act_fun(pos)
        if not self.causal:
            neg = self.act_fun(neg)
        c = torch.cat([zero, pos], dim=-2)
        r = torch.cat([zero, neg.flip(1)], dim=-2)
        vals = torch.cat([r, c[:, 1:].flip(1)], dim=-2)
        n = c.shape[-2]
        shape = self.h, n, n
        i, j = torch.ones(n, n).nonzero().T
        T = vals[:, j - i].reshape(self.h, n, n, -1)

        res = torch.einsum('h n m d, b h m d -> b h n d', T, x)
        return res

class Gtu(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        bias=True,
        act_fun="silu",
        causal=False,
        expand_ratio=3,
        resi_param=False,
        use_norm=False,
        norm_type="simplermsnorm",
        use_decay=False,
        use_multi_decay=False,
        rpe_layers=3,
        rpe_embedding=512,
        rpe_act="relu",
        normalize=False,
        par_type=1,
        residual=False,
        gamma=0.99,
        act_type="none",
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.expand_ratio = expand_ratio
        self.resi_param = resi_param
        self.num_heads = num_heads
        self.normalize = normalize
        
        if self.resi_param:
            self.d = nn.Parameter(torch.randn(embed_dim))

        d1 = int(self.expand_ratio * embed_dim)
        d1 = (d1 // self.num_heads) * self.num_heads
        self.head_dim = d1 // num_heads
        # linear projection
        self.v_proj = nn.Linear(embed_dim, d1, bias=bias)
        self.u_proj = nn.Linear(embed_dim, d1, bias=bias)
        self.o = nn.Linear(d1, embed_dim, bias=bias)
        self.act = get_activation_fn(act_fun)
        # tno
        self.toep = Tno(
            h=num_heads, 
            dim=self.head_dim,
            rpe_dim=rpe_embedding, 
            causal=causal, 
            use_decay=use_decay, 
            use_multi_decay=use_multi_decay,
            residual=residual,
            act=rpe_act,
            par_type=par_type,
            gamma=gamma,
            bias=bias,
            act_type=act_type,
            layers=rpe_layers,
            norm_type=norm_type,
        )
        # norm
        self.norm_type = norm_type
        self.use_norm = use_norm
        if self.use_norm:
            self.norm = get_norm_fn(self.norm_type)(d1)
    
    def forward(self, x):
        # x: b, h, w, d
        num_heads = self.num_heads

        if self.resi_param:
            shortcut = shortcut * self.d
        u = self.act(self.u_proj(x))
        v = self.act(self.v_proj(x))
        # reshape
        v = rearrange(v, 'b n (h d) -> b h n d', h=num_heads)
        output = self.toep(v, dim=-2, normalize=self.normalize)
        output = rearrange(output, 'b h n d -> b n (h d)')
        output = u * output
        if self.use_norm:
            output = self.norm(output)
            
        output = self.o(output)
        
        return output

class TnnLayer(nn.Module):
    def __init__(
        self, 
        dim, 
        num_heads,
        rpe_embedding,
        glu_dim,
        # model params
        prenorm=True,
        norm_type="simplermsnorm",
        # gtu params
        causal=False,
        gtu_act="silu",
        expand_ratio=3,
        use_decay=False,
        gamma=0.999,
        # rpe params
        rpe_act="relu",
        rpe_layers=3,
        # glu params
        glu_act="silu",
    ):
        super().__init__()
        self.token_mixer = Gtu(
            # gtu params
            embed_dim=dim,
            num_heads=num_heads,
            act_fun=gtu_act,
            norm_type=norm_type,
            causal=causal,
            expand_ratio=expand_ratio,
            use_decay=use_decay,
            gamma=gamma,
            # rpe params
            rpe_embedding=rpe_embedding,
            rpe_act=rpe_act,
            rpe_layers=rpe_layers,
        )

        self.token_norm = get_norm_fn(norm_type)(dim)
        self.feature_norm = get_norm_fn(norm_type)(dim)
        
        self.feature_mixer = GLU(
            d1=dim, 
            d2=glu_dim,
            act_fun=glu_act,
        )
        
        if prenorm:
            self.forward = self.forward_prenorm
        else:
            self.forward = self.forward_postnorm

    def forward_postnorm(self, x):
        x = x + self.token_norm(self.token_mixer(x))
        x = x + self.feature_norm(self.feature_mixer(x))

        return x
    
    def forward_prenorm(self, x):
        x = x + self.token_mixer(self.token_norm(x))
        x = x + self.feature_mixer(self.feature_norm(x))

        return x