"""RFA cuda.

Einsum notations:
    b: bsz
    s: seq_len
    n: num_layers
    h: num_heads
    k: proj_dim
    d: head_dim
"""


import torch


EPS = 1.0


def reverse_cumsum(x, dim):
    return torch.flip(torch.cumsum(torch.flip(x, [dim]), dim), [dim])


def rfa_debug(q, k, v):
    """
    Args:
        q: [tgt_len, bsz * num_heads, proj_dim]
        k: [tgt_len, bsz * num_heads, proj_dim]
        v: [tgt_len, bsz * num_heads, head_dim]

    Return:
        attn: [tgt_len, bsz * num_heads, head_dim]
    """
    s = torch.einsum("tbk,tbd->tbkd", k, v)
    s = torch.cumsum(s, dim=0)
    qs = torch.einsum("tbkd,tbk->tbd", s, q)

    z = torch.cumsum(k, dim=0)
    qz = torch.einsum("tbk,tbk->tb", q, z).clamp_min(EPS)
    attn = qs / qz.unsqueeze(-1)
    return attn


class RFA(torch.autograd.Function):

    @staticmethod
    def forward_torch(q, k, v):
        """
        Args:
            q: [tgt_len, bsz * num_heads, proj_dim]
            k: [tgt_len, bsz * num_heads, proj_dim]
            v: [tgt_len, bsz * num_heads, head_dim]

        Return:
            attn: [tgt_len, bsz * num_heads, head_dim]
        """
        s = torch.einsum("tbk,tbd->tbkd", k, v)
        s = torch.cumsum(s, dim=0)
        qs = torch.einsum("tbkd,tbk->tbd", s, q)

        z = torch.cumsum(k, dim=0)
        qz = torch.einsum("tbk,tbk->tb", q, z).clamp_min(EPS)
        attn = qs / qz.unsqueeze(-1)
        return attn

    @staticmethod
    def backward_torch(q, k, v, grad_attn):
        """
        Args:
            grad_attn: [tgt_len, bsz * num_heads, head_dim]
            q: [tgt_len, bsz * num_heads, proj_dim]
            k: [tgt_len, bsz * num_heads, proj_dim]
            v: [tgt_len, bsz * num_heads, head_dim]
        Return:
            grad_q: [tgt_len, bsz * num_heads, proj_dim]
            grad_k: [tgt_len, bsz * num_heads, proj_dim]
            grad_v: [tgt_len, bsz * num_heads, head_dim]
        """
        s = torch.einsum("tbk,tbd->tbkd", k, v)
        s = torch.cumsum(s, dim=0)
        qs = torch.einsum("tbkd,tbk->tbd", s, q)

        z = torch.cumsum(k, dim=0)
        qz = torch.einsum("tbk,tbk->tb", q, z).clamp_min(EPS)

        # [bsz, tgt_len, head_dim]
        grad_qs = grad_attn / qz.unsqueeze(-1)

        grad_qz = torch.einsum("tbd,tbd->tb", grad_attn, qs)
        grad_qz = -grad_qz / (qz ** 2)
        grad_qz = grad_qz * (qz > EPS)

        grad_q = torch.einsum("tbd,tbkd->tbk", grad_qs, s) \
            + grad_qz.unsqueeze(-1) * z

        grad_s = torch.einsum("tbk,tbd->tbkd", q, grad_qs)
        grad_s = reverse_cumsum(grad_s, dim=0)
        grad_k = torch.einsum("tbkd,tbd->tbk", grad_s, v)
        grad_v = torch.einsum("tbkd,tbk->tbd", grad_s, k)

        grad_k = grad_k + reverse_cumsum(q * grad_qz.unsqueeze(-1), dim=0)

        return grad_q, grad_k, grad_v

    @staticmethod
    def forward(ctx, q, k, v):
        """
        Args:
            q: [tgt_len, bsz * num_heads, proj_dim]
            k: [tgt_len, bsz * num_heads, proj_dim]
            v: [tgt_len, bsz * num_heads, head_dim]

        Return:
            attn: [tgt_len, bsz * num_heads, head_dim]
        """
        ctx.save_for_backward(q, k, v)
        attn = RFA.forward_cuda(q, k, v)
        # attn = RFA.forward_torch(q, k, v)
        return attn

    @staticmethod
    def backward(ctx, grad_attn):
        """
        Args:
            q: [tgt_len, bsz * num_heads, proj_dim]
            k: [tgt_len, bsz * num_heads, proj_dim]
            v: [tgt_len, bsz * num_heads, head_dim]
            grad_attn: [tgt_len, bsz * num_heads, head_dim]
        Return:
            grad_q: [tgt_len, bsz * num_heads, proj_dim]
            grad_k: [tgt_len, bsz * num_heads, proj_dim]
            grad_v: [tgt_len, bsz * num_heads, head_dim]
        """
        q, k, v = ctx.saved_tensors
        grad_q, grad_k, grad_v = RFA.backward_cuda(q, k, v, grad_attn)
        # grad_q, grad_k, grad_v = RFA.backward_torch(q, k, v, grad_attn)
        return grad_q, grad_k, grad_v


if __name__ == "__main__":
    device = torch.device("cuda:0")
    dtype = torch.double

    bsz, tgt_len, proj_dim, head_dim = 2, 15, 128, 8
    q = torch.rand(
        (tgt_len, bsz, head_dim),
        device=device, dtype=dtype, requires_grad=True) - 0.5
    k = torch.rand(
        (tgt_len, bsz, head_dim),
        device=device, dtype=dtype, requires_grad=True) - 0.5
    v = torch.rand(
        (tgt_len, bsz, head_dim),
        device=device, dtype=dtype, requires_grad=True)

    res = torch.autograd.gradcheck(
        RFA.apply,
        (q, k, v),
        raise_exception=True)