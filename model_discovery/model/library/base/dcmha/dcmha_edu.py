from einops import rearrange
from torch import einsum, tanh
from torch.nn.functional import gelu

from utils import causal_mask, rmsnorm

# B = batch size; S = key/value len; T = query len
# D_m = model dim; H = num. of heads; D = head dim; R = rank

def dw_proj(
    X, # B * T * D_m
    W_1, # D_m * (H*R*2)
    W_2 # (H*R*2) * (H*R*2)
):
    dw = gelu(X @ W_1) @ W_2
    dw1, dw2 = dw.chunk(2, dim=-1)
    dw1 = rmsnorm(rearrange(dw1, 'BT(RH)->BTRH'), dim=-1)
    dw2 = rearrange(dw2, 'BT(RH)->BTRH')
    return dw1, dw2

def compose(
    a, # B * H * T * S
    Q, # B * T * D_m
    K, # B * S * D_m
    theta
 ):
    W_q1, W_q2, W_k1, W_k2 = theta.W_q1, theta.W_q2, theta.W_k1, theta.W_k2
    W_qg, W_kg = theta.W_qg, theta.W_kg # D_m * H

    dw1, dw2 = dw_proj(Q, W_q1, W_q2)
    h = einsum('BHTS,BTRH->BRTS', a, dw1)
    o_qp = einsum('BRTS,BTRH->BHTS', h, dw2)

    dw1, dw2 = dw_proj(K, W_k1, W_k2)
    h = einsum('BHTS,BSRH->BRTS', a, dw1)
    o_kp = einsum('BRTS,BSRH->BHTS', h, dw2)

    o_qg = einsum('BHTS,BTH->BHTS', a, tanh(Q @ W_qg))
    o_kg = einsum('BHTS,BSH->BHTS', a, tanh(K @ W_kg))
    return a + o_qp + o_kp + o_qg + o_kg

def DCMHA(
    Q, K, V, W_q, W_k, W_v, W_o, causal,
    theta_lc, # params for pre-composition
    theta_pc # params for post-composition
):
    q, k, v = [rearrange(x, 'BT(HD)->BHTD') for x in
        [Q @ W_q, K @ W_k, V @ W_v]]
    logits = einsum('BHTD,BHSD->BHTS', q, k)
    logits = compose(logits, Q, K, theta_lc)
    if causal: logits = causal_mask(logits)
    probs = logits.softmax(-1)
    probs = compose(probs, Q, K, theta_pc)
    o = einsum('BHTS,BHSD->BHTD', probs, v)
    return rearrange(o, 'BHTD->BT(HD)') @ W_o