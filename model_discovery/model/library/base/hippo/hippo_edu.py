"""Definitions of A and B matrices for various HiPPO operators.

### HiPPO

The original HiPPO paper produced three main methods.
1. LegT is the same as the prior method [LMU (Legendre Memory Unit)](https://proceedings.neurips.cc/paper/2019/file/952285b9b7e7a1be5aa7849f32ffff05-Paper.pdf). It defines specific formulas for A and B matrices in a time-invariant ODE $x'(t) = Ax(t) + Bu(t)$.
2. LagT is another time-invariant ODE $x'(t) = Ax(t) + Bu(t)$ meant to memorize history according to a different weighting function.
3. LegS was the main new method which produces a time-varying ODE $x'(t) = 1/t Ax(t) + 1/t Bu(t)$ meant to memorize history according to a uniform measure.

These methods were incorporated into a simple RNN called HiPPO-RNN where the measures $(A, B)$ were non-trainable.

### HiPPO
The core HiPPO methods are just a set of equations and not end-to-end models.
The specific matrices are implemented in [[/src/models/hippo/hippo.py](/src/models/hippo/hippo.py)].
The connection between HiPPO/S4 matrices $(A, B)$ and convolution kernels is illustrated in [[/notebooks/ssm_kernels.ipynb](/notebooks/ssm_kernels.ipynb)].
The *online function reconstruction* theory is illustrated in [[/notebooks/hippo_function_approximation.ipynb](/notebooks/hippo_function_approximation.ipynb)].
The animation code can also be found in a [.py file](/src/models/hippo/visualizations.py) instead of notebook.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import special as ss
from einops import rearrange, repeat

contract = torch.einsum


def embed_c2r(A):
    A = rearrange(A, '... m n -> ... m () n ()')
    A = np.pad(A, ((0, 0), (0, 1), (0, 0), (0, 1))) + \
        np.pad(A, ((0, 0), (1, 0), (0, 0), (1,0)))
    return rearrange(A, 'm x n y -> (m x) (n y)')

# TODO take in 'torch' option to return torch instead of numpy, and converts the shape of B from (N, 1) to (N)
# TODO remove tlagt
def transition(measure, N, **measure_args):
    """A, B transition matrices for different measures.

    measure: the type of measure
      legt - Legendre (translated)
      legs - Legendre (scaled)
      glagt - generalized Laguerre (translated)
      lagt, tlagt - previous versions of (tilted) Laguerre with slightly different normalization
    """
    # Laguerre (translated)
    if measure == 'lagt':
        b = measure_args.get('beta', 1.0)
        A = np.eye(N) / 2 - np.tril(np.ones((N, N)))
        B = b * np.ones((N, 1))
    elif measure == 'tlagt':
        # beta = 1 corresponds to no tilt
        b = measure_args.get('beta', 1.0)
        A = (1.-b)/2 * np.eye(N) - np.tril(np.ones((N, N)))
        B = b * np.ones((N, 1))
    # Generalized Laguerre
    # alpha 0, beta small is most stable (limits to the 'lagt' measure)
    # alpha 0, beta 1 has transition matrix A = [lower triangular 1]
    elif measure == 'glagt':
        alpha = measure_args.get('alpha', 0.0)
        beta = measure_args.get('beta', 0.01)
        A = -np.eye(N) * (1 + beta) / 2 - np.tril(np.ones((N, N)), -1)
        B = ss.binom(alpha + np.arange(N), np.arange(N))[:, None]

        L = np.exp(.5 * (ss.gammaln(np.arange(N)+alpha+1) - ss.gammaln(np.arange(N)+1)))
        A = (1./L[:, None]) * A * L[None, :]
        B = (1./L[:, None]) * B * np.exp(-.5 * ss.gammaln(1-alpha)) * beta**((1-alpha)/2)
    # Legendre (translated)
    elif measure == 'legt':
        Q = np.arange(N, dtype=np.float64)
        R = (2*Q + 1) ** .5
        j, i = np.meshgrid(Q, Q)
        A = R[:, None] * np.where(i < j, (-1.)**(i-j), 1) * R[None, :]
        B = R[:, None]
        A = -A

        # Halve again for timescale correctness
        A *= 0.5
        B *= 0.5
    # LMU: equivalent to LegT up to normalization
    elif measure == 'lmu':
        Q = np.arange(N, dtype=np.float64)
        R = (2*Q + 1)[:, None] # / theta
        j, i = np.meshgrid(Q, Q)
        A = np.where(i < j, -1, (-1.)**(i-j+1)) * R
        B = (-1.)**Q[:, None] * R
    # Legendre (scaled)
    elif measure == 'legs':
        q = np.arange(N, dtype=np.float64)
        col, row = np.meshgrid(q, q)
        r = 2 * q + 1
        M = -(np.where(row >= col, r, 0) - np.diag(q))
        T = np.sqrt(np.diag(2 * q + 1))
        A = T @ M @ np.linalg.inv(T)
        B = np.diag(T)[:, None]
        B = B.copy() # Otherwise "UserWarning: given NumPY array is not writeable..." after torch.as_tensor(B)
    elif measure == 'legsd':
        q = np.arange(N, dtype=np.float64)
        col, row = np.meshgrid(q, q)
        r = 2 * q + 1
        M = -(np.where(row >= col, r, 0) - np.diag(q))
        T = np.sqrt(np.diag(2 * q + 1))
        A = T @ M @ np.linalg.inv(T)
        B = np.diag(T)[:, None]
        B = B.copy() # Otherwise "UserWarning: given NumPY array is not writeable..." after torch.as_tensor(B)
        A += .5 * B*B[None, :, 0]
        B = B / 2.0
    elif measure in ['fourier_diag', 'foud']:
        freqs = np.arange(N//2)
        d = np.stack([freqs, np.zeros(N//2)], axis=-1).reshape(-1)[:-1]
        A = 2*np.pi*(-np.diag(d, 1) + np.diag(d, -1))
        A = A - .5 * np.eye(N)
        B = np.zeros(N)
        B[0::2] = 2**.5
        B[0] = 1
        B = B[:, None]
    elif measure in ['fourier', 'fout']:
        freqs = np.arange(N//2)
        d = np.stack([np.zeros(N//2), freqs], axis=-1).reshape(-1)[1:]
        A = np.pi*(-np.diag(d, 1) + np.diag(d, -1))
        B = np.zeros(N)
        B[0::2] = 2**.5
        B[0] = 1

        # Subtract off rank correction - this corresponds to the other endpoint u(t-1) in this case
        A = A - B[:, None] * B[None, :]
        B = B[:, None]
    elif measure == 'fourier_decay':
        freqs = np.arange(N//2)
        d = np.stack([np.zeros(N//2), freqs], axis=-1).reshape(-1)[1:]
        A = np.pi*(-np.diag(d, 1) + np.diag(d, -1))
        B = np.zeros(N)
        B[0::2] = 2**.5
        B[0] = 1

        # Subtract off rank correction - this corresponds to the other endpoint u(t-1) in this case
        A = A - .5 * B[:, None] * B[None, :]
        B = .5 * B[:, None]
    elif measure == 'fourier2': # Double everything: orthonormal on [0, 1]
        freqs = 2*np.arange(N//2)
        d = np.stack([np.zeros(N//2), freqs], axis=-1).reshape(-1)[1:]
        A = np.pi*(-np.diag(d, 1) + np.diag(d, -1))
        B = np.zeros(N)
        B[0::2] = 2**.5
        B[0] = 1

        # Subtract off rank correction - this corresponds to the other endpoint u(t-1) in this case
        A = A - B[:, None] * B[None, :] * 2
        B = B[:, None] * 2
    elif measure == 'random':
        A = np.random.randn(N, N) / N
        B = np.random.randn(N, 1)
    elif measure == 'diagonal':
        A = -np.diag(np.exp(np.random.randn(N)))
        B = np.random.randn(N, 1)
    else:
        raise NotImplementedError

    return A, B

def rank_correction(measure, N, rank=1, dtype=torch.float):
    """Return low-rank matrix L such that A + L is normal."""

    if measure == 'legs':
        assert rank >= 1
        P = torch.sqrt(.5+torch.arange(N, dtype=dtype)).unsqueeze(0) # (1 N)
    elif measure == 'legt':
        assert rank >= 2
        P = torch.sqrt(1+2*torch.arange(N, dtype=dtype)) # (N)
        P0 = P.clone()
        P0[0::2] = 0.
        P1 = P.clone()
        P1[1::2] = 0.
        P = torch.stack([P0, P1], dim=0) # (2 N)
        P *= 2**(-0.5) # Halve the rank correct just like the original matrix was halved
    elif measure == 'lagt':
        assert rank >= 1
        P = .5**.5 * torch.ones(1, N, dtype=dtype)
    elif measure in ['fourier', 'fout']:
        P = torch.zeros(N)
        P[0::2] = 2**.5
        P[0] = 1
        P = P.unsqueeze(0)
    elif measure == 'fourier_decay':
        P = torch.zeros(N)
        P[0::2] = 2**.5
        P[0] = 1
        P = P.unsqueeze(0)
        P = P / 2**.5
    elif measure == 'fourier2':
        P = torch.zeros(N)
        P[0::2] = 2**.5
        P[0] = 1
        P = 2**.5 * P.unsqueeze(0)
    elif measure in ['fourier_diag', 'foud', 'legsd']:
        P = torch.zeros(1, N, dtype=dtype)
    else: raise NotImplementedError

    d = P.size(0)
    if rank > d:
        P = torch.cat([P, torch.zeros(rank-d, N, dtype=dtype)], dim=0)  # (R N)
    return P

def initial_C(measure, N, dtype=torch.float):
    """Return C that captures the other endpoint in the HiPPO approximation."""

    if measure == 'legt':
        C = (torch.arange(N, dtype=dtype)*2+1)**.5 * (-1)**torch.arange(N)
    elif measure == 'fourier':
        C = torch.zeros(N)
        C[0::2] = 2**.5
        C[0] = 1
    else:
        C = torch.zeros(N, dtype=dtype) # (N)

    return C


def nplr(measure, N, rank=1, dtype=torch.float, diagonalize_precision=True, B_clip=2.0):
    """Constructs NPLR form of HiPPO matrices.

    Returns w, p, q, V, B such that
    (w - p q^*, B) is unitarily equivalent to the original HiPPO A, B by the matrix V
    i.e. A = V[w - p q^*]V^*, B = V B

    measure: Name of HiPPO method.
    N: Size of recurrent A matrix (also known as `d_state` elsewhere).
    dtype: Single or double precision.
    diagonalize_precision: Calculate diagonalization in double precision.
    B_clip: Clip values of B, can help with stability. None for no clipping.
    """

    assert dtype == torch.float or dtype == torch.double
    cdtype = torch.cfloat if dtype == torch.float else torch.cdouble

    A, B = transition(measure, N)
    A = torch.as_tensor(A, dtype=dtype) # (N, N)
    B = torch.as_tensor(B, dtype=dtype)[:, 0] # (N,)

    P = rank_correction(measure, N, rank=rank, dtype=dtype) # (r N)
    AP = A + torch.sum(P.unsqueeze(-2)*P.unsqueeze(-1), dim=-3)

    # We require AP to be nearly skew-symmetric
    _A = AP + AP.transpose(-1, -2)
    if (err := torch.sum((_A - _A[0,0]*torch.eye(N))**2) / N) > 1e-5: # if not torch.allclose(_A - _A[0,0]*torch.eye(N), torch.zeros(N, N), atol=1e-5):
        print("WARNING: HiPPO matrix not skew symmetric", err)


    # Take advantage of identity + skew-symmetric form to calculate real and imaginary parts separately
    # Imaginary part can use eigh instead of eig
    W_re = torch.mean(torch.diagonal(AP), -1, keepdim=True)

    # Diagonalize in double precision
    if diagonalize_precision: AP = AP.to(torch.double)
    # w, V = torch.linalg.eig(AP) # (..., N) (..., N, N)
    W_im, V = torch.linalg.eigh(AP*-1j) # (..., N) (..., N, N)
    if diagonalize_precision: W_im, V = W_im.to(cdtype), V.to(cdtype)
    W = W_re + 1j * W_im
    # Check: V W V^{-1} = A
    # print("check", V @ torch.diag_embed(W) @ V.conj().transpose(-1, -2))


    # Only keep half of each conjugate pair
    _, idx = torch.sort(W.imag)
    W_sorted = W[idx]
    V_sorted = V[:, idx]

    # There is an edge case when eigenvalues can be 0, which requires some machinery to handle
    # We use a huge hack here: Assume only one pair is 0, and that it is the first row/column of A (only happens in Fourier case)
    V = V_sorted[:, :N//2]
    W = W_sorted[:N//2]  # Only keep negative imaginary components
    assert W[-2].abs() > 1e-4, "Only 1 zero eigenvalue allowed in diagonal part of A"
    if W[-1].abs() < 1e-4:
        V[:, -1] = 0.
        V[0, -1] = 2**-0.5
        V[1, -1] = 2**-0.5 * 1j

    _AP = V @ torch.diag_embed(W) @ V.conj().transpose(-1, -2)
    if ((err := torch.sum((2*_AP.real-AP)**2)/N) > 1e-5):
        print("Warning: Diagonalization of A matrix not numerically precise - error", err)
    # print("check", V @ torch.diag_embed(W) @ V.conj().transpose(-1, -2))

    V_inv = V.conj().transpose(-1, -2)

    # C = initial_C(measure, N, dtype=dtype)
    B = contract('ij, j -> i', V_inv, B.to(V)) # V^* B
    # C = contract('ij, j -> i', V_inv, C.to(V)) # V^* C
    P = contract('ij, ...j -> ...i', V_inv, P.to(V)) # V^* P

    if B_clip is not None:
        B = B.real + 1j*torch.clamp(B.imag, min=-B_clip, max=B_clip)

    # W represents the imaginary part of the DPLR form: A = W - PP^*
    # Downstream classes just call this A for simplicity,
    # which is also more consistent with the diagonal case
    return W, P, B, V