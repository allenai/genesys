"""SSM convolution kernels.

SSMKernelDPLR is the S4 kernel, implementing the 'diagonal plus low-rank' algorithm from the original S4 paper. This stores parameters A, B, C, dt, and calling it creates the SSM convolution kernel bar{K}.

SSMKernelDiag is the S4D kernel, a simpler algorithm for computing the kernel for the case of diagonal state matrices A.

SSMKernel wraps these with common options and handles the initialization.
"""

from typing import Optional, Mapping, Tuple, Union
from collections import defaultdict
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor # For type hints
import numpy as np
from einops import rearrange, repeat

import src.models.hippo.hippo as hippo
import src.models.sequence.kernels.dplr as dplr
from src.models.functional.krylov import krylov, power
import src.utils.train

log = src.utils.train.get_logger(__name__)

# Try CUDA extension
try:
    from extensions.kernels.vandermonde import log_vandermonde_cuda
    has_cuda_extension = True
    log.info("CUDA extension for structured kernels (Cauchy and Vandermonde multiplication) found.")
except:
    log.warning(
        "CUDA extension for structured kernels (Cauchy and Vandermonde multiplication) not found. Install by going to extensions/kernels/ and running `python setup.py install`, for improved speed and memory efficiency. Note that the kernel changed for state-spaces 4.0 and must be recompiled."
    )
    has_cuda_extension = False

try:
    import pykeops
    from src.models.functional.vandermonde import log_vandermonde as log_vandermonde_keops, log_vandermonde_transpose as log_vandermonde_transpose_keops

    has_pykeops = True
    log.info("Pykeops installation found.")
except ImportError:
    has_pykeops = False
    if not has_cuda_extension:
        log.warning(
            "Falling back on slow Cauchy and Vandermonde kernel. Install at least one of pykeops or the CUDA extension for better speed and memory efficiency."
        )

# Fallback versions
from src.models.functional.vandermonde import log_vandermonde_naive
from src.models.functional.vandermonde import log_vandermonde_transpose_naive

# Base Kernel class
from src.models.sequence.kernels.kernel import Kernel

# Alias torch.einsum; can easily swap to opt_einsum if desired
contract = torch.einsum

_conj = lambda x: torch.cat([x, x.conj()], dim=-1)
_c2r = torch.view_as_real
_r2c = torch.view_as_complex

if tuple(map(int, torch.__version__.split('.')[:2])) >= (1, 10):
    _resolve_conj = lambda x: x.conj().resolve_conj()
else:
    _resolve_conj = lambda x: x.conj()

def inv_transform(param, transform='none'):
    """Initialize a (positive) parameter under a transform."""
    param = torch.clamp(param, min=1e-4)
    if transform == 'none':
        return param
    elif transform == 'exp':
        return torch.log(param) # Some of the HiPPO methods have real part 0
    elif transform == 'relu':
        return param
    elif transform == 'sigmoid':
        return torch.logit(param)
    elif transform == 'softplus':
        return torch.log(torch.exp(param)-1)
    else: raise NotImplementedError

def param_transform(param, transform='none'):
    """Get a (positive) parameter under a transform."""
    if transform == 'none':
        p = param
    elif transform == 'exp':
        p = torch.exp(param)
    elif transform == 'relu':
        # JAX version seems to NaN if you allow 0's, although this code was fine without it
        p = F.relu(param)+1e-4
    elif transform == 'sigmoid':
        p = F.sigmoid(param)
    elif transform == 'softplus':
        p = F.softplus(param)
    else: raise NotImplementedError
    return p


class SSMKernel(Kernel):
    """Parent class for different SSM parameterizations.

    This class is abstract and only defines some initializations and flags that are common to all SSM variants.
    It is instantiated by subclasses SSMKernel{Dense,Real,Diag,DPLR}.

    Options:
    d_state (N): State size (dimensionality of parameters A, B, C). Generally shouldn't need to be adjusted and doens't affect speed much for most kernels (e.g. S4, S4D).
    deterministic: Use a deterministic initialization for dt, A, B, C.
        Useful for debugging as well as constructing a simple exponential decay kernel (e.g. used in S4ND image->video inflation).

    dt_min, dt_max: min and max values for the step size dt
    dt_tie: Keep dt tied across the N dimensions of the state. Although this theoretically makes more sense, models such as S5 and Mega have found slightly improvements by setting it to False.
    dt_transform: Transform function for parameterization of dt (default 'softplus', used to be 'exp')

    rank: Rank of low-rank correction for DPLR mode. Needs to be increased for init "legt".
    n_ssm: Number of independent trainable (A, B) SSMs, e.g.
        `n_ssm=1` means all A/B parameters are tied across the H different instantiations of C.
        `n_ssm=None` means all H SSMs are completely independent.
        Generally, changing this option can save parameters but doesn't affect performance or speed much.
        This parameter must divide H.
    init: Options for initialization of (A, B). For DPLR mode, recommendations are "legs", "fout", "hippo" (combination of both). For Diag mode, recommendations are "diag-inv", "diag-lin", "diag-legs", and "diag" (combination of diag-inv and diag-lin).
    init_args: Extra arguments passed into initialization function (see dplr.py for options).
    """

    def init_dt(self):
        # Generate dt
        if self.deterministic:  # Meant for debugging
            assert self.dt_tie, "Deterministic dt initialization is tied"
            assert self.dt_transform == 'exp', "Deterministic dt transform should be 'exp' for simplicity"
            inv_dt = torch.exp(torch.linspace(math.log(self.dt_min), math.log(self.dt_max), self.H)).unsqueeze(-1) # (H 1)
        else:
            shape = (self.H, 1) if self.dt_tie else (self.H, self.N//2)
            # Initialize log dt
            inv_dt = torch.rand(*shape, dtype=self.dtype) * (
                math.log(self.dt_max) - math.log(self.dt_min)
            ) + math.log(self.dt_min)
            if self.dt_transform != 'exp':
                inv_dt = inv_transform(torch.exp(inv_dt), self.dt_transform)

        return inv_dt

    def init_ssm_real(self):
        """Returns (dense, real) (A, B, C) parameters for init options."""
        # Generate A, B
        A, B = hippo.transition(self.init, self.N)
        A = torch.as_tensor(A, dtype=self.dtype)
        B = torch.as_tensor(B, dtype=self.dtype)[:, 0]
        B = repeat(B, 'n -> v n', v=self.n_ssm).clone().contiguous()
        A = repeat(A, 'n m -> v n m', v=self.n_ssm).clone().contiguous()

        # Generate C
        if self.deterministic:
            C = torch.zeros(self.channels, self.H, self.N, dtype=self.dtype)
            C[..., :1] = 1.0
        else:
            C = torch.randn(self.channels, self.H, self.N, dtype=self.dtype)

        return A, B, C

    def init_ssm_dplr(self):
        """Returns DPLR (A, P, B, C) parameters for init options."""
        A, P, B, V = dplr.combination(self.init, self.N, self.rank, self.n_ssm, **self.init_args)

        # Broadcast C to have H channels
        if self.deterministic:
            C = torch.zeros(self.channels, self.n_ssm, self.N, dtype=self.cdtype)
            C[:, :, :1] = 1.
            C = contract('hmn, chn -> chm', V.conj().transpose(-1, -2), C) # V^* C
            C = repeat(C, 'c t n -> c (v t) n', v=self.H // C.size(-2)).clone().contiguous()
        else:
            C = torch.randn(self.channels, self.H, self.N//2, dtype=self.cdtype)

        # Broadcast other parameters to have n_ssm copies
        assert self.n_ssm % B.size(-2) == 0 \
                and self.n_ssm % P.size(-2) == 0 \
                and self.n_ssm % A.size(-2) == 0

        # Broadcast tensors to n_ssm copies
        # These will be the parameters, so make sure tensors are materialized and contiguous
        B = repeat(B, 't n -> (v t) n', v=self.n_ssm // B.size(-2)).clone().contiguous()
        P = repeat(P, 'r t n -> r (v t) n', v=self.n_ssm // P.size(-2)).clone().contiguous()
        A = repeat(A, 't n -> (v t) n', v=self.n_ssm // A.size(-2)).clone().contiguous()

        # Because these complex parameterizations assume conjugate symmetry,
        # halve the value of self.N for convenience
        self.N //= 2

        return A, P, B, C

    def __init__(
        self,
        # General Kernel arguments for parent class
        d_model: int = 0,
        channels: int = 1,
        l_max: Optional[int] = None,
        lr: Union[float, Optional[Mapping]] = None,
        wd: Union[float, Optional[Mapping]] = 0.0,
        verbose: bool = True,
        # SSM arguments
        d_state: int = 64,
        deterministic: bool = False,
        # dt options
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_tie: bool = True,
        dt_transform: str = 'exp',
        # (A, B, C) options
        rank: int = 1,
        n_ssm: Optional[int] = None,
        measure: Optional[str] = None,
        init: Optional[str] = "legs",
        # Extra hyperparameters for initialization
        **init_args,
    ):
        super().__init__(d_model=d_model, channels=channels, l_max=l_max, lr=lr, wd=wd, verbose=verbose)
        self.N = d_state
        self.dtype, self.cdtype = torch.float, torch.cfloat
        self.deterministic = deterministic
        # dt options
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.dt_tie = dt_tie
        self.dt_transform = dt_transform
        # SSM options (A, B, C)
        self.rank = rank
        self.n_ssm = n_ssm if n_ssm is not None else self.H
        if measure is not None:
            log.warning("Warning: 'measure' option changed to 'init' and will be removed in a future version.")
            assert init is None, "'measure' and 'init' cannot both be passed into SSMKernel"
            init, measure = measure, init
        self.init = init
        self.init_args = init_args

    @torch.no_grad()
    def forward_state(self, u, state):
        """Forward the state through a sequence, i.e. computes the state after passing chunk through SSM

        This is a generic version of this functionality that works for SSMs.
        It is currently used by SSMKernelDense and SSMKernelDPLR.
        This is a suboptimal implementation; it is recommended to use SSMKernelDiag
        if this functionality is desired.

        state: (B, H, N)
        u: (B, H, L)

        Returns: (B, H, N)
        """

        # Construct dA, dB matrices
        dA, dB = self._setup_state() # (H N N) (H N)

        conj = state.size(-1) != dA.size(-1)
        if conj: state = _conj(state)

        v = contract('h n, b h l -> b h n l', dB, u.flip(-1))
        AL, v = power(u.size(-1), dA, v)
        next_state = contract("h m n, b h n -> b h m", AL, state)
        next_state = next_state + v

        if conj: next_state = next_state[..., : next_state.size(-1) // 2]
        return next_state

    def _setup_state(self):
        """Register dA and dB to module."""
        raise NotImplementedError

    @property
    def d_state(self):
        """d_state and state_to_tensor are used by specific decoders.

        These were used in earlier versions and should not be needed in general.
        """
        return self.H * self.N

    @property
    def state_to_tensor(self):
        return lambda state: rearrange('... h n -> ... (h n)', state)


class SSMKernelDiag(SSMKernel):
    """SSM kernel using diagonal state matrix (S4D model).

    Options:
    dt_fast:  (experimental) Parameterize inv_dt under sinh function.
        (Ohno et al. "Fast Saturating Gate for Learning Long Time Scales with RNNs")
    real_transform, imag_transform: ['none' | 'exp' | 'relu' | 'sigmoid' | 'softplus']
        Parameterize the real/imag parts of the diagonal of A under this function.
    bandlimit: Mask high frequencies of the kernel (indices corresponding to
        diagonal elements with large imaginary part). Introduced in S4ND paper.
    backend: ['cuda' | 'keops' | 'naive'] Options for Vandermonde/Cauchy kernel (in order of efficiency).
    is_real : Real-valued SSM; can be interpreted as EMA.
    """

    def __init__(
        self,
        dt_fast: bool = False,
        real_transform: str = 'exp',
        imag_transform: str = 'none',
        bandlimit: Optional[float] = None,
        backend: str = 'cuda',
        is_real: bool = False,
        **kwargs,
    ):
        # Special case: for real-valued, d_state semantics change
        if is_real and 'd_state' in kwargs:
            kwargs['d_state'] = kwargs['d_state'] * 2
        super().__init__(**kwargs)
        self.dt_fast = dt_fast
        self.real_transform = real_transform
        self.imag_transform = imag_transform
        self.bandlimit = bandlimit
        self.backend = backend
        self.is_real = is_real

        # Initialize dt, A, B, C
        inv_dt = self.init_dt()
        A, P, B, C = self.init_ssm_dplr()
        # Note that in the Diag case, P will be ignored
        # The DPLR case subclasses this and uses P
        self.register_params(A, B, C, inv_dt, P)

    def register_params(self, A, B, C, inv_dt, P):
        """Process the initialization into form of trainable parameters.

        A: (S, N) diagonal matrix
        B: (S, N)
        C: (C, H, N)
        dt: (H) timescale per feature

        Dimensions:
        N (or d_state): state size
        H (or d_model): total SSM copies
        S (or n_ssm): number of trainable copies of (A, B, dt); must divide H
        C (or channels): system is 1-dim to C-dim

        The forward pass of this Module returns a tensor of shape (C, H, L)

        Note: tensor shape N here denotes half the true state size, because of conjugate symmetry
        """

        assert self.backend in ['cuda', 'keops', 'naive']

        if self.dt_fast: inv_dt = torch.asinh(inv_dt)

        # Rank of low-rank correction
        assert self.H == inv_dt.size(0)
        assert self.N == A.size(-1) == B.size(-1) == C.size(-1)
        assert self.n_ssm == A.size(-2) == B.size(-2) # Number of independent SSMs trained
        self.repeat = self.H // A.size(0)

        # Check that diagonal part has negative real and imag part
        # (allow some tolerance for numerical precision on real part
        # since it may be constructed by a diagonalization)
        assert torch.all(A.real < 1e-4) and torch.all(A.imag <= 0.0)

        # Broadcast everything to correct shapes
        C = C.expand(torch.broadcast_shapes(C.shape, (1, self.H, self.N))) # (C, H, N)  # TODO originally this was only in DPLR, check safe for Diag
        B = B.unsqueeze(0) # (1, H, N)
        assert self.channels == C.shape[0]

        # Register dt
        self.register("inv_dt", inv_dt, self.lr_dict['dt'], self.wd_dict['dt'])
        # Register ABC
        if self.is_real:
            self.register("C", C.real, self.lr_dict['C'], None)
            self.register("B", B.real, self.lr_dict['B'], self.wd_dict['B'])
            self.register("A_real", inv_transform(-A.real, self.real_transform), self.lr_dict['A'], self.wd_dict['A'])
        else:
            self.register("C", _c2r(_resolve_conj(C)), self.lr_dict['C'], None)
            self.register("B", _c2r(B), self.lr_dict['B'], self.wd_dict['B'])
            self.register("A_real", inv_transform(-A.real, self.real_transform), self.lr_dict['A'], self.wd_dict['A'])
            self.register("A_imag", inv_transform(-A.imag, self.imag_transform), self.lr_dict['A'], self.wd_dict['A'])

    def _get_params(self, rate=1.0):
        """Process the internal parameters."""

        # (S N) where S=n_ssm
        if self.is_real:
            A = -param_transform(self.A_real, self.real_transform)
            B = self.B # (1 S N)
            C = self.C # (C H N)
        else:
            A = -param_transform(self.A_real, self.real_transform) - 1j * param_transform(self.A_imag, self.imag_transform)
            B = _r2c(self.B) # (1 S N)
            C = _r2c(self.C) # (C H N)

        if self.dt_fast: inv_dt = torch.sinh(self.inv_dt)
        else: inv_dt = self.inv_dt
        dt = param_transform(inv_dt, self.dt_transform) * rate # (H N)

        if self.bandlimit is not None:
            freqs = dt / rate * A.imag.abs() / (2*math.pi) # (H N)
            mask = torch.where(freqs < self.bandlimit * .5, 1, 0)
            C = C * mask

        # Incorporate dt into A and B
        A = repeat(A, 't n -> (v t) n', v=self.repeat)  # (H N)
        B = repeat(B, 'b t n -> b (v t) n', v=self.repeat)  # (1 H N)

        # TODO: The downstream algorithm should only need to access dt*A
        # However the current DPLR kernel still uses dt and A separately
        # Once that is fixed, this should return dtA instead of dt and A
        dtA = dt * A  # (H N)

        return dt, A, B, C

    def forward(self, L, state=None, rate=1.0):
        """See Kernel.forward() for argument documentation."""

        dt, A, B, C = self._get_params(rate)
        dtA = dt * A

        # Augment B with state
        if state is not None:
            s = state / dt
            B = torch.cat([s, B], dim=-3) # (1+B H N)


        # Combine B and C
        C = (B[:, None, :, :] * C).view(-1, self.H, self.N)

        # Dispatch which Vandermonde kernel to use
        if has_cuda_extension and C.dtype == torch.cfloat and C.device.type == 'cuda' and self.backend == 'cuda':
            log_vandermonde = log_vandermonde_cuda
        elif has_pykeops and self.backend in ['cuda', 'keops']:
            log_vandermonde = log_vandermonde_keops
        else:
            log_vandermonde = log_vandermonde_naive

        # Main kernel
        # Implementation from DSS meant for case when real eigenvalues can be positive
        P = dtA.unsqueeze(-1) * torch.arange(L, device=C.device) # [H N L]
        A_gt_0 = A.real > 0                                      # [N]
        if A_gt_0.any():
            with torch.no_grad():
                P_max = dtA * (A_gt_0 * (L-1))                   # [H N]
            P = P - P_max.unsqueeze(-1)                          # [H N L]
        S = P.exp()                                              # [H N L]

        dtA_neg = dtA * (1 - 2*A_gt_0)                           # [H N]
        num = dtA_neg.exp() - 1                                  # [H N]
        den = (dtA_neg * L).exp() - 1                            # [H N]

        # Inline reciprocal function for DSS logic
        x = den * A
        x_conj = _resolve_conj(x)
        r = x_conj / (x*x_conj + 1e-7)

        C = C * num * r             # [C H N]
        K = contract('chn,hnl->chl', C, S).float()

        K = K.view(-1, self.channels, self.H, L) # (1+B C H L)

        if state is not None:
            K_state = K[:-1, :, :, :] # (B C H L)
        else:
            K_state = None
        K = K[-1, :, :, :] # (C H L)

        return K, K_state

    def _setup_step(self):
        """Set up dA, dB, dC discretized parameters for stepping."""

        dt, A, B, C, = self._get_params()
        # Incorporate dt into A
        dtA = dt * A  # (H N)
        
        self.dB = rearrange(self.dB, '1 h n -> h n')
        self.dC = C

    def default_state(self, *batch_shape):
        C = _r2c(self.C)
        state = torch.zeros(*batch_shape, self.H, self.N, dtype=C.dtype, device=C.device)
        return state

    def step(self, u, state):
        next_state = contract("h n, b h n -> b h n", self.dA, state) \
                + contract("h n, b h -> b h n", self.dB, u)
        y = contract("c h n, b h n -> b c h", self.dC, next_state)
        return 2*y.real, next_state

    def forward_state(self, u, state):
        """Pass the state forward through an entire sequence."""
        self._setup_step()
        AL = self.dA ** u.size(-1)
        u = u.flip(-1).to(self.dA).contiguous() # (B H L)
        # Dispatch which Vandermonde kernel to use
        if has_pykeops and self.backend in ['cuda', 'keops']:
            log_vandermonde_transpose = log_vandermonde_transpose_keops
        else:
            log_vandermonde_transpose = log_vandermonde_transpose_naive
        v = log_vandermonde_transpose(u, self.dB, self.dA.log(), u.size(-1))
        next_state = AL * state + v
        return next_state
