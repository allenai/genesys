import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

try:
    from src.ops.fftconv import fftconv_func
except ImportError:
    fftconv_func = None

"""SSM convolution kernels.
SSKernel wraps different kernels with common options and handles the initialization.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
from opt_einsum import contract

from src.models.ssm.ss_kernel_diag import SSKernelDiag, EMAKernel
from src.models.ssm.ss_kernel_shift import SSKernelShift

from src.models.ssm import dplr
from src.ops.krylov import power

from src.utils.utils import get_logger

log = get_logger(__name__)


_conj = lambda x: torch.cat([x, x.conj()], dim=-1)


class SSKernel(nn.Module):
    """Wrapper around SSKernel parameterizations.

    The SSKernel is expected to support the interface
    forward()
    default_state()
    _setup_step()
    step()
    """

    def __init__(
        self,
        H,
        N=64,
        L=None,
        measure="diag-lin",
        rank=1,
        channels=1,
        dt_min=0.001,
        dt_max=0.1,
        deterministic=False,
        lr=None,
        mode="diag",
        n_ssm=None,
        verbose=False,
        measure_args={},
        **kernel_args,
    ):
        """State Space Kernel which computes the convolution kernel $\\bar{K}$

        H: Number of independent SSM copies; controls the size of the model. Also called d_model in the config.
        N: State size (dimensionality of parameters A, B, C). Also called d_state in the config. Generally shouldn't need to be adjusted and doens't affect speed much.
        L: Maximum length of convolution kernel, if known. Should work in the majority of cases even if not known.
        measure: Options for initialization of (A, B). For NPLR mode, recommendations are "legs", "fout", "hippo" (combination of both). For Diag mode, recommendations are "diag-inv", "diag-lin", "diag-legs", and "diag" (combination of diag-inv and diag-lin)
        rank: Rank of low-rank correction for NPLR mode. Needs to be increased for measure "legt"
        channels: C channels turns the SSM from a 1-dim to C-dim map; can think of it having C separate "heads" per SSM. This was partly a feature to make it easier to implement bidirectionality; it is recommended to set channels=1 and adjust H to control parameters instead
        dt_min, dt_max: min and max values for the step size dt (\Delta)
        mode: Which kernel algorithm to use. 'nplr' is the full S4 model; 'diag' is the simpler S4D; 'slow' is a dense version for testing
        n_ssm: Number of independent trainable (A, B) SSMs, e.g. n_ssm=1 means all A/B parameters are tied across the H different instantiations of C. n_ssm=None means all H SSMs are completely independent. Generally, changing this option can save parameters but doesn't affect performance or speed much. This parameter must divide H
        lr: Passing in a number (e.g. 0.001) sets attributes of SSM parameers (A, B, dt). A custom optimizer hook is needed to configure the optimizer to set the learning rates appropriately for these parameters.
        """
        super().__init__()
        self.N = N
        self.H = H
        dtype, cdtype = torch.float, torch.cfloat
        self.channels = channels
        self.n_ssm = n_ssm if n_ssm is not None else H
        self.mode = mode
        self.verbose = verbose
        self.kernel_args = kernel_args

        # Generate dt
        if deterministic:
            log_dt = torch.exp(torch.linspace(math.log(dt_min), math.log(dt_max), H))
        else:
            log_dt = torch.rand(self.H, dtype=dtype) * (
                math.log(dt_max) - math.log(dt_min)
            ) + math.log(dt_min)

        # Compute the preprocessed representation
        if mode == "ema":
            self.kernel = EMAKernel(H, N=N, channels=channels, **kernel_args)
        else:
            w, P, B, V = dplr.combination(measure, self.N, rank, self.n_ssm, **measure_args)

            # Broadcast C to have H channels
            if deterministic:
                C = torch.zeros(channels, self.n_ssm, self.N, dtype=cdtype)
                C[:, :, :1] = 1.
                C = contract('hmn, chn -> chm', V.conj().transpose(-1, -2), C) # V^* C
                C = repeat(C, 'c t n -> c (v t) n', v=self.n_ssm // C.size(-2)).clone().contiguous()
            else:
                C = torch.randn(channels, self.H, self.N//2, dtype=cdtype)

            # Broadcast other parameters to have n_ssm copies
            assert self.n_ssm % B.size(-2) == 0 \
                    and self.n_ssm % P.size(-2) == 0 \
                    and self.n_ssm % w.size(-2) == 0
            # Broadcast tensors to n_ssm copies
            # These will be the parameters, so make sure tensors are materialized and contiguous
            B = repeat(B, 't n -> (v t) n', v=self.n_ssm // B.size(-2)).clone().contiguous()
            P = repeat(P, 'r t n -> r (v t) n', v=self.n_ssm // P.size(-2)).clone().contiguous()
            w = repeat(w, 't n -> (v t) n', v=self.n_ssm // w.size(-2)).clone().contiguous()

            if mode == "diag":
                if not measure.startswith("diag"):
                    log.warning("Diagonal kernel (S4D) activated but initialization is not intended for S4D. Set `measure` to 'diag-lin', 'diag-inv', or 'diag-legs' for the main variants, or 'diag' for a combination of S4D-Lin and S4D-Inv.")
                C = C * repeat(B, 't n -> (v t) n', v=H//self.n_ssm)
                self.kernel = SSKernelDiag(
                    w, B, C, log_dt, L=L,
                    lr=lr,
                    **kernel_args,
                )
            elif mode == 'shift':
                # Initializing B to be e_1
                B = torch.zeros(self.H, self.N)
                B[..., 0] = 1.0
                # Match torch.Conv1d init
                C = torch.randn(self.H, self.channels, self.N)
                nn.init.kaiming_uniform_(C, a=math.sqrt(5))
                C = rearrange(C, 'h c n -> c h n')
                self.kernel = SSKernelShift(B, C, L=L, lr=lr, **kernel_args)
            else:
                raise NotImplementedError(f"{mode=} is not valid")

    def forward(self, state=None, L=None, rate=None):
        return self.kernel(state=state, L=L, rate=rate)

    @torch.no_grad()
    def forward_state(self, u, state):
        """ Forward the state through a sequence, i.e. computes the state after passing chunk through SSM

        state: (B, H, N)
        u: (B, H, L)

        Returns: (B, H, N)
        """

        if hasattr(self.kernel, "forward_state"):
            return self.kernel.forward_state(u, state)

        dA, dB = self.kernel._setup_state() # Construct dA, dB matrices
        # dA, dB = self.kernel.dA, self.kernel.dB # (H N N) (H N)

        conj = state.size(-1) != dA.size(-1)
        if conj: state = _conj(state)

        v = contract('h n, b h l -> b h n l', dB, u.flip(-1)) # dB.unsqueeze(-1) * u.flip(-1).unsqueeze(-2)
        AL, v = power(u.size(-1), dA, v)
        next_state = contract("h m n, b h n -> b h m", AL, state)
        next_state = next_state + v

        if conj: next_state = next_state[..., : next_state.size(-1) // 2]
        return next_state

    def _setup_step(self, **kwargs):
        # This method is intended to be private so that setting up an S4 module with
        # ```
        # if hasattr(module, 'setup_step'): module.setup_step()
        # ```
        # will not trigger this method multiple times
        self.kernel._setup_step(**kwargs)

    def step(self, u, state, **kwargs):
        y, state = self.kernel.step(u, state, **kwargs)
        return y, state

    def default_state(self, *args, **kwargs):
        return self.kernel.default_state(*args, **kwargs)
    
@torch.jit.script
def mul_sum(q, y):
    return (q * y).sum(dim=1)


class H3(nn.Module):

    def __init__(
            self,
            d_model,
            d_state=64,
            l_max=None,
            head_dim=1,
            use_fast_fftconv=False,
            dropout=0.0,   # Just to absorb the kwarg
            layer_idx=None,
            device=None, dtype=None,
            # SSM Kernel arguments
            **kernel_args,
        ):
        """
        d_state: the dimension of the state, also denoted by N
        l_max: the maximum kernel length, also denoted by L. Set l_max=None to always use a global kernel

        See the class .kernel.SSKernel for the kernel constructor which accepts kernel_args. Relevant options that are worth considering and tuning include "mode" + "measure", "dt_min", "dt_max", "lr"

        Other options are all experimental and should not need to be configured
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.d_model = d_model
        self.head_dim = head_dim
        assert d_model % head_dim == 0
        self.H = d_model // head_dim
        self.N = d_state
        self.L = l_max
        self.layer_idx = layer_idx
        self.use_fast_fftconv = use_fast_fftconv
        if self.use_fast_fftconv:
            assert fftconv_func is not None, 'Need to install fftconv'

        self.q_proj = nn.Linear(self.d_model, self.d_model, **factory_kwargs)
        self.k_proj = nn.Linear(self.d_model, self.d_model, **factory_kwargs)
        self.v_proj = nn.Linear(self.d_model, self.d_model, **factory_kwargs)

        # TODO: SSKernel doesn't take device argument yet
        self.ssm_k_kernel = SSKernel(self.d_model, N=d_state, L=self.L, mode='shift',
                                     lr=kernel_args.get('lr', None))
        self.ssm_k_D = nn.Parameter(torch.randn(self.d_model))
        # S4D Kernel
        self.kernel = SSKernel(self.H, N=self.N, L=self.L, channels=1, **kernel_args)
        self.D = nn.Parameter(torch.randn(self.H, **factory_kwargs))

        # Pointwise
        # position-wise output transform to mix features
        # Don't use FusedDense since the layout is H first
        self.output_linear = nn.Linear(self.d_model, self.d_model)

    def forward(self, u, inference_params=None):
        """
        u: (B L H)

        Returns: same shape as u
        """
        L_og = u.size(-2)
        if self.use_fast_fftconv and L_og % 2 != 0:
            u = F.pad(u, (0, 0, 0, 1))
        L = u.size(-2)

        use_fast_fftconv = self.use_fast_fftconv and inference_params is None

        state_k, state = None, None
        if inference_params is not None:
            assert self.layer_idx is not None
            if self.layer_idx not in inference_params.key_value_memory_dict:
                batch_shape = (u.shape[0] * self.head_dim * self.head_dim,)
                state_k = self.ssm_k_kernel.default_state(*batch_shape)
                state = self.kernel.default_state(*batch_shape)
                inference_params.key_value_memory_dict[self.layer_idx] = (state_k, state)
            else:
                state_k, state = inference_params.key_value_memory_dict[self.layer_idx]
            if inference_params.sequence_len_offset == 0:
                self.ssm_k_kernel._setup_step()
                self.kernel._setup_step()

        if inference_params is not None and inference_params.sequence_len_offset > 0:
            y, next_state_k, next_state = self.step(u, state_k, state)
            inference_params.key_value_memory_dict[self.layer_idx][0].copy_(next_state_k)
            inference_params.key_value_memory_dict[self.layer_idx][1].copy_(next_state)
            return y

        # Compute SS Kernel
        L_kernel = L if self.L is None else min(L, self.L )
        ssm_kernel, k_state = self.kernel(L=L_kernel, state=state, rate=1.0) # (C H L) (B C H L)
        ssm_kernel = rearrange(ssm_kernel, '1 h l -> h l')

        u = rearrange(u, 'b l h -> (b l) h')
        dtype = (self.q_proj.weight.dtype if not torch.is_autocast_enabled()
                 else torch.get_autocast_gpu_dtype())
        q = self.q_proj.weight @ u.T + self.q_proj.bias.to(dtype).unsqueeze(-1)
        k = self.k_proj.weight @ u.T + self.k_proj.bias.to(dtype).unsqueeze(-1)
        v = self.v_proj.weight @ u.T + self.v_proj.bias.to(dtype).unsqueeze(-1)
        q, k, v = [rearrange(x, 'h (b l) -> b h l', l=L) for x in [q, k, v]]

        k_og = k
        ssm_k_kernel, _ = self.ssm_k_kernel(L=L_kernel, state=state_k, rate=1.0) # (C H L) (B C H L)
        ssm_k_kernel = rearrange(ssm_k_kernel, '1 h l -> h l')
        if not use_fast_fftconv:
            fft_size = L_kernel + L
            ssm_k_kernel_f = torch.fft.rfft(ssm_k_kernel, n=fft_size) # (H 2L)
            k_f = torch.fft.rfft(k.to(ssm_kernel.dtype), n=fft_size) # (B H 2L)
            shift_k_out = torch.fft.irfft(ssm_k_kernel_f * k_f, n=fft_size)[..., :L]
            k = shift_k_out + rearrange(self.ssm_k_D, 'h -> h 1') * k
        else:
            dropout_mask = None
            # No GeLU after the SSM
            # We want output_hbl=True so that k has the same layout as q and v for the next
            # fftconv
            k = fftconv_func(k, ssm_k_kernel, self.ssm_k_D, dropout_mask, False, False, True)
            # This line below looks like it doesn't do anything, but it gets the stride right
            # for the case batch_size=1. In that case k has stride (L, L, 1), but q and v has
            # stride (H * L, L, 1). The two strides are equivalent because batch_size=1, but
            # the C++ code doesn't like that.
            k = rearrange(rearrange(k, 'b h l -> h b l'), 'h b l -> b h l')

        if not use_fast_fftconv:
            fft_size = L_kernel + L
            # kv = k * v
            kv = (rearrange(k, 'b (h d1) l -> b d1 1 h l', d1=self.head_dim)
                    * rearrange(v, 'b (h d2) l -> b 1 d2 h l', d2=self.head_dim))  # b d1 d2 h l
            kv_f = torch.fft.rfft(kv.to(dtype=ssm_kernel.dtype), n=fft_size) / fft_size
            ssm_kernel_f = torch.fft.rfft(ssm_kernel, n=fft_size)  # h L+1
            y = torch.fft.irfft(kv_f * ssm_kernel_f, n=fft_size, norm='forward')[..., :L]  # b d1 d2 h l
            y = y + kv * self.D.unsqueeze(-1)  # b d1 d2 h l
            q = rearrange(q, 'b (h d1) l -> b d1 1 h l', d1=self.head_dim)
            # einsum is way slower than multiply and then sum.
            if self.head_dim > 1:
                y = mul_sum(y, q)
                y = rearrange(y, 'b d h l -> b (d h) l')
            else:
                y = rearrange(y * q, 'b 1 1 h l -> b h l')
        else:
            dropout_mask = None
            # No GeLU after the SSM
            # Set output_hbl_layout=True since we'll be doing a matmul right after
            y = fftconv_func(k, ssm_kernel, self.D,
                             dropout_mask, False, torch.is_autocast_enabled(), True,
                             v, self.head_dim, q)

        y = rearrange(y, 'b h l -> b l h')

        if state is not None:
            assert inference_params is not None
            # TODO: This doesn't ever happen?
            # if inference_params.sequence_len_offset > 0:
            #     y = y + k_state
            inference_params.key_value_memory_dict[self.layer_idx][0].copy_(
                self.ssm_k_kernel.forward_state(k_og, state_k)
            )
            inference_params.key_value_memory_dict[self.layer_idx][1].copy_(
                self.kernel.forward_state(rearrange(kv, 'b d1 d2 h l -> (b d1 d2) h l'), state)
            )

        # y could be in fp32 because of the SSMs
        if not torch.is_autocast_enabled():
            y = y.to(dtype=self.output_linear.weight.dtype)
        y = self.output_linear(y)
        if L_og < L:
            y = y[:, :L_og, :]

        return y

    def step(self, u, state_k, state):
        q, k, v = self.q_proj(u), self.k_proj(u), self.v_proj(u)
        shift_k, next_state_k = self.ssm_k_kernel.step(rearrange(k, 'b 1 h -> b h'), state_k)
        k = shift_k + k * self.ssm_k_D
        # kv = k * v
        kv = (rearrange(k, 'b 1 (h d1) -> b d1 1 h', d1=self.head_dim)
                * rearrange(v, 'b 1 (h d2) -> b 1 d2 h', d2=self.head_dim))  # b d1 d2 h
        y, next_state = self.kernel.step(rearrange(kv, 'b d1 d2 h -> (b d1 d2) h'), state)
        y = (rearrange(y, '(b d1 d2) 1 h -> b d1 d2 h', d1=self.head_dim, d2=self.head_dim)
                + kv * self.D)
        q = rearrange(q, 'b 1 (h d1) -> b d1 1 h', d1=self.head_dim)
        if self.head_dim > 1:
            y = mul_sum(y, q)
            y = rearrange(y, 'b d h l -> b (d h) l')
        else:
            y = rearrange(y * q, 'b 1 1 h -> b 1 h')
        # y could be in fp32 because of the SSMs
        if not torch.is_autocast_enabled():
            y = y.to(dtype=self.output_linear.weight.dtype)
        return self.output_linear(y), next_state_k, next_state