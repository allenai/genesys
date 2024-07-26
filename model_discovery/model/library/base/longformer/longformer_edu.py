from typing import List
import math
import torch
from torch import nn
import torch.nn.functional as F
from transformers.modeling_roberta import RobertaConfig, RobertaModel, RobertaForMaskedLM


from typing import Union
from functools import lru_cache

import os.path


class DiagonaledMM(torch.autograd.Function):
    '''Class to encapsulate tvm code for compiling a diagonal_mm function, in addition to calling
    this function from PyTorch
    '''

    function_dict = {}  # save a list of functions, each has a different set of parameters

    @staticmethod
    def _compile_function(dtype: str, device: str, b0: int = 4, b1: int = 4, b2: int = 16):
        '''Compiles a tvm function that computes diagonal_mm
        args:
        dtype: str in ['float64', 'float32', 'float16']
        device: str in ['cpu' or 'cuda']
        b0, b1, b2: size of tensor tiles. Very important for good performance

        '''
        import tvm  # import the full tvm library here for compilation. Don't import at the top of the file in case we don't need to compile
        from tvm.contrib import nvcc
        @tvm.register_func
        def tvm_callback_cuda_compile(code):
            """Use nvcc compiler for better perf."""
            ptx = nvcc.compile_cuda(code, target="ptx", arch='sm_52')  # use old arch for this to work on old GPUs
            return ptx

        assert dtype in ['float16', 'float32', 'float64']
        assert device in ['cpu', 'cuda']
        device = None if device == 'cpu' else device
        tgt_host="llvm"

        b = tvm.var('b')  # batch size
        n = tvm.var('n')  # sequence length
        h = tvm.var('h')  # number of heads
        m = tvm.var('m')  # hidden dimension
        w = tvm.var('w')  # window size
        w_upper = tvm.var('w_upper')  # window size to the right of the word. Should be `0` or `w`
        padding = tvm.var('padding')  # padding
        transpose_t1 = tvm.var('transpose_t1')  # t1 should be transposed
        t1d3 = tvm.var('t1d3')  # last dimension of t1
        t3d3 = tvm.var('t3d3')  # last dimension of t3 (the result tensor)
        X = tvm.placeholder((b, n, h, t1d3), name='X', dtype=dtype)  # first tensor
        Y = tvm.placeholder((b, n, h, m), name='Y', dtype=dtype)  # second tensor
        k = tvm.reduce_axis((0, t1d3), name='k')  # dimension to sum over
        D = tvm.placeholder((h), name='D', dtype='int')  # dilation per head
        output_shape = (b, n, h, t3d3)  # shape of the result tensor
        algorithm = lambda l, i, q, j: tvm.sum(
            tvm.if_then_else(
                t3d3 == m,  # if output dimension == m, then t1 is diagonaled (FIXME: This breaks if t3d3 == m == t1d3)
                tvm.if_then_else(
                    transpose_t1 == 0,
                    tvm.if_then_else(
                        tvm.all(
                            i + D[q] * (k - w) >= 0,
                            i + D[q] * (k - w) < n,
                        ),
                        X[l, i, q, k] * Y[l, i + D[q] * (k - w), q, j],  # t1 is diagonaled
                        padding
                    ),
                    tvm.if_then_else(
                        tvm.all(
                            i + D[q] * (k - w_upper) >= 0,  # `w_upper` to handle the case `autoregressive=True`
                            i + D[q] * (k - w_upper) < n,
                        ),
                        X[l, i + D[q] * (k - w_upper), q, (w_upper + w) - k] * Y[l, i + D[q] * (k - w_upper), q, j],  # # t1 is diagonaled and should be transposed
                        padding
                    ),
                ),
                tvm.if_then_else(
                    tvm.all(
                        i + D[q] * (j - w) >= 0,
                        i + D[q] * (j - w) < n,
                    ),
                    X[l, i, q, k] * Y[l, i + D[q] * (j - w), q, k],  # t1 is not diagonaled, but the output tensor is going to be
                    padding
                )
            ), axis=k)

        Z = tvm.compute(output_shape, algorithm, name='Z')  # automatically generate cuda code
        s = tvm.create_schedule(Z.op)

        print('Lowering: \n ===================== \n{}'.format(tvm.lower(s, [X, Y, D], simple_mode=True)))

        # split long axis into smaller chunks and assing each one to a separate GPU thread/block
        ko, ki = s[Z].split(Z.op.reduce_axis[0], factor=b0)
        ZF = s.rfactor(Z, ki)

        j_outer, j_inner = s[Z].split(s[Z].op.axis[-1], factor=b1)
        i_outer, i_inner = s[Z].split(s[Z].op.axis[1], factor=b2)

        s[Z].bind(j_outer, tvm.thread_axis("blockIdx.x"))
        s[Z].bind(j_inner, tvm.thread_axis("threadIdx.y"))

        s[Z].bind(i_outer, tvm.thread_axis("blockIdx.y"))
        s[Z].bind(i_inner, tvm.thread_axis("threadIdx.z"))

        tx = tvm.thread_axis("threadIdx.x")
        s[Z].bind(s[Z].op.reduce_axis[0], tx)
        s[ZF].compute_at(s[Z], s[Z].op.reduce_axis[0])
        s[Z].set_store_predicate(tx.var.equal(0))

        print('Lowering with GPU splits: \n ===================== \n{}'.format(tvm.lower(s, [X, Y, D], simple_mode=True)))

        # compiling the automatically generated cuda code
        diagonaled_mm = tvm.build(s, [X, Y, Z, D, w, w_upper, padding, transpose_t1, t3d3], target=device, target_host=tgt_host, name='diagonaled_mm')
        return diagonaled_mm

    @staticmethod
    def _get_lib_filename(dtype: str, device: str):
        base_filename = 'longformer/lib/lib_diagonaled_mm'
        return '{}_{}_{}.so'.format(base_filename, dtype, device)

    @staticmethod
    def _save_compiled_function(f, dtype: str, device: str):
        if not os.path.exists('longformer/lib/'):
            os.makedirs('longformer/lib/')
        f.export_library(DiagonaledMM._get_lib_filename(dtype, device))

    @staticmethod
    def _load_compiled_function(dtype: str, device: str):
        from tvm.module import load  # this can be the small runtime python library, and doesn't need to be the whole thing
        filename = DiagonaledMM._get_lib_filename(dtype, device)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        potential_dirs = ['../../', '../', './', f'{current_dir}/', f'{current_dir}/../']
        for potential_dir in  potential_dirs:
            filepath = '{}{}'.format(potential_dir, filename)
            if os.path.isfile(filepath):
                print('Loading tvm binary from: {}'.format(filepath))
                return load(filepath)
        return None

    @staticmethod
    def _get_function(dtype: str, device: str):
        '''Loads the function from the disk or compile it'''
        # A list of arguments that define the function
        args = (dtype, device)
        if args not in DiagonaledMM.function_dict:
            diagonaled_mm = DiagonaledMM._load_compiled_function(dtype, device)  # try to load from disk
            if not diagonaled_mm:
                print('Tvm binary not found. Compiling ...')
                diagonaled_mm = DiagonaledMM._compile_function(dtype, device)  # compile
                DiagonaledMM._save_compiled_function(diagonaled_mm, dtype, device)  # save to disk
            # convert the tvm function into a pytorch function
            from tvm.contrib import dlpack
            diagonaled_mm_pytorch = dlpack.to_pytorch_func(diagonaled_mm)  # wrap it as a pytorch function
            # save the function into a dictionary to be reused
            DiagonaledMM.function_dict[args] = diagonaled_mm_pytorch  # save it in a dictionary for next time
        return DiagonaledMM.function_dict[args]

    @staticmethod
    def _diagonaled_mm(t1: torch.Tensor, t2: torch.Tensor, w: int, d: Union[torch.Tensor,int],
                       is_t1_diagonaled: bool = False, transpose_t1: bool = False, padding: int = 0,
                       autoregressive: bool = False):
        '''Calls the compiled function after checking the input format. This function is called in three different modes.
        t1 x t2 = r ==> t1 and t2 are not diagonaled, but r is. Useful for query x key = attention_scores
        t1 x t2 = r ==> t1 is diagonaled, but t2 and r are not. Useful to compuate attantion_scores x value = context
        t1 x t2 = r ==> t1 is diagonaled and it should be transposed, but t2 and r are not diagonaled. Useful in some of
                            the calculations in the backward pass.
        '''
        dtype = str(t1.dtype).split('.')[1]
        device = t1.device.type
        assert len(t1.shape) == 4
        assert len(t1.shape) == len(t2.shape)
        assert t1.shape[:3] == t2.shape[:3]
        if isinstance(d, int):  # if d is an integer, replace it with a tensor of the same length
                                # as number of heads, and it is filled with the same dilation value
            d = t1.new_full(size=(t1.shape[2],), fill_value=d, dtype=torch.int, requires_grad=False)

        assert len(d.shape) == 1
        assert d.shape[0] == t1.shape[2]  # number of dilation scores should match number of heads
        b = t1.shape[0]  # batch size
        n = t1.shape[1]  # sequence length
        h = t1.shape[2]  # number of heads
        m = t2.shape[3]  # hidden dimension
        w_upper = 0 if autoregressive else w
        c = w_upper + w + 1  # number of diagonals
        if is_t1_diagonaled:
            assert t1.shape[3] == c
            r = t1.new_empty(b, n, h, m)  # allocate spase for the result tensor
        else:
            assert not transpose_t1
            assert t1.shape[3] == m
            r = t1.new_empty(b, n, h, c)  # allocate spase for the result tensor

        # gets function from memory, from disk or compiles it from scratch
        _diagonaled_mm_function = DiagonaledMM._get_function(dtype=dtype, device=device)

        # The last argument to this function is a little hacky. It is the size of the last dimension of the result tensor
        # We use it as a proxy to tell if t1_is_diagonaled or not (if t1 is diagonaled, result is not, and vice versa).
        # The second reason is that the lambda expression in `_compile_function` is easier to express when the shape
        # of the output is known
        # This functions computes diagonal_mm then saves the result in `r`
        if m == c:
            # FIXME
            print('Error: the hidden dimension {m} shouldn\'t match number of diagonals {c}')
            assert False
        _diagonaled_mm_function(t1, t2, r, d, w, w_upper, padding, transpose_t1, m if is_t1_diagonaled else c)
        return r

    @staticmethod
    def _prepare_tensors(t):
        '''Fix `stride()` information of input tensor. This addresses some inconsistency in stride information in PyTorch.
        For a tensor t, if t.size(0) == 1, then the value of t.stride()[0] doesn't matter.
        TVM expects this value to be the `product(t.size()[1:])` but PyTorch some times sets it to `t.stride()[1]`.
        Here's an example to reporduce this issue:
            import torch
            print(torch.randn(1, 10).stride())
            > (10, 1)
            print(torch.randn(10, 1).t().contiguous().stride())
            > (1, 1)  # expected it to be (10, 1) as above
            print(torch.randn(10, 2).t().contiguous().stride())
            > (10, 1) # but gets the expected stride if the first dimension is > 1
        '''
        assert t.is_contiguous()
        t_stride = list(t.stride())
        t_size = list(t.size())
        # Fix wrong stride information for the first dimension. This occures when batch_size=1
        if t_size[0] == 1 and t_stride[0] == t_stride[1]:
            # In this case, the stride of the first dimension should be the product
            # of the sizes  of all other dimensions
            t_stride[0] = t_size[1] * t_size[2] * t_size[3]
            t = t.as_strided(size=t_size, stride=t_stride)
        return t

    min_seq_len = 16  # unexpected output if seq_len < 16

    @staticmethod
    def forward(ctx, t1: torch.Tensor, t2: torch.Tensor, w: int, d: Union[torch.Tensor,int], is_t1_diagonaled: bool = False, padding: int = 0, autoregressive: bool = False) -> torch.Tensor:
        '''Compuates diagonal_mm of t1 and t2.
        args: 
        t1: torch.Tensor = (batch_size, seq_len, num_attention_heads, hidden_size|number_of_diagonals).
            t1 can be a regular tensor (e.g. `query_layer`) or a diagonaled one (e.g. `attention_scores`)
        t2: torch.Tensor = (batch_size, seq_len, num_attention_heads, hidden_size). This is always a non-diagonaled
            tensor, e.g. `key_layer` or `value_layer`
        w: int = window size; number of attentions on each side of the word
        d: torch.Tensor or int = dilation of attentions per attention head. If int, the same dilation value will be used for all
            heads. If torch.Tensor, it should be 1D of lenth=number of attention heads
        is_t1_diagonaled: is t1 a diagonaled or a regular tensor
        padding: the padding value to use when accessing invalid locations. This is mainly useful when the padding
            needs to be a very large negative value (to compute softmax of attentions). For other usecases,
            please use zero padding.
        autoregressive: if true, return only the lower triangle
        returns: torch.Tensor = (batch_size, seq_len, num_attention_heads, hidden_size|number_of_diagonals)
            if t1 is diagonaed, result is non-diagonaled, and vice versa
        '''
        batch_size, seq_len, num_attention_heads, hidden_size = t1.size()
        assert seq_len >= DiagonaledMM.min_seq_len, 'avoid splitting errors by using seq_len >= {}'.format(DiagonaledMM.min_seq_len)  # FIXME
        ctx.save_for_backward(t1, t2)
        ctx.w = w
        ctx.d = d
        ctx.is_t1_diagonaled = is_t1_diagonaled
        ctx.autoregressive = autoregressive
        t1 = DiagonaledMM._prepare_tensors(t1)
        t2 = DiagonaledMM._prepare_tensors(t2)
        # output = t1.mm(t2)  # what would have been called if this was a regular matmul
        output = DiagonaledMM._diagonaled_mm(t1, t2, w, d, is_t1_diagonaled=is_t1_diagonaled, padding=padding, autoregressive=autoregressive)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        t1, t2 = ctx.saved_tensors
        w = ctx.w
        d = ctx.d
        is_t1_diagonaled = ctx.is_t1_diagonaled
        autoregressive = ctx.autoregressive
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()  # tvm requires all input tensors to be contiguous
        grad_output = DiagonaledMM._prepare_tensors(grad_output)
        t1 = DiagonaledMM._prepare_tensors(t1)
        t2 = DiagonaledMM._prepare_tensors(t2)
        # http://cs231n.github.io/optimization-2/
        # https://pytorch.org/docs/master/notes/extending.html
        # grad_t1 = grad_output.mm(t2)  # what would have been called if this was a regular matmul
        grad_t1 = DiagonaledMM._diagonaled_mm(grad_output, t2, w, d, is_t1_diagonaled=not is_t1_diagonaled, autoregressive=autoregressive)
        # grad_t2 = grad_output.t().mm(t1)  # or `grad_t2 = t1.t().mm(grad_output).t()` because `(AB)^T = B^TA^T`
        if is_t1_diagonaled:
            grad_t2 = DiagonaledMM._diagonaled_mm(t1, grad_output, w, d, is_t1_diagonaled=True, transpose_t1=True, autoregressive=autoregressive)
        else:
            grad_t2 = DiagonaledMM._diagonaled_mm(grad_output, t1, w, d, is_t1_diagonaled=True, transpose_t1=True, autoregressive=autoregressive)
        return grad_t1, grad_t2, None, None, None, None, None


def _get_invalid_locations_mask_fixed_dilation(seq_len: int, w: int, d: int):
    diagonals_list = []
    for j in range(-d * w, d, d):
        diagonal_mask = torch.zeros(seq_len, device='cpu', dtype=torch.uint8)
        diagonal_mask[:-j] = 1
        diagonals_list.append(diagonal_mask)
    return torch.stack(diagonals_list, dim=-1)

@lru_cache()
def _get_invalid_locations_mask(w: int, d: Union[torch.Tensor,int], autoregressive: bool, device: str):
    if isinstance(d, int):
        affected_seq_len = w * d
        mask = _get_invalid_locations_mask_fixed_dilation(affected_seq_len, w, d)
        mask = mask[None, :, None, :]
    else:
        affected_seq_len = w * d.max()
        head_masks = []
        d_list = d.cpu().numpy().tolist()
        for d in d_list:
            one_head_mask = _get_invalid_locations_mask_fixed_dilation(affected_seq_len, w, d)
            head_masks.append(one_head_mask)
        mask = torch.stack(head_masks, dim=-2)
        mask = mask[None, :, :, :]

    ending_mask = None if autoregressive else mask.flip(dims=(1, 3)).bool().to(device)
    return affected_seq_len, mask.bool().to(device), ending_mask

def mask_invalid_locations(input_tensor: torch.Tensor, w: int, d: Union[torch.Tensor, int], autoregressive: bool) -> torch.Tensor:
    affected_seq_len, beginning_mask, ending_mask = _get_invalid_locations_mask(w, d, autoregressive, input_tensor.device)
    seq_len = input_tensor.size(1)
    beginning_input = input_tensor[:, :affected_seq_len, :, :w+1]
    beginning_mask = beginning_mask[:, :seq_len].expand(beginning_input.size())
    beginning_input.masked_fill_(beginning_mask, -float('inf'))
    if not autoregressive:
        ending_input = input_tensor[:, -affected_seq_len:, :, -(w+1):]
        ending_mask = ending_mask[:, -seq_len:].expand(ending_input.size())
        ending_input.masked_fill_(ending_mask, -float('inf'))


diagonaled_mm = DiagonaledMM.apply

diagonaled_mm_tvm = diagonaled_mm
# The non-tvm implementation is the default, we don't need to load the kernel at loading time.
# DiagonaledMM._get_function('float32', 'cuda')


def _skew(x, direction, padding_value):
    '''Convert diagonals into columns (or columns into diagonals depending on `direction`'''
    x_padded = F.pad(x, direction, value=padding_value)
    x_padded = x_padded.view(*x_padded.size()[:-2], x_padded.size(-1), x_padded.size(-2))
    return x_padded


def _skew2(x, padding_value):
    '''shift every row 1 step to right converting columns into diagonals'''
    # X = B x C x M x L
    B, C, M, L = x.size()
    x = F.pad(x, (0, M + 1), value=padding_value)  # B x C x M x (L+M+1)
    x = x.view(B, C, -1)  # B x C x ML+MM+M
    x = x[:, :, :-M]  # B x C x ML+MM
    x = x.view(B, C, M, M + L)  # B x C, M x L+M
    x = x[:, :, :, :-1]
    return x


def _chunk(x, w):
    '''convert into overlapping chunkings. Chunk size = 2w, overlap size = w'''

    # non-overlapping chunks of size = 2w
    x = x.view(x.size(0), x.size(1) // (w * 2), w * 2, x.size(2))

    # use `as_strided` to make the chunks overlap with an overlap size = w
    chunk_size = list(x.size())
    chunk_size[1] = chunk_size[1] * 2 - 1

    chunk_stride = list(x.stride())
    chunk_stride[1] = chunk_stride[1] // 2
    return x.as_strided(size=chunk_size, stride=chunk_stride)


def sliding_chunks_matmul_qk(q: torch.Tensor, k: torch.Tensor, w: int, padding_value: float):
    '''Matrix multiplicatio of query x key tensors using with a sliding window attention pattern.
    This implementation splits the input into overlapping chunks of size 2w (e.g. 512 for pretrained Longformer)
    with an overlap of size w'''
    bsz, seqlen, num_heads, head_dim = q.size()
    assert seqlen % (w * 2) == 0
    assert q.size() == k.size()

    chunks_count = seqlen // w - 1

    # group bsz and num_heads dimensions into one, then chunk seqlen into chunks of size w * 2
    q = q.transpose(1, 2).reshape(bsz * num_heads, seqlen, head_dim)
    k = k.transpose(1, 2).reshape(bsz * num_heads, seqlen, head_dim)

    chunk_q = _chunk(q, w)
    chunk_k = _chunk(k, w)

    # matrix multipication
    # bcxd: bsz*num_heads x chunks x 2w x head_dim
    # bcyd: bsz*num_heads x chunks x 2w x head_dim
    # bcxy: bsz*num_heads x chunks x 2w x 2w
    chunk_attn = torch.einsum('bcxd,bcyd->bcxy', (chunk_q, chunk_k))  # multiply

    # convert diagonals into columns
    diagonal_chunk_attn = _skew(chunk_attn, direction=(0, 0, 0, 1), padding_value=padding_value)

    # allocate space for the overall attention matrix where the chunks are compined. The last dimension
    # has (w * 2 + 1) columns. The first (w) columns are the w lower triangles (attention from a word to
    # w previous words). The following column is attention score from each word to itself, then
    # followed by w columns for the upper triangle.

    diagonal_attn = diagonal_chunk_attn.new_empty((bsz * num_heads, chunks_count + 1, w, w * 2 + 1))

    # copy parts from diagonal_chunk_attn into the compined matrix of attentions
    # - copying the main diagonal and the upper triangle
    diagonal_attn[:, :-1, :, w:] = diagonal_chunk_attn[:, :, :w, :w + 1]
    diagonal_attn[:, -1, :, w:] = diagonal_chunk_attn[:, -1, w:, :w + 1]
    # - copying the lower triangle
    diagonal_attn[:, 1:, :, :w] = diagonal_chunk_attn[:, :, - (w + 1):-1, w + 1:]
    diagonal_attn[:, 0, 1:w, 1:w] = diagonal_chunk_attn[:, 0, :w - 1, 1 - w:]

    # separate bsz and num_heads dimensions again
    diagonal_attn = diagonal_attn.view(bsz, num_heads, seqlen, 2 * w + 1).transpose(2, 1)

    mask_invalid_locations(diagonal_attn, w, 1, False)
    return diagonal_attn


def sliding_chunks_matmul_pv(prob: torch.Tensor, v: torch.Tensor, w: int):
    '''Same as sliding_chunks_matmul_qk but for prob and value tensors. It is expecting the same output
    format from sliding_chunks_matmul_qk'''
    bsz, seqlen, num_heads, head_dim = v.size()
    assert seqlen % (w * 2) == 0
    assert prob.size()[:3] == v.size()[:3]
    assert prob.size(3) == 2 * w + 1
    chunks_count = seqlen // w - 1
    # group bsz and num_heads dimensions into one, then chunk seqlen into chunks of size 2w
    chunk_prob = prob.transpose(1, 2).reshape(bsz * num_heads, seqlen // w, w, 2 * w + 1)

    # group bsz and num_heads dimensions into one
    v = v.transpose(1, 2).reshape(bsz * num_heads, seqlen, head_dim)

    # pad seqlen with w at the beginning of the sequence and another w at the end
    padded_v = F.pad(v, (0, 0, w, w), value=-1)

    # chunk padded_v into chunks of size 3w and an overlap of size w
    chunk_v_size = (bsz * num_heads, chunks_count + 1, 3 * w, head_dim)
    chunk_v_stride = padded_v.stride()
    chunk_v_stride = chunk_v_stride[0], w * chunk_v_stride[1], chunk_v_stride[1], chunk_v_stride[2]
    chunk_v = padded_v.as_strided(size=chunk_v_size, stride=chunk_v_stride)

    skewed_prob = _skew2(chunk_prob, padding_value=0)

    context = torch.einsum('bcwd,bcdh->bcwh', (skewed_prob, chunk_v))
    return context.view(bsz, num_heads, seqlen, head_dim).transpose(1, 2)



# ========= "sliding_chunks_no_overlap": alternative implemenation of the sliding window attention =========
# This implementation uses non-overlapping chunks (or blocks) of size `w` with number of local attention = 3xw
# To make this implemenation comparable to "sliding_chunks" set w such that
#       w_of_sliding_chunks_no_overlap = w_of_sliding_chunks * 2 / 3
# For example,
#    w_of_sliding_chunks = 256 (this is one sided. Total attention size = 512)
#    w_of_sliding_chunks_no_overlap = 170 (Total attention size = 510)
# Performance:
# - Speed: 30% faster than "sliding_chunks"
# - Memory: 95% of the memory usage of "sliding_chunks"
# The windows are asymmetric where number of attention on each side of a token ranges between w to 2w
# while "sliding_chunks" has a symmetric window around each token.


def sliding_chunks_no_overlap_matmul_qk(q: torch.Tensor, k: torch.Tensor, w: int, padding_value: float):
    bsz, seqlen, num_heads, head_dim = q.size()
    assert seqlen % w == 0
    assert q.size() == k.size()
    # chunk seqlen into non-overlapping chunks of size w
    chunk_q = q.view(bsz, seqlen // w, w, num_heads, head_dim)
    chunk_k = k.view(bsz, seqlen // w, w, num_heads, head_dim)
    chunk_k_expanded = torch.stack((
        F.pad(chunk_k[:, :-1], (0, 0, 0, 0, 0, 0, 1, 0), value=0.0),
        chunk_k,
        F.pad(chunk_k[:, 1:], (0, 0, 0, 0, 0, 0, 0, 1), value=0.0),
    ), dim=-1)
    diagonal_attn = torch.einsum('bcxhd,bcyhde->bcxhey', (chunk_q, chunk_k_expanded))  # multiply
    return diagonal_attn.reshape(bsz, seqlen, num_heads, 3 * w)


def sliding_chunks_no_overlap_matmul_pv(prob: torch.Tensor, v: torch.Tensor, w: int):
    bsz, seqlen, num_heads, head_dim = v.size()
    chunk_prob = prob.view(bsz, seqlen // w, w, num_heads, 3, w)
    chunk_v = v.view(bsz, seqlen // w, w, num_heads, head_dim)
    chunk_v_extended = torch.stack((
        F.pad(chunk_v[:, :-1], (0, 0, 0, 0, 0, 0, 1, 0), value=0.0),
        chunk_v,
        F.pad(chunk_v[:, 1:], (0, 0, 0, 0, 0, 0, 0, 1), value=0.0),
    ), dim=-1)
    context = torch.einsum('bcwhpd,bcdhep->bcwhe', (chunk_prob, chunk_v_extended))
    return context.reshape(bsz, seqlen, num_heads, head_dim)


class Longformer(RobertaModel):
    def __init__(self, config):
        super(Longformer, self).__init__(config)
        if config.attention_mode == 'n2':
            pass  # do nothing, use BertSelfAttention instead
        else:
            for i, layer in enumerate(self.encoder.layer):
                layer.attention.self = LongformerSelfAttention(config, layer_id=i)


class LongformerForMaskedLM(RobertaForMaskedLM):
    def __init__(self, config):
        super(LongformerForMaskedLM, self).__init__(config)
        if config.attention_mode == 'n2':
            pass  # do nothing, use BertSelfAttention instead
        else:
            for i, layer in enumerate(self.roberta.encoder.layer):
                layer.attention.self = LongformerSelfAttention(config, layer_id=i)


class LongformerSelfAttention(nn.Module):
    def __init__(self, config, layer_id):
        super(LongformerSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_heads = config.num_attention_heads
        self.head_dim = int(config.hidden_size / config.num_attention_heads)
        self.embed_dim = config.hidden_size

        self.query = nn.Linear(config.hidden_size, self.embed_dim)
        self.key = nn.Linear(config.hidden_size, self.embed_dim)
        self.value = nn.Linear(config.hidden_size, self.embed_dim)

        self.query_global = nn.Linear(config.hidden_size, self.embed_dim)
        self.key_global = nn.Linear(config.hidden_size, self.embed_dim)
        self.value_global = nn.Linear(config.hidden_size, self.embed_dim)

        self.dropout = config.attention_probs_dropout_prob

        self.layer_id = layer_id
        self.attention_window = config.attention_window[self.layer_id]
        self.attention_dilation = config.attention_dilation[self.layer_id]
        self.attention_mode = config.attention_mode
        self.autoregressive = config.autoregressive
        assert self.attention_window > 0
        assert self.attention_dilation > 0
        assert self.attention_mode in ['tvm', 'sliding_chunks', 'sliding_chunks_no_overlap']
        if self.attention_mode in ['sliding_chunks', 'sliding_chunks_no_overlap']:
            assert not self.autoregressive  # not supported
            assert self.attention_dilation == 1  # dilation is not supported

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        '''
        The `attention_mask` is changed in `BertModel.forward` from 0, 1, 2 to
            -ve: no attention
              0: local attention
            +ve: global attention
        '''
        assert encoder_hidden_states is None, "`encoder_hidden_states` is not supported and should be None"
        assert encoder_attention_mask is None, "`encoder_attention_mask` is not supported and shiould be None"

        if attention_mask is not None:
            attention_mask = attention_mask.squeeze(dim=2).squeeze(dim=1)
            key_padding_mask = attention_mask < 0
            extra_attention_mask = attention_mask > 0
            remove_from_windowed_attention_mask = attention_mask != 0

            num_extra_indices_per_batch = extra_attention_mask.long().sum(dim=1)
            max_num_extra_indices_per_batch = num_extra_indices_per_batch.max()
            if max_num_extra_indices_per_batch <= 0:
                extra_attention_mask = None
            else:
                # To support the case of variable number of global attention in the rows of a batch,
                # we use the following three selection masks to select global attention embeddings
                # in a 3d tensor and pad it to `max_num_extra_indices_per_batch`
                # 1) selecting embeddings that correspond to global attention
                extra_attention_mask_nonzeros = extra_attention_mask.nonzero(as_tuple=True)
                zero_to_max_range = torch.arange(0, max_num_extra_indices_per_batch,
                                                 device=num_extra_indices_per_batch.device)
                # mask indicating which values are actually going to be padding
                selection_padding_mask = zero_to_max_range < num_extra_indices_per_batch.unsqueeze(dim=-1)
                # 2) location of the non-padding values in the selected global attention
                selection_padding_mask_nonzeros = selection_padding_mask.nonzero(as_tuple=True)
                # 3) location of the padding values in the selected global attention
                selection_padding_mask_zeros = (selection_padding_mask == 0).nonzero(as_tuple=True)
        else:
            remove_from_windowed_attention_mask = None
            extra_attention_mask = None
            key_padding_mask = None

        hidden_states = hidden_states.transpose(0, 1)
        seq_len, bsz, embed_dim = hidden_states.size()
        assert embed_dim == self.embed_dim
        q = self.query(hidden_states)
        k = self.key(hidden_states)
        v = self.value(hidden_states)
        q /= math.sqrt(self.head_dim)

        q = q.view(seq_len, bsz, self.num_heads, self.head_dim).transpose(0, 1)
        k = k.view(seq_len, bsz, self.num_heads, self.head_dim).transpose(0, 1)
        # attn_weights = (bsz, seq_len, num_heads, window*2+1)
        if self.attention_mode == 'tvm':
            q = q.float().contiguous()
            k = k.float().contiguous()
            attn_weights = diagonaled_mm_tvm(q, k, self.attention_window, self.attention_dilation, False, 0, False)
        elif self.attention_mode == "sliding_chunks":
            attn_weights = sliding_chunks_matmul_qk(q, k, self.attention_window, padding_value=0)
        elif self.attention_mode == "sliding_chunks_no_overlap":
            attn_weights = sliding_chunks_no_overlap_matmul_qk(q, k, self.attention_window, padding_value=0)
        else:
            raise False
        mask_invalid_locations(attn_weights, self.attention_window, self.attention_dilation, False)
        if remove_from_windowed_attention_mask is not None:
            # This implementation is fast and takes very little memory because num_heads x hidden_size = 1
            # from (bsz x seq_len) to (bsz x seq_len x num_heads x hidden_size)
            remove_from_windowed_attention_mask = remove_from_windowed_attention_mask.unsqueeze(dim=-1).unsqueeze(dim=-1)
            # cast to float/half then replace 1's with -inf
            float_mask = remove_from_windowed_attention_mask.type_as(q).masked_fill(remove_from_windowed_attention_mask, -10000.0)
            repeat_size = 1 if isinstance(self.attention_dilation, int) else len(self.attention_dilation)
            float_mask = float_mask.repeat(1, 1, repeat_size, 1)
            ones = float_mask.new_ones(size=float_mask.size())  # tensor of ones
            # diagonal mask with zeros everywhere and -inf inplace of padding
            if self.attention_mode == 'tvm':
                d_mask = diagonaled_mm_tvm(ones, float_mask, self.attention_window, self.attention_dilation, False, 0, False)
            elif self.attention_mode == "sliding_chunks":
                d_mask = sliding_chunks_matmul_qk(ones, float_mask, self.attention_window, padding_value=0)
            elif self.attention_mode == "sliding_chunks_no_overlap":
                d_mask = sliding_chunks_no_overlap_matmul_qk(ones, float_mask, self.attention_window, padding_value=0)

            attn_weights += d_mask
        assert list(attn_weights.size())[:3] == [bsz, seq_len, self.num_heads]
        assert attn_weights.size(dim=3) in [self.attention_window * 2 + 1, self.attention_window * 3]

        # the extra attention
        if extra_attention_mask is not None:
            selected_k = k.new_zeros(bsz, max_num_extra_indices_per_batch, self.num_heads, self.head_dim)
            selected_k[selection_padding_mask_nonzeros] = k[extra_attention_mask_nonzeros]
            # (bsz, seq_len, num_heads, max_num_extra_indices_per_batch)
            selected_attn_weights = torch.einsum('blhd,bshd->blhs', (q, selected_k))
            selected_attn_weights[selection_padding_mask_zeros[0], :, :, selection_padding_mask_zeros[1]] = -10000
            # concat to attn_weights
            # (bsz, seq_len, num_heads, extra attention count + 2*window+1)
            attn_weights = torch.cat((selected_attn_weights, attn_weights), dim=-1)
        attn_weights_float = F.softmax(attn_weights, dim=-1, dtype=torch.float32)  # use fp32 for numerical stability
        if key_padding_mask is not None:
            # softmax sometimes inserts NaN if all positions are masked, replace them with 0
            attn_weights_float = torch.masked_fill(attn_weights_float, key_padding_mask.unsqueeze(-1).unsqueeze(-1), 0.0)
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = F.dropout(attn_weights_float.type_as(attn_weights), p=self.dropout, training=self.training)
        v = v.view(seq_len, bsz, self.num_heads, self.head_dim).transpose(0, 1)
        attn = 0
        if extra_attention_mask is not None:
            selected_attn_probs = attn_probs.narrow(-1, 0, max_num_extra_indices_per_batch)
            selected_v = v.new_zeros(bsz, max_num_extra_indices_per_batch, self.num_heads, self.head_dim)
            selected_v[selection_padding_mask_nonzeros] = v[extra_attention_mask_nonzeros]
            # use `matmul` because `einsum` crashes sometimes with fp16
            # attn = torch.einsum('blhs,bshd->blhd', (selected_attn_probs, selected_v))
            attn = torch.matmul(selected_attn_probs.transpose(1, 2), selected_v.transpose(1, 2).type_as(selected_attn_probs)).transpose(1, 2)
            attn_probs = attn_probs.narrow(-1, max_num_extra_indices_per_batch, attn_probs.size(-1) - max_num_extra_indices_per_batch).contiguous()

        if self.attention_mode == 'tvm':
            v = v.float().contiguous()
            attn += diagonaled_mm_tvm(attn_probs, v, self.attention_window, self.attention_dilation, True, 0, False)
        elif self.attention_mode == "sliding_chunks":
            attn += sliding_chunks_matmul_pv(attn_probs, v, self.attention_window)
        elif self.attention_mode == "sliding_chunks_no_overlap":
            attn += sliding_chunks_no_overlap_matmul_pv(attn_probs, v, self.attention_window)
        else:
            raise False

        attn = attn.type_as(hidden_states)
        assert list(attn.size()) == [bsz, seq_len, self.num_heads, self.head_dim]
        attn = attn.transpose(0, 1).reshape(seq_len, bsz, embed_dim).contiguous()

        # For this case, we'll just recompute the attention for these indices
        # and overwrite the attn tensor. TODO: remove the redundant computation
        if extra_attention_mask is not None:
            selected_hidden_states = hidden_states.new_zeros(max_num_extra_indices_per_batch, bsz, embed_dim)
            selected_hidden_states[selection_padding_mask_nonzeros[::-1]] = hidden_states[extra_attention_mask_nonzeros[::-1]]

            q = self.query_global(selected_hidden_states)
            k = self.key_global(hidden_states)
            v = self.value_global(hidden_states)
            q /= math.sqrt(self.head_dim)

            q = q.contiguous().view(max_num_extra_indices_per_batch, bsz * self.num_heads, self.head_dim).transpose(0, 1)  # (bsz*self.num_heads, max_num_extra_indices_per_batch, head_dim)
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)  # bsz * self.num_heads, seq_len, head_dim)
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)  # bsz * self.num_heads, seq_len, head_dim)
            attn_weights = torch.bmm(q, k.transpose(1, 2))
            assert list(attn_weights.size()) == [bsz * self.num_heads, max_num_extra_indices_per_batch, seq_len]

            attn_weights = attn_weights.view(bsz, self.num_heads, max_num_extra_indices_per_batch, seq_len)
            attn_weights[selection_padding_mask_zeros[0], :, selection_padding_mask_zeros[1], :] = -10000.0
            if key_padding_mask is not None:
                attn_weights = attn_weights.masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2),
                    -10000.0,
                )
            attn_weights = attn_weights.view(bsz * self.num_heads, max_num_extra_indices_per_batch, seq_len)
            attn_weights_float = F.softmax(attn_weights, dim=-1, dtype=torch.float32)  # use fp32 for numerical stability
            attn_probs = F.dropout(attn_weights_float.type_as(attn_weights), p=self.dropout, training=self.training)
            selected_attn = torch.bmm(attn_probs, v)
            assert list(selected_attn.size()) == [bsz * self.num_heads, max_num_extra_indices_per_batch, self.head_dim]

            selected_attn_4d = selected_attn.view(bsz, self.num_heads, max_num_extra_indices_per_batch, self.head_dim)
            nonzero_selected_attn = selected_attn_4d[selection_padding_mask_nonzeros[0], :, selection_padding_mask_nonzeros[1]]
            attn[extra_attention_mask_nonzeros[::-1]] = nonzero_selected_attn.view(len(selection_padding_mask_nonzeros[0]), -1).type_as(hidden_states)

        context_layer = attn.transpose(0, 1)
        if output_attentions:
            if extra_attention_mask is not None:
                # With global attention, return global attention probabilities only
                # batch_size x num_heads x max_num_global_attention_tokens x sequence_length
                # which is the attention weights from tokens with global attention to all tokens
                # It doesn't not return local attention
                # In case of variable number of global attantion in the rows of a batch,
                # attn_weights are padded with -10000.0 attention scores
                attn_weights = attn_weights.view(bsz, self.num_heads, max_num_extra_indices_per_batch, seq_len)
            else:
                # without global attention, return local attention probabilities
                # batch_size x num_heads x sequence_length x window_size
                # which is the attention weights of every token attending to its neighbours
                attn_weights = attn_weights.permute(0, 2, 1, 3)
        outputs = (context_layer, attn_weights) if output_attentions else (context_layer,)
        return outputs