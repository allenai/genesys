import torch
import torch.nn as nn
import math

class AliBi(torch.nn.Module):
    def __init__(self, num_heads, mp_size=1, mp_rank=1):
        super().__init__()
        # megatron splits across heads, so we need to make sure each
        # head receives the correct matrix
        assert mp_size <= num_heads and mp_rank <= mp_size
        self.mp_size = mp_size
        self.mp_rank = mp_rank
        self.num_heads = num_heads
        self.slice_size = num_heads // mp_size
        self.cached_matrix = None
        self.cached_seq_len = None
        slopes = torch.Tensor(self._get_slopes(num_heads))[
            mp_rank * self.slice_size : (mp_rank + 1) * self.slice_size
        ]
        self.register_buffer("slopes", slopes)
        print(slopes.dtype)



    def _get_slopes(self, n):
        """
        Get slopes for Alibi positional embedding
        n : int = number of heads.
        For best performance, restrict n to a power of 2.
        """

        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio**i for i in range(n)]

        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            return (
                get_slopes_power_of_2(closest_power_of_2)
                + self._get_slopes(2 * closest_power_of_2)[0::2][
                    : n - closest_power_of_2
                ]
            )


    def forward(self, x):
        # [b, np, sq, sk]
        seq_len_q = x.shape[-2]
        seq_len_k = x.shape[-1]

        # Initialize the AliBi matrix to match the first provided key length; grow it exponentially
        # afterwards if longer inputs are provided. This is important for inference, where we will
        # encounter progressively longer samples; it should have no effect at training time.
        if self.cached_seq_len is not None and self.cached_seq_len >= seq_len_k:
            a = self.cached_matrix
        else:
            target_seq_len = (
                seq_len_k
            )
            a = -torch.tril(
                torch.arange(target_seq_len)
                .view(target_seq_len, 1)
                .repeat(1, target_seq_len)
                + torch.arange(0, -target_seq_len, -1)
            )
            a = a.to(x.device).to(x.dtype)
            # print(a)
            slopes = self.slopes.to(a.device).to(a.dtype)
            a = a * slopes.view(self.slopes.shape[0], 1, 1)
            self.cached_seq_len = target_seq_len
            self.cached_matrix = a

        # If the AliBi matrix is larger than the key length, clip it.
        if self.cached_seq_len > seq_len_k:
            a = self.cached_matrix[:, :seq_len_k, :seq_len_k]

        if seq_len_q != seq_len_k:
            # In the train case x has dimensionality [b, np, sq, sk] with sq == sk
            # The number of query tokens is equal to the number of key tokens
            # At inference time with cache in layer_past sq is not equal to sk. sq only contains one token (the last one in the full sequence)
            # In this case we use the appropriate token index of the cache matrix.
            # As the cache matrix could already be bigger from a past inference, not the last token index in the sq sequence is used
            assert (
                seq_len_q == 1
            ), "assumption sq == sk unless at inference time with cache in layer_past with sq == 1"
            a = a[:, seq_len_k - 1, :].view(
                a.shape[0], 1, a.shape[2]
            )  # seq_len_k - 1 points to the last token index in the current inference batch.

        return x + a


class AliBi_CAPE(torch.nn.Module):
    def __init__(self, num_heads, mp_size=1, mp_rank=1,mlp_width=32):
        super().__init__()
        # megatron splits across heads, so we need to make sure each
        # head receives the correct matrix
        assert mp_size <= num_heads and mp_rank <= mp_size
        self.mp_size = mp_size
        self.mp_rank = mp_rank
        self.num_heads = num_heads
        self.slice_size = num_heads // mp_size
        self.cached_matrix = None
        self.cached_seq_len = None
        slopes = torch.Tensor(self._get_slopes(num_heads))[
            mp_rank * self.slice_size : (mp_rank + 1) * self.slice_size
        ]
        self.register_buffer("slopes", slopes)

        self.mlp2 = nn.Sequential(
            nn.Linear(num_heads * 2, mlp_width),
            nn.LeakyReLU(),
            nn.Linear(mlp_width, num_heads))



    def _get_slopes(self, n):
        """
        Get slopes for Alibi positional embedding
        n : int = number of heads.
        For best performance, restrict n to a power of 2.
        """

        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio**i for i in range(n)]

        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            return (
                get_slopes_power_of_2(closest_power_of_2)
                + self._get_slopes(2 * closest_power_of_2)[0::2][
                    : n - closest_power_of_2
                ]
            )


    def forward(self, x):
        # [b, np, sq, sk]
        seq_len_q = x.shape[-2]
        seq_len_k = x.shape[-1]

        # Initialize the AliBi matrix to match the first provided key length; grow it exponentially
        # afterwards if longer inputs are provided. This is important for inference, where we will
        # encounter progressively longer samples; it should have no effect at training time.
        if self.cached_seq_len is not None and self.cached_seq_len >= seq_len_k:
            a = self.cached_matrix
        else:
            target_seq_len = (
                seq_len_k
            )
            a = -torch.tril(
                torch.arange(target_seq_len)
                .view(target_seq_len, 1)
                .repeat(1, target_seq_len)
                + torch.arange(0, -target_seq_len, -1)
            )
            a = a.to(x.device).to(x.dtype)
            # print(a)
            slopes = self.slopes.to(a.device).to(a.dtype)
            a = a * slopes.view(self.slopes.shape[0], 1, 1)
            self.cached_seq_len = target_seq_len
            self.cached_matrix = a

        # If the AliBi matrix is larger than the key length, clip it.
        if self.cached_seq_len > seq_len_k:
            a = self.cached_matrix[:, :seq_len_k, :seq_len_k]

        if seq_len_q != seq_len_k:
            # In the train case x has dimensionality [b, np, sq, sk] with sq == sk
            # The number of query tokens is equal to the number of key tokens
            # At inference time with cache in layer_past sq is not equal to sk. sq only contains one token (the last one in the full sequence)
            # In this case we use the appropriate token index of the cache matrix.
            # As the cache matrix could already be bigger from a past inference, not the last token index in the sq sequence is used
            assert (
                seq_len_q == 1
            ), "assumption sq == sk unless at inference time with cache in layer_past with sq == 1"
            a = a[:, seq_len_k - 1, :].view(
                a.shape[0], 1, a.shape[2]
            )  # seq_len_k - 1 points to the last token index in the current inference batch.


        x_a_bias=torch.cat((x,torch.tile(a,(x.shape[0],1,1,1))),dim=1)
        # print(x_a_bias)
        x_a_bias=torch.permute(x_a_bias,(0,2,3,1))
        x_a_bias=self.mlp2(x_a_bias)
        x_a_bias=torch.permute(x_a_bias,(0,3,1,2))

        return x + a+ x_a_bias



class ParallelKerpleLog(torch.nn.Module):
    """Kernelized T5 Relative Position Bias parallelized in the heads dimension"""

    def __init__(
            self,
            neox_args,
    ):
        super().__init__()
        self.heads = neox_args.num_attention_heads
        # self.model_parallel_size = get_model_parallel_world_size()
        # self.model_parallel_rank = get_model_parallel_rank()
        self.num_heads_per_partition = self.heads
        self.pos_emb = neox_args.pos_emb
        self.eps = 1e-2


        # megatron splits across heads, so we need to make sure each head receives the correct matrix
        # assert self.model_parallel_size <= self.heads and self.model_parallel_rank <= self.model_parallel_size

        # Allocate weights and initialize.
        # The kernel has the form -p*log(1+a*|m-n|)
        def get_parameter(scale, init_method):
            if init_method == 'ones':
                return nn.Parameter(torch.ones(
                    self.num_heads_per_partition,
                    device=torch.cuda.current_device(),
                    dtype=neox_args.params_dtype,
                )[:, None, None] * scale)
            elif init_method == 'uniform':
                return nn.Parameter(torch.rand(
                    self.num_heads_per_partition,
                    device=torch.cuda.current_device(),
                    dtype=neox_args.params_dtype,
                )[:, None, None] * scale)

        self.bias_p = get_parameter(2, 'uniform')
        self.bias_a = get_parameter(1, 'uniform')

        self.cached_matrix = None
        self.cached_seq_len = None




    def stats(self):
        def get_stats(name, obj):
            return {name + '_mean': obj.mean().detach().cpu(),
                    name + '_std': obj.std().detach().cpu(),
                    name + '_max': obj.max().detach().cpu(),
                    name + '_min': obj.min().detach().cpu()}

        dd = {}
        self.bias_a.data = self.bias_a.data.clamp(min=self.eps)
        dd.update(get_stats('bias_a', self.bias_a))
        self.bias_p.data = self.bias_p.data.clamp(min=self.eps)
        dd.update(get_stats('bias_p', self.bias_p))
        return dd

    def forward(self, x):
        # [b, np, sq, sk]
        seq_len_q = x.shape[-2]
        seq_len_k = x.shape[-1]
        if self.cached_seq_len != seq_len_k:
            diff = torch.tril(
                torch.arange(seq_len_k, device=x.device).view(seq_len_k, 1).repeat(1, seq_len_k)
                + torch.arange(0, -seq_len_k, -1, device=x.device)
            )
            diff = diff.to(x.dtype)
            self.cached_seq_len = seq_len_k
            self.cached_matrix = diff
        else:
            diff = self.cached_matrix

        self.bias_p.data = self.bias_p.data.clamp(min=self.eps)
        self.bias_a.data = self.bias_a.data.clamp(min=self.eps)
        bias = -self.bias_p * torch.log(1 + self.bias_a * diff)  # log kernel

        if seq_len_q != seq_len_k:
            # In the train case x has dimensionality [b, np, sq, sk] with sq == sk
            # The number of query tokens is equal to the number of key tokens
            # At inference time with cache in layer_past sq is not equal to sk. sq only contains one token (the last one in the full sequence)
            # In this case we use the appropriate token index of the cache matrix.
            # As the cache matrix could already be bigger from a past inference, not the last token index in the sq sequence is used
            assert (
                    seq_len_q == 1
            ), "assumption sq == sk unless at inference time with cache in layer_past with sq == 1"

            if type(bias) != float:
                # seq_len_k - 1 points to the last token index in the current inference batch.
                bias = bias[:, seq_len_k - 1, :].view(bias.shape[0], 1, bias.shape[2])

        return x + bias




class ParallelKerpleLog_CAPE(torch.nn.Module):
    """Kernelized T5 Relative Position Bias parallelized in the heads dimension"""

    def __init__(
            self,
            neox_args,
            layer_number
    ):
        super().__init__()
        self.heads = neox_args.num_attention_heads
        # self.model_parallel_size = get_model_parallel_world_size()
        # self.model_parallel_rank = get_model_parallel_rank()
        self.num_heads_per_partition = self.heads
        self.pos_emb = neox_args.pos_emb
        self.eps = 1e-2

        self.layer_number_visualization = layer_number

        # megatron splits across heads, so we need to make sure each head receives the correct matrix
        # assert self.model_parallel_size <= self.heads and self.model_parallel_rank <= self.model_parallel_size

        # Allocate weights and initialize.
        # The kernel has the form -p*log(1+a*|m-n|)
        def get_parameter(scale, init_method):
            if init_method == 'ones':
                return nn.Parameter(torch.ones(
                    self.num_heads_per_partition,
                    device=torch.cuda.current_device(),
                    dtype=neox_args.params_dtype,
                )[:, None, None] * scale)
            elif init_method == 'uniform':
                return nn.Parameter(torch.rand(
                    self.num_heads_per_partition,
                    device=torch.cuda.current_device(),
                    dtype=neox_args.params_dtype,
                )[:, None, None] * scale)

        self.bias_p = get_parameter(2, 'uniform')
        self.bias_a = get_parameter(1, 'uniform')

        self.cached_matrix = None
        self.cached_seq_len = None
        self.mlp2 = nn.Sequential(
            nn.Linear(self.num_heads_per_partition * 2, neox_args.mlp_width),
            nn.LeakyReLU(),
            nn.Linear(neox_args.mlp_width, self.num_heads_per_partition))




    def stats(self):
        def get_stats(name, obj):
            return {name + '_mean': obj.mean().detach().cpu(),
                    name + '_std': obj.std().detach().cpu(),
                    name + '_max': obj.max().detach().cpu(),
                    name + '_min': obj.min().detach().cpu()}

        dd = {}
        self.bias_a.data = self.bias_a.data.clamp(min=self.eps)
        dd.update(get_stats('bias_a', self.bias_a))
        self.bias_p.data = self.bias_p.data.clamp(min=self.eps)
        dd.update(get_stats('bias_p', self.bias_p))
        return dd

    def forward(self, x):
        # [b, np, sq, sk]
        seq_len_q = x.shape[-2]
        seq_len_k = x.shape[-1]
        if self.cached_seq_len != seq_len_k:
            diff = torch.tril(
                torch.arange(seq_len_k, device=x.device).view(seq_len_k, 1).repeat(1, seq_len_k)
                + torch.arange(0, -seq_len_k, -1, device=x.device)
            )
            diff = diff.to(x.dtype)
            self.cached_seq_len = seq_len_k
            self.cached_matrix = diff
        else:
            diff = self.cached_matrix

        self.bias_p.data = self.bias_p.data.clamp(min=self.eps)
        self.bias_a.data = self.bias_a.data.clamp(min=self.eps)
        bias = -self.bias_p * torch.log(1 + self.bias_a * diff)  # log kernel

        if seq_len_q != seq_len_k:
            # In the train case x has dimensionality [b, np, sq, sk] with sq == sk
            # The number of query tokens is equal to the number of key tokens
            # At inference time with cache in layer_past sq is not equal to sk. sq only contains one token (the last one in the full sequence)
            # In this case we use the appropriate token index of the cache matrix.
            # As the cache matrix could already be bigger from a past inference, not the last token index in the sq sequence is used
            assert (
                    seq_len_q == 1
            ), "assumption sq == sk unless at inference time with cache in layer_past with sq == 1"

            if type(bias) != float:
                # seq_len_k - 1 points to the last token index in the current inference batch.
                bias = bias[:, seq_len_k - 1, :].view(bias.shape[0], 1, bias.shape[2])



        x_a_bias=torch.cat((x,torch.tile(bias,(x.shape[0],1,1,1))),dim=1)
        # print(x_a_bias)
        x_a_bias=torch.permute(x_a_bias,(0,2,3,1))
        x_a_bias=self.mlp2(x_a_bias)
        x_a_bias=torch.permute(x_a_bias,(0,3,1,2))


        return x + bias + x_a_bias




class FIRE(nn.Module):
    def __init__(self, num_heads=12, mlp_width=32, init_c=0.1, init_L=512., eps=1e-6):
        super(FIRE, self).__init__()

        # Define the MLP layers
        self.mlp = nn.Sequential(
            nn.Linear(1, mlp_width),
            nn.ReLU(),
            nn.Linear(mlp_width, num_heads))
        # Initialize c (log transformation parameter)
        self.c = nn.Parameter(torch.tensor(init_c))


        # Initialize L (threshold)

        self.init_L = nn.Parameter(torch.tensor(init_L), requires_grad=False)
        # Learn a multiplier to L

        self.L_multiplier = nn.Parameter(torch.tensor(1.0))

        self.eps = eps

        self.cached_matrix = None
        self.cached_seq_len = None


    def forward(self, x: torch.Tensor):
        seq_length = x.size(2)

        # positions = torch.arange(seq_length, dtype=torch.float, device=x.device)


        seq_len_q = x.shape[-2]
        seq_len_k = x.shape[-1]
        if self.cached_seq_len != seq_len_k:
            rel_distance = torch.tril(
                torch.arange(seq_len_k, device=x.device).view(seq_len_k, 1).repeat(1, seq_len_k)
                + torch.arange(0, -seq_len_k, -1, device=x.device)
            )
            rel_distance = rel_distance.to(torch.float32)
            self.cached_seq_len = seq_len_k
            self.cached_matrix = rel_distance
        else:
            rel_distance = self.cached_matrix
        # print(rel_distance)

        # rel_distance = positions[:, None] - positions[None, :]

        threshold = torch.abs(self.L_multiplier * self.init_L)
        rel_distance_max = torch.max(torch.tril(rel_distance), dim=-1)[0]
        # print(rel_distance_max)

        pos_normalizer = torch.max(rel_distance_max, threshold)

        pos_normalizer = pos_normalizer[:, None]


        # self.c=self.c.to(rel_distance.device)
        rel_distance = torch.log(torch.abs(self.c * rel_distance) + 1
                                 )

        pos_normalizer = torch.log(torch.abs(self.c * pos_normalizer) + 1) + self.eps

        # Progressive interpolation

        normalized_distance = rel_distance / pos_normalizer
        normalized_distance=normalized_distance.to(x.dtype)
        # print(normalized_distance)

        fire_bias = self.mlp(normalized_distance.unsqueeze(-1))

        fire_bias = fire_bias.unsqueeze(0).permute(0, 3, 1, 2)
        # print(x)
        # print(fire_bias)

        return x + fire_bias


class FIRE_CAPE(nn.Module):
    def __init__(self, num_heads=12, mlp_width=32, init_c=0.1, init_L=512., eps=1e-6):
        super(FIRE_CAPE, self).__init__()

        # Define the MLP layers
        self.mlp = nn.Sequential(
            nn.Linear(1, mlp_width),
            nn.LeakyReLU(),
            nn.Linear(mlp_width, num_heads))
        # Initialize c (log transformation parameter)

        self.mlp2 = nn.Sequential(
            nn.Linear(num_heads*2, mlp_width),
            nn.LeakyReLU(),
            nn.Linear(mlp_width, num_heads))
        self.c = nn.Parameter(torch.tensor(init_c))

        # Initialize L (threshold)

        self.init_L = nn.Parameter(torch.tensor(init_L), requires_grad=False)
        # Learn a multiplier to L

        self.L_multiplier = nn.Parameter(torch.tensor(1.0))

        self.eps = eps

        self.cached_matrix = None
        self.cached_seq_len = None

    def forward(self, x: torch.Tensor):
        seq_length = x.size(2)

        # positions = torch.arange(seq_length, dtype=torch.float, device=x.device)

        seq_len_q = x.shape[-2]
        seq_len_k = x.shape[-1]
        if self.cached_seq_len != seq_len_k:
            rel_distance = torch.tril(
                torch.arange(seq_len_k, device=x.device).view(seq_len_k, 1).repeat(1, seq_len_k)
                + torch.arange(0, -seq_len_k, -1, device=x.device)
            )
            rel_distance = rel_distance.to(torch.float32)
            self.cached_seq_len = seq_len_k
            self.cached_matrix = rel_distance
        else:
            rel_distance = self.cached_matrix


        rel_distance=rel_distance.to(x.device)
        # print(rel_distance)

        # rel_distance = positions[:, None] - positions[None, :]

        threshold = torch.abs(self.L_multiplier * self.init_L)
        rel_distance_max=torch.max(torch.tril(rel_distance),dim=-1)[0]
        # print(rel_distance_max)

        pos_normalizer = torch.max(rel_distance_max, threshold)
        # print(threshold)

        pos_normalizer = pos_normalizer[:, None]


        # self.c=self.c.to(rel_distance.device)
        rel_distance = torch.log(torch.abs(self.c * rel_distance) + 1
                                 )

        pos_normalizer = torch.log(torch.abs(self.c * pos_normalizer) + 1) + self.eps

        # Progressive interpolation

        normalized_distance = rel_distance / pos_normalizer
        normalized_distance=normalized_distance.to(x.dtype)
        # print(normalized_distance)

        fire_bias = self.mlp(normalized_distance.unsqueeze(-1))

        fire_bias = fire_bias.unsqueeze(0).permute(0, 3, 1, 2)

        x_fire_bias=torch.cat((x,torch.tile(fire_bias,(x.shape[0],1,1,1))),dim=1)
        x_fire_bias=x_fire_bias.permute(0,2,3,1)
        x_fire_bias=self.mlp2(x_fire_bias)
        x_fire_bias = x_fire_bias.permute(0, 3, 1, 2)

        return x + fire_bias+x_fire_bias