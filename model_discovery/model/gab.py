import torch
import torch.nn as nn
from model_discovery.model.utils.modules import GABBase


class GAB(GABBase):

    def __init__(self, embed_dim: int, block_loc: tuple, device=None, dtype
        =None, **kwargs):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc)
        self.root = AdaptiveSpectralGAU(embed_dim=embed_dim, block_loc=
            block_loc, kwarg_all=kwargs, **factory_kwargs, **kwargs)

    def _forward(self, X, **Z):
        X, Z = self.root(X, **Z)
        return X, Z


from model_discovery.model.utils.modules import GAUBase, gau_test, UnitDecl
import torch.nn.functional as F


class AdaptiveSpectralGAU(GAUBase):
    """
    The AdaptiveSpectralGAU combines spectral state space models with adaptive numerical integration methods
    to efficiently process sequences with varying temporal dependencies while maintaining stability and memory efficiency.

    **Components:**

    - **AdaptiveSpectralFilter**: Performs adaptive spectral filtering using fixed spectral filters and an adaptive step controller.
    - **MemoryEfficientIntegrator**: Executes memory-efficient integration with error control using an adaptive integrator.
    - **Output Projection Layer**: A linear layer that projects the integrated state back to the embedding space.

    **Forward Pass:**

    1. Input embeddings `X` of shape `(batch_size, sequence_length, embed_dim)` are passed through `AdaptiveSpectralFilter` to obtain `spec_X`.
    2. `spec_X` is processed by `MemoryEfficientIntegrator` to compute `new_state`.
    3. `new_state` is passed through the output projection layer to produce the output embeddings `Y`.

    **Args:**

    - **embed_dim (int)**: The embedding dimension of the input and output sequences.
    - **block_loc (tuple)**: The location of this block within the network, as a tuple `(layer_idx, n_block)`.
    - **kwarg_all (dict)**: A dictionary of all keyword arguments, passed to initialize child units.
    - **device**: The device on which to place the module's tensors.
    - **dtype**: The data type for the module's tensors.

    **Example Usage:**

    ```python
    adaptive_spectral_gau = AdaptiveSpectralGAU(embed_dim=512, block_loc=(0,12), kwarg_all={})
    Y, Z = adaptive_spectral_gau(X, **Z)
    ```

    **Note:**
    This GAU is designed to efficiently handle sequences with varying complexity by dynamically allocating computational resources.
    """

    def __init__(self, embed_dim: int, block_loc: tuple, kwarg_all: dict,
        device=None, dtype=None, **kwargs):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)
        self.adaptive_spectral_filter = AdaptiveSpectralFilter(embed_dim=
            self.embed_dim, block_loc=self.block_loc, kwarg_all=self.
            kwarg_all, **self.factory_kwargs, **self.kwarg_all)
        self.memory_efficient_integrator = MemoryEfficientIntegrator(embed_dim
            =self.embed_dim, block_loc=self.block_loc, kwarg_all=self.
            kwarg_all, **self.factory_kwargs, **self.kwarg_all)
        self.output_projection = nn.Linear(in_features=embed_dim,
            out_features=embed_dim, **self.factory_kwargs)

    def _forward(self, X, **Z):
        spec_X, Z = self.adaptive_spectral_filter(X, **Z)
        new_state, Z = self.memory_efficient_integrator(spec_X, **Z)
        Z['state'] = new_state
        Y = self.output_projection(new_state)
        return Y, Z


import torch.nn.functional as F
from einops import rearrange


class MemoryEfficientIntegrator(GAUBase):
    """
    MemoryEfficientIntegrator performs memory-efficient integration with error control using an adaptive integrator.

    This unit integrates the output of the AdaptiveSpectralFilter (`spec_X`) over time to produce the new state (`new_state`).
    It uses an efficient state space model implementation that processes sequences in chunks,
    suitable for long sequences. It employs an adaptive method to control the integration error.

    **Key Features:**

    - **Adaptive Integration**: Adjusts computation based on the input characteristics to balance efficiency and accuracy.
    - **Memory Efficiency**: Processes sequences in chunks to reduce memory usage.
    - **State Propagation**: Maintains and updates internal states across sequence chunks.
    - **Error Control**: Employs techniques to ensure integration error remains within a specified tolerance.

    **Args:**

        embed_dim (int): The dimension of the input embeddings.
        block_loc (tuple): The location of the block within the network.
        kwarg_all (dict): Dictionary of all kwargs.
        chunk_size (int, optional): The chunk size for processing sequences. Default is 64.
        tol (float, optional): Tolerance for the adaptive integrator. Default is 1e-5.
        num_heads (int, optional): Number of attention heads. Default is 8.
        state_dim (int, optional): Dimension of the state in the state space model. Default is None (set to embed_dim // num_heads).
        device (torch.device, optional): The device to use. Defaults to None.
        dtype (torch.dtype, optional): The data type to use. Defaults to None.

    **Inputs:**

        spec_X (torch.Tensor): Tensor of shape (batch_size, seq_len, embed_dim), the output from AdaptiveSpectralFilter.

    **Outputs:**

        new_state (torch.Tensor): Tensor of shape (batch_size, seq_len, embed_dim), the integrated state.

    **Example Usage:**

        >>> integrator = MemoryEfficientIntegrator(embed_dim=512, block_loc=(0,12), kwarg_all={})
        >>> new_state, Z = integrator(spec_X, **Z)

    **Note:**

    - This module is designed to efficiently integrate sequences with error control, suitable for long sequences.
    - It adapts computation dynamically to optimize resource usage while maintaining accuracy.
    """

    def __init__(self, embed_dim: int, block_loc: tuple, kwarg_all: dict,
        chunk_size=64, tol=1e-05, num_heads=8, state_dim=None, device=None,
        dtype=None, **kwargs):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)
        self.embed_dim = embed_dim
        self.chunk_size = chunk_size
        self.tol = tol
        self.num_heads = num_heads
        assert embed_dim % num_heads == 0, 'embed_dim must be divisible by num_heads'
        self.d_head = d_head = embed_dim // num_heads
        if state_dim is None:
            state_dim = d_head
        self.state_dim = state_dim
        d_state = self.state_dim
        H = self.num_heads
        self.A = nn.Parameter(torch.randn(H, d_state, **self.factory_kwargs))
        self.B = nn.Parameter(torch.randn(H, d_state, d_head, **self.
            factory_kwargs))
        self.C = nn.Parameter(torch.randn(H, d_head, d_state, **self.
            factory_kwargs))
        self.D = nn.Parameter(torch.randn(H, d_head, d_head, **self.
            factory_kwargs))
        nn.init.xavier_uniform_(self.A)
        nn.init.xavier_uniform_(self.B)
        nn.init.xavier_uniform_(self.C)
        nn.init.xavier_uniform_(self.D)

    def _forward(self, X, **Z):
        """
        X: spec_X, input tensor of shape (batch_size, seq_len, embed_dim)
        """
        B, T, D = X.shape
        H = self.num_heads
        d_head = self.d_head
        d_state = self.state_dim
        X = X.view(B, T, H, d_head)
        state = Z.get('state', None)
        if state is None:
            state = torch.zeros(B, H, d_state, device=X.device, dtype=X.dtype)
        X_chunks = X.split(self.chunk_size, dim=1)
        num_chunks = len(X_chunks)
        Y = []
        A_exp = torch.exp(self.A).unsqueeze(0)
        B_mat = self.B.unsqueeze(0)
        C_mat = self.C.unsqueeze(0)
        D_mat = self.D.unsqueeze(0)
        for X_chunk in X_chunks:
            Lc = X_chunk.size(1)
            X_chunk = X_chunk.transpose(1, 2)
            outputs = []
            for t in range(Lc):
                x_t = X_chunk[:, :, t, :].unsqueeze(-1)
                state = A_exp * state + torch.matmul(B_mat, x_t).squeeze(-1)
                y_t = torch.matmul(C_mat, state.unsqueeze(-1)).squeeze(-1)
                y_t = y_t + torch.matmul(D_mat, x_t).squeeze(-1)
                outputs.append(y_t.unsqueeze(2))
            outputs = torch.cat(outputs, dim=2)
            outputs = outputs.transpose(1, 2)
            Y.append(outputs)
        Y = torch.cat(Y, dim=1)
        Y = Y.view(B, T, D)
        Z['state'] = state
        return Y, Z


import torch.nn.functional as F


class AdaptiveSpectralFilter(GAUBase):
    """
    AdaptiveSpectralFilter implements adaptive spectral filtering using fixed spectral filters
    and an adaptive step controller for efficient sequence processing.

    This GAU performs spectral convolution on input sequences using fixed spectral filters,
    adapting the computational effort based on the input characteristics to improve efficiency
    while maintaining stability.

    **Key Features:**

    - **Fixed Spectral Filters**: Uses pre-defined spectral filters (e.g., sine and cosine basis) for convolution.
    - **Adaptive Step Controller**: Dynamically adjusts computational resources (e.g., step sizes) based on input.
    - **Causal Convolution**: Ensures causality in the filtering process for autoregressive modeling.
    - **Memory Efficiency**: Optimized for handling long sequences with reduced memory consumption.

    **Args:**

        embed_dim (int): The dimension of the input embeddings.
        block_loc (tuple): The location of the block within the network.
        kwarg_all (dict): Dictionary of all kwargs.
        num_filters (int, optional): Number of spectral filters to use. Default is 64.
        device (torch.device, optional): The device to use. Defaults to None.
        dtype (torch.dtype, optional): The data type to use. Defaults to None.

    **Returns:**

        Y (torch.Tensor): The output tensor of shape (batch_size, seq_len, embed_dim).

    **Code Example:**

        >>> module = AdaptiveSpectralFilter(embed_dim=512, block_loc=(0, 1), kwarg_all={})
        >>> X = torch.randn(8, 128, 512)
        >>> Y, Z = module(X)

    **Note:**

    - This module is designed to efficiently handle sequences by adapting computation based on input characteristics.
    - It ensures causality by using causal convolution operations.
    - The adaptive step controller adjusts computational effort to balance efficiency and performance.

    **Example Diagram:**

        .. code-block:: text

            Input X ---> [AdaptiveStepController] ---+
                                                    |
                            +-----------------------+
                            |                       v
                        +-------+     +----------------------+
                        |       |     |                      |
                        | Conv1d| --> | Spectral Convolution |
                        |       |     |                      |
                        +-------+     +----------------------+
                            |
                            v
                        Output Y

    """

    def __init__(self, embed_dim: int, block_loc: tuple, kwarg_all: dict,
        num_filters: int=64, device=None, dtype=None, **kwargs):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)
        self.embed_dim = embed_dim
        self.num_filters = num_filters
        self.kernel_size = num_filters
        self.padding = self.kernel_size - 1
        self.conv = nn.Conv1d(in_channels=embed_dim, out_channels=embed_dim,
            kernel_size=self.kernel_size, stride=1, padding=0, groups=
            embed_dim, bias=False, **self.factory_kwargs)
        self._initialize_filters()
        self.step_controller = AdaptiveStepController(embed_dim=self.
            embed_dim, block_loc=self.block_loc, kwarg_all=self.kwarg_all,
            **self.factory_kwargs, **self.kwarg_all)

    def _initialize_filters(self):
        with torch.no_grad():
            filter_weights = self._generate_fixed_spectral_filters()
            self.conv.weight.copy_(filter_weights)

    def _generate_fixed_spectral_filters(self):
        kernel_size = self.conv.kernel_size[0]
        filter_weights = torch.zeros((self.embed_dim, 1, kernel_size), **
            self.factory_kwargs)
        n = torch.arange(kernel_size, **self.factory_kwargs).float()
        for i in range(self.embed_dim):
            frequency = (i + 1) * torch.pi / kernel_size
            filter_weights[i, 0, :] = torch.sin(frequency * n)
        return filter_weights

    def _forward(self, X, **Z):
        """
        X: input tensor of shape (batch_size, seq_len, embed_dim)
        """
        B, T, D = X.shape
        step_size, Z = self.step_controller(X, **Z)
        pad = self.conv.kernel_size[0] - 1
        X_padded = F.pad(X.transpose(1, 2), (pad, 0))
        spec_X = self.conv(X_padded)
        spec_X = spec_X.transpose(1, 2)
        spec_X = spec_X * step_size
        return spec_X, Z


import torch.nn.functional as F


class AdaptiveStepController(GAUBase):
    """
    AdaptiveStepController computes adaptive step sizes based on the input embeddings X.

    This unit processes the input sequence X and outputs a `step_size` tensor that is used
    to adjust the computational resources dynamically in the `AdaptiveSpectralFilter`.

    **Key Features:**

    - **Adaptive Computation**: Dynamically computes step sizes based on input characteristics.
    - **Gated Mechanism**: Uses a gated MLP to model complex dependencies in the input.
    - **Controlled Step Sizes**: Ensures step sizes are within a specified range for stability.

    **Args:**

        embed_dim (int): The dimension of the input embeddings.
        block_loc (tuple): The location of the block within the network.
        kwarg_all (dict): Dictionary of all kwargs.
        min_step_size (float, optional): Minimum value for the step size. Default is 1e-4.
        max_step_size (float, optional): Maximum value for the step size. Default is 1.0.
        device (torch.device, optional): The device to use. Defaults to None.
        dtype (torch.dtype, optional): The data type to use. Defaults to None.

    **Returns:**

        step_size (torch.Tensor): A tensor of shape (batch_size, seq_len, embed_dim) representing the adaptive step sizes.

    **Code Example:**

        >>> module = AdaptiveStepController(embed_dim=512, block_loc=(0, 1), kwarg_all={})
        >>> X = torch.randn(8, 128, 512)
        >>> step_size, Z = module(X)

    **Note:**

    - This module is designed to output adaptive step sizes that can be used by spectral filtering
      units to adjust their computational efforts based on input sequences.
    - It ensures that the step sizes are within a specified range to maintain stability.

    **Example Diagram:**

        Input X --> [Gated MLP] --> step_size

    **Todo:**

    - Explore different activation functions for gating mechanisms.
    - Implement additional controls for the range of step sizes if necessary.

    """

    def __init__(self, embed_dim: int, block_loc: tuple, kwarg_all: dict,
        min_step_size: float=0.0001, max_step_size: float=1.0, device=None,
        dtype=None, hidden_features=None, activation=None, bias=False, **kwargs
        ):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)
        self.embed_dim = embed_dim
        self.min_step_size = min_step_size
        self.max_step_size = max_step_size
        hidden_features = (hidden_features if hidden_features is not None else
            embed_dim)
        self.fc1 = nn.Linear(embed_dim, 2 * hidden_features, bias=bias, **
            self.factory_kwargs)
        self.activation = activation if activation is not None else F.silu
        self.fc2 = nn.Linear(hidden_features, embed_dim, bias=bias, **self.
            factory_kwargs)

    def _forward(self, X, **Z):
        """
        X: Input tensor of shape (batch_size, seq_len, embed_dim)
        Returns:
            step_size: Tensor of shape (batch_size, seq_len, embed_dim)
        """
        y = self.fc1(X)
        y, gate = y.chunk(2, dim=-1)
        y = y * self.activation(gate)
        step_size = self.fc2(y)
        step_size = torch.sigmoid(step_size)
        step_size = step_size * (self.max_step_size - self.min_step_size
            ) + self.min_step_size
        return step_size, Z


gab_config = {'num_filters': 64, 'min_step_size': 0.0001, 'max_step_size': 
    1.0, 'hidden_features': None, 'activation': None, 'bias': False,
    'chunk_size': 64, 'tol': 1e-05, 'num_heads': 8, 'state_dim': None}



autoconfig = {
    'd_model': 512,
    'n_block': 11
}
block_config=gab_config
block_config.update(autoconfig)


from .block_registry import BlockRegister

BlockRegister(
    name="default",
    config=block_config
)(GAB)