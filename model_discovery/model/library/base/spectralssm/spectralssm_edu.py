import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math

from convolutions import ConvolutionLayer
from autoregressive import AutoRegressiveCausalInput, AutoRegressiveCausalOutput

class AR_STULayer(nn.Module):
    def __init__(self, D_in, D_out, L, K=16, Ku=3, Ky=2, alpha=0.9):
        super(AR_STULayer, self).__init__()

        #self.bn = nn.BatchNorm1d(D_in)

        # Autoregressive sum for u (input) from Eq. 6
        self.autoregressive_u = AutoRegressiveCausalInput(D_in, D_out, Ku=Ku)

        # Autoregressive sum for y (output) from Eq. 6
        self.autoregressive_y = AutoRegressiveCausalOutput(D_out, Ky=Ky, alpha=alpha)

        # Spectral component from Eq. 6
        self.convolution_layer = ConvolutionLayer(D_in, D_out, L, K)

    def forward(self, u):
        u = u.permute(0, 2, 1) # Convert to [B, D, L]

        #u = self.bn(u)

        # Spectral channelization and mixing via convolution
        y = self.convolution_layer(u)

        # Causal auto-regressive process for inputs
        y = y + self.autoregressive_u(u)

        # Causal auto-regressive process for outputs
        y = self.autoregressive_y(y)

        y = y.permute(0, 2, 1) # Convert to [B, L, D]
        return y
    


# Basic MLP feed-forward network like in transformers
class FeedForward(nn.Module):
    def __init__(self, d_in, d_out, mult=4):
        super(FeedForward, self).__init__()

        # Default init works well for these
        hidden_size = d_in * mult
        self.proj_in = nn.Linear(d_in, hidden_size)
        self.act = nn.GELU()
        self.proj_out = nn.Linear(hidden_size, d_out)

    def forward(self, x):
        y = self.proj_in(x)
        y = self.act(y)
        y = self.proj_out(y)
        return y

# Causal average pooling: Output at time T is the average of prior values.
class CausalAveragePooling(nn.Module):
    def __init__(self):
        super(CausalAveragePooling, self).__init__()

    def forward(self, x):
        B, L, D = x.shape
        cumulative_sum = x.cumsum(dim=1)
        timesteps = torch.arange(1, L + 1, device=x.device).view(1, L, 1).expand(B, L, D)
        causal_average = cumulative_sum / timesteps
        return causal_average # [B, L, D]

class SpectralSSM(nn.Module):
    def __init__(self, d_in, d_hidden, d_out, L, num_layers=1):
        super(SpectralSSM, self).__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.d_out = d_out

        # Fig.5: Embedding Layer
        # Default init works well for these
        self.proj_in = nn.Linear(d_in, d_hidden, bias=False)

        # Fig.5: Dense (output) Layer [B, L, d_out]
        # Default init works well for these
        self.proj_out = nn.Linear(d_hidden, d_out, bias=False)

        # Repeat num_layers times:
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = nn.Sequential(
                AR_STULayer(d_hidden, d_hidden, L), # Fig.5: STU
                FeedForward(d_hidden, d_hidden) # Fig.5: MLP+Non-LIN
            )
            self.layers.append(layer)

        # Fig.5: Time Pool
        self.time_pool = CausalAveragePooling()

    def reset(self):
        for layer in self.layers:
            layer.reset()

    def forward(self, u):
        y = self.proj_in(u)

        for layer in self.layers:
            y = y + layer(y)

        # Not sure what the proper operation is here.  The paper does not describe this, and my guess seems wrong.
        #y = self.time_pool(y)

        y = self.proj_out(y) # [B, L, d_out]
        return y