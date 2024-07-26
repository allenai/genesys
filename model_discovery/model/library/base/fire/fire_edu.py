import torch
import torch.nn as nn

class FIRE(nn.Module):
    def __init__(self, num_heads=12, mlp_width=32, init_c=0.1, init_L=512., eps=1e-6):
        """
        FIRE attention bias module.

        Args:
            num_heads: number of attention heads.
            mlp_width: Width of MLP.
            init_c: initial value of log transformation parameter.
            init_L: initial value of thresholding parameter.
            eps: small constant for numerical stability.
        """
        super(FIRE, self).__init__()

        # Define the MLP layers
        self.mlp = nn.Sequential(
            nn.Linear(1, mlp_width),
            nn.ReLU(),
            nn.Linear(mlp_width, num_heads)
        )

        # Initialize c (log transformation parameter)
        self.c = nn.Parameter(torch.tensor(init_c))

        # Initialize L (threshold)
        self.init_L = nn.Parameter(torch.tensor(init_L), requires_grad=False)
        # Learn a multiplier to L
        self.L_multiplier = nn.Parameter(torch.tensor(1.0))

        self.eps = eps

    def forward(self, x: torch.Tensor):
        """
        Compute FIRE attention bias.

        Args:
            x: input sequence, shape [bsz, num_heads, seq_len, hidden_dim]

        Returns:
            attention bias, shape [1, num_heads, seq_len, seq_len]
        """
        seq_length = x.size(2)
        positions = torch.arange(seq_length, dtype=torch.float, device=x.device)
        rel_distance = positions[:, None] - positions[None, :]

        # Thresholding the normalizer
        threshold = torch.abs(self.L_multiplier * self.init_L)
        pos_normalizer = torch.max(positions, threshold)
        pos_normalizer = pos_normalizer[:, None]

        # Amplifying differences among local positions with log transform
        rel_distance = torch.log(torch.abs(self.c * rel_distance) + 1)
        pos_normalizer = torch.log(torch.abs(self.c * pos_normalizer) + 1) + self.eps

        # Progressive interpolation
        normalized_distance = rel_distance / pos_normalizer
        fire_bias = self.mlp(normalized_distance.unsqueeze(-1))
        fire_bias = fire_bias.unsqueeze(0).permute(0, 3, 1, 2)
        return fire_bias
