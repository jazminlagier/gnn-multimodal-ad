#!/usr/bin/env python3
"""
KAN layers
"""

import torch
import torch.nn as nn

class KANLinear(nn.Module):
    """
    Linear + learnable spline mixing.
    We keep a simple formulation that's numerically stable.
    """
    def __init__(self, in_features, out_features, grid_size=8, spline_order=3):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        # Base linear
        self.linear = nn.Linear(in_features, out_features, bias=True)

        # Spline coefficients per (out,in,grid)
        self.spline_weight = nn.Parameter(
            torch.randn(out_features, in_features, grid_size) * 0.05
        )
        # Grid points in [-2,2]
        self.register_buffer('grid', torch.linspace(-2.0, 2.0, grid_size))

        # Mix parameter between linear and spline
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def _rbf_basis(self, x):
        """
        Simple RBF basis over grid for each input dim.
        x: [B, in_features]
        returns: [B, in_features, grid_size]
        """
        # x -> [B, in, 1]
        x_exp = x.unsqueeze(-1)
        # grid -> [1, 1, G]
        g = self.grid.view(1, 1, -1)
        return torch.exp(-0.5 * ( (x_exp - g) ** 2 ) / 0.25)  # sigma^2=0.25

    def forward(self, x):
        # linear path
        lin = self.linear(x)                        # [B, out]

        # spline path
        basis = self._rbf_basis(x)                  # [B, in, G]
        # weight: [out, in, G] -> sum over in,G
        spl = torch.einsum('big,oig->bo', basis, self.spline_weight)

        return self.alpha * spl + (1.0 - self.alpha) * lin
