"""Neural network models for real tensor decomposition."""

import torch
import torch.nn as nn
import numpy as np


def normalize(x, dim=-1, eps=1e-8):
    """Normalize tensor by its magnitude."""
    norm = x.norm(dim=dim, keepdim=True)
    return x / (norm + eps)


class RealTensorDecompositionMLP(nn.Module):
    """MLP for decomposing real tensors into their rank-1 factors."""

    def __init__(self, config):
        super().__init__()

        self.order = config.order
        self.data_dim = config.data_dim

        # Build network with real layers
        self.layers = nn.ModuleList()

        # Input layer
        self.layers.append(nn.Linear(config.tensor_size, config.hidden_dim))

        # Hidden layers
        for _ in range(config.n_layers - 1):
            self.layers.append(nn.Linear(config.hidden_dim, config.hidden_dim))

        # Output layer
        self.output_up = nn.Linear(config.hidden_dim, config.hidden_dim * config.order)
        self.output_down = nn.Linear(config.hidden_dim * config.order, config.order * config.data_dim)

    def forward(self, T_flat):
        squeeze = False
        if T_flat.dim() == 1:
            T_flat = T_flat.unsqueeze(0)
            squeeze = True

        batch = T_flat.shape[0]
        x = T_flat

        # Forward through all layers with ReLU activations
        for i, layer in enumerate(self.layers):
            x = torch.relu(layer(x))

        x = self.output_up(x)
        # Split to a list of factors
        outputs = self.output_down(torch.relu(x)).split(self.data_dim, dim=-1)

        # Normalize each factor to lie on the unit hypersphere
        outputs = [normalize(factor, dim=-1, eps=1e-8) for factor in outputs]

        if squeeze:
            outputs = [factor.squeeze(0) for factor in outputs]

        return outputs