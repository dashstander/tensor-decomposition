"""Neural network models for complex tensor decomposition."""

import torch
import torch.nn as nn
import numpy as np


class WidelyLinear(nn.Module):
    """
    Widely Linear transform: z = W1*a + W2*a*
    This provides more degrees of freedom than strictly linear transforms.
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # W1 and W2 matrices for widely linear transform
        # Using Xavier initialization for complex numbers
        std = np.sqrt(2.0 / (in_features + out_features))

        # W1 matrix (complex)
        self.W1_real = nn.Parameter(torch.randn(out_features, in_features) * std)
        self.W1_imag = nn.Parameter(torch.randn(out_features, in_features) * std)

        # W2 matrix (complex) - operates on conjugate
        self.W2_real = nn.Parameter(torch.randn(out_features, in_features) * std)
        self.W2_imag = nn.Parameter(torch.randn(out_features, in_features) * std)

        # Bias terms
        self.bias_real = nn.Parameter(torch.zeros(out_features))
        self.bias_imag = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        # x is complex: shape (..., in_features)
        W1 = torch.complex(self.W1_real, self.W1_imag)
        W2 = torch.complex(self.W2_real, self.W2_imag)
        bias = torch.complex(self.bias_real, self.bias_imag)

        # Widely linear transform: z = W1*a + W2*a*
        output = torch.nn.functional.linear(x, W1, None) + torch.nn.functional.linear(x.conj(), W2, None)
        output = output + bias

        return output


class ComplexLinear(nn.Module):
    """Standard complex linear layer for comparison."""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Initialize complex weights and biases
        # Using Xavier initialization for complex numbers
        std = np.sqrt(2.0 / (in_features + out_features))
        self.weight_real = nn.Parameter(torch.randn(out_features, in_features) * std)
        self.weight_imag = nn.Parameter(torch.randn(out_features, in_features) * std)
        self.bias_real = nn.Parameter(torch.zeros(out_features))
        self.bias_imag = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        # x is complex: shape (..., in_features)
        weight = torch.complex(self.weight_real, self.weight_imag)
        bias = torch.complex(self.bias_real, self.bias_imag)

        # Complex matrix multiplication
        return torch.nn.functional.linear(x, weight, bias)


def complex_relu(x):
    """Apply ReLU to both real and imaginary parts."""
    return torch.complex(torch.relu(x.real), torch.relu(x.imag))


def complex_normalize(x, dim=-1, eps=1e-8):
    """Normalize complex tensor by its magnitude."""
    magnitude = torch.abs(x)
    norm = magnitude.norm(dim=dim, keepdim=True)
    return x / (norm + eps)


class ComplexTensorDecompositionMLP(nn.Module):
    """MLP for decomposing complex tensors into their rank-1 factors."""

    def __init__(self, config):
        super().__init__()

        self.order = config.order
        self.data_dim = config.data_dim
        self.use_widely_linear = config.use_widely_linear

        # Choose layer type based on configuration
        LinearLayer = WidelyLinear if config.use_widely_linear else ComplexLinear

        # Build network with complex layers
        self.layers = nn.ModuleList()

        # Input layer
        self.layers.append(LinearLayer(config.tensor_size, config.hidden_dim))

        # Hidden layers
        for _ in range(config.n_layers - 1):
            self.layers.append(LinearLayer(config.hidden_dim, config.hidden_dim))

        # Output layer
        self.output_up = LinearLayer(config.hidden_dim, config.hidden_dim * config.order)
        self.output_down = LinearLayer(config.hidden_dim * config.order, config.order * config.data_dim)

    def forward(self, T_flat):
        squeeze = False
        if T_flat.dim() == 1:
            T_flat = T_flat.unsqueeze(0)
            squeeze = True

        batch = T_flat.shape[0]
        x = T_flat

        # Forward through all layers with complex ReLU activations
        for i, layer in enumerate(self.layers):
            x = complex_relu(layer(x))

        x = self.output_up(x)
        # Split to a list of factors
        outputs = self.output_down(complex_relu(x)).split(self.data_dim, dim=-1)

        # Normalize each factor to lie on the unit hypersphere
        outputs = [complex_normalize(factor, dim=-1, eps=1e-8) for factor in outputs]

        if squeeze:
            outputs = [factor.squeeze(0) for factor in outputs]

        return outputs