"""Data generation for complex tensor decomposition."""

import torch
import random
from torch.utils.data import IterableDataset


def _fix_gauge(factors):
    """
    Fix gauge symmetry using two constraints:
    1. Make first element of first factor real and positive
    2. Make first element of second factor real

    This uses up the 2 degrees of gauge freedom for rank-1 tensors.

    Args:
        factors: List of factor tensors, each shape (dim,) for vmap or (batch, dim) for regular use

    Returns:
        List of gauge-fixed factors
    """
    if len(factors) < 2:
        return factors

    a, b = factors[0], factors[1]

    # Handle both vmap case (1D) and regular case (2D)
    if a.dim() == 1:
        # vmap case - each factor is (dim,)
        # Step 1: Make a[0] real and positive
        alpha = a[0].conj() / (torch.abs(a[0]) + 1e-8)

        # Step 2: Make b[0] real (after applying alpha transformation)
        b_after_alpha = b  # b doesn't change from alpha transformation
        beta = b_after_alpha[0].conj() / (torch.abs(b_after_alpha[0]) + 1e-8)

        # Step 3: gamma is determined by constraint αβγ = 1
        gamma = 1.0 / (alpha * beta)

        # Apply transformations to all factors
        fixed_factors = []
        transformations = [alpha, beta, gamma]

        for factor, transformation in zip(factors, transformations):
            fixed_factor = factor * transformation
            fixed_factors.append(fixed_factor)
    else:
        # Regular case - each factor is (batch, dim)
        # Step 1: Make a[0] real and positive
        alpha = a[:, 0].conj() / (torch.abs(a[:, 0]) + 1e-8)
        alpha = alpha.unsqueeze(-1)

        # Step 2: Make b[0] real (after applying alpha transformation)
        b_after_alpha = b  # b doesn't change from alpha transformation
        beta = b_after_alpha[:, 0].conj() / (torch.abs(b_after_alpha[:, 0]) + 1e-8)
        beta = beta.unsqueeze(-1)

        # Step 3: gamma is determined by constraint αβγ = 1
        gamma = 1.0 / (alpha * beta)

        # Apply transformations to all factors
        fixed_factors = []
        transformations = [alpha, beta, gamma]

        for factor, transformation in zip(factors, transformations):
            fixed_factor = factor * transformation.squeeze(-1).unsqueeze(-1)
            fixed_factors.append(fixed_factor)

    return fixed_factors


class Rank1ComplexTensorDataset(IterableDataset):
    """Dataset for generating rank-1 complex tensors."""

    def __init__(self, config, batch_size):
        self.config = config
        self.order = config.order
        self.dim = config.data_dim
        self.batch_size = batch_size
        self.canonicalize_fn = torch.vmap(_fix_gauge, in_dims=([0 for _ in range(self.order)],))

    def sample(self, batch_size):
        """Sample factors uniformly on the complex unit hypersphere."""
        # Sample from complex Gaussian, then normalize to unit magnitude
        real = torch.randn((batch_size, self.config.data_dim), dtype=torch.float32)
        imag = torch.randn((batch_size, self.config.data_dim), dtype=torch.float32)
        x = torch.complex(real, imag)

        # Normalize to unit magnitude (uniform on hypersphere)
        return x / (torch.abs(x).norm(dim=-1, keepdim=True) + 1e-8)

    def fix_gauge(self, factors):
        """Apply gauge fixing to factors."""
        return self.canonicalize_fn(factors)

    def __iter__(self):
        """Generate batches of tensors with different symmetry patterns."""
        # Sample factor vectors (one per mode)
        three_factor_bs = int(self.batch_size * 0.75)
        two_factor_bs = int(self.batch_size * 0.125)
        one_factor_bs = self.batch_size - three_factor_bs - two_factor_bs

        while True:
            # Generate three different factors (order 3: x ⊗ y ⊗ z)
            three_factors = [self.sample(three_factor_bs) for _ in range(3)]

            # Generate two different factors (order 2: x ⊗ x ⊗ y or permutations)
            two_factors = [self.sample(two_factor_bs) for _ in range(2)]
            two_factors.append(two_factors[-1])
            random.shuffle(two_factors)

            # Generate one factor (order 1: x ⊗ x ⊗ x)
            single_factor = self.sample(one_factor_bs)
            one_factors = [single_factor, single_factor, single_factor]

            # Combine all factors
            factors = [
                torch.concat([first, second, third], dim=0)
                for first, second, third in zip(three_factors, two_factors, one_factors)
            ]
            canon_factors = self.fix_gauge(factors)

            # Construct tensor
            T = torch.einsum('...i,...j,...k->...ijk', canon_factors[0], canon_factors[1], canon_factors[2])

            # Flatten tensor for network input
            T_flat = T.reshape(self.batch_size, -1).pin_memory()
            # Pin factors for ground truth
            canon_factors = [factor.pin_memory() for factor in canon_factors]

            yield T_flat, canon_factors