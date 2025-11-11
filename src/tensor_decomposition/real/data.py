"""Data generation for real tensor decomposition."""

import torch
import random
from torch.utils.data import IterableDataset


def _fix_gauge_simple(factors):
    """
    Simple gauge fixing for real tensors without vmap complications.

    Args:
        factors: List of factor tensors, each shape (batch, dim)

    Returns:
        List of gauge-fixed factors
    """
    if len(factors) < 2:
        return factors

    fixed_factors = []

    # Make first factor's first element positive
    sign1 = torch.sign(factors[0][:, 0])
    sign1 = torch.where(sign1 == 0, torch.ones_like(sign1), sign1)

    # Make second factor's first element positive
    sign2 = torch.sign(factors[1][:, 0] * sign1)
    sign2 = torch.where(sign2 == 0, torch.ones_like(sign2), sign2)

    # Third factor's sign is determined by constraint that product of signs = 1
    sign3 = 1.0 / (sign1 * sign2) if len(factors) > 2 else torch.ones_like(sign1)

    signs = [sign1, sign2, sign3]

    # Apply sign corrections
    for i, (factor, sign) in enumerate(zip(factors, signs)):
        if i < len(signs):
            fixed_factor = factor * sign.unsqueeze(-1)
            fixed_factors.append(fixed_factor)
        else:
            fixed_factors.append(factor)

    return fixed_factors


class Rank1RealTensorDataset(IterableDataset):
    """Dataset for generating rank-1 real tensors."""

    def __init__(self, config, batch_size):
        self.config = config
        self.order = config.order
        self.dim = config.data_dim
        self.batch_size = batch_size

    def sample(self, batch_size):
        """Sample factors uniformly on the real unit hypersphere."""
        # Sample from Gaussian, then normalize to unit magnitude
        x = torch.randn((batch_size, self.config.data_dim), dtype=torch.float32)

        # Normalize to unit magnitude (uniform on hypersphere)
        return x / (x.norm(dim=-1, keepdim=True) + 1e-8)

    def fix_gauge(self, factors):
        """Apply gauge fixing to factors."""
        return _fix_gauge_simple(factors)

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