"""Utility functions for tensor decomposition."""

import torch
import numpy as np


def haar_measure(n, batch_size=None):
    """
    Generate random unitary matrices distributed according to Haar measure.

    Args:
        n: Dimension of the unitary matrices
        batch_size: If provided, generates batch_size matrices of shape (batch_size, n, n)
                   If None, generates a single matrix of shape (n, n)

    Returns:
        Unitary matrix/matrices sampled from Haar measure
    """
    if batch_size is None:
        # Single matrix
        z = (torch.randn(n, n, dtype=torch.complex64) + 1j * torch.randn(n, n, dtype=torch.complex64)) / np.sqrt(2.0)
        q, r = torch.linalg.qr(z)
        d = torch.diagonal(r)
        ph = d / torch.abs(d)
        q = q * ph
        return q
    else:
        # Batch of matrices
        z = (torch.randn(batch_size, n, n, dtype=torch.complex64) + 1j * torch.randn(batch_size, n, n, dtype=torch.complex64)) / np.sqrt(2.0)
        q, r = torch.linalg.qr(z)
        d = torch.diagonal(r, dim1=-2, dim2=-1)  # Get diagonals for each matrix in the batch
        ph = d / torch.abs(d)
        # Broadcast multiplication: q has shape (batch_size, n, n), ph has shape (batch_size, n)
        q = q * ph.unsqueeze(-2)  # Add dimension to make ph (batch_size, 1, n) for broadcasting
        return q


def get_lr_schedule_factor(batch, config):
    """
    Get learning rate scaling factor for given batch.

    Schedule:
    - Linear warmup from 0 to 1.0 over warmup_batches
    - Constant at 1.0 until decay_start_batch
    - Linear decay from 1.0 to 0.1 from decay_start_batch to n_train_batches

    Args:
        batch: Current batch (0-indexed)
        config: Configuration object

    Returns:
        Learning rate scaling factor
    """
    if not config.use_lr_schedule:
        return 1.0

    # Warmup phase: linear from 0 to 1
    if batch < config.warmup_batches:
        return (batch + 1) / config.warmup_batches

    # Determine decay start batch
    decay_start = config.decay_start_batch
    if decay_start is None:
        decay_start = config.warmup_batches

    # Constant phase
    if batch < decay_start:
        return 1.0

    # Decay phase: linear from 1.0 to 0.1
    if batch < config.n_train_batches:
        decay_progress = (batch - decay_start) / (config.n_train_batches - decay_start)
        return 1.0 - 0.9 * decay_progress  # Decay from 1.0 to 0.1

    return 0.1  # Minimum LR