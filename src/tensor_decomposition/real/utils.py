"""Utility functions for real tensor decomposition."""

import torch


def haar_measure(n, batch_size=1):
    """
    Generate random SO(n) matrices from the Haar measure.

    Uses QR decomposition of Gaussian matrices to generate
    uniformly distributed orthogonal matrices, then ensures
    they have determinant +1 (SO(n) rather than O(n)).

    Args:
        n: Dimension of the rotation matrices
        batch_size: Number of rotation matrices to generate

    Returns:
        Batch of SO(n) matrices with shape (batch_size, n, n)
    """
    # Generate random Gaussian matrices
    A = torch.randn(batch_size, n, n, dtype=torch.float32)

    # QR decomposition
    Q, R = torch.linalg.qr(A)

    # Ensure Q is in SO(n) (det = +1) not just O(n)
    # Fix sign based on diagonal of R
    sign = torch.sign(torch.diagonal(R, dim1=-2, dim2=-1))
    sign[sign == 0] = 1
    Q = Q * sign.unsqueeze(-2)

    # Final check: ensure det(Q) = +1
    det = torch.det(Q)
    # If det = -1, flip the first column
    mask = det < 0
    if mask.any():
        Q[mask, :, 0] *= -1

    return Q


def get_lr_schedule_factor(batch_idx, config):
    """
    Get learning rate schedule factor.

    Args:
        batch_idx: Current batch index
        config: Configuration object

    Returns:
        Learning rate multiplication factor
    """
    if not config.use_lr_schedule:
        return 1.0

    # Linear warmup
    if batch_idx < config.warmup_batches:
        return batch_idx / config.warmup_batches

    # Linear decay after decay_start_batch
    if batch_idx >= config.decay_start_batch:
        decay_steps = batch_idx - config.decay_start_batch
        total_decay_steps = config.n_train_batches - config.decay_start_batch
        return max(0.01, 1.0 - 0.99 * (decay_steps / total_decay_steps))

    return 1.0