"""Loss functions for real tensor decomposition."""

import torch


def factor_cosine_distance(pred_factors, true_factors):
    """
    Compute cosine distance directly on factors (gauge-invariant).

    Args:
        pred_factors: List of predicted factors, each shape (batch, dim)
        true_factors: List of true factors, each shape (batch, dim)

    Returns:
        Scalar loss
    """
    total_loss = 0

    for pred_factor, true_factor in zip(pred_factors, true_factors):
        # Factors are already normalized, compute cosine similarity
        cosine_sim = (pred_factor * true_factor).sum(dim=-1)
        total_loss += torch.mean(1 - torch.abs(cosine_sim))  # Use abs for real tensors (sign ambiguity)

    return total_loss / len(pred_factors)  # Average across factors


def reconstruction_mse(pred_factors, T_flat, config):
    """
    Reconstruct tensor from predicted factors and measure MSE.

    Args:
        pred_factors: List of predicted factor vectors (already normalized to unit magnitude)
        T_flat: Target tensor (flattened)
        config: Configuration object

    Returns:
        Scalar MSE loss
    """
    batch_size = T_flat.shape[0]

    # Reconstruct tensor from list of factors
    T_reconstructed = torch.einsum('bi,bj,bk->bijk', pred_factors[0], pred_factors[1], pred_factors[2])

    T_reconstructed_flat = T_reconstructed.reshape(batch_size, -1)
    T_target = T_flat.reshape(batch_size, -1)

    # Compute MSE
    diff = T_reconstructed_flat - T_target
    return torch.mean(torch.sum(diff ** 2, dim=-1))


def compute_loss(pred_factors, true_factors, T_flat, config):
    """
    Compute loss based on configuration.

    Args:
        pred_factors: List of predicted factors
        true_factors: List of true factors
        T_flat: Target tensor (flattened)
        config: Configuration object

    Returns:
        Scalar loss
    """
    if config.loss_type == 'factor':
        return factor_cosine_distance(pred_factors, true_factors)
    elif config.loss_type == 'reconstruction':
        return reconstruction_mse(pred_factors, T_flat, config)
    elif config.loss_type == 'combined':
        factor_loss = factor_cosine_distance(pred_factors, true_factors)
        recon_loss = reconstruction_mse(pred_factors, T_flat, config)
        return config.loss_weight * factor_loss + (1 - config.loss_weight) * recon_loss
    else:
        raise ValueError(f"Unknown loss_type: {config.loss_type}")