"""Device utilities for tensor decomposition."""

import torch


def get_best_device():
    """
    Auto-detect the best available device.

    Returns:
        str: 'cuda', 'mps', or 'cpu'
    """
    if torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'


def get_recommended_config_for_device(device):
    """
    Get recommended configuration adjustments for a device.

    Args:
        device: Device string ('cpu', 'cuda', 'mps')

    Returns:
        dict: Configuration adjustments
    """
    config_adjustments = {}

    if device == 'cpu':
        # Smaller configurations for CPU
        config_adjustments.update({
            'batch_size': 128,
            'hidden_dim': 256,
        })
    elif device == 'mps':
        # MPS (Apple Silicon) configurations
        config_adjustments.update({
            'batch_size': 512,
        })
    elif device == 'cuda':
        # CUDA configurations can handle larger models
        pass

    return config_adjustments

