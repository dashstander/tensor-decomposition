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

