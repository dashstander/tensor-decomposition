"""Real tensor decomposition module."""

from .config import Config
from .model import RealTensorDecompositionMLP, normalize
from .data import Rank1RealTensorDataset
from .losses import compute_loss, factor_cosine_distance, reconstruction_mse
from .evaluation import evaluate_comprehensive, evaluate_equivariance
from .utils import get_lr_schedule_factor, haar_measure

__all__ = [
    'Config',
    'RealTensorDecompositionMLP',
    'normalize',
    'Rank1RealTensorDataset',
    'compute_loss',
    'factor_cosine_distance',
    'reconstruction_mse',
    'evaluate_comprehensive',
    'evaluate_equivariance',
    'get_lr_schedule_factor',
    'haar_measure',
]