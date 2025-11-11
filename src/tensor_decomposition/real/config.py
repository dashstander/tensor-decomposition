"""Configuration for real tensor decomposition training."""

from dataclasses import dataclass
import torch


@dataclass
class Config:
    # Required parameters (no defaults)
    seed: int
    data_dim: int      # dimension of each mode (real)
    order: int         # number of copies of the vector space R^data_dim
    sigma: float      # isotropic Gaussian scale
    n_train_batches: int  # Total number of training batches
    batch_size: int
    lr: float
    hidden_dim: int
    n_layers: int
    dtype: torch.dtype = torch.float32  # Real tensors use float32

    # Optional parameters (with defaults)
    device: str = 'cpu'  # 'cpu', 'cuda', or 'mps' for Apple Silicon
    log_interval: int = 10  # Log metrics every N batches
    eval_interval: int = 500  # Evaluate on test set every N batches
    loss_type: str = 'factor'  # 'factor', 'reconstruction', 'combined'
    loss_weight: float = 0.5  # Weight for factor loss when using 'combined' (1-weight for reconstruction)
    use_wandb: bool = False  # Enable wandb logging
    wandb_project: str = "real-tensor-decomposition"  # Wandb project name
    wandb_name: str = None  # Run name (auto-generated if None)
    use_lr_schedule: bool = True  # Enable learning rate scheduling
    warmup_batches: int = 1000  # Number of batches for linear warmup
    decay_start_batch: int = 5000  # Batch to start linear decay (if None, starts after warmup)

    # S3 Checkpoint parameters
    use_s3_checkpoints: bool = False  # Enable S3 checkpoint saving
    s3_bucket: str = None  # S3 bucket name for checkpoints
    s3_prefix: str = "real-tensor-decomposition"  # S3 key prefix
    checkpoint_interval: int = 1000  # Save checkpoint every N batches
    keep_last_n_checkpoints: int = 5  # Keep only last N checkpoints
    aws_access_key_id: str = None  # AWS access key (optional, can use env vars)
    aws_secret_access_key: str = None  # AWS secret key (optional, can use env vars)
    aws_region: str = "us-east-1"  # AWS region

    # Derived parameters
    @property
    def shape(self):
        return tuple([self.data_dim] * self.order)

    @property
    def tensor_size(self):
        return self.data_dim ** self.order

    @property
    def factor_size(self):
        return self.data_dim * self.order