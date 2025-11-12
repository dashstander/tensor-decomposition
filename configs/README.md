# Configuration Files

This directory contains YAML configuration files for training tensor decomposition models.

## Available Configurations

### Complex Tensor Decomposition
- `complex_default.yaml` - Default configuration for full training
- `complex_small.yaml` - Small configuration for testing/debugging

### Real Tensor Decomposition
- `real_default.yaml` - Default configuration for full training
- `real_small.yaml` - Small configuration for testing/debugging

## Usage

### Basic Usage
```bash
# Train with default complex config
python complex_train.py --config configs/complex_default.yaml

# Train with default real config
python real_train.py --config configs/real_default.yaml

# Train with small configs for testing
python complex_train.py --config configs/complex_small.yaml
python real_train.py --config configs/real_small.yaml
```

### Override Parameters
You can override any configuration parameter from the command line:

```bash
# Override learning rate and batch size
python complex_train.py --config configs/complex_default.yaml --override lr=0.001 batch_size=512

# Override multiple parameters
python real_train.py --config configs/real_default.yaml --override \
    n_train_batches=5000 \
    hidden_dim=256 \
    use_wandb=true
```

## Configuration Parameters

### Basic Parameters
- `seed`: Random seed for reproducibility
- `data_dim`: Dimension of each tensor mode
- `order`: Number of tensor modes (typically 3)
- `sigma`: Gaussian noise scale

### Training Parameters
- `n_train_batches`: Total number of training batches
- `batch_size`: Batch size for training
- `lr`: Learning rate
- `hidden_dim`: Hidden dimension of MLP
- `n_layers`: Number of hidden layers
- `dtype`: Data type ("complex64", "float32", etc.)

### Loss Configuration
- `loss_type`: "factor", "reconstruction", or "combined"
- `loss_weight`: Weight for factor loss when using "combined"

### Model Configuration (Complex only)
- `use_widely_linear`: Whether to use widely linear layers

### Learning Rate Scheduling
- `use_lr_schedule`: Enable learning rate scheduling
- `warmup_batches`: Number of warmup batches
- `decay_start_batch`: Batch to start decay

### Logging and Evaluation
- `log_interval`: Log metrics every N batches
- `eval_interval`: Evaluate every N batches
- `use_wandb`: Enable Weights & Biases logging
- `wandb_project`: W&B project name

### Device Configuration
- `device`: "auto" (auto-detect), "cpu", "cuda", or "mps"

### S3 Checkpointing (Optional)
- `use_s3_checkpoints`: Enable S3 checkpoint saving
- `s3_bucket`: S3 bucket name
- `s3_prefix`: S3 key prefix
- `checkpoint_interval`: Save checkpoint every N batches

## Creating Custom Configurations

Copy an existing configuration file and modify the parameters as needed:

```bash
cp configs/complex_default.yaml configs/my_custom_config.yaml
# Edit my_custom_config.yaml
python complex_train.py --config configs/my_custom_config.yaml
```