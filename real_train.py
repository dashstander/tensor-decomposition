#!/usr/bin/env python3
"""Training script for real tensor decomposition."""

import time
import random
import numpy as np
import torch
from tqdm import tqdm
import wandb

from src.tensor_decomposition.real import (
    Config,
    RealTensorDecompositionMLP,
    Rank1RealTensorDataset,
    compute_loss,
    evaluate_comprehensive,
    get_lr_schedule_factor
)
from src.tensor_decomposition.device_utils import get_best_device, get_recommended_config_for_device
from src.tensor_decomposition.checkpoints import create_checkpoint_manager


def train(config):
    """Train the real tensor decomposition model."""
    # Set random seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    # Initialize wandb if enabled
    if config.use_wandb:
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_name,
            config=vars(config)
        )

    # Initialize S3 checkpoint manager
    checkpoint_manager = create_checkpoint_manager(config)
    if checkpoint_manager:
        print(f"✅ S3 checkpoints enabled: s3://{config.s3_bucket}/{config.s3_prefix}")
    else:
        print("ℹ️ S3 checkpoints disabled")

    # Data
    dataset = Rank1RealTensorDataset(config, config.batch_size)
    loader = iter(dataset)

    # Model
    model = RealTensorDecompositionMLP(config)
    model = model.to(config.device)
    print(f"Using Real MLP model")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    # Learning rate scheduler info
    if config.use_lr_schedule:
        print(f"Using LR schedule: warmup for {config.warmup_batches} batches, decay starts at batch {config.decay_start_batch}")
    else:
        print("Using constant learning rate")

    print("Using real tensor decomposition")

    # Training loop with tqdm
    pbar = tqdm(total=config.n_train_batches, desc="Training", unit="batch")

    losses = []
    batch_times = []

    for batch_idx in range(config.n_train_batches):
        batch_start = time.time()

        # Update learning rate
        if config.use_lr_schedule:
            lr_factor = get_lr_schedule_factor(batch_idx, config)
            current_lr = config.lr * lr_factor
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
        else:
            current_lr = config.lr

        # Get batch
        T_flat, factors_true = next(loader)
        T_flat = T_flat.to(config.device)
        factors_true = [factor.to(config.device) for factor in factors_true]

        # Forward pass timing
        forward_start = time.time()
        factors_pred = model(T_flat)
        loss = compute_loss(factors_pred, factors_true, T_flat, config)
        forward_time = time.time() - forward_start

        # Backward pass timing
        backward_start = time.time()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        backward_time = time.time() - backward_start

        # Total batch time
        batch_time = time.time() - batch_start
        batch_times.append(batch_time)

        # Update metrics
        loss_value = loss.item()
        losses.append(loss_value)

        # Update progress bar
        pbar.update(1)
        pbar.set_postfix({
            'loss': f'{loss_value:.4f}',
            'lr': f'{current_lr:.2e}',
            'batch_time': f'{batch_time:.3f}s'
        })

        # Log to wandb
        if config.use_wandb and (batch_idx % config.log_interval == 0):
            wandb.log({
                'train/loss': loss_value,
                'train/lr': current_lr,
                'timing/forward_ms': forward_time * 1000,
                'timing/backward_ms': backward_time * 1000,
                'timing/total_batch_ms': batch_time * 1000,
                'batch': batch_idx
            })

        # Save checkpoint periodically
        if checkpoint_manager and (batch_idx + 1) % config.checkpoint_interval == 0:
            try:
                checkpoint_key = checkpoint_manager.save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    batch_idx=batch_idx + 1,
                    loss=loss_value,
                    config=config,
                    metadata={
                        'lr': current_lr,
                        'device': config.device,
                        'avg_batch_time': np.mean(batch_times[-100:]) if batch_times else 0
                    }
                )

                # Clean up old checkpoints
                checkpoints = checkpoint_manager.list_checkpoints()
                if len(checkpoints) > config.keep_last_n_checkpoints:
                    old_checkpoints = checkpoints[config.keep_last_n_checkpoints:]
                    for old_checkpoint in old_checkpoints:
                        checkpoint_manager.delete_checkpoint(old_checkpoint['name'])
                        pbar.write(f"🗑️ Deleted old checkpoint: {old_checkpoint['name']}")

                pbar.write(f"💾 Checkpoint saved at batch {batch_idx + 1}")

            except Exception as e:
                pbar.write(f"❌ Failed to save checkpoint: {e}")

        # Evaluate periodically
        if (batch_idx + 1) % config.eval_interval == 0:
            eval_results = evaluate_comprehensive(model, config)

            # Print key metrics
            pbar.write(f"\n{'='*60}")
            pbar.write(f"Evaluation at batch {batch_idx + 1}:")
            pbar.write(f"Overall - Factor: {eval_results['overall/factor_loss']:.4f}, Recon: {eval_results['overall/recon_loss']:.4f}")
            pbar.write(f"Order 1 (x⊗x⊗x) - Factor: {eval_results.get('order_1/factor_loss', 0):.4f}, Recon: {eval_results.get('order_1/recon_loss', 0):.4f}")
            pbar.write(f"Order 2 (x⊗x⊗y) - Factor: {eval_results.get('order_2/factor_loss', 0):.4f}, Recon: {eval_results.get('order_2/recon_loss', 0):.4f}")
            pbar.write(f"Order 3 (x⊗y⊗z) - Factor: {eval_results.get('order_3/factor_loss', 0):.4f}, Recon: {eval_results.get('order_3/recon_loss', 0):.4f}")
            pbar.write(f"Factor correlations - (0,1): {eval_results.get('correlations/factor_0_1', 0):.3f}, (0,2): {eval_results.get('correlations/factor_0_2', 0):.3f}, (1,2): {eval_results.get('correlations/factor_1_2', 0):.3f}")
            pbar.write(f"SO(n) Equivariance - Mean variance: {eval_results.get('equivariance/mean_variance', 0):.5f}, Max variance: {eval_results.get('equivariance/max_variance', 0):.5f}")
            pbar.write(f"                     Variance ratio (within/between): {eval_results.get('equivariance/variance_ratio', 0):.3f}")
            pbar.write(f"{'='*60}\n")

            if config.use_wandb:
                # Log all metrics with proper prefixes
                wandb_metrics = {'batch': batch_idx}
                for key, value in eval_results.items():
                    wandb_metrics[f'test/{key}'] = value
                wandb.log(wandb_metrics)

    pbar.close()

    # Save final checkpoint
    if checkpoint_manager:
        try:
            final_key = checkpoint_manager.save_checkpoint(
                model=model,
                optimizer=optimizer,
                batch_idx=config.n_train_batches,
                loss=losses[-1] if losses else 0.0,
                config=config,
                checkpoint_name="final_model",
                metadata={'final': True, 'total_batches': config.n_train_batches}
            )
            print(f"💾 Final checkpoint saved: {final_key}")
        except Exception as e:
            print(f"❌ Failed to save final checkpoint: {e}")

    if config.use_wandb:
        wandb.finish()

    return model, losses


def main():
    """Main training function."""
    # Auto-detect best device or override manually
    device = get_best_device()  # Change this to 'cpu', 'cuda', or 'mps' to override
    device_config = get_recommended_config_for_device(device)

    print(f"Using device: {device}")
    print(f"Recommended config adjustments: {device_config}")

    # Base configuration
    base_config = {
        'seed': 0,
        'data_dim': 2,
        'order': 3,
        'sigma': 1.0,
        'n_train_batches': 10_000,
        'batch_size': 2**10,  # 1024
        'lr': 5.0e-3,
        'hidden_dim': 512,
        'n_layers': 4,
        'dtype': torch.float32,  # Real tensors
        'loss_type': 'factor',  # 'factor', 'reconstruction', or 'combined'
        'loss_weight': 0.5,  # Weight for factor loss when using 'combined'
        'use_lr_schedule': True,
        'warmup_batches': 1000,
        'decay_start_batch': 5000,
        'log_interval': 10,
        'eval_interval': 500,
        'device': device,
        'use_wandb': False,  # Set to True to enable wandb logging
        'wandb_project': "real-tensor-decomposition",
        'wandb_name': None
    }

    # Apply device-specific overrides
    base_config.update(device_config)

    # Create config
    config = Config(**base_config)

    print(f"\nTraining Configuration:")
    print(f"  Device: {config.device}")
    print(f"  Tensor shape: {config.shape}")
    print(f"  Tensor size: {config.tensor_size}")
    print(f"  Factor size: {config.factor_size}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Total batches: {config.n_train_batches}")
    print(f"  Loss function: {config.loss_type}")
    if config.loss_type == 'combined':
        print(f"  Loss weight (factor/reconstruction): {config.loss_weight}/{1-config.loss_weight}")
    print()

    model, losses = train(config)

    # Final comprehensive evaluation
    print("\nFinal evaluation:")
    eval_results = evaluate_comprehensive(model, config, n_test_batches=50)
    print(f"Overall - Factor: {eval_results['overall/factor_loss']:.4f}, Recon: {eval_results['overall/recon_loss']:.4f}")
    print(f"Order 1 (x⊗x⊗x) - Factor: {eval_results.get('order_1/factor_loss', 0):.4f}, Recon: {eval_results.get('order_1/recon_loss', 0):.4f}")
    print(f"Order 2 (x⊗x⊗y) - Factor: {eval_results.get('order_2/factor_loss', 0):.4f}, Recon: {eval_results.get('order_2/recon_loss', 0):.4f}")
    print(f"Order 3 (x⊗y⊗z) - Factor: {eval_results.get('order_3/factor_loss', 0):.4f}, Recon: {eval_results.get('order_3/recon_loss', 0):.4f}")
    print(f"\nSO(n) Equivariance Analysis:")
    print(f"  Mean loss: {eval_results.get('equivariance/mean_loss', 0):.4f} ± {eval_results.get('equivariance/std_loss', 0):.4f}")
    print(f"  Mean variance within orbits: {eval_results.get('equivariance/mean_variance', 0):.5f}")
    print(f"  Max variance within orbits: {eval_results.get('equivariance/max_variance', 0):.5f}")
    print(f"  Variance ratio (within/between): {eval_results.get('equivariance/variance_ratio', 0):.3f}")
    print(f"  (Low variance and ratio ≈ 0 indicates good equivariance)")


if __name__ == '__main__':
    main()