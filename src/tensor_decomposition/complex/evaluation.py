"""Evaluation functions for tensor decomposition models."""

import torch
import numpy as np
import random
from .data import Rank1ComplexTensorDataset
from .losses import factor_cosine_distance, reconstruction_mse
from .utils import haar_measure


def evaluate_comprehensive(model, config, n_test_batches=20):
    """
    Comprehensive evaluation with detailed metrics.

    Args:
        model: Trained model
        config: Configuration object
        n_test_batches: Number of test batches

    Returns:
        Dictionary of evaluation metrics
    """
    # Set random seed for reproducibility
    torch.manual_seed(config.seed + 1)
    np.random.seed(config.seed + 1)

    model.eval()

    # We need to generate specific types of tensors for analysis
    metrics = {
        'overall': {'factor_loss': [], 'recon_loss': []},
        'order_1': {'factor_loss': [], 'recon_loss': [], 'count': 0},  # x ⊗ x ⊗ x
        'order_2': {'factor_loss': [], 'recon_loss': [], 'count': 0},  # x ⊗ x ⊗ y or permutations
        'order_3': {'factor_loss': [], 'recon_loss': [], 'count': 0},  # x ⊗ y ⊗ z
        'per_factor': {'factor_0': [], 'factor_1': [], 'factor_2': []},
        'equivariance': {'losses': [], 'variances': []},
    }

    with torch.no_grad():
        for batch_idx in range(n_test_batches):
            # Generate batch with known structure
            batch_size = config.batch_size

            # Split batch into three types
            order_1_size = batch_size // 3
            order_2_size = batch_size // 3
            order_3_size = batch_size - order_1_size - order_2_size

            dataset = Rank1ComplexTensorDataset(config, batch_size)

            # Generate order-1 tensors (x ⊗ x ⊗ x)
            if order_1_size > 0:
                x = dataset.sample(order_1_size)
                factors_1 = [x, x, x]
                factors_1 = dataset.fix_gauge(factors_1)
                T_1 = torch.einsum('...i,...j,...k->...ijk', factors_1[0], factors_1[1], factors_1[2])
                T_1_flat = T_1.reshape(order_1_size, -1)

            # Generate order-2 tensors (x ⊗ x ⊗ y and permutations)
            if order_2_size > 0:
                x = dataset.sample(order_2_size)
                y = dataset.sample(order_2_size)
                factors_2 = [x, x, y]  # Could randomize order
                random.shuffle(factors_2)
                factors_2 = dataset.fix_gauge(factors_2)
                T_2 = torch.einsum('...i,...j,...k->...ijk', factors_2[0], factors_2[1], factors_2[2])
                T_2_flat = T_2.reshape(order_2_size, -1)

            # Generate order-3 tensors (x ⊗ y ⊗ z)
            if order_3_size > 0:
                factors_3 = [dataset.sample(order_3_size) for _ in range(3)]
                factors_3 = dataset.fix_gauge(factors_3)
                T_3 = torch.einsum('...i,...j,...k->...ijk', factors_3[0], factors_3[1], factors_3[2])
                T_3_flat = T_3.reshape(order_3_size, -1)

            # Combine batches
            T_flat_list = []
            factors_true_list = [[], [], []]
            tensor_types = []

            if order_1_size > 0:
                T_flat_list.append(T_1_flat)
                for i in range(3):
                    factors_true_list[i].append(factors_1[i])
                tensor_types.extend([1] * order_1_size)

            if order_2_size > 0:
                T_flat_list.append(T_2_flat)
                for i in range(3):
                    factors_true_list[i].append(factors_2[i])
                tensor_types.extend([2] * order_2_size)

            if order_3_size > 0:
                T_flat_list.append(T_3_flat)
                for i in range(3):
                    factors_true_list[i].append(factors_3[i])
                tensor_types.extend([3] * order_3_size)

            # Concatenate
            T_flat = torch.cat(T_flat_list, dim=0).to(config.device)
            factors_true = [torch.cat(factor_list, dim=0).to(config.device) for factor_list in factors_true_list]
            tensor_types = torch.tensor(tensor_types).to(config.device)

            # Get predictions
            factors_pred = model(T_flat)

            # Compute losses
            factor_loss = factor_cosine_distance(factors_pred, factors_true)
            recon_loss = reconstruction_mse(factors_pred, T_flat, config)

            # Overall metrics
            metrics['overall']['factor_loss'].append(factor_loss.item())
            metrics['overall']['recon_loss'].append(recon_loss.item())

            # Per-factor cosine distances
            for i, (pred, true) in enumerate(zip(factors_pred, factors_true)):
                cosine_sim = (pred * true.conj()).sum(dim=-1).real
                per_factor_loss = 1 - cosine_sim  # Per-sample losses
                metrics['per_factor'][f'factor_{i}'].extend(per_factor_loss.cpu().numpy())

            # Compute losses per tensor type
            for tensor_type in [1, 2, 3]:
                mask = tensor_types == tensor_type
                if mask.any():
                    # Factor loss for this type
                    type_factor_loss = 0
                    for pred, true in zip(factors_pred, factors_true):
                        cosine_sim = (pred[mask] * true[mask].conj()).sum(dim=-1).real
                        type_factor_loss += torch.mean(1 - cosine_sim)
                    type_factor_loss /= len(factors_pred)

                    # Reconstruction loss for this type
                    type_factors = [f[mask] for f in factors_pred]
                    type_T_flat = T_flat[mask]
                    type_recon_loss = reconstruction_mse(type_factors, type_T_flat, config)

                    key = f'order_{tensor_type}'
                    metrics[key]['factor_loss'].append(type_factor_loss.item())
                    metrics[key]['recon_loss'].append(type_recon_loss.item())
                    metrics[key]['count'] += mask.sum().item()

    # Compute summary statistics
    results = {}

    # Overall metrics
    results['overall/factor_loss'] = np.mean(metrics['overall']['factor_loss'])
    results['overall/recon_loss'] = np.mean(metrics['overall']['recon_loss'])

    # Per tensor type metrics
    for tensor_type in [1, 2, 3]:
        key = f'order_{tensor_type}'
        if metrics[key]['factor_loss']:
            results[f'{key}/factor_loss'] = np.mean(metrics[key]['factor_loss'])
            results[f'{key}/recon_loss'] = np.mean(metrics[key]['recon_loss'])
            results[f'{key}/count'] = metrics[key]['count']

    # Per-factor metrics and correlations
    for i in range(3):
        factor_losses = np.array(metrics['per_factor'][f'factor_{i}'])
        if len(factor_losses) > 0:
            results[f'per_factor/factor_{i}_mean'] = np.mean(factor_losses)
            results[f'per_factor/factor_{i}_std'] = np.std(factor_losses)

    # Compute correlations between factor losses
    if all(len(metrics['per_factor'][f'factor_{i}']) > 0 for i in range(3)):
        factor_0_losses = np.array(metrics['per_factor']['factor_0'])
        factor_1_losses = np.array(metrics['per_factor']['factor_1'])
        factor_2_losses = np.array(metrics['per_factor']['factor_2'])

        # Ensure same length
        min_len = min(len(factor_0_losses), len(factor_1_losses), len(factor_2_losses))
        factor_0_losses = factor_0_losses[:min_len]
        factor_1_losses = factor_1_losses[:min_len]
        factor_2_losses = factor_2_losses[:min_len]

        results['correlations/factor_0_1'] = np.corrcoef(factor_0_losses, factor_1_losses)[0, 1]
        results['correlations/factor_0_2'] = np.corrcoef(factor_0_losses, factor_2_losses)[0, 1]
        results['correlations/factor_1_2'] = np.corrcoef(factor_1_losses, factor_2_losses)[0, 1]

    # U(2) Equivariance Test
    evaluate_equivariance(model, config, metrics, results)

    model.train()
    return results


def evaluate_equivariance(model, config, metrics, results):
    """
    Test U(2) equivariance of the model.

    Args:
        model: Model to evaluate
        config: Configuration object
        metrics: Dictionary to store metrics
        results: Dictionary to store results
    """

    n_equivariance_samples = 100  # Number of base tensors to test
    n_orbit_samples = 1000  # Number of U(2) transformations per tensor

    equivariance_losses_per_tensor = []

    with torch.no_grad():
        for sample_idx in range(n_equivariance_samples):
            # Generate a random test tensor
            dataset = Rank1ComplexTensorDataset(config, 1)
            factors_base = [dataset.sample(1) for _ in range(3)]
            factors_base = dataset.fix_gauge(factors_base)

            # Generate all U(2) matrices for this orbit at once
            U_batch = haar_measure(config.data_dim, batch_size=n_orbit_samples).to(config.device)

            # Prepare base factors for batched transformation
            factors_base_device = [factor.to(config.device) for factor in factors_base]

            # Apply U to each factor in batch: U_batch @ factor
            # U_batch shape: (n_orbit_samples, data_dim, data_dim)
            # factor shape: (1, data_dim) -> need to broadcast
            orbit_losses = []

            # Transform factors: einsum 'bij,kj->bki' where b=batch, i,j=matrix dims, k=samples
            factors_transformed = []
            for factor in factors_base_device:
                # factor shape: (1, data_dim), U_batch shape: (n_orbit_samples, data_dim, data_dim)
                # Want: (n_orbit_samples, 1, data_dim)
                factor_expanded = factor.expand(n_orbit_samples, -1, -1)  # (n_orbit_samples, 1, data_dim)
                transformed = torch.bmm(factor_expanded, U_batch.transpose(-2, -1))  # (n_orbit_samples, 1, data_dim)
                factors_transformed.append(transformed)

            # Construct transformed tensors in batch
            T_transformed = torch.einsum('bki,bkj,bkl->bkijl',
                                       factors_transformed[0],
                                       factors_transformed[1],
                                       factors_transformed[2])
            # Reshape for model input: (n_orbit_samples, 1, tensor_size)
            T_transformed_flat = T_transformed.reshape(n_orbit_samples, 1, -1)

            # Process each sample in the orbit
            for i in range(n_orbit_samples):
                sample_tensor = T_transformed_flat[i]  # (1, tensor_size)
                sample_factors = [f[i] for f in factors_transformed]  # Each (1, data_dim)

                # Get model predictions
                factors_pred = model(sample_tensor)

                # Compute loss against transformed factors
                factor_loss = factor_cosine_distance(factors_pred, sample_factors)
                orbit_losses.append(factor_loss.item())

            # Compute mean and variance of losses across the orbit
            orbit_losses = np.array(orbit_losses)
            mean_loss = np.mean(orbit_losses)
            var_loss = np.var(orbit_losses)

            metrics['equivariance']['losses'].append(mean_loss)
            metrics['equivariance']['variances'].append(var_loss)
            equivariance_losses_per_tensor.append(orbit_losses)

    # Compute equivariance statistics
    if metrics['equivariance']['losses']:
        results['equivariance/mean_loss'] = np.mean(metrics['equivariance']['losses'])
        results['equivariance/std_loss'] = np.std(metrics['equivariance']['losses'])
        results['equivariance/mean_variance'] = np.mean(metrics['equivariance']['variances'])
        results['equivariance/max_variance'] = np.max(metrics['equivariance']['variances'])

        # Compute overall variance (across all orbits and tensors)
        all_orbit_losses = np.concatenate(equivariance_losses_per_tensor)
        results['equivariance/overall_variance'] = np.var(all_orbit_losses)

        # Ratio of within-orbit variance to between-orbit variance
        # High ratio means the model is inconsistent within orbits
        within_orbit_var = np.mean(metrics['equivariance']['variances'])
        between_orbit_var = np.var(metrics['equivariance']['losses'])
        if between_orbit_var > 0:
            results['equivariance/variance_ratio'] = within_orbit_var / between_orbit_var
        else:
            results['equivariance/variance_ratio'] = 0.0