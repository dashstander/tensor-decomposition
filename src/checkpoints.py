"""Checkpoint saving and loading utilities for S3."""

import os
import io
import time
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import boto3
from safetensors.torch import save, load
from botocore.exceptions import NoCredentialsError, ClientError


class S3CheckpointManager:
    """Manages saving and loading checkpoints to/from S3 using safetensors."""

    def __init__(self, bucket_name: str, prefix: str = "checkpoints",
                 aws_access_key_id: Optional[str] = None,
                 aws_secret_access_key: Optional[str] = None,
                 region_name: str = "us-east-1"):
        """
        Initialize S3 checkpoint manager.

        Args:
            bucket_name: S3 bucket name
            prefix: Prefix for checkpoint keys in S3
            aws_access_key_id: AWS access key (optional, can use env vars or IAM)
            aws_secret_access_key: AWS secret key (optional, can use env vars or IAM)
            region_name: AWS region
        """
        self.bucket_name = bucket_name
        self.prefix = prefix.rstrip('/')

        # Initialize S3 client
        session_kwargs = {'region_name': region_name}
        if aws_access_key_id and aws_secret_access_key:
            session_kwargs.update({
                'aws_access_key_id': aws_access_key_id,
                'aws_secret_access_key': aws_secret_access_key
            })

        session = boto3.Session(**session_kwargs)
        self.s3_client = session.client('s3')

        # Test connection
        try:
            self.s3_client.head_bucket(Bucket=bucket_name)
            print(f"Successfully connected to S3 bucket: {bucket_name}")
        except NoCredentialsError:
            print(f"AWS credentials not found. Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY or use IAM roles.")
            raise
        except ClientError as e:
            print(f"Error accessing S3 bucket {bucket_name}: {e}")
            raise

    def _get_checkpoint_key(self, checkpoint_name: str) -> str:
        """Generate S3 key for checkpoint."""
        return f"{self.prefix}/{checkpoint_name}.safetensors"

    def save_checkpoint(self,
                       model: torch.nn.Module,
                       optimizer: torch.optim.Optimizer,
                       batch_idx: int,
                       loss: float,
                       config: Any,
                       checkpoint_name: Optional[str] = None,
                       metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Save model and optimizer state to S3 using safetensors format.

        Args:
            model: PyTorch model
            optimizer: PyTorch optimizer
            batch_idx: Current batch index
            loss: Current loss value
            config: Training configuration
            checkpoint_name: Custom checkpoint name (auto-generated if None)
            metadata: Additional metadata to save

        Returns:
            S3 key of saved checkpoint
        """
        if checkpoint_name is None:
            timestamp = int(time.time())
            checkpoint_name = f"checkpoint_batch_{batch_idx:06d}_{timestamp}"

        # Prepare state dict
        state_dict = {
            # Model parameters (safetensors format)
            **{f"model.{k}": v for k, v in model.state_dict().items()},
        }

        # Prepare metadata
        checkpoint_metadata = {
            "batch_idx": str(batch_idx),
            "loss": str(loss),
            "timestamp": str(int(time.time())),
            "model_class": model.__class__.__name__,
            "optimizer_class": optimizer.__class__.__name__,
            "device": str(next(model.parameters()).device),
        }

        # Add config info
        if hasattr(config, '__dict__'):
            for k, v in config.__dict__.items():
                if isinstance(v, (str, int, float, bool)):
                    checkpoint_metadata[f"config.{k}"] = str(v)

        # Add custom metadata
        if metadata:
            for k, v in metadata.items():
                checkpoint_metadata[f"metadata.{k}"] = str(v)

        try:
            # Save model state to bytes buffer using safetensors
            model_buffer = io.BytesIO()
            save(state_dict, model_buffer, metadata=checkpoint_metadata)
            model_buffer.seek(0)

            # Save optimizer state separately (pickle format for now)
            optimizer_buffer = io.BytesIO()
            torch.save(optimizer.state_dict(), optimizer_buffer)
            optimizer_buffer.seek(0)

            # Upload model to S3
            model_key = self._get_checkpoint_key(f"{checkpoint_name}_model")
            self.s3_client.upload_fileobj(
                model_buffer,
                self.bucket_name,
                model_key,
                ExtraArgs={'ContentType': 'application/octet-stream'}
            )

            # Upload optimizer to S3
            optimizer_key = self._get_checkpoint_key(f"{checkpoint_name}_optimizer")
            self.s3_client.upload_fileobj(
                optimizer_buffer,
                self.bucket_name,
                optimizer_key,
                ExtraArgs={'ContentType': 'application/octet-stream'}
            )

            print(f"Checkpoint saved to S3: s3://{self.bucket_name}/{model_key}")
            return model_key

        except Exception as e:
            print(f"Failed to save checkpoint to S3: {e}")
            raise

    def load_checkpoint(self,
                       checkpoint_name: str,
                       model: torch.nn.Module,
                       optimizer: Optional[torch.optim.Optimizer] = None,
                       device: Optional[torch.device] = None) -> Dict[str, Any]:
        """
        Load checkpoint from S3.

        Args:
            checkpoint_name: Name of checkpoint to load
            model: PyTorch model to load state into
            optimizer: PyTorch optimizer to load state into (optional)
            device: Device to map tensors to

        Returns:
            Dictionary with metadata and loading info
        """
        try:
            # Download model checkpoint
            model_key = self._get_checkpoint_key(f"{checkpoint_name}_model")
            model_buffer = io.BytesIO()
            self.s3_client.download_fileobj(self.bucket_name, model_key, model_buffer)
            model_buffer.seek(0)

            # Load model state
            state_dict = load(model_buffer)
            metadata = getattr(state_dict, 'metadata', {})

            # Extract model parameters
            model_state = {}
            for key, value in state_dict.items():
                if key.startswith('model.'):
                    model_key_name = key[6:]  # Remove 'model.' prefix
                    model_state[model_key_name] = value

            # Load into model
            if device:
                model_state = {k: v.to(device) for k, v in model_state.items()}
            model.load_state_dict(model_state)

            # Load optimizer if provided
            if optimizer:
                try:
                    optimizer_key = self._get_checkpoint_key(f"{checkpoint_name}_optimizer")
                    optimizer_buffer = io.BytesIO()
                    self.s3_client.download_fileobj(self.bucket_name, optimizer_key, optimizer_buffer)
                    optimizer_buffer.seek(0)

                    optimizer_state = torch.load(optimizer_buffer, map_location=device)
                    optimizer.load_state_dict(optimizer_state)
                    print(f"✅ Loaded optimizer state from S3")
                except Exception as e:
                    print(f"⚠️ Failed to load optimizer state: {e}")

            print(f"✅ Checkpoint loaded from S3: s3://{self.bucket_name}/{model_key}")
            return {
                'metadata': metadata,
                'batch_idx': int(metadata.get('batch_idx', 0)),
                'loss': float(metadata.get('loss', 0.0)),
                'timestamp': int(metadata.get('timestamp', 0)),
            }

        except Exception as e:
            print(f"❌ Failed to load checkpoint from S3: {e}")
            raise

    def list_checkpoints(self) -> list:
        """List available checkpoints in S3."""
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=f"{self.prefix}/",
                Delimiter='/'
            )

            checkpoints = []
            for obj in response.get('Contents', []):
                key = obj['Key']
                if key.endswith('_model.safetensors'):
                    checkpoint_name = key.replace(f"{self.prefix}/", "").replace('_model.safetensors', '')
                    checkpoints.append({
                        'name': checkpoint_name,
                        'key': key,
                        'size': obj['Size'],
                        'last_modified': obj['LastModified']
                    })

            return sorted(checkpoints, key=lambda x: x['last_modified'], reverse=True)

        except Exception as e:
            print(f"❌ Failed to list checkpoints: {e}")
            return []

    def delete_checkpoint(self, checkpoint_name: str) -> bool:
        """Delete a checkpoint from S3."""
        try:
            # Delete both model and optimizer files
            model_key = self._get_checkpoint_key(f"{checkpoint_name}_model")
            optimizer_key = self._get_checkpoint_key(f"{checkpoint_name}_optimizer")

            # Delete model file
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=model_key)

            # Delete optimizer file (might not exist)
            try:
                self.s3_client.delete_object(Bucket=self.bucket_name, Key=optimizer_key)
            except:
                pass

            print(f"✅ Deleted checkpoint: {checkpoint_name}")
            return True

        except Exception as e:
            print(f"❌ Failed to delete checkpoint {checkpoint_name}: {e}")
            return False


def create_checkpoint_manager(config) -> Optional[S3CheckpointManager]:
    """
    Create S3 checkpoint manager from config, with proper error handling.

    Args:
        config: Configuration object with S3 settings

    Returns:
        S3CheckpointManager instance or None if S3 is disabled/unavailable
    """
    if not getattr(config, 'use_s3_checkpoints', False):
        return None

    bucket_name = getattr(config, 's3_bucket', None)
    if not bucket_name:
        print("⚠️ S3 checkpoints enabled but no bucket specified")
        return None

    try:
        return S3CheckpointManager(
            bucket_name=bucket_name,
            prefix=getattr(config, 's3_prefix', 'tensor-decomposition'),
            aws_access_key_id=getattr(config, 'aws_access_key_id', None),
            aws_secret_access_key=getattr(config, 'aws_secret_access_key', None),
            region_name=getattr(config, 'aws_region', 'us-east-1')
        )
    except Exception as e:
        print(f"⚠️ Failed to initialize S3 checkpoint manager: {e}")
        print("💡 Continuing without S3 checkpoints...")
        return None