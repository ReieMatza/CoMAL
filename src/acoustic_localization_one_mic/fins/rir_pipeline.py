from pathlib import Path
import multiprocessing as mp
import os
import glob
import re
from typing import Optional

import torch
import wandb
from torch.utils.data import DataLoader
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import lightning as pl
# Add import for SimpleProfiler (not strictly needed for 'simple' string usage, but for clarity)
# from lightning.pytorch.profilers import SimpleProfiler

from acoustic_localization_one_mic.fins.model.model import FilteredNoiseShaper
from acoustic_localization_one_mic.config_model import FinsModelConfigParams, TrainParams, \
    WandbConfigParams, DatasetParams
from acoustic_localization_one_mic.fins.model.lightning_module import FinsLightningModule
from acoustic_localization_one_mic.rir_dataset.rir_dataset import RirDataset
from acoustic_localization_one_mic.rir_dataset.genereate_dataset import RirDataEntry, HSample

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_tensor_cores():
    """
    Setup optimal float32 matmul precision for GPUs with Tensor Cores.
    Falls back gracefully for GPUs without Tensor Cores.
    """
    if torch.cuda.is_available():
        try:
            # Get GPU compute capability
            major, minor = torch.cuda.get_device_capability()
            
            # Tensor Cores are available on compute capability 7.0+ (V100, RTX series, A100, etc.)
            if major >= 7:
                # Set medium precision for better performance on Tensor Core GPUs
                torch.set_float32_matmul_precision('medium')
                print(f"GPU with Tensor Cores detected (compute capability {major}.{minor}). "
                      f"Set float32 matmul precision to 'medium' for optimal performance.")
            else:
                print(f"GPU without Tensor Cores detected (compute capability {major}.{minor}). "
                      f"Using default float32 matmul precision.")
        except Exception as e:
            print(f"Could not determine GPU capabilities: {e}. Using default precision.")
    else:
        print("No CUDA device available. Using CPU.")


def find_best_checkpoint(checkpoint_dir: str, wandb_run_name: str, monitor_metric: str = 'valid/error/angular'):
    """
    Find the best checkpoint for a given wandb run name based on the monitored metric.
    
    Expected directory structure:
    {checkpoint_dir}/{wandb_run_name}/{filename}.ckpt
    
    Example:
    checkpoints/run-name/epoch=45-step=1234-valid_loss=0.25.ckpt
    
    Args:
        checkpoint_dir: Base checkpoint directory (e.g., 'checkpoints')
        wandb_run_name: The wandb run name (e.g., 'atomic-resonance-703')
        monitor_metric: The metric being monitored (e.g., 'valid/error/angular')
        
    Returns:
        tuple: (checkpoint_path, epoch) or (None, 0) if no checkpoint found
        - checkpoint_path: Full path to the best checkpoint file
        - epoch: Next epoch number to resume from (best_epoch + 1)
    """
    run_checkpoint_dir = os.path.join(checkpoint_dir, wandb_run_name)
    
    if not os.path.exists(run_checkpoint_dir):
        print(f"No checkpoint directory found for wandb run: {wandb_run_name}")
        return None, 0
    
    # Find all checkpoint files in the directory
    checkpoint_files = glob.glob(os.path.join(run_checkpoint_dir, "*.ckpt"))
    
    if not checkpoint_files:
        # Try old .pt format as fallback
        checkpoint_files = glob.glob(os.path.join(run_checkpoint_dir, "*.pt"))
    
    if not checkpoint_files:
        print(f"No checkpoint files found for wandb run: {wandb_run_name}")
        return None, 0
    
    # For best checkpoint, we need to load each checkpoint and check the monitored metric
    best_checkpoint = None
    best_metric_value = float('inf')  # Lower is better for error metrics
    best_epoch = -1
    
    print(f"Searching for best checkpoint based on metric: {monitor_metric}")
    
    for checkpoint_file in checkpoint_files:
        try:
            # Load checkpoint to get metric value
            # First try with weights_only=True for security
            try:
                checkpoint = torch.load(checkpoint_file, map_location='cpu', weights_only=True)
            except Exception as weights_only_error:
                # If weights_only fails, try without it (for trusted checkpoints)
                print(f"Warning: weights_only=True failed for {checkpoint_file}, trying without it...")
                checkpoint = torch.load(checkpoint_file, map_location='cpu', weights_only=False)
            
            # Extract metric value from checkpoint
            metric_value = None
            if 'callbacks' in checkpoint and 'ModelCheckpoint' in checkpoint['callbacks']:
                # Get the monitored metric value from the checkpoint
                callbacks = checkpoint['callbacks']['ModelCheckpoint']
                if 'monitor' in callbacks and callbacks['monitor'] == monitor_metric:
                    metric_value = callbacks.get('best_model_score', float('inf'))
                else:
                    # Try to find the metric in the checkpoint state
                    if 'state_dict' in checkpoint:
                        # Look for the metric in the lightning module state
                        for key in checkpoint['state_dict'].keys():
                            if monitor_metric.replace('/', '_') in key:
                                metric_value = checkpoint['state_dict'][key].item()
                                break
            
            # If we couldn't find the metric, try to extract epoch and use that as fallback
            if metric_value is None:
                # Extract epoch from filename
                filename = os.path.basename(checkpoint_file)
                epoch_match = re.search(r'epoch[=]?(\d+)', filename)
                if epoch_match:
                    epoch = int(epoch_match.group(1))
                    metric_value = epoch  # Use epoch as fallback metric (lower = better)
                else:
                    continue
            
            # Update best checkpoint if this one is better
            if metric_value < best_metric_value:
                best_metric_value = metric_value
                best_checkpoint = checkpoint_file
                
                # Extract epoch for return value
                filename = os.path.basename(checkpoint_file)
                epoch_match = re.search(r'epoch[=]?(\d+)', filename)
                if epoch_match:
                    best_epoch = int(epoch_match.group(1))
                else:
                    best_epoch = 0
                    
        except Exception as e:
            print(f"Warning: Could not load checkpoint {checkpoint_file}: {e}")
            continue
    
    if best_checkpoint:
        print(f"Found best checkpoint for {wandb_run_name}: {best_checkpoint} (epoch {best_epoch}, metric: {best_metric_value:.4f})")
        return best_checkpoint, best_epoch + 1  # Resume from next epoch
    else:
        print(f"No valid checkpoint files found for wandb run: {wandb_run_name}")
        return None, 0


def find_latest_checkpoint(checkpoint_dir: str, wandb_run_name: str):
    """
    Find the latest checkpoint for a given wandb run name by epoch number.
    
    Expected directory structure:
    {checkpoint_dir}/{wandb_run_name}/{filename}.ckpt
    
    Example:
    checkpoints/run-name/epoch=45-step=1234-valid_loss=0.25.ckpt
    
    Args:
        checkpoint_dir: Base checkpoint directory (e.g., 'checkpoints')
        wandb_run_name: The wandb run name (e.g., 'atomic-resonance-703')
        
    Returns:
        tuple: (checkpoint_path, epoch) or (None, 0) if no checkpoint found
        - checkpoint_path: Full path to the latest checkpoint file
        - epoch: Next epoch number to resume from (latest_epoch + 1)
    """
    run_checkpoint_dir = os.path.join(checkpoint_dir, wandb_run_name)
    
    if not os.path.exists(run_checkpoint_dir):
        print(f"No checkpoint directory found for wandb run: {wandb_run_name}")
        return None, 0
    
    # Find all checkpoint files in the directory
    checkpoint_files = glob.glob(os.path.join(run_checkpoint_dir, "*.ckpt"))
    
    if not checkpoint_files:
        # Try old .pt format as fallback
        checkpoint_files = glob.glob(os.path.join(run_checkpoint_dir, "*.pt"))
    
    if not checkpoint_files:
        print(f"No checkpoint files found for wandb run: {wandb_run_name}")
        return None, 0
    
    # Extract epoch numbers and find the latest one
    latest_epoch = -1
    latest_checkpoint = None
    
    for checkpoint_file in checkpoint_files:
        # Extract epoch number from filename like "epoch=45-step=1234-valid_loss=0.25.ckpt"
        filename = os.path.basename(checkpoint_file)
        epoch_match = re.search(r'epoch[=]?(\d+)', filename)
        if epoch_match:
            epoch = int(epoch_match.group(1))
            if epoch > latest_epoch:
                latest_epoch = epoch
                latest_checkpoint = checkpoint_file
    
    if latest_checkpoint:
        print(f"Found latest checkpoint for {wandb_run_name}: {latest_checkpoint} (epoch {latest_epoch})")
        return latest_checkpoint, latest_epoch + 1  # Resume from next epoch
    else:
        print(f"No valid checkpoint files found for wandb run: {wandb_run_name}")
        return None, 0


def train(train_dataloader, valid_dataloader, train_config, model_config:FinsModelConfigParams):
    # Setup optimal precision for Tensor Cores if available
    setup_tensor_cores()

    # Handle resuming from wandb run name
    checkpoint_path = None
    
    if train_config.resume_from_wandb_run:
        if train_config.resume_checkpoint_type.lower() == "best":
            checkpoint_path, _ = find_best_checkpoint(
                train_config.checkpoint_dir, 
                train_config.resume_from_wandb_run,
                monitor_metric='valid/error/angular'
            )
            print(f"Looking for best checkpoint based on 'valid/error/angular' metric")
        else:
            checkpoint_path, _ = find_latest_checkpoint(
                train_config.checkpoint_dir, 
                train_config.resume_from_wandb_run
            )
            print(f"Looking for latest checkpoint by epoch number")
            
        if checkpoint_path:
            print(f"Resuming training from wandb run '{train_config.resume_from_wandb_run}' using {train_config.resume_checkpoint_type} checkpoint")
        else:
            print(f"Could not find checkpoint for wandb run '{train_config.resume_from_wandb_run}', starting fresh training")

    model = FinsLightningModule(model_config=model_config, train_config=train_config)

    # Setup wandb logger
    wandb_config = WandbConfigParams()

    # Handle wandb resuming
    wandb_resume_id = None
    if train_config.resume_from_wandb_run and checkpoint_path:
        # Try to extract the wandb run ID from the checkpoint
        try:
            # First try with weights_only=True for security
            try:
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            except Exception as weights_only_error:
                # If weights_only fails, try without it (for trusted checkpoints)
                print(f"Warning: weights_only=True failed for wandb run ID extraction, trying without it...")
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            
            # Look for wandb run ID in various possible locations
            if 'callbacks' in checkpoint and 'WandbLogger' in checkpoint['callbacks']:
                wandb_resume_id = checkpoint['callbacks']['WandbLogger'].get('_wandb_init', {}).get('id')
            elif 'wandb_run_id' in checkpoint:
                wandb_resume_id = checkpoint['wandb_run_id']
            elif 'hyper_parameters' in checkpoint and 'wandb_run_id' in checkpoint['hyper_parameters']:
                wandb_resume_id = checkpoint['hyper_parameters']['wandb_run_id']
            
            if wandb_resume_id:
                print(f"Found wandb run ID in checkpoint: {wandb_resume_id}")
                print(f"Will resume the same wandb run: {wandb_resume_id}")
            else:
                print(f"No wandb run ID found in checkpoint. Will create a new wandb run.")
        except Exception as e:
            print(f"Warning: Could not extract wandb run ID from checkpoint: {e}")
            print("Will create a new wandb run.")
    
    # Log all configuration parameters to wandb
    # Use the config parameter in WandbLogger to pass all configuration
    wandb_logger = WandbLogger(
        project=wandb_config.project,
        save_dir=wandb_config.dir,
        resume='allow' if wandb_resume_id else None,
        id=wandb_resume_id,
        config=wandb_config.config
    )

    # Setup checkpointing
    # Use the wandb run id for the checkpoint directory (guaranteed to exist and unique)
    # Wait for wandb experiment to be properly initialized
    if hasattr(wandb_logger.experiment, 'name') and wandb_logger.experiment.name is not None:
        wandb_run_dir = wandb_logger.experiment.name
    elif hasattr(wandb_logger.experiment, 'id') and wandb_logger.experiment.id is not None:
        wandb_run_dir = wandb_logger.experiment.id
    else:
        # Fallback to a default name if neither is available
        wandb_run_dir = f"run_{wandb_resume_id if wandb_resume_id else 'new'}"
    
    # Debug print to check types and values
    print("checkpoint_dir:", train_config.checkpoint_dir, type(train_config.checkpoint_dir))
    print("wandb_run_dir:", wandb_run_dir, type(wandb_run_dir))
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(str(train_config.checkpoint_dir), str(wandb_run_dir)),
        filename='epoch={epoch}-step={step}-{valid/error/angular:.4f}',
        save_top_k=3,
        monitor='valid/error/angular',
        mode='min'  # Lower is better for error metrics
    )

    # Setup trainer
    trainer_kwargs = dict(
        max_epochs=train_config.num_epochs,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        precision=train_config.precision,  # Use configurable precision from config
    )
    if train_config.profiler:
        trainer_kwargs['profiler'] = train_config.profiler
    # Add multi-GPU/distributed support
    trainer_kwargs['devices'] = train_config.devices
    if train_config.strategy is not None:
        trainer_kwargs['strategy'] = train_config.strategy
    trainer = pl.Trainer(**trainer_kwargs)

    # Train the model
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader,
                ckpt_path=checkpoint_path)


def load_model_from_file(filepath, config):
    """
    Load a model from a PyTorch Lightning checkpoint file.
    
    Args:
        filepath: Path to the checkpoint file
        config: Model configuration parameters
        
    Returns:
        Loaded model in evaluation mode
    """
    # Load the checkpoint
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    
    # Create the model instance
    model = FilteredNoiseShaper(config=config).to(device)
    
    # Handle different checkpoint formats
    if 'state_dict' in checkpoint:
        # Standard PyTorch Lightning checkpoint format
        state_dict = checkpoint['state_dict']
        # Remove 'model.' prefix from state dict keys if present
        # (Lightning modules often prefix the actual model with 'model.')
        cleaned_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('model.'):
                cleaned_key = key[6:]  # Remove 'model.' prefix
                cleaned_state_dict[cleaned_key] = value
            else:
                cleaned_state_dict[key] = value
        
        model.load_state_dict(cleaned_state_dict, strict=False)
    elif 'model_state_dict' in checkpoint:
        # Legacy format
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    elif 'model_with_arc' in checkpoint:
        # Special case for model with architecture
        model = checkpoint['model_with_arc']
    else:
        # Assume the checkpoint is just the state dict
        model.load_state_dict(checkpoint, strict=False)
    
    model.eval()
    return model


def load_lightning_module_from_checkpoint(filepath, model_config, train_config):
    """
    Load a PyTorch Lightning module from a checkpoint file.
    
    Args:
        filepath: Path to the checkpoint file
        model_config: Model configuration parameters
        train_config: Training configuration parameters
        
    Returns:
        Loaded FinsLightningModule in evaluation mode
    """
    # Load the checkpoint
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    
    # Create the model instance
    model = FinsLightningModule(model_config=model_config, train_config=train_config)
    
    # Load the state dict
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    else:
        raise ValueError("Checkpoint does not contain 'state_dict'")
    
    model.eval()
    return model


def load_dataset_and_get_dataloader(name: str, dataset_params: DatasetParams, batch_size: int, num_workers: int = 1,
                                    prefetch_factor = None, persistent_workers = True, shuffle: bool = True, **kwargs):
    dataset_path = Path(dataset_params.rir_dataset_path) / name
    dataset = RirDataset(root_dir=dataset_path, librispeech_path=dataset_params.librispeech_path)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        pin_memory=True,
        multiprocessing_context='spawn',
    )
    return dataloader, dataset.config

def load_train_valid_data_sets(dataset_params: DatasetParams, train_config: Optional[TrainParams] = None):
    if train_config is None:
        train_config = TrainParams()

    train_dataloader, train_dataset_config = load_dataset_and_get_dataloader('train', dataset_params,
                                                                             num_workers=train_config.train_num_workers,
                                                                             shuffle=True,
                                                                             **train_config.__dict__)
    valid_dataloader, valid_dataset_config = load_dataset_and_get_dataloader('valid', dataset_params,
                                                                             num_workers=train_config.valid_num_workers,
                                                                             shuffle=False,
                                                                             **train_config.__dict__)

    return train_dataloader, valid_dataloader, train_config, train_dataset_config


def main_pipline(train_config: Optional[TrainParams] = None, dataset_params: Optional[DatasetParams] = None):
    """
    Main training pipeline.

    To resume training from a previous wandb run, set the resume_from_wandb_run
    parameter in TrainParams to the wandb run name (e.g., 'atomic-resonance-703').

    Example usage for resuming:
        train_config = TrainParams(resume_from_wandb_run='atomic-resonance-703')
        main_pipline(train_config)

    Example usage for training without contrastive learning (more efficient):
        train_config = TrainParams(
            use_contrastive_learning=False,
            batch_size=100,
            gradient_accumulation_steps=2,
        )
        main_pipline(train_config)
    """
    if train_config is None:
        train_config = TrainParams()
    if dataset_params is None:
        dataset_params = DatasetParams()

    train_dataloader, valid_dataloader, train_config, train_dataset_config = load_train_valid_data_sets(
        dataset_params, train_config)

    config = FinsModelConfigParams()
    train(train_dataloader, valid_dataloader, train_config, config)


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main_pipline()
