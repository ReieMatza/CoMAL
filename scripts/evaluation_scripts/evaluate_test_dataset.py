#!/usr/bin/env python3
"""
Script to evaluate a model checkpoint on test/validation dataset and output statistics
for angular and radial error measurements.

Usage:
    python scripts/evaluate_test_dataset.py <checkpoint_path> [--dataset_path <path>] [--dataset_split <split>] [--batch_size <size>]
    
Example:
    python scripts/evaluate_test_dataset.py "path/to/checkpoint.ckpt" --batch_size 64
    python scripts/evaluate_test_dataset.py "path/to/checkpoint.ckpt" --dataset_split valid --batch_size 64
"""

import argparse
import sys
import os
from pathlib import Path
import numpy as np
import torch
import torchaudio
from tqdm import tqdm
from collections import defaultdict
import json

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from acoustic_localization_one_mic.fins.common_functions import calculate_theta, calculate_radius
from acoustic_localization_one_mic.fins.model.lightning_module import FinsLightningModule, generate_colored_noise_gpu
from acoustic_localization_one_mic.fins.rir_pipeline import load_dataset_and_get_dataloader, device
from acoustic_localization_one_mic.config_model import DatasetParams, FinsModelConfigParams, TrainParams
from acoustic_localization_one_mic.fins.model.utils.audio import add_noise_batch, audio_normalize_batch


def deterministic_get_noise_and_snr_torch(source: torch.Tensor, use_noise=True, input_signal_length=131070, seed=42):
    """
    Deterministic version of get_noise_and_snr_torch for reproducible evaluation.
    """
    device = source.device
    if use_noise:
        # Use fixed seed for deterministic behavior
        torch.manual_seed(seed)
        np.random.seed(seed)
        import random
        random.seed(seed)
        
        # Always add noise (no random probability)
        min_snr = 0.0
        max_snr = 30.0
        beta = 1.5  # Fixed beta value
        noise = generate_colored_noise_gpu(beta, input_signal_length, device)
        snr_db = torch.tensor([15.0], dtype=torch.float32, device=device)  # Fixed SNR
    else:
        noise = torch.zeros_like(source, device=device)
        snr_db = torch.tensor([0.0], dtype=torch.float32, device=device)

    return noise, snr_db


def deterministic_make_batch_data(model, reverberated_speech, seed=42):
    """
    Deterministic version of make_batch_data for reproducible evaluation.
    """
    reverberated_source = reverberated_speech.unsqueeze(1)
    
    # Use deterministic noise generation
    noise, snr_db = deterministic_get_noise_and_snr_torch(
        reverberated_source, 
        use_noise=True,
        input_signal_length=model.model_config.input_length,
        seed=seed
    )

    batch_size = reverberated_speech.shape[0]

    reverberated_source = audio_normalize_batch(reverberated_source, "rms", model.train_config.rms_level)

    # Noise SNR
    reverberated_source_with_noise = add_noise_batch(reverberated_source, noise, snr_db)

    # Deterministic noise for late part
    torch.manual_seed(seed)
    rir_length = int(model.model_config.rir_length)
    stochastic_noise = torch.randn((batch_size, 1, rir_length), device=model.device)
    batch_stochastic_noise = stochastic_noise.repeat(1, model.model_config.num_filters, 1)

    # Deterministic noise for decoder conditioning
    torch.manual_seed(seed + 1)  # Different seed for different noise
    batch_noise_condition = torch.randn((batch_size, model.model_config.noise_condition_length), device=model.device)

    return (
        reverberated_source_with_noise,
        reverberated_source,
        batch_stochastic_noise,
        batch_noise_condition,
    )


def load_fins_lightning_module_from_checkpoint(checkpoint_path: str, model_config: FinsModelConfigParams, train_config: TrainParams):
    """
    Load a FinsLightningModule from a PyTorch Lightning checkpoint file.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        model_config: Model configuration parameters
        train_config: Training configuration parameters
        
    Returns:
        Loaded FinsLightningModule in evaluation mode
    """
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Create the FinsLightningModule instance
    model = FinsLightningModule(model_config=model_config, train_config=train_config)
    
    # Load the state dict
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    else:
        # If no state_dict key, assume the checkpoint is the state dict
        model.load_state_dict(checkpoint, strict=False)
    
    model.eval()
    model = model.to(device)
    return model


def evaluate_model_on_dataset(model: FinsLightningModule, dataloader, model_config: FinsModelConfigParams, deterministic=True):
    """
    Evaluate the model on a dataset and collect error statistics.
    
    Args:
        model: The loaded FinsLightningModule
        dataloader: DataLoader for the dataset
        model_config: Model configuration parameters
        deterministic: If True, set random seeds for reproducible results
        
    Returns:
        dict: Dictionary containing error statistics
    """
    model.eval()
    
    # Set deterministic behavior if requested
    if deterministic:
        # Set PyTorch to deterministic mode
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Set random seeds for reproducibility
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        np.random.seed(42)
        import random
        random.seed(42)
        
        print("🔒 Running in deterministic mode for reproducible results")
    
    # Initialize lists to store all errors
    all_angular_errors = []
    all_radial_errors = []
    all_true_angles = []
    all_predicted_angles = []
    all_true_radii = []
    all_predicted_radii = []
    
    # Statistics by radius ranges
    radius_ranges = defaultdict(list)
    
    print("Evaluating model on dataset...")
    
    with torch.no_grad():
        for batch_idx, (rir, source_location, receiver_position, room_dimensions, source, reverberation_time, sample_rate) in enumerate(tqdm(dataloader, desc="Processing batches")):
            # Move data to device
            rir = rir.to(model.device)
            source_location = source_location.to(model.device)
            receiver_position = receiver_position.to(model.device)
            room_dimensions = room_dimensions.to(model.device)
            source = source.to(model.device)
            reverberation_time = reverberation_time.to(model.device)
            sample_rate = sample_rate.to(model.device)

            # Convolve source with RIR to get reverberated speech
            source_squeezed = source.squeeze()
            if source_squeezed.dim() == 1:
                source_squeezed = source_squeezed.unsqueeze(0)
            reverberated_speech = torchaudio.functional.convolve(source_squeezed.float(), rir, mode='same')

            # Prepare batch data using deterministic or regular method
            if deterministic:
                (
                    reverberated_source_with_noise,
                    reverberated_source,
                    batch_stochastic_noise,
                    batch_noise_condition,
                ) = deterministic_make_batch_data(model, reverberated_speech, seed=batch_idx)
            else:
                (
                    reverberated_source_with_noise,
                    reverberated_source,
                    batch_stochastic_noise,
                    batch_noise_condition,
                ) = model.make_batch_data(reverberated_speech)

            # Prepare metadata if needed
            metadata = None
            if model.model_config.use_metadata:
                metadata = torch.cat((room_dimensions, receiver_position, reverberation_time.unsqueeze(1)), dim=1)

            # Model forward pass
            predicted_rir, angle_predictions, rad_prediction, z = model(
                reverberated_source_with_noise, batch_stochastic_noise, batch_noise_condition, metadata=metadata
            )

            # Calculate true angles and radii
            theta_degrees_int = calculate_theta(receiver_position, source_location).int().long()
            true_radius = calculate_radius(receiver_position, source_location, rounded=False)
            
            # Get predictions
            predicted_angles = torch.argmax(angle_predictions, dim=1)
            predicted_radii = torch.argmax(rad_prediction, dim=1) * model_config.rad_resolution
            
            # Calculate errors
            angular_error = torch.abs(predicted_angles - theta_degrees_int).float()
            radial_error = torch.abs(predicted_radii - true_radius).float()
            
            # Store results
            all_angular_errors.extend(angular_error.cpu().numpy())
            all_radial_errors.extend(radial_error.cpu().numpy())
            all_true_angles.extend(theta_degrees_int.cpu().numpy())
            all_predicted_angles.extend(predicted_angles.cpu().numpy())
            all_true_radii.extend(true_radius.cpu().numpy())
            all_predicted_radii.extend(predicted_radii.cpu().numpy())
            
            # Group by radius ranges for detailed analysis
            for i, true_rad in enumerate(true_radius.cpu().numpy()):
                if true_rad < 1.5:
                    radius_ranges["0-1.5m"].append({
                        'angular_error': angular_error[i].cpu().numpy(),
                        'radial_error': radial_error[i].cpu().numpy(),
                        'true_angle': theta_degrees_int[i].cpu().numpy(),
                        'predicted_angle': predicted_angles[i].cpu().numpy(),
                        'true_radius': true_rad,
                        'predicted_radius': predicted_radii[i].cpu().numpy()
                    })
                elif true_rad < 2.5:
                    radius_ranges["1.5-2.5m"].append({
                        'angular_error': angular_error[i].cpu().numpy(),
                        'radial_error': radial_error[i].cpu().numpy(),
                        'true_angle': theta_degrees_int[i].cpu().numpy(),
                        'predicted_angle': predicted_angles[i].cpu().numpy(),
                        'true_radius': true_rad,
                        'predicted_radius': predicted_radii[i].cpu().numpy()
                    })
                elif true_rad < 3.5:
                    radius_ranges["2.5-3.5m"].append({
                        'angular_error': angular_error[i].cpu().numpy(),
                        'radial_error': radial_error[i].cpu().numpy(),
                        'true_angle': theta_degrees_int[i].cpu().numpy(),
                        'predicted_angle': predicted_angles[i].cpu().numpy(),
                        'true_radius': true_rad,
                        'predicted_radius': predicted_radii[i].cpu().numpy()
                    })
                else:
                    radius_ranges["3.5m+"].append({
                        'angular_error': angular_error[i].cpu().numpy(),
                        'radial_error': radial_error[i].cpu().numpy(),
                        'true_angle': theta_degrees_int[i].cpu().numpy(),
                        'predicted_angle': predicted_angles[i].cpu().numpy(),
                        'true_radius': true_rad,
                        'predicted_radius': predicted_radii[i].cpu().numpy()
                    })
    
    # Convert to numpy arrays for easier analysis
    all_angular_errors = np.array(all_angular_errors)
    all_radial_errors = np.array(all_radial_errors)
    all_true_angles = np.array(all_true_angles)
    all_predicted_angles = np.array(all_predicted_angles)
    all_true_radii = np.array(all_true_radii)
    all_predicted_radii = np.array(all_predicted_radii)
    
    # Calculate overall statistics
    stats = {
        'overall': {
            'angular_error': {
                'mean': float(np.mean(all_angular_errors)),
                'std': float(np.std(all_angular_errors)),
                'median': float(np.median(all_angular_errors)),
                'min': float(np.min(all_angular_errors)),
                'max': float(np.max(all_angular_errors)),
                'q25': float(np.percentile(all_angular_errors, 25)),
                'q75': float(np.percentile(all_angular_errors, 75)),
                'rmse': float(np.sqrt(np.mean(all_angular_errors**2))),
                'mae': float(np.mean(all_angular_errors))
            },
            'radial_error': {
                'mean': float(np.mean(all_radial_errors)),
                'std': float(np.std(all_radial_errors)),
                'median': float(np.median(all_radial_errors)),
                'min': float(np.min(all_radial_errors)),
                'max': float(np.max(all_radial_errors)),
                'q25': float(np.percentile(all_radial_errors, 25)),
                'q75': float(np.percentile(all_radial_errors, 75)),
                'rmse': float(np.sqrt(np.mean(all_radial_errors**2))),
                'mae': float(np.mean(all_radial_errors))
            },
            'total_samples': len(all_angular_errors)
        },
        'radius_ranges': {}
    }
    
    # Calculate statistics for each radius range
    for range_name, samples in radius_ranges.items():
        if samples:
            angular_errors = [s['angular_error'] for s in samples]
            radial_errors = [s['radial_error'] for s in samples]
            
            stats['radius_ranges'][range_name] = {
                'angular_error': {
                    'mean': float(np.mean(angular_errors)),
                    'std': float(np.std(angular_errors)),
                    'median': float(np.median(angular_errors)),
                    'min': float(np.min(angular_errors)),
                    'max': float(np.max(angular_errors)),
                    'q25': float(np.percentile(angular_errors, 25)),
                    'q75': float(np.percentile(angular_errors, 75)),
                    'rmse': float(np.sqrt(np.mean(np.array(angular_errors)**2))),
                    'mae': float(np.mean(angular_errors))
                },
                'radial_error': {
                    'mean': float(np.mean(radial_errors)),
                    'std': float(np.std(radial_errors)),
                    'median': float(np.median(radial_errors)),
                    'min': float(np.min(radial_errors)),
                    'max': float(np.max(radial_errors)),
                    'q25': float(np.percentile(radial_errors, 25)),
                    'q75': float(np.percentile(radial_errors, 75)),
                    'rmse': float(np.sqrt(np.mean(np.array(radial_errors)**2))),
                    'mae': float(np.mean(radial_errors))
                },
                'sample_count': len(samples)
            }
    
    return stats


def print_statistics(stats, dataset_split='test'):
    """
    Print the evaluation statistics in a formatted way.
    
    Args:
        stats: Dictionary containing the evaluation statistics
        dataset_split: The dataset split that was evaluated ('test' or 'valid')
    """
    print("\n" + "="*80)
    print(f"{dataset_split.upper()} DATASET EVALUATION RESULTS")
    print("="*80)
    
    # Overall statistics
    overall = stats['overall']
    print(f"\nOVERALL STATISTICS ({overall['total_samples']} samples):")
    print("-" * 50)
    
    print("\nANGULAR ERROR (degrees):")
    print(f"  Mean ± Std:     {overall['angular_error']['mean']:.3f} ± {overall['angular_error']['std']:.3f}")
    print(f"  Median:         {overall['angular_error']['median']:.3f}")
    print(f"  Range:          [{overall['angular_error']['min']:.3f}, {overall['angular_error']['max']:.3f}]")
    print(f"  Q25-Q75:        [{overall['angular_error']['q25']:.3f}, {overall['angular_error']['q75']:.3f}]")
    print(f"  RMSE:           {overall['angular_error']['rmse']:.3f}")
    print(f"  MAE:            {overall['angular_error']['mae']:.3f}")
    
    print("\nRADIAL ERROR (meters):")
    print(f"  Mean ± Std:     {overall['radial_error']['mean']:.3f} ± {overall['radial_error']['std']:.3f}")
    print(f"  Median:         {overall['radial_error']['median']:.3f}")
    print(f"  Range:          [{overall['radial_error']['min']:.3f}, {overall['radial_error']['max']:.3f}]")
    print(f"  Q25-Q75:        [{overall['radial_error']['q25']:.3f}, {overall['radial_error']['q75']:.3f}]")
    print(f"  RMSE:           {overall['radial_error']['rmse']:.3f}")
    print(f"  MAE:            {overall['radial_error']['mae']:.3f}")
    
    # Statistics by radius range
    print(f"\nSTATISTICS BY RADIUS RANGE:")
    print("-" * 50)
    
    for range_name, range_stats in stats['radius_ranges'].items():
        print(f"\n{range_name.upper()} ({range_stats['sample_count']} samples):")
        print(f"  Angular Error:  {range_stats['angular_error']['mean']:.3f} ± {range_stats['angular_error']['std']:.3f} (RMSE: {range_stats['angular_error']['rmse']:.3f})")
        print(f"  Radial Error:   {range_stats['radial_error']['mean']:.3f} ± {range_stats['radial_error']['std']:.3f} (RMSE: {range_stats['radial_error']['rmse']:.3f})")
    
    print("\n" + "="*80)


def save_statistics(stats, output_path):
    """
    Save the evaluation statistics to a JSON file.
    
    Args:
        stats: Dictionary containing the evaluation statistics
        output_path: Path to save the JSON file
    """
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\nStatistics saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate model checkpoint on test/validation dataset')
    parser.add_argument('checkpoint_path', type=str, help='Path to the model checkpoint file')
    parser.add_argument('--dataset_path', type=str, default=None, 
                       help='Path to the dataset directory (default: uses DatasetParams default)')
    parser.add_argument('--dataset_split', type=str, default='test', choices=['test', 'valid'],
                       help='Dataset split to evaluate on (default: test, falls back to valid if test not available)')
    parser.add_argument('--batch_size', type=int, default=32, 
                       help='Batch size for evaluation (default: 32)')
    parser.add_argument('--num_workers', type=int, default=4, 
                       help='Number of workers for data loading (default: 4)')
    parser.add_argument('--output_file', type=str, default=None,
                       help='Path to save statistics JSON file (default: checkpoint_name_stats.json)')
    parser.add_argument('--deterministic', action='store_true', default=True,
                       help='Run in deterministic mode for reproducible results (default: True)')
    parser.add_argument('--non-deterministic', action='store_true', default=False,
                       help='Allow non-deterministic behavior (overrides --deterministic)')
    
    args = parser.parse_args()
    
    # Check if checkpoint file exists
    if not os.path.exists(args.checkpoint_path):
        print(f"Error: Checkpoint file not found: {args.checkpoint_path}")
        sys.exit(1)
    
    # Setup configurations
    model_config = FinsModelConfigParams()
    train_config = TrainParams()
    train_config.batch_size = args.batch_size
    train_config.valid_num_workers = args.num_workers
    
    dataset_params = DatasetParams()
    if args.dataset_path:
        dataset_params.rir_dataset_path = args.dataset_path
    
    print(f"Loading model from: {args.checkpoint_path}")
    print(f"Dataset path: {dataset_params.rir_dataset_path}")
    # Determine if we should run deterministically
    deterministic = args.deterministic and not args.non_deterministic
    
    print(f"Dataset split: {args.dataset_split}")
    print(f"Batch size: {args.batch_size}")
    print(f"Number of workers: {args.num_workers}")
    print(f"Deterministic mode: {deterministic}")
    
    try:
        # Load the model
        model = load_fins_lightning_module_from_checkpoint(args.checkpoint_path, model_config, train_config)
        print("✅ Model loaded successfully")
        
        # Try to load the requested dataset split
        dataset_split = args.dataset_split
        dataloader = None
        dataset_config = None
        
        try:
            print(f"Loading {dataset_split} dataset...")
            dataloader, dataset_config = load_dataset_and_get_dataloader(
                dataset_split, 
                dataset_params, 
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                shuffle=False
            )
            print(f"✅ {dataset_split.capitalize()} dataset loaded: {len(dataloader.dataset)} samples")
        except Exception as e:
            print(f"⚠️  Could not load {dataset_split} dataset: {e}")
            
            # If test dataset is not available, try validation dataset
            if dataset_split == 'test':
                print("🔄 Attempting to load validation dataset instead...")
                try:
                    dataloader, dataset_config = load_dataset_and_get_dataloader(
                        'valid', 
                        dataset_params, 
                        batch_size=args.batch_size,
                        num_workers=args.num_workers,
                        shuffle=False
                    )
                    dataset_split = 'valid'
                    print(f"✅ Validation dataset loaded: {len(dataloader.dataset)} samples")
                except Exception as e2:
                    print(f"❌ Could not load validation dataset either: {e2}")
                    raise Exception(f"Neither test nor validation dataset could be loaded. Please check your dataset path: {dataset_params.rir_dataset_path}")
            else:
                raise e
        
        # Evaluate the model
        stats = evaluate_model_on_dataset(model, dataloader, model_config, deterministic=deterministic)
        
        # Print results
        print_statistics(stats, dataset_split)
        
        # Save results
        if args.output_file:
            output_path = args.output_file
        else:
            checkpoint_name = Path(args.checkpoint_path).stem
            output_path = f"{checkpoint_name}_{dataset_split}_stats.json"
        
        save_statistics(stats, output_path)
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main() 