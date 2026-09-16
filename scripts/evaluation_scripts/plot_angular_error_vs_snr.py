#!/usr/bin/env python3
"""
Script to evaluate one or two FINS checkpoints at different SNR levels and plot angular & radial error vs SNR.

Usage (single):
    python scripts/evaluation_scripts/plot_angular_error_vs_snr.py <checkpoint_path> [--dataset_path <path>] [--dataset_split <split>] [--batch_size <size>]

Usage (comparison of two checkpoints):
    python scripts/evaluation_scripts/plot_angular_error_vs_snr.py <checkpoint_path> --checkpoint_path_b <checkpoint_b> \
        [--label_a <label_for_a>] [--label_b <label_for_b>] [other args...]

Examples:
    python scripts/evaluation_scripts/plot_angular_error_vs_snr.py "path/to/contrastive.ckpt" --batch_size 64 --include_random_baseline
    python scripts/evaluation_scripts/plot_angular_error_vs_snr.py "path/to/contrastive.ckpt" --checkpoint_path_b "path/to/non_contrastive.ckpt" \
        --label_a "with-contrastive" --label_b "no-contrastive" --dataset_split valid --batch_size 64
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torchaudio
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import json

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from acoustic_localization_one_mic.fins.common_functions import calculate_theta, calculate_radius
from acoustic_localization_one_mic.fins.model.lightning_module import FinsLightningModule, generate_colored_noise_gpu
from acoustic_localization_one_mic.fins.rir_pipeline import load_dataset_and_get_dataloader, device
from acoustic_localization_one_mic.config_model import DatasetParams, FinsModelConfigParams, TrainParams
from acoustic_localization_one_mic.fins.model.utils.audio import add_noise_batch, audio_normalize_batch


def calculate_fixed_angle_baseline_stats(model_config: FinsModelConfigParams, fixed_angle: int = 90, num_samples: int = 10000):
    """
    Calculate theoretical performance statistics for fixed angle baseline (e.g., always predicting 90 degrees).
    
    This function computes the expected mean and standard deviation for angular and radial
    errors when predictions are always set to a fixed angle (default: 90 degrees), providing 
    a simple heuristic baseline for comparison with trained models.
    
    Args:
        model_config: Model configuration containing num_angles, max_rad_value, rad_resolution
        fixed_angle: Fixed angle to always predict (default: 90 degrees)
        num_samples: Number of samples to simulate for statistical calculation
        
    Returns:
        dict: Dictionary containing angular and radial error statistics for fixed angle baseline
    """
    # Angular error calculation
    # Always predict the fixed angle (e.g., 90 degrees)
    # True angles are uniformly distributed across the range (0 to num_angles-1)
    num_angles = model_config.num_angles  # 181 classes (0-180 degrees)
    
    # Ensure fixed angle is within valid range
    fixed_angle = max(0, min(fixed_angle, num_angles - 1))
    
    # Generate predictions and true values
    np.random.seed(42)  # For reproducible baseline
    
    # Fixed predictions (always the same angle)
    fixed_angle_predictions = np.full(num_samples, fixed_angle, dtype=np.int32)
    # Random true angles (uniform across all angle classes) 
    random_true_angles = np.random.randint(0, num_angles, num_samples)
    
    # Calculate angular errors (absolute difference)
    angular_errors = np.abs(fixed_angle_predictions - random_true_angles).astype(np.float32)
    
    # Radial error calculation
    # For fixed angle baseline, use average radius prediction (middle of range)
    max_rad_value = model_config.max_rad_value  # 6.0 meters
    rad_resolution = model_config.rad_resolution  # 0.1 meters
    num_rad_classes = int(max_rad_value / rad_resolution) + 1  # 61 classes (0.0 to 6.0 in 0.1m steps)
    
    # Fixed radius prediction (middle of range) and random true values
    fixed_rad_class = num_rad_classes // 2  # Middle of the range
    fixed_rad_predictions = np.full(num_samples, fixed_rad_class, dtype=np.int32)
    random_true_rad_indices = np.random.randint(0, num_rad_classes, num_samples)
    
    # Convert indices to actual radial values (in meters)
    predicted_radii = fixed_rad_predictions * rad_resolution
    true_radii = random_true_rad_indices * rad_resolution
    
    # Calculate radial errors (absolute difference in meters)
    radial_errors = np.abs(predicted_radii - true_radii).astype(np.float32)
    
    # Calculate radial accuracy (for the default threshold)
    rad_acc_threshold_m = 0.5  # Default threshold
    radial_accuracy = float(np.mean(radial_errors <= rad_acc_threshold_m))
    
    # Compile statistics
    fixed_angle_baseline_stats = {
        'angular_error': {
            'mean': float(np.mean(angular_errors)),
            'std': float(np.std(angular_errors)),
            'median': float(np.median(angular_errors)),
            'rmse': float(np.sqrt(np.mean(angular_errors**2))),
            'mae': float(np.mean(angular_errors))
        },
        'radial_error': {
            'mean': float(np.mean(radial_errors)),
            'std': float(np.std(radial_errors)),
            'median': float(np.median(radial_errors)),
            'rmse': float(np.sqrt(np.mean(radial_errors**2))),
            'mae': float(np.mean(radial_errors))
        },
        'total_samples': num_samples,
        'radial_accuracy': radial_accuracy,
        'num_angles': num_angles,
        'num_rad_classes': num_rad_classes,
        'max_rad_value': max_rad_value,
        'rad_resolution': rad_resolution,
        'fixed_angle': fixed_angle,
        'fixed_radius': float(predicted_radii[0])  # All predictions are the same
    }
    
    return fixed_angle_baseline_stats


def snr_get_noise_and_snr_torch(source: torch.Tensor, snr_db: float, input_signal_length=131070, seed=42):
    """
    Generate noise with a specific SNR value for evaluation.
    
    Args:
        source: Source signal tensor
        snr_db: SNR value in dB
        input_signal_length: Length of the input signal
        seed: Random seed for reproducibility
        
    Returns:
        tuple: (noise, snr_tensor)
    """
    device = source.device
    
    # Use fixed seed for deterministic behavior
    torch.manual_seed(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    
    # Generate colored noise
    beta = 1.5  # Fixed beta value
    noise = generate_colored_noise_gpu(beta, input_signal_length, device)
    snr_tensor = torch.tensor([snr_db], dtype=torch.float32, device=device)

    return noise, snr_tensor


def snr_make_batch_data(model, reverberated_speech, snr_db: float, seed=42):
    """
    Create batch data with a specific SNR value.
    
    Args:
        model: The model instance
        reverberated_speech: Reverberated speech signal
        snr_db: SNR value in dB
        seed: Random seed for reproducibility
        
    Returns:
        tuple: Batch data components
    """
    reverberated_source = reverberated_speech.unsqueeze(1)
    
    # Use SNR-specific noise generation
    noise, snr_tensor = snr_get_noise_and_snr_torch(
        reverberated_source, 
        snr_db,
        input_signal_length=model.model_config.input_length,
        seed=seed
    )

    batch_size = reverberated_speech.shape[0]

    reverberated_source = audio_normalize_batch(reverberated_source, "rms", model.train_config.rms_level)

    # Add noise with specified SNR
    reverberated_source_with_noise = add_noise_batch(reverberated_source, noise, snr_tensor)

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


def evaluate_model_at_snr(model: FinsLightningModule,
                          dataloader,
                          model_config: FinsModelConfigParams,
                          snr_db: float,
                          max_batches: int = None,
                          rad_acc_threshold_m: Optional[float] = None):
    """
    Evaluate the model at a specific SNR level.
    
    Args:
        model: The loaded FinsLightningModule
        dataloader: DataLoader for the dataset
        model_config: Model configuration parameters
        snr_db: SNR value in dB
        max_batches: Maximum number of batches to process (None for all)
        
    Returns:
        dict: Dictionary containing error statistics for this SNR
    """
    model.eval()
    
    # Set PyTorch to deterministic mode
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Initialize lists to store errors
    all_angular_errors = []
    all_radial_errors = []
    
    print(f"Evaluating at SNR = {snr_db:.1f} dB...")
    
    with torch.no_grad():
        for batch_idx, (rir, source_location, receiver_position, room_dimensions, source, reverberation_time, sample_rate) in enumerate(tqdm(dataloader, desc=f"SNR {snr_db:.1f} dB")):
            # Stop if max_batches is reached
            if max_batches is not None and batch_idx >= max_batches:
                break
                
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

            # Prepare batch data with specific SNR
            (
                reverberated_source_with_noise,
                reverberated_source,
                batch_stochastic_noise,
                batch_noise_condition,
            ) = snr_make_batch_data(model, reverberated_speech, snr_db, seed=batch_idx)

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
    
    # Convert to numpy arrays
    all_angular_errors = np.array(all_angular_errors)
    all_radial_errors = np.array(all_radial_errors)
    
    # Calculate statistics
    # Radial accuracy at threshold
    radial_accuracy = None
    if rad_acc_threshold_m is not None:
        radial_accuracy = float(np.mean(all_radial_errors <= rad_acc_threshold_m)) if all_radial_errors.size > 0 else 0.0

    stats = {
        'snr_db': snr_db,
        'angular_error': {
            'mean': float(np.mean(all_angular_errors)),
            'std': float(np.std(all_angular_errors)),
            'median': float(np.median(all_angular_errors)),
            'rmse': float(np.sqrt(np.mean(all_angular_errors**2))),
            'mae': float(np.mean(all_angular_errors))
        },
        'radial_error': {
            'mean': float(np.mean(all_radial_errors)),
            'std': float(np.std(all_radial_errors)),
            'median': float(np.median(all_radial_errors)),
            'rmse': float(np.sqrt(np.mean(all_radial_errors**2))),
            'mae': float(np.mean(all_radial_errors))
        },
        'total_samples': len(all_angular_errors),
        'radial_accuracy': radial_accuracy,
    }
    
    return stats


def plot_angular_error_vs_snr(snr_results, output_path=None, figure_width_cm=8.8, dpi=600, include_fixed_angle_baseline=False, model_config=None, fixed_angle_baseline_label="90° Baseline", fixed_angle=90):
    """
    Create a plot showing angular error vs SNR following IEEE academic paper requirements.
    
    Args:
        snr_results: Either a list of dicts (single checkpoint) or a dict mapping label -> list of dicts (multi)
        output_path: Path to save the plot (optional)
        figure_width_cm: Figure width in cm (default: 8.8 cm for single-column)
        dpi: Resolution for PNG output (default: 600 dpi)
        include_fixed_angle_baseline: Whether to include theoretical fixed angle baseline
        model_config: Model configuration for fixed angle baseline calculation
        fixed_angle_baseline_label: Label for the fixed angle baseline series
        fixed_angle: Fixed angle to use for baseline (default: 90 degrees)
    """
    # Normalize input to a mapping: label -> list of per-SNR dicts
    if isinstance(snr_results, dict):
        series_map = snr_results
    else:
        series_map = {"model": snr_results}

    # Add fixed angle baseline if requested
    if include_fixed_angle_baseline and model_config is not None:
        # Determine SNR axis from the first series
        first_key = next(iter(series_map))
        snr_values = [result['snr_db'] for result in series_map[first_key]]
        
        # Calculate fixed angle baseline stats (SNR-independent)
        fixed_angle_stats = calculate_fixed_angle_baseline_stats(model_config, fixed_angle)
        
        # Create fixed angle baseline series with same SNR values but constant error
        fixed_angle_baseline_series = []
        for snr_db in snr_values:
            baseline_result = {
                'snr_db': snr_db,
                'angular_error': fixed_angle_stats['angular_error'].copy(),
                'radial_error': fixed_angle_stats['radial_error'].copy(),
                'total_samples': fixed_angle_stats['total_samples'],
                'radial_accuracy': fixed_angle_stats['radial_accuracy']
            }
            fixed_angle_baseline_series.append(baseline_result)
        
        # Add to series map
        series_map[fixed_angle_baseline_label] = fixed_angle_baseline_series
    else:
        # Determine SNR axis from the first series (fallback for when no baseline is added)
        first_key = next(iter(series_map))
        snr_values = [result['snr_db'] for result in series_map[first_key]]
    
    # Convert cm to inches for matplotlib
    width_inches = figure_width_cm / 2.54
    height_inches = width_inches * 0.68  # Maintain aspect ratio (~6 cm height)
    
    # Create the plot with IEEE-compliant settings
    plt.figure(figsize=(width_inches, height_inches))
    
    # Set IEEE-compliant style
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    plt.rcParams['font.size'] = 8
    plt.rcParams['axes.linewidth'] = 0.5
    plt.rcParams['axes.labelpad'] = 2
    plt.rcParams['xtick.major.pad'] = 2
    plt.rcParams['ytick.major.pad'] = 2
    plt.rcParams['lines.linewidth'] = 1.0
    plt.rcParams['lines.markersize'] = 4
    plt.rcParams['legend.fontsize'] = 8
    plt.rcParams['legend.frameon'] = False
    plt.rcParams['legend.borderpad'] = 0.5
    plt.rcParams['legend.labelspacing'] = 0.3
    
    # Colors/markers for up to six series (color palette) - added gray for random baseline
    series_styles = [
        {"color": "#1f77b4", "linestyle": "-",  "marker": "o"},  # blue
        {"color": "#ff7f0e", "linestyle": "--", "marker": "s"},  # orange
        {"color": "#2ca02c", "linestyle": ":",  "marker": "^"},  # green
        {"color": "#d62728", "linestyle": "-.", "marker": "D"},  # red
        {"color": "#9467bd", "linestyle": "-",  "marker": "v"},  # purple
        {"color": "#7f7f7f", "linestyle": "-",  "marker": "x", "alpha": 0.7},  # gray for random baseline
    ]

    # Ensure the first series (typically contrastive/label A, styled black) is drawn on top
    enumerated_items = list(enumerate(series_map.items()))
    if len(enumerated_items) > 1:
        plot_order = enumerated_items[1:] + [enumerated_items[0]]
    else:
        plot_order = enumerated_items

    for style_idx, (label, series) in plot_order:
        angular_errors_mean = [item['angular_error']['mean'] for item in series]
        angular_errors_std = [item['angular_error']['std'] for item in series]
        style = series_styles[min(style_idx, len(series_styles) - 1)]
        # Higher zorder for the first series (style_idx == 0), lower for fixed angle baseline
        z = 3 if style_idx == 0 else 1 if fixed_angle_baseline_label in label else 2
        # Use style-specific alpha if available, otherwise default
        alpha = style.get("alpha", 0.9)
        plt.errorbar(
            snr_values,
            angular_errors_mean,
            yerr=angular_errors_std,
            marker=style["marker"],
            linewidth=1.0,
            markersize=4,
            capsize=2,
            capthick=0.5,
            color=style["color"],
            linestyle=style["linestyle"],
            alpha=alpha,
            zorder=z,
            label=label,
        )
    
    # Customize the plot for IEEE format
    plt.xlabel('SNR (dB)', fontsize=8, fontweight='normal')
    plt.ylabel('Mean Absolute Angular Error (degrees)', fontsize=8, fontweight='normal')
    
    # Set tick parameters
    plt.tick_params(axis='both', which='major', labelsize=8, width=0.5, length=3)
    plt.tick_params(axis='both', which='minor', width=0.5, length=1.5)
    
    # Add grid (subtle)
    plt.grid(True, alpha=0.3, linewidth=0.5)
    
    # Invert x-axis (lower SNR to higher SNR)
    plt.gca().invert_xaxis()
    
    # Add legend if multiple series
    if len(series_map) > 1:
        plt.legend(loc='best')
    
    # Tight layout with minimal padding
    plt.tight_layout(pad=0.5)
    
    # Save the plot if output path is provided
    if output_path:
        # Determine file format from extension
        file_ext = Path(output_path).suffix.lower()
        
        # Save in the specified format
        if file_ext in ['.pdf', '.eps']:
            # Vector format
            plt.savefig(output_path, format=file_ext[1:], bbox_inches='tight', 
                       dpi=300, transparent=False)
        elif file_ext == '.png':
            # High-resolution raster format
            plt.savefig(output_path, format='png', bbox_inches='tight', 
                       dpi=dpi, transparent=False)
        else:
            # Default to PNG
            plt.savefig(output_path, format='png', bbox_inches='tight', 
                       dpi=dpi, transparent=False)
        
        print(f"Plot saved to: {output_path}")
        print(f"Figure size: {width_inches:.2f} x {height_inches:.2f} inches ({figure_width_cm} x {figure_width_cm*0.68:.1f} cm)")
        print(f"Resolution: {dpi} dpi")
        
        # Also save as PNG if the original format is not PNG
        if file_ext != '.png':
            png_path = str(output_path).replace(file_ext, '.png')
            plt.savefig(png_path, format='png', bbox_inches='tight', 
                       dpi=dpi, transparent=False)
            print(f"PNG version also saved to: {png_path}")
    
    # Show the plot
    plt.show()


def save_snr_results(snr_results, output_path):
    """
    Save the SNR evaluation results to a JSON file.
    
    Args:
        snr_results: Either a list (single model) or dict[label->list] (multi-model)
        output_path: Path to save the JSON file
    """
    with open(output_path, 'w') as f:
        json.dump(snr_results, f, indent=2)
    print(f"Results saved to: {output_path}")


def main():
    """
    Main function to run the SNR evaluation and plotting.
    """
    parser = argparse.ArgumentParser(description='Evaluate model performance at different SNR levels (supports comparing two checkpoints)')
    parser.add_argument('checkpoint_path', type=str, help='Path to the primary model checkpoint file (A)')
    parser.add_argument('--checkpoint_path_b', type=str, default=None, help='Path to the secondary model checkpoint file (B) for comparison')
    parser.add_argument('--label_a', type=str, default=None, help='Label for checkpoint A (default: filename)')
    parser.add_argument('--label_b', type=str, default=None, help='Label for checkpoint B (default: filename)')
    parser.add_argument('--dataset_path', type=str, default=None, help='Path to the dataset directory (optional)')
    parser.add_argument('--dataset_split', type=str, default='test', choices=['test', 'valid'], 
                       help='Dataset split to evaluate (default: test)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation (default: 32)')
    parser.add_argument('--snr_range', type=str, default='0,30,6', 
                       help='SNR range as "min,max,step" in dB (default: 0,30,6)')
    parser.add_argument('--max_batches', type=int, default=None, 
                       help='Maximum number of batches to process per SNR (default: all)')
    parser.add_argument('--output_dir', type=str, default='./', 
                       help='Output directory for plots and results (default: current directory)')
    parser.add_argument('--no_plot', action='store_true', help='Skip plotting and only save results')
    parser.add_argument('--figure_width_cm', type=float, default=8.8, 
                       help='Figure width in cm (default: 8.8 cm for single-column, use 18.0 for double-column)')
    parser.add_argument('--dpi', type=int, default=600, 
                       help='Resolution for PNG output (default: 600 dpi)')
    parser.add_argument('--output_format', type=str, default='png', choices=['png', 'pdf', 'eps'], 
                       help='Output format (default: png)')
    parser.add_argument('--rad_acc_threshold_m', type=float, default=0.5, help='Threshold in meters for radial accuracy (default: 0.5m)')
    parser.add_argument('--include_random_baseline', action='store_true', help='Include theoretical random-choice baseline in plots and results')
    # Fixed angle baseline options
    parser.add_argument('--include_fixed_angle_baseline', action='store_true', help='Include theoretical fixed angle baseline in plots and results')
    parser.add_argument('--fixed_angle_baseline_label', type=str, default='90° Baseline', help='Label for fixed angle baseline series')
    parser.add_argument('--fixed_angle', type=int, default=90, help='Fixed angle to use for baseline (default: 90 degrees)')
    
    args = parser.parse_args()
    
    # Parse SNR range
    try:
        snr_min, snr_max, snr_step = map(float, args.snr_range.split(','))
        snr_values = np.arange(snr_min, snr_max + snr_step, snr_step)
    except ValueError:
        print("Error: Invalid SNR range format. Use 'min,max,step' (e.g., '0,30,6')")
        sys.exit(1)

    if not Path(args.checkpoint_path).exists():
        print(f"Error: Checkpoint file not found: {args.checkpoint_path}")
        sys.exit(1)
    if args.checkpoint_path_b and not Path(args.checkpoint_path_b).exists():
        print(f"Error: Checkpoint file not found: {args.checkpoint_path_b}")
        sys.exit(1)
    
    # Load configuration
    model_config = FinsModelConfigParams()
    train_config = TrainParams()
    dataset_params = DatasetParams()
    
    # Override dataset path if provided
    if args.dataset_path:
        dataset_params.rir_dataset_path = args.dataset_path
    
    print(f"Loading model A from: {args.checkpoint_path}")
    model_a = load_fins_lightning_module_from_checkpoint(args.checkpoint_path, model_config, train_config)
    label_a = args.label_a or Path(args.checkpoint_path).stem
    model_b = None
    label_b = None
    if args.checkpoint_path_b:
        print(f"Loading model B from: {args.checkpoint_path_b}")
        model_b = load_fins_lightning_module_from_checkpoint(args.checkpoint_path_b, model_config, train_config)
        label_b = args.label_b or Path(args.checkpoint_path_b).stem

    print(f"Loading {args.dataset_split} dataset...")
    dataloader, dataset_config = load_dataset_and_get_dataloader(
        args.dataset_split, dataset_params, args.batch_size,
        num_workers=1, prefetch_factor=None, persistent_workers=False, shuffle=False
    )
    
    # Evaluate at each SNR level for model A (and B if provided)
    print(f"\nEvaluating at {len(snr_values)} SNR levels: {snr_values}")
    print("=" * 60)

    results_map = {}

    print(f"\nModel A [{label_a}] results:")
    snr_results_a = []
    for snr_db in snr_values:
        result = evaluate_model_at_snr(model_a, dataloader, model_config, snr_db, args.max_batches, args.rad_acc_threshold_m)
        snr_results_a.append(result)
        print(f"SNR {snr_db:.1f} dB: Angular Error = {result['angular_error']['mean']:.2f} ± {result['angular_error']['std']:.2f}° "
              f"(median: {result['angular_error']['median']:.2f}°)")
    results_map[label_a] = snr_results_a

    if model_b is not None:
        print(f"\nModel B [{label_b}] results:")
        snr_results_b = []
        for snr_db in snr_values:
            result = evaluate_model_at_snr(model_b, dataloader, model_config, snr_db, args.max_batches, args.rad_acc_threshold_m)
            snr_results_b.append(result)
            print(f"SNR {snr_db:.1f} dB: Angular Error = {result['angular_error']['mean']:.2f} ± {result['angular_error']['std']:.2f}° "
                  f"(median: {result['angular_error']['median']:.2f}°)")
        results_map[label_b] = snr_results_b
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results
    if not results_map:
        print("Error: No results to save. Make sure at least one model is evaluated.")
        sys.exit(1)
    
    # Determine which dataset split to use in file naming
    file_split = args.dataset_split
    
    results_path = output_dir / f"snr_evaluation_results_{file_split}.json"
    save_snr_results(results_map, results_path)
    
    # Create and save plot
    if not args.no_plot:
        plot_path = output_dir / f"angular_error_vs_snr_{file_split}.{args.output_format}"
        plot_angular_error_vs_snr(results_map, plot_path, args.figure_width_cm, args.dpi, args.include_fixed_angle_baseline, model_config, args.fixed_angle_baseline_label, args.fixed_angle)
        # Also plot radial error vs SNR 
        def plot_radial_error_vs_snr(snr_data, output_path, include_fixed_angle_baseline=False, model_config=None, fixed_angle_baseline_label="90° Baseline", fixed_angle=90):
            if isinstance(snr_data, dict):
                series_map = snr_data
            else:
                series_map = {"model": snr_data}
            
            # Add fixed angle baseline if requested
            if include_fixed_angle_baseline and model_config is not None:
                # Get SNR values from first series
                first_key = next(iter(series_map))
                snr_vals = [entry['snr_db'] for entry in series_map[first_key]]
                
                # Calculate fixed angle baseline stats
                fixed_angle_stats = calculate_fixed_angle_baseline_stats(model_config, fixed_angle)
                
                # Create fixed angle baseline series for radial error plot
                fixed_angle_baseline_series = []
                for snr_db in snr_vals:
                    baseline_result = {
                        'snr_db': snr_db,
                        'radial_error': fixed_angle_stats['radial_error'].copy()
                    }
                    fixed_angle_baseline_series.append(baseline_result)
                
                # Add to series map
                series_map[fixed_angle_baseline_label] = fixed_angle_baseline_series
            # Determine SNR axis
            first_key = next(iter(series_map))
            snr_axis = [entry['snr_db'] for entry in series_map[first_key]]
            width_inches = args.figure_width_cm / 2.54
            height_inches = width_inches * 0.68
            plt.figure(figsize=(width_inches, height_inches))
            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
            plt.rcParams['font.size'] = 8
            plt.rcParams['axes.linewidth'] = 0.5
            plt.rcParams['lines.linewidth'] = 1.0
            plt.rcParams['lines.markersize'] = 4
            enumerated = list(enumerate(series_map.items()))
            # keep same style ordering as angular plot
            styles = [
                {"color": "#1f77b4", "linestyle": "-",  "marker": "o"},
                {"color": "#ff7f0e", "linestyle": "--", "marker": "s"},
                {"color": "#2ca02c", "linestyle": ":",  "marker": "^"},
                {"color": "#d62728", "linestyle": "-.", "marker": "D"},
                {"color": "#9467bd", "linestyle": "-",  "marker": "v"},
                {"color": "#7f7f7f", "linestyle": "-",  "marker": "x", "alpha": 0.7},  # gray for random baseline
            ]
            for idx, (label, series) in enumerated:
                # Get radial error mean and std for each series
                rad_err_mean = [entry.get('radial_error', {}).get('mean') for entry in series]
                rad_err_std = [entry.get('radial_error', {}).get('std') for entry in series]
                
                # Filter out None values
                x = [s for s, v in zip(snr_axis, rad_err_mean) if v is not None]
                y_mean = [v for v in rad_err_mean if v is not None]
                y_std = [std for std, mean in zip(rad_err_std, rad_err_mean) if mean is not None]
                
                if not x:  # Skip if no valid data
                    continue
                    
                style = styles[min(idx, len(styles) - 1)]
                alpha = style.get("alpha", 1.0)
                
                # Plot with error bars
                plt.errorbar(x, y_mean, yerr=y_std, label=label, 
                           marker=style['marker'], linestyle=style['linestyle'], 
                           color=style['color'], alpha=alpha, linewidth=1.0, 
                           markersize=4, capsize=2, capthick=0.5)
            
            plt.xlabel('SNR (dB)', fontsize=8)
            plt.ylabel('Mean Absolute Radial Error (m)', fontsize=8)
            plt.grid(True, alpha=0.3, linewidth=0.5)
            plt.gca().invert_xaxis()
            if len(series_map) > 1:
                plt.legend(loc='best')
            plt.tight_layout(pad=0.5)
            if output_path:
                file_ext = Path(output_path).suffix.lower()
                if file_ext == '.eps':
                    plt.savefig(output_path, format='eps', bbox_inches='tight', dpi=300, transparent=False)
                elif file_ext in ['.png', '.jpg', '.jpeg']:
                    format_name = 'jpeg' if file_ext in ['.jpg', '.jpeg'] else 'png'
                    plt.savefig(output_path, format=format_name, bbox_inches='tight', dpi=args.dpi, transparent=False)
                else:
                    plt.savefig(output_path, format='png', bbox_inches='tight', dpi=args.dpi, transparent=False)
                
                # Note: JPG saving removed for radial plots to keep only PNG
                # Keeping only the primary format for radial error plots
        rad_plot_path = output_dir / f"radial_error_vs_snr_{file_split}.{args.output_format}"
        plot_radial_error_vs_snr(results_map, rad_plot_path, args.include_fixed_angle_baseline, model_config, args.fixed_angle_baseline_label, args.fixed_angle)
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Dataset split: {file_split}")
    print(f"SNR range: {snr_min} to {snr_max} dB (step: {snr_step} dB)")
    
    # Get total samples from the first available series
    first_series = next(iter(results_map.values()))
    total_samples = first_series[0]['total_samples'] if first_series else 0
    print(f"Total samples per SNR: {total_samples}")
    print(f"Results saved to: {results_path}")
    if not args.no_plot:
        print(f"Angular error plot saved to: {output_dir / f'angular_error_vs_snr_{file_split}.{args.output_format}'}")
        print(f"Radial error plot saved to: {output_dir / f'radial_error_vs_snr_{file_split}.{args.output_format}'}")
        print(f"Figure size: {args.figure_width_cm} cm width (single-column format)")
        if args.figure_width_cm >= 18.0:
            print("Note: Using double-column figure size")
        print(f"Output format: {args.output_format.upper()}")
        if args.output_format in ['png', 'jpg']:
            print(f"Resolution: {args.dpi} dpi")
        print("Note: JPG versions are automatically saved for high compatibility")
    
    # Find best and worst performance for each model
    for model_label, series_results in results_map.items():
        if series_results and all('angular_error' in result for result in series_results):
            # Filter results that have valid angular_error data
            valid_results = [r for r in series_results if 'angular_error' in r and 'mean' in r['angular_error']]
            if valid_results:
                best_snr = min(valid_results, key=lambda x: x['angular_error']['mean'])
                worst_snr = max(valid_results, key=lambda x: x['angular_error']['mean'])
                print(f"\n[{model_label}] Best: SNR {best_snr['snr_db']:.1f} dB (Angular Error: {best_snr['angular_error']['mean']:.2f}°)")
                print(f"[{model_label}] Worst: SNR {worst_snr['snr_db']:.1f} dB (Angular Error: {worst_snr['angular_error']['mean']:.2f}°)")
                
                # Add fixed angle baseline summary if included
                if args.include_fixed_angle_baseline and model_label == args.fixed_angle_baseline_label:
                    print(f"[{model_label}] Fixed {args.fixed_angle}° baseline angular error: {best_snr['angular_error']['mean']:.2f} ± {best_snr['angular_error']['std']:.2f}° (constant across all SNR)")
                    print(f"[{model_label}] Fixed {args.fixed_angle}° baseline radial accuracy: {best_snr.get('radial_accuracy', 0.0)*100:.1f}% (<= {args.rad_acc_threshold_m:.2f}m)")
        else:
            print(f"\n[{model_label}] Skipping performance summary (incomplete data)")


if __name__ == '__main__':
    main() 