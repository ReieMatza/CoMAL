#!/usr/bin/env python3
"""
Example script showing how to use plot_angular_error_vs_snr.py.

Evaluates a FINS checkpoint at several SNR levels and writes angular/radial error plots.
"""

import subprocess
import sys
from pathlib import Path


def run_snr_evaluation(
    checkpoint_path: str,
    dataset_path: str = None,
    dataset_split: str = 'test',
    batch_size: int = 32,
    snr_range: str = '0,30,6',
    max_batches: int = None,
    output_dir: str = './',
    no_plot: bool = False,
    figure_width_cm: float = 8.8,
    dpi: int = 600,
    output_format: str = 'png',
    checkpoint_path_b: str = None,
    label_a: str = None,
    label_b: str = None,
    include_fixed_angle_baseline: bool = False,
    fixed_angle_baseline_label: str = "90° Baseline",
    fixed_angle: int = 90,
    include_random_baseline: bool = False,
):
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent / "plot_angular_error_vs_snr.py"),
        checkpoint_path,
        "--dataset_split", dataset_split,
        "--batch_size", str(batch_size),
        "--snr_range", snr_range,
        "--output_dir", output_dir,
        "--figure_width_cm", str(figure_width_cm),
        "--dpi", str(dpi),
        "--output_format", output_format,
    ]

    if checkpoint_path_b:
        cmd.extend(["--checkpoint_path_b", checkpoint_path_b])
    if label_a:
        cmd.extend(["--label_a", label_a])
    if label_b:
        cmd.extend(["--label_b", label_b])
    if dataset_path:
        cmd.extend(["--dataset_path", dataset_path])
    if max_batches:
        cmd.extend(["--max_batches", str(max_batches)])
    if no_plot:
        cmd.append("--no_plot")
    if include_random_baseline:
        cmd.append("--include_random_baseline")
    if include_fixed_angle_baseline:
        cmd.append("--include_fixed_angle_baseline")
        if fixed_angle_baseline_label != "90° Baseline":
            cmd.extend(["--fixed_angle_baseline_label", fixed_angle_baseline_label])
        if fixed_angle != 90:
            cmd.extend(["--fixed_angle", str(fixed_angle)])

    print(f"Running command: {' '.join(cmd)}")
    print("-" * 80)

    try:
        subprocess.run(cmd, check=True, capture_output=False)
        print("\nSNR evaluation completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\nSNR evaluation failed with error code: {e.returncode}")
        sys.exit(1)


def main():
    """
    Usage:
        python scripts/evaluation_scripts/example_snr_evaluation.py <checkpoint_path> [--dataset_path PATH]
    """
    if len(sys.argv) < 2:
        print("Usage: python scripts/evaluation_scripts/example_snr_evaluation.py <checkpoint_path> [--dataset_path PATH]")
        sys.exit(1)

    checkpoint_path = sys.argv[1]
    dataset_path = None
    if "--dataset_path" in sys.argv:
        idx = sys.argv.index("--dataset_path")
        if idx + 1 >= len(sys.argv):
            print("Error: --dataset_path requires a value")
            sys.exit(1)
        dataset_path = sys.argv[idx + 1]

    dataset_split = 'test'
    batch_size = 100
    snr_range = "0,30,6"
    max_batches = 30
    output_dir = "./snr_evaluation_results"
    no_plot = False
    checkpoint_path_b = None
    label_a = None
    label_b = None
    include_fixed_angle_baseline = False
    include_random_baseline = True
    figure_width_cm = 8.8
    dpi = 600
    output_format = "png"

    if not Path(checkpoint_path).exists():
        print(f"Checkpoint file not found: {checkpoint_path}")
        sys.exit(1)
    if dataset_path and not Path(dataset_path).exists():
        print(f"Dataset path not found: {dataset_path}")
        sys.exit(1)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    run_snr_evaluation(
        checkpoint_path,
        dataset_path,
        dataset_split,
        batch_size,
        snr_range,
        max_batches,
        output_dir,
        no_plot,
        figure_width_cm,
        dpi,
        output_format,
        checkpoint_path_b=checkpoint_path_b,
        label_a=label_a,
        label_b=label_b,
        include_fixed_angle_baseline=include_fixed_angle_baseline,
        include_random_baseline=include_random_baseline,
    )


if __name__ == '__main__':
    main()
