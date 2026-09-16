#!/usr/bin/env python3
"""
Example script showing how to use the evaluate_test_dataset.py script.

This script demonstrates how to evaluate a model checkpoint on the test/validation dataset
and get comprehensive error statistics.

NOTE: By default, the evaluation runs in deterministic mode for reproducible results.
If you want to test the model's robustness to noise variations, use --non-deterministic flag.
"""

import subprocess
import sys
from pathlib import Path

def run_evaluation(checkpoint_path: str, dataset_path: str = None, dataset_split: str = 'test', 
                   batch_size: int = 32, deterministic: bool = True):
    """
    Run the evaluation script with the given parameters.
    
    Args:
        checkpoint_path: Path to the model checkpoint file
        dataset_path: Path to the dataset directory (optional)
        dataset_split: Dataset split to evaluate ('test' or 'valid')
        batch_size: Batch size for evaluation
        deterministic: If True, run in deterministic mode for reproducible results
    """
    
    # Build the command
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent / "evaluate_test_dataset.py"),
        checkpoint_path,
        "--dataset_split", dataset_split,
        "--batch_size", str(batch_size)
    ]
    
    if dataset_path:
        cmd.extend(["--dataset_path", dataset_path])
    
    if not deterministic:
        cmd.append("--non-deterministic")
    
    print(f"Running command: {' '.join(cmd)}")
    print("-" * 80)
    
    # Run the evaluation
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("\n✅ Evaluation completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Evaluation failed with error code: {e.returncode}")
        sys.exit(1)


def main():
    """
    Example usage of the evaluation script.

    Usage:
        python scripts/evaluation_scripts/example_evaluation.py <checkpoint_path> [--dataset_path PATH]
    """
    if len(sys.argv) < 2:
        print("Usage: python scripts/evaluation_scripts/example_evaluation.py <checkpoint_path> [--dataset_path PATH]")
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
    deterministic = False
    
    print("Example: Evaluating Model on Dataset")
    print("=" * 50)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Dataset: {dataset_path if dataset_path else 'Default'}")
    print(f"Dataset Split: {dataset_split}")
    print(f"Batch size: {batch_size}")
    print(f"Deterministic mode: {deterministic}")
    print()
    
    if deterministic:
        print("🔒 Running in deterministic mode - results will be reproducible across runs")
        print("   This is useful for comparing different models or configurations")
    else:
        print("🎲 Running in non-deterministic mode - results may vary between runs")
        print("   This tests the model's robustness to noise variations")
    print()
    
    # Check if checkpoint exists
    if not Path(checkpoint_path).exists():
        print(f"❌ Checkpoint file not found: {checkpoint_path}")
        print("Please pass a valid checkpoint path.")
        sys.exit(1)
    
    # Check if dataset exists (if specified)
    if dataset_path and not Path(dataset_path).exists():
        print(f"❌ Dataset path not found: {dataset_path}")
        print("Please update the dataset_path variable in this script or set it to None to use default.")
        sys.exit(1)
    
    # Run the evaluation
    run_evaluation(checkpoint_path, dataset_path, dataset_split, batch_size, deterministic)


if __name__ == '__main__':
    main() 