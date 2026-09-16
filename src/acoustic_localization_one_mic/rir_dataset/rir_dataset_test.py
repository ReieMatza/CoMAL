#!/usr/bin/env python3
"""
Test script to verify that the RIR dataset works with both old and new formats.
"""

import sys
import os
from pathlib import Path
import torch

# Fix the import to use the correct module path
from acoustic_localization_one_mic.config_model import DatasetParams
from acoustic_localization_one_mic.rir_dataset.rir_dataset import RirDataset


def test_dataset_loading(dataset_path: str, librispeech_path: str = None):
    """Test dataset loading and basic functionality."""
    
    print(f"Testing dataset at: {dataset_path}")
    dataset_path = Path(dataset_path)
    
    # Check if dataset path exists
    if not dataset_path.exists():
        print(f"❌ Dataset path does not exist: {dataset_path}")
        return False
    
    # Check for required files
    csv_file = dataset_path / 'rir_dataset.csv'
    if not csv_file.exists():
        print(f"❌ Missing rir_dataset.csv file: {csv_file}")
        return False
    
    # Check dataset format
    config_json = dataset_path / 'config.json'
    config_pkl = dataset_path / 'config.pkl'
    h_metadata_json = dataset_path / 'h_metadata.json'
    h_dict_pkl = dataset_path / 'h_dict.pkl'
    h_tensors_dir = dataset_path / 'h_tensors'
    
    if config_json.exists() and h_metadata_json.exists() and h_tensors_dir.exists():
        print("✅ New format detected (JSON + individual tensors)")
        format_type = "new"
    elif config_pkl.exists() and h_dict_pkl.exists():
        print("✅ Old format detected (pickle files)")
        format_type = "old"
    else:
        print("❌ Incomplete dataset - missing required files")
        print(f"   Looking for: config.json + h_metadata.json + h_tensors/ OR config.pkl + h_dict.pkl")
        return False
    
    try:
        # Test dataset initialization
        print("\n🔄 Initializing dataset...")
        import time
        start_time = time.time()
        
        dataset = RirDataset(root_dir=dataset_path, librispeech_path=librispeech_path)
        
        init_time = time.time() - start_time
        print(f"✅ Dataset initialized in {init_time:.3f} seconds")
        
        # Test basic properties
        print(f"✅ Dataset length: {len(dataset)} entries")
        
        # Handle different config types
        if hasattr(dataset.config, 'nsample'):
            nsample = dataset.config.nsample
        elif hasattr(dataset.config, 'get'):
            nsample = dataset.config.get('nsample', 'N/A')
        else:
            nsample = getattr(dataset.config, 'nsample', 'N/A')
        
        print(f"✅ Config nsample: {nsample}")
        
        # Test lazy loading specific features
        if hasattr(dataset, 'h_metadata'):
            print(f"✅ Lazy loading enabled: {len(dataset.h_metadata)} H samples metadata loaded")
            cache_info = dataset.get_cache_info()
            print(f"✅ Cache info: {cache_info['cached_tensors']}/{cache_info['cache_limit']} tensors cached")
        
        # Test accessing a few samples (without LibriSpeech dependency)
        print("\n🔄 Testing metadata access...")
        
        # Get first few data entries
        sample_entries = dataset.data_list[:min(3, len(dataset.data_list))]
        
        for i, entry in enumerate(sample_entries):
            print(f"  Entry {i}: h_uid={entry.h_uid}, source_location={entry.source_location}")
            
            # Test lazy loading of H tensor
            if hasattr(dataset, '_get_h_tensor'):
                try:
                    h_tensor = dataset._get_h_tensor(entry.h_uid)
                    print(f"    ✅ H tensor loaded: shape={h_tensor.shape}, dtype={h_tensor.dtype}")
                except Exception as e:
                    print(f"    ❌ Failed to load H tensor: {e}")
                    return False
        
        # Test cache functionality
        if hasattr(dataset, 'get_cache_info'):
            cache_info = dataset.get_cache_info()
            print(f"✅ After accessing {len(sample_entries)} samples: {cache_info['cached_tensors']} tensors cached")
        
        print(f"\n✅ Dataset test completed successfully!")
        print(f"   Format: {format_type}")
        print(f"   Initialization time: {init_time:.3f}s")
        print(f"   Total entries: {len(dataset)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Dataset test failed: {e}")
        print(f"   Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function."""
    print("=== RIR Dataset Test ===\n")
    
    dataset_params = DatasetParams()
    default_dataset_path = str(Path(dataset_params.rir_dataset_path) / "train")
    default_librispeech_path = dataset_params.librispeech_path
    
    # Allow command line arguments
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
        librispeech_path = sys.argv[2] if len(sys.argv) > 2 else None
    else:
        dataset_path = default_dataset_path
        librispeech_path = default_librispeech_path
    
    # Test the dataset
    success = test_dataset_loading(dataset_path, librispeech_path)
    
    if success:
        print("\n🎉 All tests passed! The dataset is working correctly.")
    else:
        print("\n❌ Tests failed. Please check the error messages above.")
        sys.exit(1)


if __name__ == '__main__':
    main()