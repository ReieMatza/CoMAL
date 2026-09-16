import csv
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import soundfile as sf
import copy
import torch
import json
from torch.utils.data import Dataset
from acoustic_localization_one_mic.rir_dataset.genereate_dataset import RirDataEntry, HSample
from acoustic_localization_one_mic.config_model import RirDatasetGeneratorConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RirDataset(Dataset):
    def __init__(self, root_dir: Path, librispeech_path: str = None):
        self.root_dir = root_dir
        self.data_list = self._load_rir_data_entries(Path(self.root_dir) / 'rir_dataset.csv')
        self.librispeech_path = Path(librispeech_path) if librispeech_path is not None else None
        
        # Lazy loading setup - only load metadata, not actual tensors
        self.h_metadata, self.h_tensors_dir = self._load_h_metadata(Path(self.root_dir))
        self._tensor_cache = {}  # Cache for loaded tensors to avoid reloading
        self._cache_limit = 100  # Limit cache size to avoid memory issues
        
        self.config = self._load_config(Path(self.root_dir))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]

        filepath = self.librispeech_path / 'LibriSpeech' / Path(item.path)

        # Read only the relevant section of the audio file for efficiency
        frames_to_read = item.end_sample - item.start_sample
        waveform, sample_rate = sf.read(
            str(filepath).replace('\\', '/'), 
            start=item.start_sample, 
            frames=frames_to_read, 
            always_2d=True
        )

        # Lazy load the specific tensor we need
        h = self._get_h_tensor(item.h_uid)

        source_location = torch.FloatTensor(item.source_location)
        receiver_position = torch.FloatTensor(item.receiver_position)
        room_dimensions = torch.FloatTensor(item.room_dimensions)
        reverberation_time = torch.as_tensor(item.reverberation_time)
        source = waveform

        return h, source_location, receiver_position, room_dimensions, source, reverberation_time, sample_rate

    def _get_h_tensor(self, uid: str) -> torch.Tensor:
        """
        Lazy load a specific h tensor by UID with caching.
        """
        # Check cache first
        if uid in self._tensor_cache:
            return self._tensor_cache[uid]
        
        # Check if we have metadata for this UID
        if uid not in self.h_metadata:
            raise KeyError(f"No metadata found for h_uid: {uid}")
        
        # Load tensor from file
        if self.h_tensors_dir:
            # New format: load from individual tensor file
            tensor_path = self.h_tensors_dir / f'{uid}.pt'
            if not tensor_path.exists():
                raise FileNotFoundError(f"Tensor file not found: {tensor_path}")
            
            h_tensor = torch.load(tensor_path, map_location='cpu', weights_only=True)
        else:
            # Fallback: this shouldn't happen with lazy loading, but handle gracefully
            raise RuntimeError(f"No tensor loading method available for uid: {uid}")
        
        # Cache the tensor (with size limit)
        if len(self._tensor_cache) >= self._cache_limit:
            # Remove oldest entry (simple FIFO eviction)
            oldest_key = next(iter(self._tensor_cache))
            del self._tensor_cache[oldest_key]
        
        self._tensor_cache[uid] = h_tensor
        return h_tensor

    def get_h_sample(self, uid: str) -> HSample:
        """
        Get a complete HSample object for a given UID (reconstructed from metadata and tensor).
        This is useful for compatibility with existing code that expects HSample objects.
        """
        if uid not in self.h_metadata:
            raise KeyError(f"No metadata found for h_uid: {uid}")
        
        metadata = self.h_metadata[uid]
        h_tensor = self._get_h_tensor(uid)
        
        return HSample(
            h=h_tensor,
            x=metadata['x'],
            y=metadata['y'],
            z=metadata['z'],
            uid=metadata['uid'],
            receiver_position=metadata['receiver_position'],
            room_dimensions=metadata['room_dimensions'],
            reverberation_time=metadata['reverberation_time']
        )

    def clear_tensor_cache(self):
        """Clear the tensor cache to free memory."""
        self._tensor_cache.clear()

    def set_cache_limit(self, limit: int):
        """Set the maximum number of tensors to keep in cache."""
        self._cache_limit = limit
        # Trim cache if it's over the new limit
        while len(self._tensor_cache) > limit:
            oldest_key = next(iter(self._tensor_cache))
            del self._tensor_cache[oldest_key]

    def get_cache_info(self) -> Dict:
        """Get information about the current cache state."""
        return {
            'cached_tensors': len(self._tensor_cache),
            'cache_limit': self._cache_limit,
            'cached_uids': list(self._tensor_cache.keys())
        }

    def _load_rir_data_entries(self, csv_path: Path) -> list[RirDataEntry]:
        rir_data_entries = []
        with open(csv_path, mode='r', newline='') as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                rir_data_entry = RirDataEntry(
                    entry_id=int(row['entry_id']),
                    start_sample=int(row['start_sample']),
                    end_sample=int(row['end_sample']),
                    length_in_seconds=float(row['length_in_seconds']),
                    path=Path(row['path']),
                    h_uid=row['h_uid'],
                    receiver_position=[float(x) for x in row['receiver_position'].strip('[]').split(',')],
                    room_dimensions=[float(x) for x in row['room_dimensions'].strip('[]').split(',')],
                    source_location=[float(x) for x in row['source_location'].strip('[]').split(',')],
                    reverberation_time=float(row['reverberation_time']),
                )

                rir_data_entries.append(rir_data_entry)
        return rir_data_entries

    def _load_config(self, rir_dataset_path: Path) -> RirDatasetGeneratorConfig:
        """Load config from JSON instead of pickle."""
        config_path = rir_dataset_path / 'config.json'
        if config_path.exists():
            with open(config_path, 'r') as json_file:
                config_dict = json.load(json_file)
                return RirDatasetGeneratorConfig(**config_dict)
        else:
            # Fallback to pickle if JSON doesn't exist (for backward compatibility)
            config_path_pkl = rir_dataset_path / 'config.pkl'
            if config_path_pkl.exists():
                import pickle
                with open(config_path_pkl, 'rb') as config_file:
                    return pickle.load(config_file)
            else:
                raise FileNotFoundError(f"No config file found at {rir_dataset_path}")

    def _load_h_metadata(self, rir_dataset_path: Path) -> Tuple[Dict, Optional[Path]]:
        """
        Load h_dict metadata for lazy loading. Returns metadata dict and tensors directory path.
        """
        # Check if new format exists
        metadata_path = rir_dataset_path / 'h_metadata.json'
        tensors_dir = rir_dataset_path / 'h_tensors'
        
        if metadata_path.exists() and tensors_dir.exists():
            # Load from new format - only metadata, not tensors
            with open(metadata_path, 'r') as json_file:
                metadata_dict = json.load(json_file)
            
            print(f"[DEBUG] Loaded metadata for {len(metadata_dict)} HSamples (lazy loading enabled)")
            return metadata_dict, tensors_dir
        else:
            # Fallback to pickle if new format doesn't exist (for backward compatibility)
            pickle_path = rir_dataset_path / 'h_dict.pkl'
            if pickle_path.exists():
                import pickle
                print(f"[DEBUG] Falling back to pickle format (consider migrating to new format for better performance)")
                with open(pickle_path, 'rb') as file:
                    h_dict = copy.deepcopy(pickle.load(file))
                
                # Convert pickle format to metadata format for lazy loading
                metadata_dict = {}
                for uid, h_sample in h_dict.items():
                    metadata_dict[uid] = {
                        'x': h_sample.x,
                        'y': h_sample.y,
                        'z': h_sample.z,
                        'uid': h_sample.uid,
                        'receiver_position': h_sample.receiver_position,
                        'room_dimensions': h_sample.room_dimensions,
                        'reverberation_time': h_sample.reverberation_time
                    }
                    # Cache the tensors from pickle since we have them
                    self._tensor_cache[uid] = h_sample.h
                
                return metadata_dict, None  # No tensors dir for pickle format
            else:
                raise FileNotFoundError(f"No h_dict file found at {rir_dataset_path}")
