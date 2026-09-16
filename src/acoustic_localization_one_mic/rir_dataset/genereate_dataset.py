import csv
import json
import logging
import os.path
import random
import sys
import uuid
import warnings
from pathlib import Path
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

import torch
import torchaudio

import numpy as np
import rir_generator as rir
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from dataclasses import dataclass

from acoustic_localization_one_mic.config_model import RirDatasetGeneratorConfig


@dataclass
class RirDataEntry:
    entry_id: int
    start_sample: int
    end_sample: int
    length_in_seconds: float
    path: Path
    h_uid: str
    receiver_position: List[float]
    room_dimensions: List[float]
    source_location: List[float]
    reverberation_time : float

@dataclass
class HSample:
    h: torch.Tensor
    x: float
    y: float
    z: float
    uid : str
    receiver_position: List[float]
    room_dimensions: List[float]
    reverberation_time : float


def generate_positions(receiver_position: list, room_dimensions: tuple, wall_backoff: float = 0.5,
                       theta_range: Tuple[int, int] = (-90, 90),
                       r_range: Tuple[float, float] = (0.2, 4.5), num_of_position_to_generate: int = 100) -> Tuple[List[float], List[float]]:
    all_positions_x = []
    all_positions_y = []

    while len(all_positions_x) < num_of_position_to_generate:
        # Draw a single radius value from a uniform distribution
        radius = np.random.uniform(max(r_range[0], 0), min(r_range[1], room_dimensions[0]))

        # Calculate valid theta range based on room dimensions and wall backoff
        if (room_dimensions[0] - receiver_position[0]) > radius:
            max_theta = np.pi / 2
        else:
            max_theta = np.arcsin((room_dimensions[0] - wall_backoff - receiver_position[0]) / radius)
        if (receiver_position[0] - radius) > 0:
            min_theta = -np.pi / 2
        else:
            min_theta = np.arcsin(-(receiver_position[0] - wall_backoff) / radius)

        # Draw a single theta value from a uniform distribution
        theta_pos = np.random.uniform(max(np.radians(theta_range[0]), min_theta),
                                      min(np.radians(theta_range[1]), max_theta))

        # Calculate position
        pos_x = receiver_position[0] + radius * np.sin(theta_pos)
        pos_y = receiver_position[1] + radius * np.cos(theta_pos)

        # Check if the position is valid
        if wall_backoff <= pos_x <= (room_dimensions[0] - wall_backoff) and \
           wall_backoff <= pos_y <= (room_dimensions[1] - wall_backoff):
            all_positions_x.append(pos_x)
            all_positions_y.append(pos_y)

    return all_positions_x, all_positions_y


def plot_room_and_positions(positions_x, positions_y, test_positions_x, test_positions_y, receiver_position,
                            room_dimensions, room_number:int = 0):
    fig, ax = plt.subplots(1, 1)
    if test_positions_x:
        plt.scatter(test_positions_x, test_positions_y, label='Test locations', zorder=1)
    plt.scatter(receiver_position[0], receiver_position[1], label='Mic location', zorder=2)
    plt.scatter(positions_x, positions_y, label='Train locations', s=10, zorder=3)  # Adjust the size of train locations
    ax.add_patch(Rectangle((0, 0), room_dimensions[0], room_dimensions[1], alpha=0.1, color='red', zorder=0))
    ax.set_xlabel('Room X')
    ax.set_ylabel('Room Y')
    # Set the aspect ratio to be equal
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f"Room and positions for room {room_number}")

    ax.legend()
    plt.show()
    return fig

def plot_impulse_responses(h_list):
    """Plot the impulse responses."""
    fig, ax = plt.subplots(len(h_list), 1, figsize=(10, 2 * len(h_list)))
    if len(h_list) == 1:
        ax = [ax]
    for i, h in enumerate(h_list):
        ax[i].plot(np.abs(h.numpy()))
        ax[i].set_title(f'Impulse Response {i + 1}')
        ax[i].set_xlabel('Sample')
        ax[i].set_ylabel('Amplitude')
    plt.tight_layout()
    plt.show()


def generate_rir_h(positions_x, positions_y, z:float, C:int, fs:float, receiver_position:List[List[float]],
                   room_dimensions:tuple[float,float,float],reverberation_time_range:tuple[float,float],
                   nsample:int) -> List[HSample]:

    with tqdm(total=len(positions_x)) as pbar:
        def generate_single_rir(i_x, i_y):
            nonlocal pbar
            # print(f'Generating H for location {i_x, i_y}')
            reverberation_time = random.uniform(reverberation_time_range[0], reverberation_time_range[1])
            try:
                h = rir.generate(
                    c=C,  # Sound velocity (m/s)
                    fs=fs,  # Sample frequency (samples/s)
                    r=receiver_position,  # Receiver position(s) [x y z] (m)
                    s=[i_x, i_y, z],  # Source position [x y z] (m)
                    L=room_dimensions,  # Room dimensions [x y z] (m)
                    reverberation_time=reverberation_time,  # Reverberation time (s)
                    nsample=nsample,  # Number of output samples
                )
            except ValueError:
                return None
            pbar.update(1)
            uid = str(uuid.uuid4())
            h_sample = HSample(h=torch.tensor(h, dtype=torch.float32).squeeze(), x=i_x, y=i_y, z=z, uid=uid,
                               receiver_position=receiver_position, room_dimensions=room_dimensions,
                               reverberation_time=reverberation_time)
            return h_sample

        h_list = []
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(generate_single_rir, i_x, i_y) for i_x, i_y in zip(positions_x, positions_y)]
            for future in as_completed(futures):
                if future.result() is not None:
                    h_list.append(future.result())

        return h_list


def generate_audio_samples(h_list, fs, rir_dataset_path, librispeech_dataset_subset, librispeech_dataset,
                           audio_length: float = 2, num_overall_samples: int = 2000):
    os.makedirs(rir_dataset_path, exist_ok=False)

    # Number of samples for audio_length seconds
    samples_per_audio_length = int(audio_length * fs)
    created_datapoints_num = 0
    libri_indices = list(librispeech_dataset_subset.indices)
    current_libri_idx = 0
    cycles_through_dataset = 0

    # Open CSV file for writing
    with open(rir_dataset_path / 'rir_dataset.csv', mode='w', newline='') as csv_file:
        fieldnames = ['entry_id', 'librispeech_index', 'start_sample', 'end_sample', 'length_in_seconds', 'path',
                      'h_uid', 'receiver_position', 'room_dimensions', 'source_location', 'reverberation_time', 'source_location_index']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        with tqdm(total=num_overall_samples) as pbar:
            while created_datapoints_num < num_overall_samples:
                # Cycle through LibriSpeech indices
                if current_libri_idx >= len(libri_indices):
                    current_libri_idx = 0
                    cycles_through_dataset += 1
                    if cycles_through_dataset == 1:
                        warnings.warn(f"Cycling through LibriSpeech dataset to reach {num_overall_samples} samples. Currently at {created_datapoints_num} samples.")
                
                libri_index = libri_indices[current_libri_idx]
                sample = librispeech_dataset.get_metadata(libri_index)
                (waveform, sample_rate, _, _, _, _) = librispeech_dataset.__getitem__(libri_index)
                waveform_tensor = waveform.reshape(waveform.shape[0], -1).T.squeeze()

                # Split the scaled signal into 2-second chunks
                num_chunks = len(waveform_tensor) // samples_per_audio_length
                for chunk_idx in range(num_chunks):
                    if created_datapoints_num >= num_overall_samples:
                        break
                        
                    # Select a different h_sample for each chunk
                    h_sample = random.choice(h_list)
                    h, i_x, i_y, i_z, h_uid = h_sample.h, h_sample.x, h_sample.y, h_sample.z, h_sample.uid

                    start_idx = chunk_idx * samples_per_audio_length
                    end_idx = start_idx + samples_per_audio_length
                    rir_data_entry = RirDataEntry(entry_id=created_datapoints_num, start_sample=start_idx,
                                                  end_sample=end_idx, length_in_seconds=2, path=Path(sample[0]), h_uid=h_uid,
                                                  receiver_position=h_sample.receiver_position, room_dimensions=h_sample.room_dimensions,
                                                  source_location=[i_x, i_y, i_z], reverberation_time=h_sample.reverberation_time)
                    writer.writerow({
                        'entry_id': rir_data_entry.entry_id,
                        'start_sample': rir_data_entry.start_sample,
                        'end_sample': rir_data_entry.end_sample,
                        'length_in_seconds': rir_data_entry.length_in_seconds,
                        'path': rir_data_entry.path,
                        'h_uid': rir_data_entry.h_uid,
                        'receiver_position': rir_data_entry.receiver_position,
                        'room_dimensions': rir_data_entry.room_dimensions,
                        'source_location': rir_data_entry.source_location,
                        'reverberation_time': rir_data_entry.reverberation_time,
                    })
                    created_datapoints_num += 1
                    pbar.update(1)
                
                current_libri_idx += 1

    if cycles_through_dataset == 0:
        print(f"Created {created_datapoints_num} samples using the dataset once.")
    else:
        print(f"Created {created_datapoints_num} samples, cycling through the dataset {cycles_through_dataset + 1} times to reach the target.")


def save_config(rir_dataset_path: Path, config: RirDatasetGeneratorConfig):
    # Remove pickle saving, only use JSON
    with open(rir_dataset_path / 'config.json', 'w') as json_file:
        json.dump(config.model_dump(), json_file, indent=4)


def save_h_dict(rir_dataset_path: Path, h_dict: dict):
    """
    Save h_dict using PyTorch tensors and JSON metadata instead of pickle.
    Each HSample is saved as separate files:
    - h_tensors/{uid}.pt for the tensor data
    - h_metadata.json for all metadata
    """
    # Create h_tensors directory
    tensors_dir = rir_dataset_path / 'h_tensors'
    os.makedirs(tensors_dir, exist_ok=True)
    
    # Save tensors and collect metadata
    metadata_dict = {}
    for uid, h_sample in h_dict.items():
        # Save tensor
        torch.save(h_sample.h, tensors_dir / f'{uid}.pt')
        
        # Collect metadata
        metadata_dict[uid] = {
            'x': h_sample.x,
            'y': h_sample.y,
            'z': h_sample.z,
            'uid': h_sample.uid,
            'receiver_position': h_sample.receiver_position,
            'room_dimensions': h_sample.room_dimensions,
            'reverberation_time': h_sample.reverberation_time
        }
    
    # Save metadata as JSON
    with open(rir_dataset_path / 'h_metadata.json', 'w') as json_file:
        json.dump(metadata_dict, json_file, indent=4)


def creat_and_save_audio_samples(config, h_list, dataset_name, librispeech_dataset, types, fig_list, num_overall_samples):
    save_path = Path(config.dataset_params.rir_dataset_path)


    rir_dataset_path = save_path / config.dataset_id / dataset_name
    generate_audio_samples(h_list, config.fs, rir_dataset_path, types[dataset_name], librispeech_dataset,
                           config.audio_length, num_overall_samples)
    save_config(rir_dataset_path, config)
    h_dict = {h_sample.uid: h_sample for h_sample in h_list}
    save_h_dict(Path(rir_dataset_path), h_dict)

    # Save the figure
    for i, fig in enumerate(fig_list):
        fig_save_path = rir_dataset_path / f'room_and_positions_{i}.png'
        os.makedirs(rir_dataset_path, exist_ok=True)
        fig.savefig(fig_save_path)

    print(f"Saved {dataset_name} dataset to {rir_dataset_path}")

def setup_logging():
    log_file_path = Path("logs") / "dataset_generation.log"
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger()
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(file_handler)
    logger.info("Logging to both terminal and log file")

def main():
    setup_logging()
    config = RirDatasetGeneratorConfig()
    librispeech_dataset = torchaudio.datasets.LIBRISPEECH(config.dataset_params.librispeech_path, url='train-clean-100',
                                                          download=True)


    # Define the sizes for the train and validation datasets
    train_libri_size = int(0.8 * len(librispeech_dataset))
    val_libri_size = int(0.1 * len(librispeech_dataset))
    test_libri_size = int(len(librispeech_dataset) - train_libri_size - val_libri_size)
    train_dataset, val_dataset, test_dataset = random_split(librispeech_dataset, [train_libri_size, val_libri_size, test_libri_size])
    types = {"train": train_dataset, "valid": val_dataset, "test": test_dataset}
    valid_size = int(config.train_data_set_size * (config.validation_size_percent / 100.0))

    n_rooms = len(config.room_dimensions)
    h_sample_list = []
    h_sample_list_valid = []
    fig_list_train = []
    fig_list_valid = []

    for i_room in range(n_rooms):
        receiver_position = [config.room_dimensions[i_room][0] * config.receiver_position_percent[0][0] / 100,
                             config.room_dimensions[i_room][1] * config.receiver_position_percent[0][1] / 100,
                             config.receiver_position_percent[0][2]]

        # Generate train positions
        positions_x_train, positions_y_train = generate_positions(receiver_position, config.room_dimensions[i_room],
                                                      config.wall_backoff, config.theta_range,
                                                      config.r_range, config.num_of_position_to_generate_train)
        # Generate validation positions (ensure different random seed or call)
        positions_x_valid, positions_y_valid = generate_positions(receiver_position, config.room_dimensions[i_room],
                                                      config.wall_backoff, config.theta_range,
                                                      config.r_range, valid_size)

        fig_train = plot_room_and_positions(positions_x_train, positions_y_train, None, None, receiver_position,
                                config.room_dimensions[i_room], i_room)
        fig_valid = plot_room_and_positions(positions_x_valid, positions_y_valid, None, None, receiver_position,
                                config.room_dimensions[i_room], i_room)

        print(f'Generating RIR for {len(positions_x_train)} positions Train')
        z = receiver_position[2]
        room_h_sample_list_train = generate_rir_h(positions_x_train, positions_y_train, z, config.C, config.fs, receiver_position,
                                       config.room_dimensions[i_room],config.reverberation_time_range, config.nsample)
        print(f'Generating RIR for {len(positions_x_valid)} positions Validation')
        room_h_sample_list_valid = generate_rir_h(positions_x_valid, positions_y_valid, z, config.C, config.fs, receiver_position,
                                       config.room_dimensions[i_room],config.reverberation_time_range, config.nsample)

        h_sample_list.extend(room_h_sample_list_train)
        h_sample_list_valid.extend(room_h_sample_list_valid)
        fig_list_train.append(fig_train)
        fig_list_valid.append(fig_valid)


    for i in types.keys():
        if i.lower() == "train":
            creat_and_save_audio_samples(
                config, h_sample_list, i, librispeech_dataset, types, fig_list_train, config.train_data_set_size
            )
        elif i.lower() == "valid":

            creat_and_save_audio_samples(
                config, h_sample_list, i, librispeech_dataset, types, fig_list_valid, valid_size
            )


if __name__ == '__main__':
    main()
