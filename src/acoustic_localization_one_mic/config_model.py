import os
import uuid
from dataclasses import fields
from typing import List, Tuple, Optional
import torch

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings

def get_cuda_memory_based_defaults():
    """
    Returns batch_size and gradient_accumulation_steps based on available CUDA memory.
    If CUDA memory > 30GB: batch_size=200, gradient_accumulation_steps=2
    If CUDA memory <= 30GB or CUDA not available: batch_size=50, gradient_accumulation_steps=5
    """
    if torch.cuda.is_available():
        # Get total memory in GB
        total_memory_bytes = torch.cuda.get_device_properties(0).total_memory
        total_memory_gb = total_memory_bytes / (1024**3)
        
        if total_memory_gb > 30:
            return 70, 4  # High memory configuration
        else:
            return 20, 8   # Low memory configuration
    else:
        return 2, 5  # No CUDA available, use conservative settings
    

class FinsModelConfigParams(BaseSettings):
    num_filters: int = 10
    filter_order: int = 1023
    sr: float = 16e3
    rir_length: int = Field(default=int(0.5 * 16e3), description="Number of samples (max(reverberation_time) * fs)")
    input_length: int = 32000
    early_length: int = 2400
    decoder_input_length: int = 400
    noise_condition_length: int = 16
    z_size: int = 128
    min_snr: int = 15
    max_snr: int = 50
    normalize: str = "rms"
    rms_level: float = 0.01
    num_angles: int = 181
    max_rad_value: float = 6 #[m]
    rad_resolution: float = 0.1 #[m]
    encoder_dropout_rate: float = 0.2
    use_metadata: bool = False
    metadata_len: int = 0

class DatasetParams(BaseSettings):
    rir_dataset_path: str = "data/rir_dataset"
    librispeech_path: str = "data/librispeech"


class TrainParams(BaseSettings):
    sr: float = 16e3
    batch_size: int = Field(default_factory=lambda: get_cuda_memory_based_defaults()[0])
    train_num_workers: int = 16  # Number of workers for training dataloader
    valid_num_workers: int = 4   # Number of workers for validation dataloader
    prefetch_factor: int = 2
    persistent_workers : bool = True
    lr: float = 0.0001
    lr_step_size: int = 50
    lr_decay_factor: float = 0.8
    gradient_clip_value: float = 5.0
    gradient_accumulation_steps: int = Field(default_factory=lambda: get_cuda_memory_based_defaults()[1])
    num_epochs: int = 500
    validation_interval: int = 5
    checkpoint_interval: int = 5
    logging_dir: str = "logs"
    checkpoint_dir: str = "checkpoints"
    early_length: int = 2400
    input_length: int = 32000
    rir_length: int =  Field(default=int(0.5 * 16e3), description="Number of samples (max(reverberation_time) * fs)")
    rms_level: float = 0.01
    noise_condition_length: int = 16
    num_filters: int = 10
    # Training precision configuration
    precision: str = Field(default="16-mixed", description="Training precision: '32-true', '16-mixed', 'bf16-mixed'")
    # Loss weights
    stft_loss_weight: float = 1.0  # STFT loss quickly converges and shouldn't dominate
    angle_loss_weight: float = 1.0  # Classification loss is central
    radius_loss_weight: float = 1.0  # Rad classification is also important
    discriminator_loss_weight: float = 1.0  # Discriminator loss weight

    # Number of views per RIR for supervised contrastive loss
    contrastive_n_views: int = 6
    # Weight for supervised contrastive loss
    contrastive_loss_weight: float = 1.0
    # Enable/disable contrastive learning
    use_contrastive_learning: bool = Field(default=False, description="Enable or disable contrastive learning during training")

    # Resume training from a previous wandb run
    resume_from_wandb_run: Optional[str] = Field(default=None, description="Wandb run name to resume training from (e.g., 'atomic-resonance-703')")
    resume_checkpoint_type: str = Field(default="best", description="Type of checkpoint to resume from: 'best' (best metric) or 'latest' (latest epoch)")

    # Profiler option: None/False disables, 'simple' or 'advanced' enables
    profiler: Optional[str] = Field(default=None, description="Enable PyTorch Lightning profiler: 'simple', 'advanced', or None/False for off.")

    # Multi-GPU/distributed training options
    devices: int = Field(default=1, description="Number of devices (GPUs/CPUs) to use for training. Set to 2 for 2 GPUs.")
    strategy: Optional[str] = Field(default=None, description="Distributed training strategy, e.g., 'ddp' for multi-GPU. Set to None for auto.")




class WandbConfigParams(BaseSettings):
    project: str = "acoustic-localization"
    tags: str = "model"
    dir: str = "logs"
    config: dict = {FinsModelConfigParams.__name__: FinsModelConfigParams().model_dump(),
              TrainParams.__name__: TrainParams().model_dump(),
                DatasetParams.__name__: DatasetParams().model_dump(),}

class WandbConfigLocationParams(BaseSettings):
    project: str = "fins_location"
    tags: str = "fins_location"
    dir: str = "logs"
    config: dict = {FinsModelConfigParams.__name__: FinsModelConfigParams().model_dump(),
              TrainParams.__name__: TrainParams().model_dump(),
                DatasetParams.__name__: DatasetParams().model_dump()}


class RirDatasetGeneratorConfig(BaseModel):
    C: int = Field(default=340, description="Speed of sound in m/s")
    fs: float = Field(default=16e3, description="Sampling frequency in Hz")
    receiver_position_percent: List[List[float]] = Field(default=[[60, 20, 1]], description="Receiver position [x, y, z]")
    room_dimensions: List[List[int]] = Field(default=[[6, 5, 3],[6, 7, 3],[8, 5, 3]], description="Room dimensions [x, y, z]")
    reverberation_time_range: tuple[float, float] = Field(default=(0.2, 0.5),
                                                          description="Reverberation time in seconds")
    nsample: int = Field(default=int(0.5 * 16e3), description="Number of samples (max(reverberation_time) * fs)")
    dataset_params: DatasetParams = Field(default=DatasetParams(), description="Dataset parameters")
    wall_backoff: float = 0.5
    theta_range: Tuple[int,int] = (-90,90)
    train_data_set_size: int = 150000  # replaces max_overall_samples
    validation_size_percent: float = 10.0  # percent of train size for validation
    dataset_id: str = Field(default_factory=lambda: f"{str(uuid.uuid4())[:8]}")
    audio_length: float = 2.0
    r_range: Tuple[float, float] = (0.2, 6)
    num_of_position_to_generate_train:int = 30000
    num_of_position_to_generate_test:int = 5000
