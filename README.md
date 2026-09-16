# CoMAL

Publication snapshot of single-microphone acoustic localization with a Filtered Noise Shaper (FINS).

This repository is a copy of the last single-speaker FINS tree from [Acoustic_Localization_one_mic](https://github.com/ReieMatza/Acoustic_Localization_one_mic), frozen at:

```
281dda489612f083dc1b20d3a953bf7135456206
```

See [SOURCE_COMMIT.txt](SOURCE_COMMIT.txt). The internal Python package name is still `acoustic_localization_one_mic` so imports match that snapshot.

## Features

- **Single Microphone Acoustic Localization**: Estimate source location using only one microphone
- **FINS Model**: Filtered Noise Shaper for robust RIR estimation
- **Flexible Training**: Support for both contrastive and non-contrastive learning modes
- **Efficient Training**: Optimized for different GPU memory configurations
- **Wandb Integration**: Comprehensive logging and experiment tracking
- **Checkpoint Resuming**: Easy resumption of interrupted training sessions

## Installation

```bash
pip install -e .
```

Or:

```bash
pip install -r requirements.txt
```

The default dataset, checkpoint, and log paths are `data/rir_dataset`, `data/librispeech`, `checkpoints`, and `logs`. Override them with environment variables (pydantic-settings), for example:

```bash
export rir_dataset_path=/path/to/rir_dataset
export librispeech_path=/path/to/librispeech
export checkpoint_dir=/path/to/checkpoints
export logging_dir=/path/to/logs
```

## Dataset generation

Generate the image-source RIR dataset (requires LibriSpeech):

```bash
python -m acoustic_localization_one_mic.rir_dataset.genereate_dataset
```

## Training

Entry point: `acoustic_localization_one_mic.fins.rir_pipeline.main_pipline`.

```bash
python -m acoustic_localization_one_mic.fins.rir_pipeline
```

### Training Without Contrastive Learning (Recommended for efficiency)

For faster training when you want to focus on the main task losses:

```python
from acoustic_localization_one_mic.config_model import TrainParams
from acoustic_localization_one_mic.fins.rir_pipeline import main_pipline

train_config = TrainParams(
    use_contrastive_learning=False,
    batch_size=100,
    gradient_accumulation_steps=2,
)

main_pipline(train_config)
```

### Training With Contrastive Learning

For best model performance with sufficient computational resources:

```python
from acoustic_localization_one_mic.config_model import TrainParams
from acoustic_localization_one_mic.fins.rir_pipeline import main_pipline

train_config = TrainParams(
    use_contrastive_learning=True,
    batch_size=50,
    gradient_accumulation_steps=4,
    contrastive_n_views=6,
    contrastive_loss_weight=1.0,
)

main_pipline(train_config)
```

To resume a previous wandb run, set `TrainParams(resume_from_wandb_run="run-name", resume_checkpoint_type="best")`.

## Evaluation

```bash
python scripts/evaluation_scripts/evaluate_test_dataset.py /path/to/checkpoint.ckpt --dataset_split valid --batch_size 64

python scripts/evaluation_scripts/plot_angular_error_vs_snr.py /path/to/checkpoint.ckpt --include_random_baseline --dataset_split valid
```

## Performance Considerations

### Training Speed
- **Without contrastive learning**: ~2-3x faster training
- **With contrastive learning**: Slower due to additional forward passes

### Memory Usage
- **Without contrastive learning**: Lower memory usage, larger batch sizes possible
- **With contrastive learning**: Higher memory usage due to multiple views per RIR

### Recommended Settings by GPU Memory

| GPU Memory | Contrastive Learning | Batch Size | Gradient Accumulation |
|------------|---------------------|------------|----------------------|
| 24GB+      | Enabled             | 70         | 4                     |
| 8-16GB     | Disabled            | 50         | 4                     |
| <8GB       | Disabled            | 20         | 8                     |

## Configuration

The model uses Pydantic for configuration management. Key configuration classes:

- `TrainParams`: Training hyperparameters and settings
- `FinsModelConfigParams`: Model architecture parameters
- `DatasetParams`: Dataset paths and settings

## License

See [LICENSE](LICENSE) file for details.
