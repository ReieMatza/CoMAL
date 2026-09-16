# CoMAL

Publication CoMAL, single-microphone acoustic localization.

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
## Evaluation

```bash
python scripts/evaluation_scripts/evaluate_test_dataset.py /path/to/checkpoint.ckpt --dataset_split valid --batch_size 64

python scripts/evaluation_scripts/plot_angular_error_vs_snr.py /path/to/checkpoint.ckpt --include_random_baseline --dataset_split valid
```
## License

See [LICENSE](LICENSE) file for details.
