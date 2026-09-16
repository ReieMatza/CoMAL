import pickle
import sys
from pathlib import Path
import sounddevice as sd


def play_rir_data_sample(pickle_path: Path):
    # Load the RirDataSample from the pickle file
    with open(pickle_path, 'rb') as f:
        rir_data_sample = pickle.load(f)

    # Extract the signal
    signal = rir_data_sample.reverberated_speech

    # Ensure the signal is a 1D tensor
    if signal.ndim > 1:
        signal = signal.squeeze()

    # Play the signal using torchaudio
    sd.play(signal.numpy(), samplerate=16000)
    sd.wait()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python -m acoustic_localization_one_mic.rir_dataset.play_sample <pickle_path>")
    play_rir_data_sample(Path(sys.argv[1]))
