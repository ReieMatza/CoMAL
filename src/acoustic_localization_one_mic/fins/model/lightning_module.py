import lightning as pl
import torch
import torch.nn as nn
import torchaudio
from audiomentations import Compose, AddGaussianNoise, PitchShift, Shift, Gain
import random
import numpy as np

from acoustic_localization_one_mic.fins.common_functions import calculate_theta, calculate_radius
from acoustic_localization_one_mic.fins.model.loss import MultiResolutionSTFTLoss
from acoustic_localization_one_mic.fins.model.model import FilteredNoiseShaper
from acoustic_localization_one_mic.fins.model.utils.audio import add_noise_batch, audio_normalize_batch


def generate_colored_noise_gpu(beta: float, length: int, device: torch.device):
    # Generate white noise in frequency domain
    freqs = torch.fft.rfftfreq(length, d=1.0).to(device)
    freqs[0] = 1e-6  # avoid division by zero
    amplitude = 1.0 / (freqs ** (beta / 2.0))  # Power-law spectrum
    phase = 2 * torch.pi * torch.rand_like(amplitude)
    spectrum = amplitude * torch.exp(1j * phase)

    # Transform back to time domain and normalize
    noise = torch.fft.irfft(spectrum, n=length)
    noise = noise.unsqueeze(0)  # shape (1, length)
    noise = noise / noise.std()  # normalize to unit variance
    return noise


def get_noise_and_snr_torch(source: torch.Tensor, use_noise=True, input_signal_length=131070):
    device = source.device
    if use_noise:
        if random.random() < 0.9:
            min_snr = 0.0
            max_snr = 30.0
            beta = random.random() + 1.0
            noise = generate_colored_noise_gpu(beta, input_signal_length, device)
            snr_db = torch.tensor([random.random() * (max_snr - min_snr) + min_snr], dtype=torch.float32,
                                  device=device)
        else:
            noise = torch.zeros_like(source, device=device)
            snr_db = torch.tensor([0.0], dtype=torch.float32, device=device)
    else:
        noise = torch.zeros_like(source, device=device)
        snr_db = torch.tensor([0.0], dtype=torch.float32, device=device)

    return noise, snr_db


class FinsLightningModule(pl.LightningModule):
    def __init__(self, model_config, train_config):
        super().__init__()
        self.save_hyperparameters()
        self.model_config = model_config
        self.train_config = train_config

        self.model = FilteredNoiseShaper(config=model_config)

        # Loss functions
        fft_sizes = [64, 512, 2048, 8192]
        hop_sizes = [32, 256, 1024, 4096]
        win_lengths = [64, 512, 2048, 8192]
        sc_weight = 1.0
        mag_weight = 1.0

        self.classification_criterion = nn.CrossEntropyLoss(reduction='mean', label_smoothing=0.1)
        self.stft_loss_fn = MultiResolutionSTFTLoss(
            fft_sizes=fft_sizes,
            hop_sizes=hop_sizes,
            win_lengths=win_lengths,
            sc_weight=sc_weight,
            mag_weight=mag_weight,
        )

        self.augment = Compose([
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.01, p=0.5),
            PitchShift(min_semitones=-2, max_semitones=2, p=0.2),
            Shift(min_shift=-0.5, max_shift=0.5, p=0.5),
            Gain(min_gain_db=-6, max_gain_db=6, p=0.5)
        ])
        
    def forward(self, reverberated_source_with_noise, batch_stochastic_noise, batch_noise_condition, metadata=None):
        return self.model(reverberated_source_with_noise, batch_stochastic_noise, batch_noise_condition, metadata=metadata)

    def _shared_step(self, batch, batch_idx, stage: str):
        rir, source_location, receiver_position, room_dimensions, source, reverberation_time, sample_rate = batch

        if stage == 'train':
            # Augmentation
            sr_scalar = sample_rate[0].item() if sample_rate.numel() > 1 else sample_rate.item()
            # Convert to numpy for augmentation, then back to tensor
            source_np = source.squeeze().cpu().numpy().astype(np.float32)
            source_augmented = self.augment(source_np, sample_rate=sr_scalar)
            source = torch.tensor(source_augmented, dtype=torch.float32, device=self.device)
        else:
            # For validation, just squeeze and ensure it's on the right device
            source = source.squeeze().clone().detach().to(self.device)

        # Convolve
        reverberated_speech = torchaudio.functional.convolve(source.float(), rir, mode='same')

        # Prepare batch data
        (reverberated_source_with_noise, _, batch_stochastic_noise, batch_noise_condition) = self.make_batch_data(
            reverberated_speech)

        # Metadata
        metadata = None
        if self.model_config.use_metadata:
            metadata = torch.cat((room_dimensions, receiver_position, reverberation_time.unsqueeze(1)), dim=1)

        # === Main forward pass for main loss ===
        predicted_rir, angle_predictions, rad_prediction, z_main = self(
            reverberated_source_with_noise, batch_stochastic_noise, batch_noise_condition, metadata=metadata
        )

        # === Main task loss calculation (as before) ===
        theta_degrees_int = calculate_theta(receiver_position, source_location).int().long()
        radius_tensor = (calculate_radius(receiver_position, source_location) / self.model_config.rad_resolution).long()

        stft_loss = self.stft_loss_fn(predicted_rir, rir)["total"]
        angle_classification_loss = self.classification_criterion(angle_predictions, theta_degrees_int)
        rad_classification_loss = self.classification_criterion(rad_prediction, radius_tensor)

        main_loss = (self.train_config.stft_loss_weight * stft_loss +
                     self.train_config.angle_loss_weight * angle_classification_loss +
                     self.train_config.radius_loss_weight * rad_classification_loss)

        # === Contrastive learning (optional) ===
        contrastive_loss = torch.tensor(0.0, device=self.device)
        if self.train_config.use_contrastive_learning:
            # === For contrastive loss only: expand batch with N views per RIR (efficient vectorized version) ===
            n_views = self.train_config.contrastive_n_views
            batch_size = rir.shape[0]
            device = rir.device
            speech_pool = source.squeeze()  # (batch_size, T)
            if speech_pool.dim() == 1:
                speech_pool = speech_pool.unsqueeze(0)
            rir_expanded = rir.repeat_interleave(n_views, dim=0)  # (B*N, 1, T)
            sampled_indices = torch.randint(0, batch_size, (batch_size * n_views,), device=device)
            speech_expanded = speech_pool[sampled_indices]  # (B*N, T)
            # Convolve in batch
            reverberated_speech = torchaudio.functional.convolve(speech_expanded.float(), rir_expanded, mode='same')
            # Prepare batch data for all (B*N) pairs
            (contrastive_reverb_with_noise, _, contrastive_stochastic_noise, contrastive_noise_condition) = self.make_batch_data(
                reverberated_speech)
            # Metadata for each RIR (if used)
            meta = None
            if self.model_config.use_metadata:
                meta = torch.cat((
                    room_dimensions.repeat_interleave(n_views, dim=0),
                    receiver_position.repeat_interleave(n_views, dim=0),
                    reverberation_time.repeat_interleave(n_views, dim=0).unsqueeze(1)
                ), dim=1)
            # Forward pass for all (B*N) pairs
            _, _, _, z = self(
                contrastive_reverb_with_noise, contrastive_stochastic_noise, contrastive_noise_condition, metadata=meta
            )
            embeddings = z  # (B*N, latent_dim)
            labels = torch.arange(batch_size, device=device).repeat_interleave(n_views)  # (B*N,)
            from acoustic_localization_one_mic.fins.model.loss import supervised_contrastive_loss
            contrastive_loss = supervised_contrastive_loss(embeddings, labels)
        
        weighted_contrastive_loss = self.train_config.contrastive_loss_weight * contrastive_loss
        total_loss = main_loss + weighted_contrastive_loss

        # Logging
        self.log(f'{stage}/loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log(f'{stage}/loss/main', main_loss, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        if self.train_config.use_contrastive_learning:
            self.log(f'{stage}/loss/contrastive', contrastive_loss, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log(f'{stage}/loss/stft', stft_loss, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log(f'{stage}/loss/angle_classification', angle_classification_loss, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log(f'{stage}/loss/rad_classification', rad_classification_loss, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log(f'{stage}/z_norm', z_main.norm(dim=1).mean(), on_step=True, on_epoch=True, logger=True, sync_dist=True)


        # Error metrics
        angular_error = torch.abs((torch.argmax(angle_predictions, dim=1) - theta_degrees_int)).float().mean()
        rad_error = torch.abs((torch.argmax(rad_prediction, dim=1) * self.model.config.rad_resolution -
                               calculate_radius(receiver_position, source_location,
                                                rounded=False))).float().mean()
        self.log(f'{stage}/error/angular', angular_error, on_epoch=True, logger=True, sync_dist=True)
        self.log(f'{stage}/error/rad', rad_error, on_epoch=True, logger=True, sync_dist=True)

        return total_loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, 'valid')

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.train_config.lr, weight_decay=1e-6)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.train_config.lr_step_size,
            gamma=self.train_config.lr_decay_factor,
        )
        return [optimizer], [scheduler]

    def on_save_checkpoint(self, checkpoint):
        """Store wandb run ID in checkpoint for resuming."""
        # Get the current wandb run ID if available
        if hasattr(self.logger, 'experiment') and hasattr(self.logger.experiment, 'id'):
            checkpoint['wandb_run_id'] = self.logger.experiment.id
    def make_batch_data(self, reverberated_speech):
        reverberated_source = reverberated_speech.unsqueeze(1)
        noise, snr_db = get_noise_and_snr_torch(reverberated_source, use_noise=True,
                                                     input_signal_length=self.model_config.input_length)

        batch_size = reverberated_speech.shape[0]

        reverberated_source = audio_normalize_batch(reverberated_source, "rms", self.train_config.rms_level)

        # Noise SNR
        reverberated_source_with_noise = add_noise_batch(reverberated_source, noise, snr_db)

        # Noise for late part
        rir_length = int(self.model_config.rir_length)
        stochastic_noise = torch.randn((batch_size, 1, rir_length), device=self.device)
        batch_stochastic_noise = stochastic_noise.repeat(1, self.model_config.num_filters, 1)

        # Noise for decoder conditioning
        batch_noise_condition = torch.randn((batch_size, self.model_config.noise_condition_length), device=self.device)

        return (
            reverberated_source_with_noise,
            reverberated_source,
            batch_stochastic_noise,
            batch_noise_condition,
        )
