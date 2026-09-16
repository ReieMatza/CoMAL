from typing import Tuple

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

from acoustic_localization_one_mic.config_model import FinsModelConfigParams
from acoustic_localization_one_mic.fins.fc_classification_model import AngleClassification, RadiusClassification
from acoustic_localization_one_mic.fins.model.utils.audio import (
    get_octave_filters,
)

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        super(EncoderBlock, self).__init__()
        if use_batchnorm:
            self.conv = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=15, stride=2, padding=7),
                nn.BatchNorm1d(out_channels, track_running_stats=True),
                nn.PReLU(),
            )
            self.skip_conv = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=2, padding=0),
                nn.BatchNorm1d(out_channels, track_running_stats=True),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=15, stride=2, padding=7),
                nn.PReLU(),
            )
            self.skip_conv = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=2, padding=0),
            )

    def forward(self, x):
        out = self.conv(x)
        skip_out = self.skip_conv(x)
        skip_out = out + skip_out
        return skip_out


class Encoder(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super(Encoder, self).__init__()
        block_list = []
        channels = [1, 32, 32, 64, 64, 128, 128, 256, 256, 512]

        for i in range(0, len(channels) - 1):
            if (i + 1) % 3 == 0:
                use_batchnorm = True
            else:
                use_batchnorm = False
            in_channels = channels[i]
            out_channels = channels[i + 1]
            curr_block = EncoderBlock(in_channels, out_channels, use_batchnorm)

            # Add dropout to the final 3 EncoderBlocks
            if i >= len(channels) - 4:
                block_list.append(nn.Sequential(
                    curr_block,
                    nn.Dropout(dropout_rate)
                ))
            else:
                block_list.append(curr_block)

        self.encode = nn.Sequential(*block_list)
        self.pooling = nn.AdaptiveAvgPool1d(1)
        # Add dropout before the final fully connected layer
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(512, 128)

    def forward(self, x):
        b = x.shape[0]
        out = self.encode(x)
        out = self.pooling(out)
        out = out.view(b, -1)
        # Apply dropout before the final fully connected layer
        out = self.dropout(out)
        out = self.fc(out)
        return out


class UpsampleNet(nn.Module):
    def __init__(self, input_size, output_size, upsample_factor):
        super(UpsampleNet, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.upsample_factor = upsample_factor

        layer = nn.ConvTranspose1d(
            input_size,
            output_size,
            upsample_factor * 2,
            upsample_factor,
            padding=upsample_factor // 2,
        )
        nn.init.orthogonal_(layer.weight)
        self.layer = spectral_norm(layer)

    def forward(self, inputs):
        outputs = self.layer(inputs)
        outputs = outputs[:, :, : inputs.size(-1) * self.upsample_factor]
        return outputs


class ConditionalBatchNorm1d(nn.Module):

    """Conditional Batch Normalization"""

    def __init__(self, num_features, condition_length):
        super().__init__()

        self.num_features = num_features
        self.condition_length = condition_length
        self.norm = nn.BatchNorm1d(num_features, affine=True, track_running_stats=True)

        self.layer = spectral_norm(nn.Linear(condition_length, num_features * 2))
        self.layer.weight.data.normal_(1, 0.02)  # Initialise scale at N(1, 0.02)
        self.layer.bias.data.zero_()  # Initialise bias at 0

    def forward(self, inputs, noise):
        outputs = self.norm(inputs)
        gamma, beta = self.layer(noise).chunk(2, 1)
        gamma = gamma.view(-1, self.num_features, 1)
        beta = beta.view(-1, self.num_features, 1)

        outputs = gamma * outputs + beta

        return outputs


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, upsample_factor, condition_length):
        super(DecoderBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.condition_length = condition_length
        self.upsample_factor = upsample_factor

        # Block A
        self.condition_batchnorm1 = ConditionalBatchNorm1d(in_channels, condition_length)

        self.first_stack = nn.Sequential(
            nn.PReLU(),
            UpsampleNet(in_channels, in_channels, upsample_factor),
            nn.Conv1d(in_channels, out_channels, kernel_size=15, dilation=1, padding=7),
        )

        self.condition_batchnorm2 = ConditionalBatchNorm1d(out_channels, condition_length)

        self.second_stack = nn.Sequential(
            nn.PReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=15, dilation=1, padding=7),
        )

        self.residual1 = nn.Sequential(
            UpsampleNet(in_channels, in_channels, upsample_factor),
            nn.Conv1d(in_channels, out_channels, kernel_size=1, padding=0),
        )
        # Block B
        self.condition_batchnorm3 = ConditionalBatchNorm1d(out_channels, condition_length)

        self.third_stack = nn.Sequential(
            nn.PReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=15, dilation=4, padding=28),
        )

        self.condition_batchnorm4 = ConditionalBatchNorm1d(out_channels, condition_length)

        self.fourth_stack = nn.Sequential(
            nn.PReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=15, dilation=8, padding=56),
        )

    def forward(self, enc_out, condition):
        inputs = enc_out

        outputs = self.condition_batchnorm1(inputs, condition)
        outputs = self.first_stack(outputs)
        outputs = self.condition_batchnorm2(outputs, condition)
        outputs = self.second_stack(outputs)

        residual_outputs = self.residual1(inputs) + outputs

        outputs = self.condition_batchnorm3(residual_outputs, condition)
        outputs = self.third_stack(outputs)
        outputs = self.condition_batchnorm4(outputs, condition)
        outputs = self.fourth_stack(outputs)

        outputs = outputs + residual_outputs

        return outputs


class Decoder(nn.Module):
    def __init__(self, num_filters, cond_length):
        super(Decoder, self).__init__()

        self.preprocess = nn.Conv1d(1, 512, kernel_size=15, padding=7)
        self.blocks = nn.ModuleList(
            [
                DecoderBlock(512, 512, 1, cond_length),
                DecoderBlock(512, 512, 1, cond_length),
                DecoderBlock(512, 256, 2, cond_length),
                DecoderBlock(256, 256, 2, cond_length),
                DecoderBlock(256, 256, 1, cond_length),
                # DecoderBlock(256, 128, 3, cond_length),
                DecoderBlock(256, 64, 5, cond_length),
            ]
        )

        self.postprocess = nn.Sequential(nn.Conv1d(64, num_filters + 1, kernel_size=15, padding=7))

        self.sigmoid = nn.Sigmoid()

    def forward(self, v, condition):
        inputs = self.preprocess(v)
        outputs = inputs
        for i, layer in enumerate(self.blocks):
            outputs = layer(outputs, condition)
        outputs = self.postprocess(outputs)

        direct_early = outputs[:, 0:1]
        late = outputs[:, 1:]
        late = self.sigmoid(late)

        return direct_early, late


class FilteredNoiseShaper(nn.Module):
    def __init__(self, config:FinsModelConfigParams):
        super(FilteredNoiseShaper, self).__init__()

        self.config = config

        self.rir_length = int(self.config.rir_length)
        self.min_snr, self.max_snr = config.min_snr, config.max_snr

        # Learned decoder input
        self.decoder_input = nn.Parameter(torch.randn((1, 1, config.decoder_input_length)))  # 1,1,400
        self.encoder = Encoder(dropout_rate=config.encoder_dropout_rate)

        self.decoder = Decoder(config.num_filters, config.noise_condition_length + config.z_size)

        # Learned "octave-band" like filter
        self.filter = nn.Conv1d(
            config.num_filters,
            config.num_filters,
            kernel_size=config.filter_order,
            stride=1,
            padding='same',
            groups=config.num_filters,
            bias=False,
        )

        self.angle_classification = AngleClassification(config.z_size, config.num_angles, config.metadata_len)
        self.radius_classification = RadiusClassification(config.z_size, int(config.max_rad_value/config.rad_resolution)+1, config.metadata_len)

        # Octave band pass initialization
        octave_filters = get_octave_filters()
        self.filter.weight.data = torch.FloatTensor(octave_filters)

        # self.filter.bias.data.zero_()

        # Mask for direct and early part
        mask = torch.zeros((1, 1, self.rir_length))
        mask[:, :, : self.config.early_length] = 1.0
        self.register_buffer("mask", mask)
        self.output_conv = nn.Conv1d(config.num_filters + 1, 1, kernel_size=1, stride=1)

    def forward(self, x, stochastic_noise, noise_condition, metadata=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        args:
            x : Reverberant speech. shape=(batch_size, 1, input_samples)
            stochastic_noise : Random normal noise for late reverb synthesis. shape=(batch_size, n_freq_bands, length_of_rir)
            noise_condition : Noise used for conditioning. shape=(batch_size, noise_cond_length)
        return
            rir: shape=(batch_size, 1, rir_samples)
            angle_predictions: shape=(batch_size, num_angles)
            rad_predictions: shape=(batch_size, num_radii)
            z: latent representation (batch_size, z_size)
        """
        b = x.shape[0]

        # Filter random noise signal
        filtered_noise = self.filter(stochastic_noise)

        # Encode the reverberated speech
        z = self.encoder(x)

        # Classify the location
        angle_predictions = self.angle_classification(z, metadata)

        rad_predictions = self.radius_classification(z, metadata)

        # Make condition vector
        condition = torch.cat([z, noise_condition], dim=-1)

        # Learnable decoder input. Repeat it in the batch dimension.
        decoder_input = self.decoder_input.repeat(b, 1, 1)

        # Generate RIR
        direct_early, late_mask = self.decoder(decoder_input, condition)

        # Apply mask to the filtered noise to get the late part
        late_part = filtered_noise * late_mask

        # Zero out sample beyond 2400 for direct early part
        direct_early = torch.mul(direct_early, self.mask)
        # Concat direct,early with late and perform convolution
        rir = torch.cat((direct_early, late_part), 1)

        # Sum
        rir = self.output_conv(rir)

        return rir, angle_predictions, rad_predictions, z

    def get_latent_representation(self, x):
        z = self.encoder(x)
        return z

if __name__ == "__main__":


    batch_size = 1
    input_size = 32000
    noise_size = 16
    target_size = 16000

    device = 'cpu'

    # load config
    config = FinsModelConfigParams()
    print(config)

    x = torch.randn((batch_size, 1, input_size)).to(device)
    stochastic_noise = torch.randn((batch_size, 10, target_size)).to(device)
    noise_condition = torch.randn((batch_size, noise_size)).to(device)

    model = FilteredNoiseShaper(config).to(device)

    rir_estimated = model(x, stochastic_noise, noise_condition)
    print(rir_estimated.shape)
