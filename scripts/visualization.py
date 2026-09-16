import csv
import os
import pickle
from pathlib import Path

import numpy as np
import torch
import torchaudio
import matplotlib.pyplot as plt
import umap
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import seaborn as sns
from torch.utils.data import DataLoader
from torchviz import make_dot
from tqdm import tqdm

from acoustic_localization_one_mic.fins.common_functions import calculate_theta, calculate_radius
from acoustic_localization_one_mic.fins.model.model import FilteredNoiseShaper
from acoustic_localization_one_mic.fins.fc_classification_model import AngleClassification, \
    load_angle_model_from_file
from acoustic_localization_one_mic.fins.rir_pipeline import load_model_from_file, load_train_valid_data_sets, \
    load_dataset_and_get_dataloader, device
from acoustic_localization_one_mic.config_model import DatasetParams, FinsModelConfigParams, \
    TrainParams, RirDatasetGeneratorConfig
from acoustic_localization_one_mic.rir_dataset.genereate_dataset import generate_positions, generate_rir_h, HSample
from acoustic_localization_one_mic.fins.model.lightning_module import FinsLightningModule


def load_fins_lightning_module_from_checkpoint(checkpoint_path: str, model_config: FinsModelConfigParams, train_config: TrainParams):
    """
    Load a FinsLightningModule from a PyTorch Lightning checkpoint file.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        model_config: Model configuration parameters
        train_config: Training configuration parameters
        
    Returns:
        Loaded FinsLightningModule in evaluation mode
    """
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Create the FinsLightningModule instance
    model = FinsLightningModule(model_config=model_config, train_config=train_config)
    
    # Load the state dict
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    else:
        # If no state_dict key, assume the checkpoint is the state dict
        model.load_state_dict(checkpoint, strict=False)
    
    model.eval()
    model = model.to(device)
    return model


def plot_room_and_positions_with_probabilities(receiver_position, room_dimensions,
                                               probabilities, source_location):
    fig, ax = plt.subplots(1, 1)
    plt.scatter(source_location[0], source_location[1], label='Source_location', color='blue', zorder=4)
    plt.scatter(receiver_position[0], receiver_position[1], label='Mic location', color='red', zorder=3)
    ax.add_patch(Rectangle((0, 0), room_dimensions[0], room_dimensions[1], alpha=0.1, color='red', zorder=0))
    ax.set_xlabel('Room X')
    ax.set_ylabel('Room Y')

    # Set the aspect ratio to be equal
    ax.set_aspect('equal', adjustable='box')

    # Define the angles (0 to 180 degrees in radians)
    angles = np.linspace(0 , np.pi , len(probabilities))

    # Normalize probabilities for colormap
    norm = mcolors.Normalize(vmin=0, vmax=np.max(probabilities))
    cmap = cm.get_cmap("viridis")

    # Draw the probability arc
    for i, angle in enumerate(angles):
        direction = np.array([np.cos(angle), np.sin(angle)])
        end_point = receiver_position[0:2] + 1 * direction  # Scale for visibility
        color = cmap(norm(probabilities[i]))
        ax.plot([receiver_position[0], end_point[0]],
                [receiver_position[1], end_point[1]],
                color=color, alpha=0.8, lw=2)

    # Add colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("Probability")
    plt.title("Azimuth Probability Estimation")
    # plt.show()


def is_misclassified_due_to_symmetry(probabilities, true_angle, tolerance=20):
    """
    Determine if a misclassified sample is due to symmetry issues.

    Args:
        probabilities (np.ndarray): Array of probabilities for each angle.
        true_angle (float): The true angle of the sample.
        threshold (float): The probability threshold to consider a high probability.
        tolerance (int): The degree of freedom for considering symmetry.

    Returns:
        bool: True if misclassified due to symmetry, False otherwise.
    """
    num_angles = len(probabilities)
    opposite_angle = (180-true_angle) % 180

    lower_bound = int(max(0, opposite_angle - tolerance))
    upper_bound = int(min(180, opposite_angle + tolerance))


    predicted_angle = probabilities.argmax()

    if lower_bound < upper_bound:
        return lower_bound <= predicted_angle < upper_bound
    else:
        return predicted_angle >= lower_bound or predicted_angle < upper_bound


def plot_classifications_errors(train_dataloader: DataLoader, valid_dataloader: DataLoader, train_config: TrainParams, model:FilteredNoiseShaper):
    trainer = Trainer(model, train_dataloader, valid_dataloader, train_config, save_name='fins_model', device=device)

    misclassified_counts = {}
    symmetry_issue_count = 0
    total_misclassified = 0
    total_exact = 0
    total_plus_one = 0

    # Create a folder to save the plots
    plots_folder = 'misclassified_plots'
    os.makedirs(plots_folder, exist_ok=True)

    for reverberated_speech, rir, source_location, receiver_position, room_dimensions, source  in tqdm(valid_dataloader, desc="Processing validation data"):
        (
            reverberated_source_with_noise,
            reverberated_source,
            batch_stochastic_noise,
            batch_noise_condition,
        ) = trainer.make_batch_data(reverberated_speech)

        # Model forward
        with torch.no_grad():
            _, location_predictions, _ = model(
                reverberated_source_with_noise, batch_stochastic_noise, batch_noise_condition
            )
            probabilities = torch.softmax(location_predictions, dim=1).cpu().numpy()

        # Calculate theta angles
        theta = calculate_theta(receiver_position, source_location).cpu().numpy()

        # Calculate radii
        radii = calculate_radius(receiver_position, source_location).cpu().numpy()

        for i_smple in range(len(theta)):
            # Check if the sample is misclassified
            predicted_location_index = probabilities[i_smple,:].argmax()
            if predicted_location_index == int(theta[i_smple].item()):
                total_exact += 1
            elif predicted_location_index == int(theta[i_smple].item())+1:
                total_plus_one += 1
            else:
                total_misclassified += 1
                # Plot the misclassified sample
                plot_room_and_positions_with_probabilities(receiver_position[i_smple], room_dimensions[i_smple], probabilities[i_smple], source_location[i_smple])
                plt.savefig(os.path.join(plots_folder, f'misclassified_sample_{total_misclassified}.png'))
                plt.close()

                radius = int(radii[i_smple])
                if str(radius) not in misclassified_counts:
                    misclassified_counts[str(radius)] = 0
                misclassified_counts[str(radius)] += 1

                # Check for symmetry issues
                if is_misclassified_due_to_symmetry(probabilities[i_smple], theta[i_smple]):
                    symmetry_issue_count += 1

    # Save the data to a CSV file
    csv_file = os.path.join(plots_folder, 'misclassification_data.csv')
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Radius', 'Misclassified Count'])
        for radius, count in misclassified_counts.items():
            writer.writerow([radius, count])
        writer.writerow([])
        writer.writerow(['Symmetry Issue Count', symmetry_issue_count])
        writer.writerow(['Total Misclassified', total_misclassified])
        writer.writerow(['Total Exact', total_exact])
        writer.writerow(['Total Plus One', total_plus_one])

def plot_confusion_matrix(valid_dataloader: DataLoader, model: FinsLightningModule):
    # Set matplotlib style for IEEE format academic paper
    plt.style.use('default')
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 10  # Base font size for IEEE format
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['xtick.major.width'] = 0.8
    plt.rcParams['ytick.major.width'] = 0.8
    plt.rcParams['xtick.major.size'] = 3
    plt.rcParams['ytick.major.size'] = 3
    all_preds = []
    all_labels = []
    all_rads = []

    # Set model to evaluation mode
    model.eval()
    # Iterate through the validation dataset
    for rir, source_location, receiver_position, room_dimensions, source, reverberation_time, sample_rate in tqdm(valid_dataloader, desc="Processing validation data"):
        # Move data to device
        rir = rir.to(model.device)
        source_location = source_location.to(model.device)
        receiver_position = receiver_position.to(model.device)
        room_dimensions = room_dimensions.to(model.device)
        source = source.to(model.device)
        reverberation_time = reverberation_time.to(model.device)
        sample_rate = sample_rate.to(model.device)

        # Convolve source with RIR to get reverberated speech
        source_squeezed = source.squeeze()
        if source_squeezed.dim() == 1:
            source_squeezed = source_squeezed.unsqueeze(0)
        reverberated_speech = torchaudio.functional.convolve(source_squeezed.float(), rir, mode='same')

        # Prepare batch data using the model's make_batch_data method
        (
            reverberated_source_with_noise,
            reverberated_source,
            batch_stochastic_noise,
            batch_noise_condition,
        ) = model.make_batch_data(reverberated_speech)

        # Prepare metadata if needed
        metadata = None
        if model.model_config.use_metadata:
            metadata = torch.cat((room_dimensions, receiver_position, reverberation_time.unsqueeze(1)), dim=1)

        # Model forward
        with torch.no_grad():
            predicted_rir, angle_predictions, rad_prediction, z = model(
                reverberated_source_with_noise, batch_stochastic_noise, batch_noise_condition, metadata=metadata
            )
            probabilities = torch.softmax(angle_predictions, dim=1).cpu().numpy()

        # Calculate theta angles
        theta = calculate_theta(receiver_position, source_location).cpu().numpy().astype(int)
        rad = calculate_radius(receiver_position, source_location).cpu().numpy()

        # Store predictions and true labels
        all_preds.extend(probabilities.argmax(axis=1))
        all_labels.extend(theta)
        all_rads.extend(rad)

    # Convert lists to numpy arrays for easier slicing
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_rads = np.array(all_rads)

    # Determine the range of radii and divide into four equal parts
    min_rad, max_rad = all_rads.min(), all_rads.max()
    rad_ranges = np.linspace(min_rad, max_rad, 5)

    for i in range(4):
        # Get the indices for the current range
        indices = (all_rads >= rad_ranges[i]) & (all_rads < rad_ranges[i + 1])
        preds_subset = all_preds[indices]
        labels_subset = all_labels[indices]

        # Compute confusion matrix
        cm = confusion_matrix(labels_subset, preds_subset)
        cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

        # Calculate accuracy, precision, and recall
        accuracy = accuracy_score(labels_subset, preds_subset)
        precision = precision_score(labels_subset, preds_subset, average='weighted')
        recall = recall_score(labels_subset, preds_subset, average='weighted')

        # Print accuracy, precision, and recall
        print(f'Range {rad_ranges[i]:.2f} - {rad_ranges[i + 1]:.2f}')
        print(f'Accuracy: {accuracy:.2f}')
        print(f'Precision: {precision:.2f}')
        print(f'Recall: {recall:.2f}')
        print(f'Max cell value: {cm_percentage.max():.1f}%')
        print(f'Min cell value: {cm_percentage.min():.1f}%')

        # Plot confusion matrix with IEEE format specifications
        # Width: 3.45 inches (8.8 cm), height scaled proportionally
        fig_width = 3.45  # inches
        fig_height = 3.0  # inches (scaled proportionally)
        
        plt.figure(figsize=(fig_width, fig_height), dpi=300)
        
        # Use viridis colormap for better grayscale interpretation
        # Dynamic scale that adapts to actual data range
        sns.heatmap(cm_percentage, 
                   annot=False, 
                   fmt='.1f', 
                   cmap='viridis', 
                   cbar_kws={'shrink': 0.8, 'aspect': 20, 'label': 'Percentage (%)'},
                   xticklabels=True,
                   yticklabels=True)
        
        plt.xlabel('Predicted Angle (deg)', fontsize=9, fontweight='normal')
        plt.ylabel('True Angle (deg)', fontsize=9, fontweight='normal')
        
        # # Adjust tick parameters for better readability with dense matrices
        # # Calculate appropriate tick spacing based on matrix size
        # matrix_size = cm_percentage.shape[0]
        # if matrix_size > 50:
        #     # For very large matrices, show only key angle markers (0°, 90°, 180°)
        #     key_angles = [0, 90, 180]
        #     # Convert angles to matrix indices (assuming 0-180 degree range)
        #     tick_positions = [int(angle * matrix_size / 180) for angle in key_angles]
        #     # Ensure we don't exceed matrix bounds
        #     tick_positions = [pos if pos < matrix_size else matrix_size - 1 for pos in tick_positions]
        #     tick_labels = ['0°', '90°', '180°']
        #
        #     plt.xticks(tick_positions, tick_labels, fontsize=8, rotation=0)
        #     plt.yticks(tick_positions, tick_labels, fontsize=8, rotation=0)
        # if matrix_size > 20:
        #     # For large matrices, show every nth tick
        #     tick_spacing = max(1, matrix_size // 10)
        #     tick_positions = np.arange(0, matrix_size, tick_spacing)
        #     if tick_positions[-1] != matrix_size - 1:
        #         tick_positions = np.append(tick_positions, matrix_size - 1)
        #
        #     plt.xticks(tick_positions, fontsize=7, rotation=0)
        #     plt.yticks(tick_positions, fontsize=7, rotation=0)
        # else:
        #     # For smaller matrices, show all ticks
        #     plt.xticks(fontsize=8, rotation=0)
        #     plt.yticks(fontsize=8, rotation=0)
        #
        # # Tight layout to minimize white space
        # plt.tight_layout()
        #
        # # Save with high DPI and tight bounding box
        # plt.savefig(f'confusion_matrix_{i}.png',
        #            dpi=300,
        #            bbox_inches='tight',
        #            facecolor='white',
        #            edgecolor='none')
        # plt.savefig(f'confusion_matrix_{i}.pdf',
        #            dpi=300,
        #            bbox_inches='tight',
        #            facecolor='white',
        #            edgecolor='none')
        # plt.show()

    # Plot confusion matrix for all radius ranges with IEEE format
    cm_all = confusion_matrix(all_labels, all_preds)
    cm_all_percentage = cm_all.astype('float') / cm_all.sum(axis=1)[:, np.newaxis] * 100

    # IEEE format specifications
    fig_width = 3.45  # inches
    fig_height = 3.0  # inches (scaled proportionally)
    
    plt.figure(figsize=(fig_width, fig_height), dpi=300)
    
    # Use viridis colormap for better grayscale interpretation
    # Dynamic scale that adapts to actual data range
    sns.heatmap(cm_all_percentage, 
               annot=False, 
               fmt='.1f', 
               cmap='viridis', 
               cbar_kws={'shrink': 0.8, 'aspect': 20, 'label': 'Percentage (%)'},
               xticklabels=True,
               yticklabels=True)
    
    plt.xlabel('Predicted Angle (deg)', fontsize=9, fontweight='normal')
    plt.ylabel('True Angle (deg)', fontsize=9, fontweight='normal')
    
    # Adjust tick parameters for better readability with dense matrices
    # Calculate appropriate tick spacing based on matrix size
    matrix_size = cm_all_percentage.shape[0]
    if matrix_size > 50:
        # For very large matrices, show only key angle markers (0°, 90°, 180°)
        key_angles = [0, 90, 180]
        # Convert angles to matrix indices (assuming 0-180 degree range)
        tick_positions = [int(angle * matrix_size / 180) for angle in key_angles]
        # Ensure we don't exceed matrix bounds
        tick_positions = [pos if pos < matrix_size else matrix_size - 1 for pos in tick_positions]
        tick_labels = ['0°', '90°', '180°']
        
        plt.xticks(tick_positions, tick_labels, fontsize=8, rotation=0)
        plt.yticks(tick_positions, tick_labels, fontsize=8, rotation=0)
    elif matrix_size > 20:
        # For large matrices, show every nth tick
        tick_spacing = max(1, matrix_size // 10)
        tick_positions = np.arange(0, matrix_size, tick_spacing)
        if tick_positions[-1] != matrix_size - 1:
            tick_positions = np.append(tick_positions, matrix_size - 1)
        
        plt.xticks(tick_positions, fontsize=7, rotation=0)
        plt.yticks(tick_positions, fontsize=7, rotation=0)
    else:
        # For smaller matrices, show all ticks
        plt.xticks(fontsize=8, rotation=0)
        plt.yticks(fontsize=8, rotation=0)
    
    # Tight layout to minimize white space
    plt.tight_layout()
    
    # Save with high DPI and tight bounding box
    plt.savefig('confusion_matrix_all.png', 
               dpi=300, 
               bbox_inches='tight', 
               facecolor='white',
               edgecolor='none')
    plt.savefig('confusion_matrix_all.pdf', 
               dpi=300, 
               bbox_inches='tight', 
               facecolor='white',
               edgecolor='none')
    plt.show()

def plot_latent_space(valid_dataloader: DataLoader, model: FinsLightningModule):
    all_predicted_rir = []
    all_source_location = []
    all_receiver_position = []

    # Set model to evaluation mode
    model.eval()
    
    # Iterate through the validation dataset
    for rir, source_location, receiver_position, room_dimensions, source, reverberation_time, sample_rate in tqdm(valid_dataloader, desc="Processing validation data for latent space"):
        # Move data to device
        rir = rir.to(model.device)
        source_location = source_location.to(model.device)
        receiver_position = receiver_position.to(model.device)
        room_dimensions = room_dimensions.to(model.device)
        source = source.to(model.device)
        reverberation_time = reverberation_time.to(model.device)
        sample_rate = sample_rate.to(model.device)

        # Convolve source with RIR to get reverberated speech
        source_squeezed = source.squeeze()
        if source_squeezed.dim() == 1:
            source_squeezed = source_squeezed.unsqueeze(0)
        reverberated_speech = torchaudio.functional.convolve(source_squeezed.float(), rir, mode='same')

        with torch.no_grad():
            # Prepare batch data using the model's make_batch_data method
            (
                reverberated_source_with_noise,
                reverberated_source,
                batch_stochastic_noise,
                batch_noise_condition,
            ) = model.make_batch_data(reverberated_speech)

            # Get latent representation using the model's get_latent_representation method
            latent_space = model.model.get_latent_representation(reverberated_source_with_noise)
            all_predicted_rir.append(latent_space.view(latent_space.size(0), -1).cpu().numpy())
            all_source_location.append(source_location)
            all_receiver_position.append(receiver_position)

    # Concatenate all batches
    all_predicted_rir = np.concatenate(all_predicted_rir, axis=0)
    all_source_location = torch.concatenate(all_source_location, dim=0)
    all_receiver_position = torch.concatenate(all_receiver_position, dim=0)

    # Calculate theta angles
    thetas = calculate_theta(all_receiver_position, all_source_location).cpu().numpy()

    # Calculate radii
    radii = calculate_radius(all_receiver_position, all_source_location).cpu().numpy()

    # Apply UMAP
    reducer = umap.UMAP(n_components=2)
    embedding = reducer.fit_transform(all_predicted_rir)

    # Create colormaps
    norm_theta = plt.Normalize(vmin=thetas.min(), vmax=thetas.max())
    norm_radius = plt.Normalize(vmin=radii.min(), vmax=radii.max())
    cmap = plt.cm.viridis



    # Set IEEE format specifications
    plt.style.use('default')
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 8
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['xtick.major.width'] = 0.8
    plt.rcParams['ytick.major.width'] = 0.8
    plt.rcParams['xtick.major.size'] = 3
    plt.rcParams['ytick.major.size'] = 3

    # IEEE format specifications
    fig_width = 3.5  # inches (IEEE single-column width)
    fig_height = 2.5  # inches (proportional height)
    
    # Plot the embedding with colors based on theta angles
    plt.figure(figsize=(fig_width, fig_height), dpi=300)
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=thetas, cmap='viridis', norm=norm_theta, s=15, alpha=0.7)
    cbar = plt.colorbar(scatter, shrink=0.8, aspect=20)
    cbar.set_label('Theta (degrees)', fontsize=8, fontfamily='Times New Roman')
    cbar.ax.tick_params(labelsize=7)
    plt.title('UMAP Projection by Theta', fontsize=9, fontfamily='Times New Roman', fontweight='normal', pad=8)
    plt.xlabel('Latent Dimension 1', fontsize=8, fontfamily='Times New Roman')
    plt.ylabel('Latent Dimension 2', fontsize=8, fontfamily='Times New Roman')
    plt.xticks(fontsize=7, fontfamily='Times New Roman')
    plt.yticks(fontsize=7, fontfamily='Times New Roman')
    plt.tight_layout()
    plt.savefig('latent_space_theta.pdf', dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.savefig('latent_space_theta.png', dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.show()

    # Plot the embedding with colors based on radii
    plt.figure(figsize=(fig_width, fig_height), dpi=300)
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=radii, cmap='viridis', norm=norm_radius, s=15, alpha=0.7)
    cbar = plt.colorbar(scatter, shrink=0.8, aspect=20)
    cbar.set_label('Radius (m)', fontsize=8, fontfamily='Times New Roman')
    cbar.ax.tick_params(labelsize=7)
    plt.title('UMAP Projection by Radius', fontsize=9, fontfamily='Times New Roman', fontweight='normal', pad=8)
    plt.xlabel('Latent Dimension 1', fontsize=8, fontfamily='Times New Roman')
    plt.ylabel('Latent Dimension 2', fontsize=8, fontfamily='Times New Roman')
    plt.xticks(fontsize=7, fontfamily='Times New Roman')
    plt.yticks(fontsize=7, fontfamily='Times New Roman')
    plt.tight_layout()
    plt.savefig('latent_space_radius.pdf', dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.savefig('latent_space_radius.png', dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.show()


def plot_predicted_vs_measured_air(valid_dataloader: DataLoader, train_dataloader: DataLoader, train_config: TrainParams,
                                    model: FilteredNoiseShaper):

    trainer = Trainer(model, train_dataloader, valid_dataloader, train_config, save_name='fins_model', device=device)


    reverberated_speech, rir, source_location, receiver_position, room_dimensions, source = next(
        iter(valid_dataloader))
    with torch.no_grad():
        (
            reverberated_source_with_noise,
            reverberated_source,
            batch_stochastic_noise,
            batch_noise_condition,
        ) = trainer.make_batch_data(reverberated_speech)

        # Model forward
        predicted_rir, _, _ = model(
            reverberated_source_with_noise, batch_stochastic_noise, batch_noise_condition
        )
        # Plotting
    for i in range(3):
        plt.figure(figsize=(10, 6))
        plt.plot(rir[i,:].flatten().cpu(), label='Measured RIR')
        plt.plot(predicted_rir[i,:,:].flatten().cpu(), label='Predicted RIR')
        plt.title(f'Measured vs Predicted RIR - Sample {i + 1}')
        plt.xlabel('Sample Index')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.show()


def plot_predicted_vs_predicted(valid_dataloader: DataLoader, train_dataloader: DataLoader,
                                   train_config: TrainParams,
                                   model: FilteredNoiseShaper):
    trainer = Trainer(model, train_dataloader, valid_dataloader, train_config, save_name='fins_model', device=device)

    reverberated_speech, rir, source_location, receiver_position, room_dimensions, source = next(
        iter(valid_dataloader))
    with torch.no_grad():
        (
            reverberated_source_with_noise,
            reverberated_source,
            batch_stochastic_noise,
            batch_noise_condition,
        ) = trainer.make_batch_data(reverberated_speech)

        # Model forward
        predicted_rir, _, _ = model(
            reverberated_source_with_noise, batch_stochastic_noise, batch_noise_condition
        )


    # Plotting the first 10 predicted RIRs overlaid
    plt.figure(figsize=(10, 6))
    for i in range(3):
        plt.plot(predicted_rir[i, :, :].flatten().cpu(), label=f'Predicted RIR {i+1}')
    plt.title('Overlay of First 10 Predicted RIRs')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()


def plot_model_architecture(valid_dataloader: DataLoader, train_dataloader: DataLoader,
                                   train_config: TrainParams,
                                   model: FilteredNoiseShaper):
    trainer = Trainer(model, train_dataloader, valid_dataloader, train_config, save_name='fins_model', device=device)

    reverberated_speech, rir, source_location, receiver_position, room_dimensions, source = next(
        iter(valid_dataloader))

    (
        reverberated_source_with_noise,
        reverberated_source,
        batch_stochastic_noise,
        batch_noise_condition,
    ) = trainer.make_batch_data(reverberated_speech)

    # Model forward
    predicted_rir, location_prediction = model(
        reverberated_source_with_noise, batch_stochastic_noise, batch_noise_condition
    )
    make_dot(location_prediction, params=dict(model.named_parameters())).render("model_architecture", directory=".", format="png")


def calculate_angle(position, receiver_position):
    delta_x = position[0] - receiver_position[0]
    delta_y = position[1] - receiver_position[1]
    return np.arctan2(delta_y, delta_x)


def plot_h_difference_matrix(distribution: str='uniform', path_dataset: str=None):
    if path_dataset:
        with open(Path(path_dataset) / 'h_dict.pkl', 'rb') as file:
            h_dict = pickle.load(file)
        h_list = [i for i in h_dict.values()]

        with open(Path(path_dataset) / 'config.pkl', 'rb') as file:
            config = pickle.load(file)
        receiver_position = config.receiver_position
    else:
        receiver_position = [3.5, 1, 1]
        room_dimensions = [6, 5, 3]
        wall_backoff = 0.5
        samples_in_theta = 1000
        config = RirDatasetGeneratorConfig()
        config.reverberation_time_range = (0.2,1.0)
        theta_range = (-90, 90)
        preset_radii = [1.0, 2.5, 4.0]
        positions_x, positions_y = generate_positions(receiver_position, room_dimensions, wall_backoff, theta_range,
                                                      'uniform', preset_radii, samples_in_theta)

        h_list = generate_rir_h(positions_x, positions_y, config)


    # Calculate angles
    angles = [calculate_angle((h.x, h.y), receiver_position) for h in h_list]

    # Sort h_list_zipped by angles
    sorted_h_list = [h for _, h in sorted(zip(angles, h_list))]
    sorted_angles = sorted(angles)

    # Convert angles to degrees
    sorted_angles_degrees = [np.degrees(angle) for angle in sorted_angles]

    num_h = len(sorted_h_list)
    matrix = np.zeros((num_h, num_h))

    for i in range(num_h):
        for j in range(num_h):
            h_i = sorted_h_list[i].h
            h_j = sorted_h_list[j].h
            matrix[i, j] = np.linalg.norm(h_i - h_j)

    plt.figure(figsize=(10, 8))
    plt.imshow(matrix, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Difference')
    plt.title(f'Difference Matrix of h (Sorted by Angle) {distribution}')
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Angle (degrees)')

    # Set tick marks
    tick_positions = np.arange(0, num_h, 100)
    if tick_positions[-1] != num_h - 1:
        tick_positions = np.append(tick_positions, num_h - 1)
    tick_labels = [f'{sorted_angles_degrees[i]:.1f}' for i in tick_positions]
    plt.xticks(tick_positions, tick_labels, rotation=90)
    plt.yticks(tick_positions, tick_labels)

    plt.show()



def main():
    import argparse

    parser = argparse.ArgumentParser(description="Plot confusion matrix and latent space for a FINS checkpoint.")
    parser.add_argument("checkpoint_path", help="Path to a Lightning .ckpt file")
    parser.add_argument("--dataset_path", default=None, help="RIR dataset root (default: DatasetParams.rir_dataset_path)")
    parser.add_argument("--split", default="test", help="Dataset split to load (default: test)")
    args = parser.parse_args()

    dataset_params = DatasetParams()
    if args.dataset_path:
        dataset_params = DatasetParams(rir_dataset_path=args.dataset_path)
    train_config = TrainParams()
    train_config.batch_size = 100
    valid_dataloader, valid_dataset_config = load_dataset_and_get_dataloader(args.split, dataset_params,
                                                                             num_workers=2,
                                                                             shuffle=False,
                                                                             **train_config.__dict__)

    model_config = FinsModelConfigParams()
    model = load_fins_lightning_module_from_checkpoint(args.checkpoint_path, model_config, train_config)

    plot_confusion_matrix(valid_dataloader, model)
    plot_latent_space(valid_dataloader, model)

if __name__ == '__main__':
    main()
