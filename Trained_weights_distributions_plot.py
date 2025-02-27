import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from matplotlib.colors import LinearSegmentedColormap

def weight_mapping(weights, r_hrs, r_lrs):
    """Map weights to conductances.
    """
    weight_min = torch.min(weights)
    weight_max = torch.max(weights)

    return ((torch.clamp(weights, weight_min, weight_max) - weight_min)
            * (1 / r_lrs - 1 / r_hrs) / (weight_max - weight_min)) + (1 / r_hrs)

def plot_conductance_distributions():
    # Load model weights
    trained_models_folder = "TrainedModels"
    available_models = [d for d in os.listdir(trained_models_folder) if os.path.isdir(os.path.join(trained_models_folder, d))]
    print("Available trained models:")
    for i, model_name in enumerate(available_models):
        print(f"{i}: {model_name}")
    model_choice = int(input("Select the model number for evaluation: "))
    selected_model_name = available_models[model_choice]

    folder_path = f"{trained_models_folder}/{selected_model_name}"
    available_rpar_folders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    
    if not available_rpar_folders:
        print(f"No subfolders found in {folder_path}")
        return
    
    # Create a list to store folder information with tile sizes
    folder_info = []
    
    # Extract tile sizes and folder names
    for rpar_folder in available_rpar_folders:
        size_match = re.search(r'Size(\d+)', rpar_folder)
        if size_match:
            tile_size = int(size_match.group(1))
            folder_info.append((rpar_folder, tile_size))
    
    # Sort folders by tile size (from smallest to largest)
    folder_info.sort(key=lambda x: x[1])
    
    # Recalculate number of folders after potential filtering
    n_folders = len(folder_info)
    if n_folders == 0:
        print("No valid folders with tile size information found.")
        return
    
    # Calculate grid size for subplots
    n_cols = 2  # 2 columns
    n_rows = (n_folders + 1) // 2  # Ceiling division to get required rows
    
    # Create a single figure with subplots for each configuration
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    
    # Flatten axes array for easier indexing
    if n_rows > 1 or n_cols > 1:
        axes = axes.flatten()
    else:
        axes = [axes]  # Make a list if there's only one subplot
    
    # Custom colormap - purple
    color = "#6a0dad"
    
    # Process each subfolder in order of tile size
    for i, (rpar_folder, tile_size) in enumerate(folder_info):
        print(f"Processing {i+1}/{n_folders}: {rpar_folder} (Tile Size: {tile_size})")
        
        # Extract R_hrs, R_lrs from folder name
        lrs_match = re.search(r'LRS(\d+)', rpar_folder)
        hrs_match = re.search(r'HRS(\d+)', rpar_folder)
        
        if not (lrs_match and hrs_match):
            print(f"  Skipping {rpar_folder}: Could not extract resistance parameters from folder name")
            continue
        
        r_lrs = float(lrs_match.group(1))
        r_hrs = float(hrs_match.group(1))
        
        weights_path = f"{trained_models_folder}/{selected_model_name}/{rpar_folder}/fc_layers_weights.pth"
        if not os.path.exists(weights_path):
            print(f"  Skipping {rpar_folder}: Weights file not found")
            continue
        
        # Load weights
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            weights_dict = torch.load(weights_path, map_location=device)
        except Exception as e:
            print(f"  Error loading {weights_path}: {e}")
            continue
        
        # Extract weights (excluding bias)
        all_weights = []
        for key, value in weights_dict.items():
            if isinstance(value, torch.Tensor):
                if 'weight' in key and 'bias' not in key:
                    all_weights.extend(value.cpu().detach().numpy().flatten())
        
        if len(all_weights) == 0:
            print(f"  Skipping {rpar_folder}: No weight tensors found")
            continue
        
        # Convert weights to conductances
        all_weights_tensor = torch.tensor(all_weights)
        all_conductances = weight_mapping(all_weights_tensor, r_hrs, r_lrs).numpy()
        
        # Plot conductance distribution
        ax = axes[i]
        sns.histplot(all_conductances, kde=True, color=color, ax=ax)
        
        # Set title with tile size
        ax.set_title(f'Tile Size: {tile_size}x{tile_size}', fontweight='bold')
        ax.set_xlabel('Conductance (S)')
        ax.set_ylabel('Frequency')
        
        # Add statistics
        g_min = 1/r_hrs
        g_max = 1/r_lrs
        stats_text = (f"Mean: {np.mean(all_conductances):.4e} S\n"
                     f"Std Dev: {np.std(all_conductances):.4e} S\n"
                     f"G_min (1/HRS): {g_min:.4e} S\n"
                     f"G_max (1/LRS): {g_max:.4e} S")
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, 
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Hide any unused subplots
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    
    # Add main title
    plt.suptitle(f'Conductance Distributions for {selected_model_name}\n(Ordered by Tile Size)', fontsize=16, y=0.98)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.94)  # Make room for suptitle
    
    # Save the plot
    save_path = f"{trained_models_folder}/{selected_model_name}/conductance_distributions_by_size.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    plot_conductance_distributions()