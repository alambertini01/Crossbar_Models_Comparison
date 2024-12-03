import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load and visualize model weights and biases as heatmaps
if __name__ == "__main__":
    # Model selection
    available_models = [
        "jeong_model", "dmr_model", "gamma_model", "solve_passive_model", "crosssim_model", "IdealModel"
    ]
    
    print("Available models:")
    for i, model_name in enumerate(available_models):
        print(f"{i}: {model_name}")
    
    model_choice = int(input("Select the model number to visualize: "))
    selected_model_name = available_models[model_choice]

    # Folder path
    folder_path = f'Models/Trained/{selected_model_name}'

    # Check if the model exists
    if not os.path.exists(folder_path):
        print("No models found in the specified folder. Please train the model first and try again.")
    else:
        # List all model files in the folder
        model_files = [f for f in os.listdir(folder_path) if f.endswith('.pth')]

        if not model_files:
            print("No model files found in the folder. Please train the model first and try again.")
        else:
            # Load the weights file
            weights_path = f'{folder_path}/fc_layers_weights_positive_v2.pth'
            if not os.path.exists(weights_path):
                print("The specified weights file does not exist. Please ensure the model is trained and saved correctly.")
            else:
                # Load weights and biases
                weights_data = torch.load(weights_path)
                fc1_weights = weights_data['fc1_weights']
                fc1_bias = weights_data['fc1_bias']
                fc2_weights = weights_data['fc2_weights']
                fc2_bias = weights_data['fc2_bias']

                # Plot heatmaps
                fig, axes = plt.subplots(2, 2, figsize=(10, 8))

                sns.heatmap(fc1_weights, ax=axes[0, 0], cmap='viridis')
                axes[0, 0].set_title('FC1 Weights')

                sns.heatmap(np.expand_dims(fc1_bias, axis=0), ax=axes[0, 1], cmap='viridis', cbar=False)
                axes[0, 1].set_title('FC1 Bias')

                sns.heatmap(fc2_weights, ax=axes[1, 0], cmap='viridis')
                axes[1, 0].set_title('FC2 Weights')

                sns.heatmap(np.expand_dims(fc2_bias, axis=0), ax=axes[1, 1], cmap='viridis', cbar=False)
                axes[1, 1].set_title('FC2 Bias')

                plt.tight_layout()
                plt.show()