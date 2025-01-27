import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import datetime
import pytz
import os
import gc
import math
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import csv
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from CrossbarModels.Crossbar_Models_pytorch import jeong_model, dmr_model, gamma_model, solve_passive_model, crosssim_model, IdealModel
from Crossbar_net import CustomNet, evaluate_model

if __name__ == '__main__':
    # *************** User Parameters ***************
    # The code assumes the user picks a model from a folder structure as before
    # You can hardcode choices if desired. For now, we keep user input for model selection.

    # Primary fixed parameters
    R_lrs = 1e3
    
    parasitic_resistances = torch.arange(2.00001, 2.1, 0.1).tolist()
    max_array_size = 64
    model_functions = [ crosssim_model]
    bias_correction = False
    debug_plot = False
    debug_index = 0
    plot_confusion = False

    batch_size=64
    test_samples = 1000

    # Parameters to potentially sweep
    #R_hrs_values = torch.linspace(10000, 200000, steps=20).tolist()
    R_hrs_values = 40000
    bits_values = 6

    # *************** Determine Sweeps ***************
    # Convert R_hrs_values and bits_values into lists if they aren't already
    if not isinstance(R_hrs_values, list):
        R_hrs_values = [R_hrs_values]
    if not isinstance(bits_values, list):
        bits_values = [bits_values]

    # Only one of R_hrs_values or bits_values can have more than one element
    if len(R_hrs_values) > 1 and len(bits_values) > 1:
        raise ValueError("Only one parameter among R_hrs and bits can be swept. Both cannot be lists at the same time.")

    # Identify which parameter (if any) is being swept
    # If R_hrs_values has multiple elements, we sweep R_hrs
    # If bits_values has multiple elements, we sweep bits
    # If both have single element, no second sweep
    if len(R_hrs_values) > 1:
        sweeping_param = 'R_hrs'
        param_values = R_hrs_values
        fixed_bits = bits_values[0]
    elif len(bits_values) > 1:
        sweeping_param = 'bits'
        param_values = bits_values
        fixed_R_hrs = R_hrs_values[0]
    else:
        sweeping_param = None
        param_values = [None]  # no sweep, just a dummy list

    # *************** Setup Device & Data ***************
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

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
    print("Available Rpar values:")
    for i, rpar_folder in enumerate(available_rpar_folders):
        print(f"{i}: {rpar_folder}")
    rpar_choice = int(input("Select the Rpar value for evaluation: "))
    selected_rpar_folder = available_rpar_folders[rpar_choice]

    weights_path = f"{trained_models_folder}/{selected_model_name}/{selected_rpar_folder}/fc_layers_weights.pth"
    if not os.path.exists(weights_path):
        print("The specified weights file does not exist.")
        exit()
    weights = torch.load(weights_path, map_location=device)

    # # Data loading
    # transform = transforms.Compose([
    #    transforms.ToTensor(),
    #    transforms.Lambda(lambda x: x / 2)  # Scale to [0,0.5]
    # ])

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    small_test_dataset = Subset(test_dataset, torch.arange(test_samples))
    small_test_loader = DataLoader(small_test_dataset, batch_size=batch_size, shuffle=False)

    # Results folder
    end = datetime.datetime.now(pytz.timezone('Europe/Rome'))
    results_folder = f'Results/{end.year}{end.month}{end.day}_{end.hour}_{end.minute}_NN_benchmark'
    os.makedirs(results_folder, exist_ok=True)


    # *************** Run Evaluations ***************

    # Colors for models
    colors = {
        'crosssim_model': 'blue',
        'jeong_model': 'cyan',
        'gamma_model': 'red',
        'dmr_model': 'green',
        'solve_passive_model': 'orange',
        'IdealModel': 'black'
    }
    
    # Initialize accuracies structure
    # Rows = parasiticResistance, Cols = param_values if sweeping_param
    n_pr = len(parasitic_resistances)
    n_pv = len(param_values)
    custom_accuracies = {m.__name__: np.zeros((n_pr, n_pv)) for m in model_functions}
    confusion_data = []

    # CSV logger
    csv_file = os.path.join(results_folder, "accuracy_data.csv")
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        if sweeping_param:
            writer.writerow(["parasiticResistance", sweeping_param, "paramValue", "model", "accuracy"])
        else:
            writer.writerow(["parasiticResistance", "model", "accuracy"])

        total_iterations = n_pr * n_pv * len(model_functions)
        current_iter = 0

        for j, pv in enumerate(param_values):
            # Assign current R_hrs and bits depending on sweep
            if sweeping_param == 'R_hrs':
                current_R_hrs = pv
                current_bits = fixed_bits
            elif sweeping_param == 'bits':
                current_bits = pv
                current_R_hrs = R_hrs_values[0]  # fixed since single
            else:
                # No sweeping param
                current_R_hrs = R_hrs_values[0]
                current_bits = bits_values[0]

            for i, pr in enumerate(parasitic_resistances):
                if debug_plot:
                     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
                for model_function in model_functions:
                    # Print progress
                    current_iter += 1

                    model_custom = CustomNet(
                        weights,
                        pr,
                        current_R_hrs,
                        R_lrs,
                        model_function,
                        device=device,
                        debug=debug_plot,
                        bits=current_bits,
                        correction=bias_correction,
                        max_array_size=max_array_size
                    )

                    acc, preds, targets = evaluate_model(model_custom, small_test_loader, device)
                    custom_accuracies[model_function.__name__][i, j] = acc

                    print(f"Processing {current_iter}/{total_iterations} | "
                        f"Rpar={pr:.2f}, "
                        f"{sweeping_param + '=' + str(pv) if sweeping_param else ''}, "
                        f"model={model_function.__name__}, "
                        f"acc={acc:.2f}")
                    
                    if sweeping_param:
                        writer.writerow([pr, sweeping_param, pv, model_function.__name__, acc])
                    else:
                        writer.writerow([pr, model_function.__name__, acc])

                    if plot_confusion and model_function == model_functions[0]:
                        confusion_data.append((preds, targets, f"Rpar={pr:.2f}{', '+sweeping_param+'='+str(pv) if sweeping_param else ''}, model={model_function.__name__}"))
                    
                    # Debugging plot
                    if debug_plot:
                        color = colors[model_function.__name__]
                        # Plotting Layer 1 Currents
                        ax1.plot(
                            model_custom.fc1.currents[debug_index].numpy(),
                            color=color,
                            label=model_function.__name__
                        )
                        # Plotting Layer 2 Currents
                        ax2.plot(
                            model_custom.fc2.currents[debug_index].numpy(),
                            color=color,
                            label=model_function.__name__
                        )
                        if bias_correction and model_function != IdealModel:
                            # Plotting Layer 1 Currents
                            ax1.plot(
                                model_custom.fc1.currents_corrected[debug_index].numpy(),
                                color=color,
                                linestyle='--',
                                label=model_function.__name__ + " corrected"
                            )
                            # Plotting Layer 2 Currents
                            ax2.plot(
                                model_custom.fc2.currents_corrected[debug_index].numpy(),
                                color=color,
                                linestyle='--',
                                label=model_function.__name__ + " corrected"
                            )
                    # Clear memory after each iteration to prevent memory buildup
                    del model_custom
                    torch.cuda.empty_cache()
                    gc.collect()
                if debug_plot:
                    ax1.set_title(f'Layer 1 Currents (parasiticResistance={pr:.2f})')
                    ax1.set_xlabel('Index')
                    ax1.set_ylabel('Current')
                    ax2.set_title(f'Layer 2 Currents (parasiticResistance={pr:.2f})')
                    ax2.set_xlabel('Index')
                    ax2.set_ylabel('Current')
                    ax1.legend()
                    ax2.legend()
                    plt.show()

    # *************** Plotting ***************

    for model_name, acc_matrix in custom_accuracies.items():
        if n_pv > 1:
            # Create a 3D surface plot for each model
            X, Y = np.meshgrid(param_values, parasitic_resistances)
            Z = acc_matrix
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            # Plot the surface
            surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)
            # Add a color bar which maps values to colors
            cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
            cbar.set_label('Accuracy (%)', fontsize=12)
            # Set labels
            ax.set_xlabel(sweeping_param, fontsize=12, labelpad=10)
            ax.set_ylabel('Parasitic Resistance', fontsize=12, labelpad=10)
            ax.set_zlabel('Accuracy (%)', fontsize=12, labelpad=10)
            # Set title
            ax.set_title(f'Accuracy Surface Plot - {model_name}', fontsize=14, pad=20)
            # Improve layout and appearance
            ax.view_init(elev=30, azim=225)  # Adjust the viewing angle for better visualization
            plt.tight_layout()
            # Save the figure
            plt.savefig(os.path.join(results_folder, f"Accuracy_Surface_{model_name}_sweep_{sweeping_param}.png"), dpi=300)
            # Show the plot
            plt.show()
        else:
            # When not sweeping an additional parameter, plot all models on the same figure
            # Initialize the plot only once
            if model_name == list(custom_accuracies.keys())[0]:
                plt.figure(figsize=(12, 8))
            # Plot each model's accuracy curve
            plt.plot(
                parasitic_resistances,
                acc_matrix[:, 0],  # Assuming acc_matrix has shape (parasitic_resistances, 1)
                marker='o',
                linestyle='-',
                color=colors.get(model_name, 'blue'),  # Default to 'blue' if model not in colors
                label=model_name
            )

    # handle the 2D plot configurations
    if n_pv <= 1:
        plt.xlabel('Parasitic Resistance', fontsize=14)
        plt.ylabel('Accuracy (%)', fontsize=14)
        plt.title('Accuracy vs Parasitic Resistance', fontsize=16)
        plt.grid(True)
        plt.legend(title='Models', fontsize=12, title_fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(results_folder, "Accuracy_vs_Parasitic_Resistance_All_Models.png"), dpi=300)
        plt.show()

    # Confusion matrices if required
    if plot_confusion and confusion_data:
        num_confusions = len(confusion_data) + 1
        num_rows = math.ceil(num_confusions / 4)
        fig, axes = plt.subplots(num_rows, 4, figsize=(20, 5 * num_rows))
        axes = axes.ravel()
        for i, (custom_preds, custom_targets, title) in enumerate(confusion_data, start=1):
            if i < len(axes):
                cm = confusion_matrix(custom_targets, custom_preds, labels=sorted(set(custom_targets) | set(custom_preds)))
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=sorted(set(custom_targets) | set(custom_preds)))
                disp.plot(cmap=plt.cm.Blues, values_format='d', ax=axes[i])
                axes[i].set_title(title)

        for j in range(len(confusion_data) + 1, len(axes)):
            fig.delaxes(axes[j])
        plt.tight_layout()
        plt.savefig(os.path.join(results_folder, "Selected_Confusion_Matrices.png"))
        plt.show()

    print("All results saved in:", results_folder)
    print("Accuracy data saved in:", csv_file)
