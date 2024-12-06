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
from CrossbarModels.Crossbar_Models_pytorch import jeong_model, dmr_model, gamma_model, solve_passive_model, crosssim_model, IdealModel
from Crossbar_net import CustomNet, evaluate_model


# Main Function
if __name__ == '__main__':

    # Test dataset parameters
    batch_size = 32
    subset_indices = torch.arange(60)

    # Crossbar parameters
    R_lrs = 1e3 
    R_hrs = 5e6
    parasitic_resistances = torch.arange(0.0001, 2.1, 0.1).tolist()
    bits=0

    # Enabled models for the accuracy test
    model_functions = [crosssim_model, jeong_model, dmr_model, gamma_model, IdealModel]

    # Plotting parameters
    debug_plot = False  # Set to True to enable debugging plots
    debug_index = 0  # Set the index of the currents to plot for debugging
    bias_correction=False
    selected_model = model_functions[0]  # Change this to select a different model for confusion matrix plotting

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # List available trained models
    trained_models_folder = "Models/Trained"
    available_models = [d for d in os.listdir(trained_models_folder) if os.path.isdir(os.path.join(trained_models_folder, d))]
    print("Available trained models:")
    for i, model_name in enumerate(available_models):
        print(f"{i}: {model_name}")
    model_choice = int(input("Select the model number for evaluation: "))
    selected_model_name = available_models[model_choice]
    # List subfolders for different parasitic resistance values
    folder_path = f"{trained_models_folder}/{selected_model_name}"
    available_rpar_folders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    print("Available Rpar values:")
    for i, rpar_folder in enumerate(available_rpar_folders):
        print(f"{i}: {rpar_folder}")
    rpar_choice = int(input("Select the Rpar value for evaluation: "))
    selected_rpar_folder = available_rpar_folders[rpar_choice]
    # Load weights and biases from the selected model's folder
    folder_path = f"{trained_models_folder}/{selected_model_name}/{selected_rpar_folder}"
    weights_path = f"{folder_path}/fc_layers_weights.pth"
    # Check if the model exists
    if not os.path.exists(weights_path):
        print("The specified weights file does not exist. Please ensure the model is trained and saved correctly.")
        exit()
    else:
        weights = torch.load(weights_path, map_location=device)

    # Data transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x / 2)  # Scale to range [0, 0.5]
    ])

    # Load test dataset
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    small_test_dataset = Subset(test_dataset, subset_indices)
    small_test_loader = DataLoader(small_test_dataset, batch_size=batch_size, shuffle=False)

    # results
    end = datetime.datetime.now(pytz.timezone('Europe/Rome'))
    folder = f'Results/{end.year}{end.month}{end.day}_{end.hour}_{end.minute}_NN_benchmark'
    if not os.path.exists(folder):
        os.makedirs(folder)
    # Colors for models
    colors = {
        'crosssim_model': 'blue',
        'jeong_model': 'cyan',
        'gamma_model': 'red',
        'dmr_model': 'green',
        'solve_passive_model': 'orange',
        'IdealModel': 'black'
    }

    custom_accuracies = {model_function.__name__: [] for model_function in model_functions}
    confusion_data = []

    for parasiticResistance in parasitic_resistances:
        if debug_plot:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        for model_function in model_functions:
            model_custom = CustomNet(weights, parasiticResistance, R_hrs, R_lrs, model_function, device=device, debug=debug_plot, bits=bits, bias_correction=bias_correction)
            custom_accuracy, custom_preds, custom_targets = evaluate_model(model_custom, small_test_loader, device)
            custom_accuracies[model_function.__name__].append(custom_accuracy)
            if model_function == selected_model:
                confusion_data.append((
                    custom_preds,
                    custom_targets,
                    f"Rpar={parasiticResistance:.2f}, model={model_function.__name__})"
                ))
            print(f"Accuracy of custom network (parasiticResistance={parasiticResistance:.2f}, model={model_function.__name__}): {custom_accuracy:.2f}%")
            # Debugging plot
            if debug_plot:
                color = colors[model_function.__name__]
                # Plotting Layer 1 Currents
                ax1.plot(
                    model_custom.fc1.currents[debug_index].numpy(),
                    color=color,
                    label=model_function.__name__
                )
                ax1.set_title(f'Layer 1 Currents (parasiticResistance={parasiticResistance:.2f})')
                ax1.set_xlabel('Index')
                ax1.set_ylabel('Current')
                ax1.legend()
                # Plotting Layer 2 Currents
                ax2.plot(
                    model_custom.fc2.currents[debug_index].numpy(),
                    color=color,
                    label=model_function.__name__
                )
                ax2.set_title(f'Layer 2 Currents (parasiticResistance={parasiticResistance:.2f})')
                ax2.set_xlabel('Index')
                ax2.set_ylabel('Current')
                ax2.legend()
            # Clear memory after each iteration to prevent memory buildup
            del model_custom
            torch.cuda.empty_cache()
            gc.collect()
        if debug_plot:
            plt.show()

    # Normalize accuracies by ideal model accuracy and plot multiple lines
    plt.figure(figsize=(10, 6))
    for model_name, accuracies in custom_accuracies.items():
        plt.plot(parasitic_resistances, accuracies, marker='o', linestyle='-', label=f'Accuracy ({model_name})', color=colors[model_name])
    plt.xlabel('Parasitic Resistance')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Parasitic Resistance')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(folder + "/Accuracy_vs_Parasitic_Resistance.png")
    plt.show()

    # Plot selected confusion matrices as subplots
    num_confusions = len(confusion_data) + 1
    num_rows = math.ceil(num_confusions / 4)
    fig, axes = plt.subplots(num_rows, 4, figsize=(20, 5 * num_rows))
    axes = axes.ravel()
    # Plot confusion matrices for selected custom models
    for i, (custom_preds, custom_targets, title) in enumerate(confusion_data, start=1):
        if i < len(axes):
            cm = confusion_matrix(custom_targets, custom_preds, labels=sorted(set(custom_targets) | set(custom_preds)))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=sorted(set(custom_targets) | set(custom_preds)))
            disp.plot(cmap=plt.cm.Blues, values_format='d', ax=axes[i])
            axes[i].set_title(title)
    # Hide any unused subplots
    for j in range(len(confusion_data) + 1, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.savefig(folder + "/Selected_Confusion_Matrices.png")
    plt.show()