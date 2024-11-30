import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import datetime
import pytz
import os
import gc
from CrossbarModels.Models import memtorch_bindings # type: ignore
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import math

# Custom Layer implementation
class CustomLayer(nn.Module):
    def __init__(self, weight, bias, parasiticResistance, R_lrs, model_function):
        super(CustomLayer, self).__init__()
        self.weight = torch.tensor(weight).to('cpu')  # Keep on CPU to save GPU memory
        self.bias = torch.tensor(bias).to('cpu')  # Keep on CPU to save GPU memory
        self.mapping_coefficient = 1 / R_lrs
        self.parasiticResistance = parasiticResistance
        self.model_function = model_function

    def forward(self, x):
        Currents = self.model_function(
            self.weight.T * self.mapping_coefficient,
            x.cpu(),  # Move input to CPU
            self.parasiticResistance
        )
        output = Currents / self.mapping_coefficient
        return output.to(x.device) + self.bias.to(x.device)  # Move output to original device

# Placeholder model functions
def solve_passive_model(weight, x, parasiticResistance):
    return memtorch_bindings.solve_passive(
        weight,
        x,
        torch.zeros(weight.shape[0]),
        parasiticResistance,
        parasiticResistance,
        n_input_batches=x.shape[0]
    )

def IdealModel(weight, x, parasiticResistance):
    # Placeholder for another model's implementation
    if len(x.shape) == 1:
        x = x.unsqueeze(0)  # Add batch dimension if input is a single sample
    return torch.matmul(x, weight)  # Supports batched input

# Standard Neural Network
class NetStandard(nn.Module):
    def __init__(self):
        super(NetStandard, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Custom Neural Network
class NetCustom(nn.Module):
    def __init__(self, weights, parasiticResistance, R_lrs, model_function):
        super(NetCustom, self).__init__()
        self.fc1 = CustomLayer(weights['fc1_weights'], weights['fc1_bias'], parasiticResistance, R_lrs, model_function)
        self.fc2 = CustomLayer(weights['fc2_weights'], weights['fc2_bias'], parasiticResistance, R_lrs, model_function)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Evaluate Model
def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            all_preds.extend(pred.cpu().numpy().flatten())
            all_targets.extend(target.cpu().numpy().flatten())
            correct += pred.eq(target.view_as(pred)).sum().item()
    accuracy = 100. * correct / len(test_loader.dataset)
    return accuracy, all_preds, all_targets

# Plot Confusion Matrix
def plot_confusion_matrix_subplot(ax, y_true, y_pred, title):
    classes = sorted(set(y_true).union(set(y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap=plt.cm.Blues, values_format='d', ax=ax)
    ax.set_title(title)

# Main Function
if __name__ == '__main__':
    # Load weights
    weights = torch.load('fc_layers_weights_positive_v2.pth', weights_only=False)  # Use weights_only=False for compatibility, ensure you trust the source

    # Data transformation
    transform = transforms.Compose([transforms.ToTensor()])

    # Load test dataset
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    subset_indices = torch.arange(30)
    small_test_dataset = Subset(test_dataset, subset_indices)
    small_test_loader = DataLoader(small_test_dataset, batch_size=1, shuffle=False)

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Parameters
    R_lrs = 1e3  # Low resistance state in Ohms

    # Save results
    end = datetime.datetime.now(pytz.timezone('Europe/Rome'))
    folder = f'Results/{end.year}{end.month}{end.day}_{end.hour}_{end.minute}_NN_benchmark'
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Standard network evaluation
    model_standard = NetStandard().to(device)
    model_standard.load_state_dict(torch.load('model_standard_positive_v2.pth', weights_only=True))
    standard_accuracy, standard_preds, standard_targets = evaluate_model(model_standard, small_test_loader, device)
    print(f"Accuracy of standard network: {standard_accuracy:.2f}%")
 
    # Custom networks with varying parasiticResistance and models
    parasitic_resistances = torch.arange(10, 10.01, 0.05).tolist()  # Sweep from 0.01 to 0.3 with step of 0.05
    model_functions = [solve_passive_model]
    custom_accuracies = {model_function.__name__: [] for model_function in model_functions}
    confusion_data = []

    for model_function in model_functions:
        for parasiticResistance in parasitic_resistances:
            model_custom = NetCustom(weights, parasiticResistance, R_lrs, model_function).to(device)
            custom_accuracy, custom_preds, custom_targets = evaluate_model(model_custom, small_test_loader, device)
            custom_accuracies[model_function.__name__].append(custom_accuracy)
            confusion_data.append((custom_preds, custom_targets, f"Custom Network (parasiticResistance={parasiticResistance}, model={model_function.__name__})"))
            print(f"Accuracy of custom network (parasiticResistance={parasiticResistance}, model={model_function.__name__}): {custom_accuracy:.2f}%")
            # Clear memory after each iteration to prevent memory buildup
            del model_custom
            torch.cuda.empty_cache()
            gc.collect()

    # Normalize accuracies by standard model accuracy and plot multiple lines
    plt.figure(figsize=(10, 6))
    for model_name, accuracies in custom_accuracies.items():
        normalized_accuracies = [acc / standard_accuracy for acc in accuracies]
        plt.plot(parasitic_resistances, normalized_accuracies, marker='o', linestyle='-', label=f'Normalized Accuracy ({model_name})')
    
    plt.xlabel('Parasitic Resistance')
    plt.ylabel('Normalized Accuracy')
    plt.title('Normalized Accuracy vs Parasitic Resistance')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(folder + "/Normalized_Accuracy_vs_Parasitic_Resistance.png")
    plt.show()

    # Plot all confusion matrices as subplots
    num_confusions = len(confusion_data) + 1
    num_rows = math.ceil(num_confusions / 4)
    fig, axes = plt.subplots(num_rows, 4, figsize=(20, 5 * num_rows))
    axes = axes.ravel()

    # Plot confusion matrix for standard model
    plot_confusion_matrix_subplot(axes[0], standard_targets, standard_preds, "Confusion Matrix: Standard Network")

    # Plot confusion matrices for custom models
    for i, (custom_preds, custom_targets, title) in enumerate(confusion_data, start=1):
        if i < len(axes):
            plot_confusion_matrix_subplot(axes[i], custom_targets, custom_preds, title)

    # Hide any unused subplots
    for j in range(len(confusion_data) + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(folder + "/All_Confusion_Matrices.png")
    plt.show()
