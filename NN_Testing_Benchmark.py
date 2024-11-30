import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import datetime
import pytz
from sklearn.linear_model import LinearRegression

from CrossbarModels.Crossbar_Models import *
from CrossbarModels.Models import memtorch_bindings # type: ignore


def tune_layer( original_layer, conductance_layer):
    """
    Determine a linear mapping between conductance-based outputs and original layer outputs.

    Returns
    -------
    function
        A function that transforms the output of the conductance-based layer to match the original layer output.
    """
    # Infer input shape from the layer's weight
    input_shape = (1, conductance_layer.weight.shape[1])
    device = conductance_layer.weight.device

    # Generate random input
    input_data = torch.rand(input_shape, device=device).uniform_(0, 1)

    # Compute outputs for both layers
    original_output = original_layer(input_data).detach().cpu().numpy().reshape(-1, 1)
    conductance_output = conductance_layer(input_data).detach().cpu().numpy().reshape(-1, 1)


    print("Conductance Output:", conductance_output)
    print("Original Output:", original_output)

    # Plot the outputs for visual comparison
    plt.figure(figsize=(8, 6))
    plt.scatter(conductance_output, original_output, alpha=0.7, label="Data Points")
    plt.plot(
        conductance_output,
        conductance_output,  # y = x line for reference
        color="red",
        linestyle="--",
        label="y = x (Reference Line)",
    )
    plt.xlabel("Conductance Layer Output")
    plt.ylabel("Original Layer Output")
    plt.title("Conductance vs Original Output")
    plt.legend()
    plt.grid(True)
    plt.show()
    input()

    # Perform linear regression
    reg = LinearRegression(fit_intercept=True).fit(conductance_output, original_output)

    # Create the transformation function
    coef, intercept = reg.coef_.item(), reg.intercept_.item()
    print(f"Layer tuned: Coef={coef}, Intercept={intercept}")
    input()
    return lambda x: x * coef + intercept



import seaborn as sns

def map_weights_to_conductances(weights, R_hrs, R_lrs):
    """
    Map weight values to conductance matrices using memristive device characteristics.
    Biases are not scaled.

    Parameters
    ----------
    weights : dict
        Dictionary containing the layer weights and biases.
    R_hrs : float
        High resistance state of the memristive device.
    R_lrs : float
        Low resistance state of the memristive device.

    Returns
    -------
    dict
        Dictionary containing conductance matrices for weights and unscaled biases.
    """
    # Calculate the corresponding conductance range
    G_min, G_max = 1 / R_hrs, 1 / R_lrs

    def convert_range(old_value, old_min, old_max, new_min, new_max):
        """Convert values from one range to another."""
        return ((old_value - old_min) * (new_max - new_min)) / (old_max - old_min) + new_min

    conductance_matrices = {}
    for key, value in weights.items():
        if "bias" in key:
            # Keep biases unscaled
            conductance_matrices[key] = torch.tensor(value) if isinstance(value, np.ndarray) else value
        else:
            # Convert weights to PyTorch tensor if it's a numpy array
            weight = torch.tensor(value) if isinstance(value, np.ndarray) else value
            # Find the min and max values of the weights
            weight_min, weight_max = weight.min().item(), weight.max().item()
            # Clamp and map weights to the conductance range
            crossbar = torch.clamp(weight, weight_min, weight_max)  # Ensure weights stay in range
            crossbar = convert_range(crossbar, weight_min, weight_max, G_min, G_max)  # Map to conductance
            conductance_matrices[key] = crossbar  # Store the mapped conductance matrix

            # Visualize weights and conductances as heatmaps
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            sns.heatmap(weight.cpu().numpy(), ax=axes[0], cmap="viridis", cbar=True)
            axes[0].set_title(f"Original Weights ({key})")
            sns.heatmap(crossbar.cpu().numpy(), ax=axes[1], cmap="viridis", cbar=True)
            axes[1].set_title(f"Mapped Conductances ({key})")
            plt.tight_layout()
            plt.show()

    return conductance_matrices


def map_weights_to_conductances_simple(weights, R_lrs):
    scale_factor = 1 / R_lrs

    scaled_weights = {}
    for key, value in weights.items():
        if "bias" in key:
            scaled_weights[key] = value  # Keep biases unscaled
        else:
            weight = torch.tensor(value) if isinstance(value, np.ndarray) else value
            scaled_weights[key] = weight * (scale_factor / weight.max().item())

    return scaled_weights

def tune_layer_simple(conductance_layer):
    scale_factor = conductance_layer.weight.max().item()
    coef = 1 / (1 / scale_factor)  # Inverse of the original scale
    print(f"Layer tuned: Coef={coef}")
    return lambda x: x * coef




# # Custom layer implementation
# class CustomLayer(nn.Module):
#     def __init__(self, weight, bias):
#         super(CustomLayer, self).__init__()
#         # Load pre-trained weights and bias
#         self.weight = torch.tensor(weight)
#         self.bias = torch.tensor(bias)
#         self.parasiticModel = MemtorchModelCpp("Memtorch")

#     def forward(self, x):
#         # # Custom forward computation (matrix multiplication + bias addition)
#         # batch_size = x.shape[0]  # Number of samples in the batch
#         # results = []
#         # # Loop over each sample in the batch
#         # for i in range(batch_size):
#         #     # Extract the i-th sample as a NumPy array
#         #     sample = x[i].cpu().numpy()  # Shape: (in_features,)
#         #     # Perform calculation for this sample
#         #     voltage_drops = self.parasiticModel.calculate(self.weight, 0.0001, sample)  # Shape: (out_features,)
#         #     voltage_drops_np = voltage_drops
#         #     output_currents_np = np.sum(voltage_drops_np*self.weight,axis=0)
#         #     results.append(output_currents_np)
#         #     print(output_currents_np)
#         # # Convert the list of results to a PyTorch tensor
#         # output = torch.tensor(results, device=x.device)  # Shape: (batch_size, out_features)
#         # # Add bias (broadcasting ensures correct addition)
#         # return output + self.bias

#         parasiticResistance=0.0000001
#         Currents = memtorch_bindings.solve_passive(
#             self.weight.T,
#             x,
#             torch.zeros(self.weight.shape[0]),
#             parasiticResistance,
#             parasiticResistance,
#             n_input_batches=x.shape[0]
#         )
#         print("Currents:",Currents)
#         input()
#         print("ideal currents:",self.weight@x[0,:])
#         input()
#         return Currents + self.bias


class CustomLayer(nn.Module):
    def __init__(self, weight, bias, parasiticResistance, R_lrs):
        super(CustomLayer, self).__init__()
        self.weight = torch.tensor(weight)
        self.bias = torch.tensor(bias)
        self.parasiticModel = MemtorchModelCpp("Memtorch")
        self.correction_function = None  # Placeholder for tuning correction
        self.mapping_coefficient = 1 / R_lrs
        self.parasiticResistance = parasiticResistance



    def forward(self, x):
        Currents = memtorch_bindings.solve_passive(
            self.weight.T * self.mapping_coefficient,
            x,
            torch.zeros(self.weight.shape[0]),
            self.parasiticResistance,
            self.parasiticResistance,
            n_input_batches=x.shape[0]
        )
        # print("conductances",self.weight.T * self.mapping_coefficient)
        # input()
        # print("Currents:",Currents)
        # input()
        # print("ideal currents:",self.mapping_coefficient*self.weight@x[0,:])
        # input()
        # Apply correction function if available
        output = Currents/self.mapping_coefficient
        return output + self.bias


class NetStandard(nn.Module):
    def __init__(self):
        super(NetStandard, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 64)  # Standard fully connected layer
        self.fc2 = nn.Linear(64, 10)      # Standard fully connected layer

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class NetCustom(nn.Module):
    def __init__(self, weights, parasiticResistance, R_lrs):
        super(NetCustom, self).__init__()
        self.fc1 = CustomLayer(weights['fc1_weights'], weights['fc1_bias'], parasiticResistance, R_lrs)  # Custom layer
        self.fc2 = CustomLayer(weights['fc2_weights'], weights['fc2_bias'], parasiticResistance, R_lrs)  # Custom layer

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        # print(x)
        # input("x value printed - first layer outputs")
        x = self.fc2(x)
        # print(x)
        # input("x value printed - second layer outputs")
        return x






def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            all_preds.extend(pred.cpu().numpy().flatten())
            all_targets.extend(target.cpu().numpy().flatten())
            correct += pred.eq(target.view_as(pred)).sum().item()
            print(correct)
    accuracy = 100. * correct / len(test_loader.dataset)
    return accuracy, all_preds, all_targets

def plot_confusion_matrix(y_true, y_pred, folder, title):
    """Plot confusion matrix."""
    # Dynamically determine the unique classes from the data
    classes = sorted(set(y_true).union(set(y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title(title)
    plt.savefig(folder)
    plt.show()






if __name__ == '__main__':
    # original_layer = nn.Linear(100, 20)
    # conductance_layer = nn.Linear(100, 20)
    # conductance_layer.weight.data = original_layer.weight.data.clone() * 2  # Known relationship
    # conductance_layer.bias.data = original_layer.bias.data.clone() + 1

    # correction_func = tune_layer(original_layer, conductance_layer)
    # test_output = conductance_layer(torch.rand(1, 3))
    # corrected_output = correction_func(test_output)
    # print("Corrected Output:", corrected_output)


    # Load the weights
    weights = torch.load('fc_layers_weights_positive_v2.pth')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x)  # Scale to range [0, 0.5]
    ])

    # # Load the test dataset
    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.1307,), (0.3081,)),
    # ])

    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    # Create a smaller subset
    subset_size = 50
    subset_indices = torch.arange(subset_size)
    small_test_dataset = Subset(test_dataset, subset_indices)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)
    small_test_loader = DataLoader(small_test_dataset, batch_size=1, shuffle=False)

    # Test both models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define memristive device parameters
    R_hrs = 1e4  # High resistance state in Ohms
    R_lrs = 1e3  # Low resistance state in Ohms
    parasiticResistance = 0.03
    # Map weights to conductances
    conductance_matrices = map_weights_to_conductances_simple(weights, R_lrs)

    # Standard network
    model_standard = NetStandard().to(device)
    state_dict = model_standard.state_dict()  # Get the current state dict of the model

    model_standard.load_state_dict(torch.load('model_standard_positive_v2.pth', weights_only=True))  # Load trained state
    # Update the state_dict with the mapped weights
    # state_dict['fc1.weight'] = torch.tensor(conductance_matrices['fc1_weights']).to(device)
    # state_dict['fc1.bias'] = torch.tensor(conductance_matrices['fc1_bias']).to(device)
    # state_dict['fc2.weight'] = torch.tensor(conductance_matrices['fc2_weights']).to(device)
    # state_dict['fc2.bias'] = torch.tensor(conductance_matrices['fc2_bias']).to(device)

    end = datetime.datetime.now(pytz.timezone('Europe/Rome'))
    folder = 'Results/'+str(end.year)+ str(end.month)+  str(end.day) + '_'+ str(end.hour) +'_'+ str(end.minute)+"_NN_benchmark"
    if not (os.path.exists(folder)):
            os.makedirs(folder)

    # Load the updated state dict into the model
    standard_accuracy, standard_preds, standard_targets = evaluate_model(model_standard, small_test_loader, device)
    print(f"Accuracy of standard network: {standard_accuracy:.2f}%")
    plot_confusion_matrix(standard_targets, standard_preds, folder + "\Confusion_Matrix_Standard_Network.png", "Confusion Matrix: Standard Network")

    # Custom network
    model_custom = NetCustom(weights, parasiticResistance, R_lrs).to(device)
    # # Tune each layer
    # for (name_std, module_std), (name_cust, module_cust) in zip(
    #     model_standard.named_modules(), model_custom.named_modules()
    # ):
    #     if isinstance(module_std, nn.Linear) and isinstance(module_cust, CustomLayer):
    #         print(f"Tuning layer: {name_std}")
    #         module_cust.correction_function = tune_layer_simple(module_cust)


    # Evaluate the tuned model
    custom_accuracy, custom_preds, custom_targets = evaluate_model(model_custom, small_test_loader, device)
    print(f"Accuracy of tuned custom network: {custom_accuracy:.2f}%")
    plot_confusion_matrix(custom_targets, custom_preds, folder + "\Confusion_Matrix_Custom_Network.png", "Confusion Matrix: Custom Network")