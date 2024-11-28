import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import Subset

from CrossbarModels.Crossbar_Models import *
from CrossbarModels.Models import memtorch_bindings # type: ignore

# Custom layer implementation
class CustomLayer(nn.Module):
    def __init__(self, weight, bias):
        super(CustomLayer, self).__init__()
        # Load pre-trained weights and bias
        self.weight = torch.tensor(abs(weight))
        self.bias = torch.tensor(bias)
        self.parasiticModel = MemtorchModelCpp("Memtorch")

    def forward(self, x):
        # # Custom forward computation (matrix multiplication + bias addition)
        # batch_size = x.shape[0]  # Number of samples in the batch
        # results = []
        # # Loop over each sample in the batch
        # for i in range(batch_size):
        #     # Extract the i-th sample as a NumPy array
        #     sample = x[i].cpu().numpy()  # Shape: (in_features,)
        #     # Perform calculation for this sample
        #     voltage_drops = self.parasiticModel.calculate(self.weight, 0.0001, sample)  # Shape: (out_features,)
        #     voltage_drops_np = voltage_drops
        #     output_currents_np = np.sum(voltage_drops_np*self.weight,axis=0)
        #     results.append(output_currents_np)
        #     print(output_currents_np)
        # # Convert the list of results to a PyTorch tensor
        # output = torch.tensor(results, device=x.device)  # Shape: (batch_size, out_features)
        # # Add bias (broadcasting ensures correct addition)
        # return output + self.bias

        parasiticResistance=0.0001
        print("weight shape:",self.weight.shape[1])
        print("Batches:",x.shape[0])
        Currents = memtorch_bindings.solve_passive(
            self.weight,
            x.T,
            torch.zeros(self.weight.shape[0]),
            parasiticResistance,
            parasiticResistance,
            n_input_batches=x.shape[0]
        )
        print(Currents.T)
        return Currents.T + self.bias

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
    def __init__(self, weights):
        super(NetCustom, self).__init__()
        self.fc1 = CustomLayer(weights['fc1_weights'], weights['fc1_bias'])  # Custom layer
        self.fc2 = CustomLayer(weights['fc2_weights'], weights['fc2_bias'])  # Custom layer

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x



def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            print("Correct:", correct)
    accuracy = 100. * correct / len(test_loader.dataset)
    return accuracy


if __name__ == '__main__':

    # Load the weights
    weights = torch.load('fc_layers_weights.pth')

    # Load the test dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    # Create a smaller subset
    subset_size = 200
    subset_indices = torch.arange(subset_size)
    small_test_dataset = Subset(test_dataset, subset_indices)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)
    small_test_loader = DataLoader(small_test_dataset, batch_size=1, shuffle=False)

    # Test both models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Standard network
    model_standard = NetStandard().to(device)
    model_standard.load_state_dict(torch.load('model_standard.pth', weights_only=True))  # Load trained state
    standard_accuracy = evaluate_model(model_standard, small_test_loader, device)
    print(f"Accuracy of standard network: {standard_accuracy:.2f}%")

    # Custom network
    model_custom = NetCustom(weights).to(device)
    custom_accuracy = evaluate_model(model_custom, small_test_loader, device)
    print(f"Accuracy of custom network: {custom_accuracy:.2f}%")