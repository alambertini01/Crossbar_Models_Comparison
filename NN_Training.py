import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np
from CrossbarModels.Crossbar_Models_pytorch import jeong_model, dmr_model, gamma_model, solve_passive_model, crosssim_model, IdealModel

# Define the network with a selectable model function
class CustomNet(nn.Module):
    def __init__(self, model_function, parasiticResistance, R_lrs):
        super(CustomNet, self).__init__()
        self.model_function = model_function
        self.parasiticResistance = parasiticResistance
        self.R_lrs = R_lrs
        # Two fully connected layers
        self.fc1 = nn.Linear(28 * 28, 64)  # Input: 784, Hidden Layer: 64 neurons
        self.fc2 = nn.Linear(64, 10)       # Hidden Layer: 64 neurons, Output: 10 classes

        # Initialize weights to be positive
        nn.init.uniform_(self.fc1.weight, a=0.01, b=0.1)
        nn.init.uniform_(self.fc2.weight, a=0.01, b=0.1)
        nn.init.constant_(self.fc1.bias, 0.01)
        nn.init.constant_(self.fc2.bias, 0.01)

    def forward(self, x):
        # Flatten the input image
        x = x.view(-1, 28 * 28)
        # First fully connected layer with ReLU activation
        x = F.relu(self.model_function(self.fc1.weight.T * (1 / self.R_lrs), x, self.parasiticResistance) / (1 / self.R_lrs) + self.fc1.bias)
        # Output layer
        x = self.model_function(self.fc2.weight.T * (1 / self.R_lrs), x, self.parasiticResistance) / (1 / self.R_lrs) + self.fc2.bias
        # x = torch.clamp(x, min=-1e3, max=1e3)  # Clamp values to prevent extreme values
        return x

# Training and evaluation script
if __name__ == "__main__":
    # Model selection
    available_models = {
        "jeong_model": jeong_model,
        "dmr_model": dmr_model,
        "gamma_model": gamma_model,
        "solve_passive_model": solve_passive_model,
        "crosssim_model": crosssim_model,
        "IdealModel": IdealModel
    }
    
    print("Available models:")
    for i, model_name in enumerate(available_models.keys()):
        print(f"{i}: {model_name}")
    
    model_choice = int(input("Select the model number for training: "))
    selected_model_name = list(available_models.keys())[model_choice]
    selected_model_function = available_models[selected_model_name]

    # Select parasitic resistance and R_lrs
    parasiticResistance = float(input("Enter parasitic resistance value: "))
    R_lrs = float(input("Enter R_lrs value: "))

    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Hyperparameters
    batch_size = 128
    learning_rate = 0.001
    epochs = 20

    # Data loading and preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x / 2)  # Scale to range [0, 0.5]
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model, optimizer, and loss function
    model = CustomNet(selected_model_function, parasiticResistance, R_lrs).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Save the trained weights
    save_folder = f'Models/Trained/{selected_model_name}/Rpar_{parasiticResistance}'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Training loop with accuracy tracking
    train_accuracies = []
    plt.figure(figsize=(10, 5))

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            # Forward pass
            output = model(data)
            loss = criterion(output, target)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # # Print gradients for debugging
            # for name, param in model.named_parameters():
            #     if param.grad is not None:
            #         print(f'Gradient for {name}: {param.grad.norm()}')

            optimizer.step()

            # Enforce positive weights
            with torch.no_grad():
                model.fc1.weight.data = abs(model.fc1.weight.data)
                model.fc2.weight.data = abs(model.fc2.weight.data)

            train_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

        train_loss /= len(train_loader.dataset)
        train_accuracy = 100. * correct / len(train_loader.dataset)
        train_accuracies.append(train_accuracy)
        print(f'Epoch: {epoch + 1}, Loss: {train_loss:.6f}, Accuracy: {train_accuracy:.2f}%')

            # Save checkpoint
        checkpoint_path = f'{save_folder}/checkpoint_epoch_{epoch}.pth'
        torch.save(model.state_dict(), checkpoint_path)
        print(f'Model checkpoint saved at {checkpoint_path}')

        # Plot accuracy dynamically
        plt.clf()
        plt.plot(range(1, epoch + 2), train_accuracies, marker='o', linestyle='-', color='b')
        plt.xlabel('Epoch')
        plt.ylabel('Training Accuracy (%)')
        plt.title('Training Accuracy Over Epochs')
        plt.pause(0.1)


    torch.save({
        'fc1_weights': model.fc1.weight.data.cpu().numpy(),
        'fc1_bias': model.fc1.bias.data.cpu().numpy(),
        'fc2_weights': model.fc2.weight.data.cpu().numpy(),
        'fc2_bias': model.fc2.bias.data.cpu().numpy(),
    }, f'{save_folder}/fc_layers_weights_positive_v2.pth')

    # Save the entire state_dict of the model
    torch.save(model.state_dict(), f'{save_folder}/model_{selected_model_name}_positive_full_statedict.pth')

    # Final plot showing the training accuracy
    plt.savefig(f'{save_folder}/training_accuracy.png')
    plt.show()

    # Evaluation loop
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            # Forward pass
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest Loss: {test_loss:.6f}, Test Accuracy: {test_accuracy:.2f}%')

    # Save heatmaps of weights and biases
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    sns.heatmap(model.fc1.weight.data.cpu().numpy(), ax=axes[0, 0], cmap='viridis')
    axes[0, 0].set_title('FC1 Weights')
    
    sns.heatmap(np.expand_dims(model.fc1.bias.data.cpu().numpy(), axis=0), ax=axes[0, 1], cmap='viridis', cbar=False)
    axes[0, 1].set_title('FC1 Bias')
    
    sns.heatmap(model.fc2.weight.data.cpu().numpy(), ax=axes[1, 0], cmap='viridis')
    axes[1, 0].set_title('FC2 Weights')
    
    sns.heatmap(np.expand_dims(model.fc2.bias.data.cpu().numpy(), axis=0), ax=axes[1, 1], cmap='viridis', cbar=False)
    axes[1, 1].set_title('FC2 Bias')
    
    plt.tight_layout()
    plt.savefig(f'{save_folder}/weights_biases_heatmap.png')
    plt.close(fig)
