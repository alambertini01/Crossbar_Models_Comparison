import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define the network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Two fully connected layers
        self.fc1 = nn.Linear(28 * 28, 64)  # Input: 784, Hidden Layer: 256 neurons
        self.fc2 = nn.Linear(64, 10)       # Hidden Layer: 256 neurons, Output: 10 classes

    def forward(self, x):
        # Flatten the input image
        x = x.view(-1, 28 * 28)
        # First fully connected layer with ReLU activation
        x = F.relu(self.fc1(x))
        # Output layer
        x = self.fc2(x)
        return x

# Training and evaluation script
def main():
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Hyperparameters
    batch_size = 64
    learning_rate = 0.001
    epochs = 10

    # Data loading and preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # Normalize to mean=0.1307, std=0.3081
    ])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model, optimizer, and loss function
    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Training loop
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
            optimizer.step()

            train_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

        train_loss /= len(train_loader.dataset)
        train_accuracy = 100. * correct / len(train_loader.dataset)
        print(f'Epoch: {epoch + 1}, Loss: {train_loss:.6f}, Accuracy: {train_accuracy:.2f}%')


        # Save the trained weights
    torch.save({
        'fc1_weights': model.fc1.weight.data.numpy(),
        'fc1_bias': model.fc1.bias.data.numpy(),
        'fc2_weights': model.fc2.weight.data.numpy(),
        'fc2_bias': model.fc2.bias.data.numpy(),
    }, 'fc_layers_weights.pth')

    # Save the entire state_dict of the model
    torch.save(model.state_dict(), 'model_standard.pth')

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








if __name__ == "__main__":
    main()
