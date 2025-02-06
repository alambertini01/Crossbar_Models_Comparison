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
import time
import glob
from CrossbarModels.Crossbar_Models_pytorch import jeong_model, dmr_model, alpha_beta_model, Memtorch_model, crosssim_model, IdealModel
from NN_Crossbar.Crossbar_net import CustomNet, evaluate_model

print("Available models:")
available_models = {
    "jeong_model": jeong_model,
    "dmr_model": dmr_model,
    "alpha_beta_model": alpha_beta_model,
    "solve_passive_model": Memtorch_model,
    "crosssim_model": crosssim_model,
    "IdealModel": IdealModel
}
for i, model_name in enumerate(available_models.keys()):
    print(f"{i}: {model_name}")

model_choice = int(input("Select the model number for training: "))
selected_model_name = list(available_models.keys())[model_choice]
selected_model_function = available_models[selected_model_name]

parasiticResistance = float(input("Enter parasitic resistance value: "))
R_lrs = float(input("Enter R_lrs value: "))
R_hrs = float(input("Enter R_hrs value: "))
max_array_size = int(input("Enter max_array_size value: "))
quant_bits = 0

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
hiddenLayer = 128
batch_size = 64
learning_rate = 0.002
epochs = 20
save_checkpoint = True
Fix_positive_weights = True
early_stop_acc = 97

# Data loading and preprocessing - using normalization for stability
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize model weights
weights = {
    'fc1_weights': torch.randn(28 * 28, hiddenLayer),
    'fc1_bias': torch.randn(hiddenLayer),
    'fc2_weights': torch.randn(hiddenLayer, 10),
    'fc2_bias': torch.randn(10)
}

if Fix_positive_weights:
    weights['fc1_weights'] = torch.abs(weights['fc1_weights'])
    weights['fc2_weights'] = torch.abs(weights['fc2_weights'])

# Apply Xavier initialization
nn.init.xavier_uniform_(weights['fc1_weights'])
weights['fc1_bias'].fill_(0.0)
nn.init.xavier_uniform_(weights['fc2_weights'])
weights['fc2_bias'].fill_(0.0)

# Initialize the model and move to GPU
model = CustomNet(weights, parasiticResistance, R_hrs, R_lrs, selected_model_function, device,
                  debug=False, bits=quant_bits, correction=False, max_array_size=max_array_size).to(device)

# Initialize optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# Save folder setup
save_folder = f'TrainedModels/{selected_model_name}/Rpar{parasiticResistance}__LRS{R_lrs}__HRS{R_hrs}__Size{max_array_size}'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

start_epoch = 0
# Look for checkpoint files using the naming pattern we already use.
checkpoint_files = glob.glob(os.path.join(save_folder, 'checkpoint_epoch_*.pth'))
if checkpoint_files:
    latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
    try:
        model.load_state_dict(torch.load(latest_checkpoint))
        # Assuming a filename like 'checkpoint_epoch_3.pth'
        start_epoch = int(os.path.basename(latest_checkpoint).split('_')[-1].split('.')[0]) + 1
        print(f"Resuming training from checkpoint: {latest_checkpoint} (starting at epoch {start_epoch})")
    except Exception as e:
        print(f"Error loading checkpoint {latest_checkpoint}: {e}\nStarting training from scratch.")
elif os.path.exists(os.path.join(save_folder, f'model_{selected_model_name}_full_statedict.pth')):
    final_model_path = os.path.join(save_folder, f'model_{selected_model_name}_full_statedict.pth')
    try:
        model.load_state_dict(torch.load(final_model_path))
        print(f"Found final model weights at {final_model_path}.")
        cont = input("Training appears complete. Do you want to continue training? (y/n): ")
        if cont.lower() == 'y':
            additional_epochs = int(input("Enter additional epochs to train: "))
            start_epoch = epochs  # current training was already complete
            epochs += additional_epochs
            print(f"Resuming training from epoch {start_epoch} to {epochs}")
        else:
            print("Exiting training.")
            exit(0)
    except Exception as e:
        print(f"Error loading final model weights: {e}\nStarting training from scratch.")
else:
    print("No saved model found. Starting training from scratch.")

train_accuracies = []
plt.figure(figsize=(10, 5))

train_time_start = time.time()

for epoch in range(start_epoch, epochs):
    model.train()
    train_loss = 0
    correct = 0

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)

        output = model(data)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if Fix_positive_weights:
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if 'weight' in name:
                        param.clamp_(min=0)

        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

    train_loss /= len(train_loader.dataset)
    train_accuracy = 100. * correct / len(train_loader.dataset)
    train_accuracies.append(train_accuracy)
    print(f'Epoch: {epoch + 1}, Loss: {train_loss:.6f}, Accuracy: {train_accuracy:.2f}%')

    # Early stopping check
    if train_accuracy >= early_stop_acc:
        print("Early stopping criterion reached.")
        break

    if save_checkpoint:
        checkpoint_path = f'{save_folder}/checkpoint_epoch_{epoch}.pth'
        torch.save(model.state_dict(), checkpoint_path)
        print(f'Model checkpoint saved at {checkpoint_path}')

    plt.clf()
    plt.plot(range(1, epoch + 2), train_accuracies, marker='o', linestyle='-', color='b')
    plt.xlabel('Epoch')
    plt.ylabel('Training Accuracy (%)')
    plt.title('Training Accuracy Over Epochs')
    plt.pause(0.1)

train_time_end = time.time()
total_training_time = train_time_end - train_time_start

# Save final weights
torch.save({
    'fc1_weights': model.fc1.weight.data.cpu(),
    'fc1_bias': model.fc1.bias.data.cpu(),
    'fc2_weights': model.fc2.weight.data.cpu(),
    'fc2_bias': model.fc2.bias.data.cpu(),
}, f'{save_folder}/fc_layers_weights.pth')

# Save the entire state_dict of the model
torch.save(model.state_dict(), f'{save_folder}/model_{selected_model_name}_full_statedict.pth')

plt.savefig(f'{save_folder}/training_accuracy.png')
plt.close()

accuracy, all_preds, all_targets = evaluate_model(model, test_loader, device)
print(f'Test Accuracy: {accuracy:.2f}%')

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
sns.heatmap(model.fc1.weight.data.cpu().numpy(), ax=axes[0, 0], cmap='viridis')
axes[0, 0].set_title('FC1 Weights')
sns.heatmap(
    np.expand_dims(model.fc1.bias.data.cpu().numpy(), axis=0),
    ax=axes[0, 1],
    cmap='viridis',
    cbar=True,
    cbar_kws={"shrink": 0.5}
)
axes[0, 1].set_title('FC1 Bias')
sns.heatmap(model.fc2.weight.data.cpu().numpy(), ax=axes[1, 0], cmap='viridis')
axes[1, 0].set_title('FC2 Weights')
sns.heatmap(
    np.expand_dims(model.fc2.bias.data.cpu().numpy(), axis=0),
    ax=axes[1, 1],
    cmap='viridis',
    cbar=True,
    cbar_kws={"shrink": 0.5}
)
axes[1, 1].set_title('FC2 Bias')
plt.tight_layout()
plt.savefig(f'{save_folder}/weights_biases_heatmap.png')
plt.show()

log_path = os.path.join(save_folder, "train_log.txt")
with open(log_path, 'w') as f:
    f.write("Training Log\n")
    f.write(f"Model: {selected_model_name}\n")
    f.write(f"Parasitic Resistance: {parasiticResistance}\n")
    f.write(f"R_lrs: {R_lrs}\n")
    f.write(f"R_hrs: {R_hrs}\n")
    f.write(f"Max Array Size: {max_array_size}\n")
    f.write(f"Hidden Layer Size: {hiddenLayer}\n")
    f.write(f"Batch Size: {batch_size}\n")
    f.write(f"Learning Rate: {learning_rate}\n")
    f.write(f"Epochs: {epochs}\n")
    f.write(f"Early Stop Accuracy: {early_stop_acc}%\n")
    f.write(f"Final Training Accuracy: {train_accuracies[-1]:.2f}%\n")
    f.write(f"Test Accuracy: {accuracy:.2f}%\n")
    f.write(f"Total Training Time (s): {total_training_time:.4f}\n")