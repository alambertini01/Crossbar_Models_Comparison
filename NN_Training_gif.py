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
from PIL import Image
from CrossbarModels.Crossbar_Models_pytorch import jeong_model, dmr_model, gamma_model, alpha_beta_model, solve_passive_model, crosssim_model, IdealModel
from Crossbar_net import CustomNet, evaluate_model

print("Available models:")
available_models = {
    "jeong_model": jeong_model,
    "dmr_model": dmr_model,
    "gamma_model": gamma_model,
    "alpha_beta_model": alpha_beta_model,
    "solve_passive_model": solve_passive_model,
    "crosssim_model": crosssim_model,
    "IdealModel": IdealModel
}
for i, model_name in enumerate(available_models.keys()):
    print(f"{i}: {model_name}")

model_choice = int(input("Select the model number for training: "))
selected_model_name = list(available_models.keys())[model_choice]
selected_model_function = available_models[selected_model_name]

parasiticResistance = float(input("Enter parasitic resistance value: "))
R_lrs = 1000
R_hrs = 40000
max_array_size = int(input("Enter max_array_size value: "))
quant_bits = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
hiddenLayer = 128
batch_size = 128
learning_rate = 0.01
epochs = 20
save_checkpoint = False
Fix_positive_weights = False
early_stop_acc = 90
# Add visualization flag
ENABLE_WEIGHT_VIS = True  # Set to False to disable dynamic weight visualization
VISUALIZATION_FREQ = 20  # Update visualization every N batches
vmin, vmax = -1, 2  # heatmap values

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

weights = {
    'fc1_weights': torch.randn(28 * 28, hiddenLayer),
    'fc1_bias': torch.randn(hiddenLayer),
    'fc2_weights': torch.randn(hiddenLayer, 10),
    'fc2_bias': torch.randn(10)
}

if Fix_positive_weights:
    weights['fc1_weights'] = torch.abs(weights['fc1_weights'])
    weights['fc2_weights'] = torch.abs(weights['fc2_weights'])

nn.init.xavier_uniform_(weights['fc1_weights'])
weights['fc1_bias'].fill_(0.0)
nn.init.xavier_uniform_(weights['fc2_weights'])
weights['fc2_bias'].fill_(0.0)

model = CustomNet(weights, parasiticResistance, R_hrs, R_lrs, selected_model_function, device,
                 debug=False, bits=quant_bits, correction=False, max_array_size=max_array_size).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

save_folder = f'TrainedModels/{selected_model_name}/Rpar{parasiticResistance}__LRS{R_lrs}__HRS{R_hrs}__Size{max_array_size}'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

train_accuracies = []
weight_frames = []
batch_accuracies = []  # Track accuracies at batch level
running_correct = 0
running_total = 0

if ENABLE_WEIGHT_VIS:
    plt.ion()  # Enable interactive mode
    fig = plt.figure(figsize=(15, 8))
    
    # Create layout with GridSpec
    gs = plt.GridSpec(2, 8, height_ratios=[1, 1], width_ratios=[10, 1, 10, 1, 10, 1, 10, 1])
    ax1 = plt.subplot(gs[0, :])  # Accuracy plot spanning all columns
    
    # Create subplots with dedicated colorbar axes
    ax2 = plt.subplot(gs[1, 0])  # FC1 weights
    cax2 = plt.subplot(gs[1, 1])  # FC1 weights colorbar
    ax3 = plt.subplot(gs[1, 2])  # FC1 bias
    cax3 = plt.subplot(gs[1, 3])  # FC1 bias colorbar
    ax4 = plt.subplot(gs[1, 4])  # FC2 weights
    cax4 = plt.subplot(gs[1, 5])  # FC2 weights colorbar
    ax5 = plt.subplot(gs[1, 6])  # FC2 bias
    cax5 = plt.subplot(gs[1, 7])  # FC2 bias colorbar
    
    # Initialize the colormaps and colorbars
    norm1 = plt.Normalize(vmin=-1, vmax=1)
    norm2 = plt.Normalize(vmin=-1, vmax=1)
    norm3 = plt.Normalize(vmin=-1, vmax=1)
    norm4 = plt.Normalize(vmin=-1, vmax=1)
    
    plt.colorbar(plt.cm.ScalarMappable(norm=norm1, cmap='viridis'), cax=cax2)
    plt.colorbar(plt.cm.ScalarMappable(norm=norm2, cmap='viridis'), cax=cax3)
    plt.colorbar(plt.cm.ScalarMappable(norm=norm3, cmap='viridis'), cax=cax4)
    plt.colorbar(plt.cm.ScalarMappable(norm=norm4, cmap='viridis'), cax=cax5)

train_time_start = time.time()


for epoch in range(epochs):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
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
        correct_batch = pred.eq(target.view_as(pred)).sum().item()
        correct += correct_batch
        total += target.size(0)
        
        # Update running statistics
        running_correct += correct_batch
        running_total += target.size(0)
        
        # Visualize every VISUALIZATION_FREQ batches
        if ENABLE_WEIGHT_VIS and (batch_idx + 1) % VISUALIZATION_FREQ == 0:
            running_acc = 100. * running_correct / running_total
            batch_accuracies.append(running_acc)

            # Clear main axes but keep colorbar axis
            ax1.clear()
            ax2.clear()
            ax3.clear()
            ax4.clear()
            ax5.clear()
            cax2.clear()

            # Plot running accuracy
            ax1.plot(batch_accuracies, marker='', linestyle='-', color='b')
            ax1.set_xlabel('Updates')
            ax1.set_ylabel('Running Accuracy (%)')
            ax1.set_title(f'Training Progress (Epoch {epoch + 1})')
            ax1.grid(True, linestyle='--', alpha=0.7)
            ax1.set_ylim(0, 100)

            # Get current weight data
            fc1_weights = model.fc1.weight.data.cpu().numpy()
            fc1_bias = np.expand_dims(model.fc1.bias.data.cpu().numpy(), axis=0)
            fc2_weights = model.fc2.weight.data.cpu().numpy()
            fc2_bias = np.expand_dims(model.fc2.bias.data.cpu().numpy(), axis=0)

            # Fixed normalization range for all heatmaps
            vmin, vmax = -1, 2

            # Plot heatmaps without creating new colorbars
            im1 = ax2.imshow(fc1_weights, cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax)
            im2 = ax3.imshow(fc1_bias, cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax)
            im3 = ax4.imshow(fc2_weights, cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax)
            im4 = ax5.imshow(fc2_bias, cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax)

            # Set titles
            ax2.set_title(f'FC1 Weights (Epoch {epoch + 1})')
            ax3.set_title(f'FC1 Bias (Epoch {epoch + 1})')
            ax4.set_title(f'FC2 Weights (Epoch {epoch + 1})')
            ax5.set_title(f'FC2 Bias (Epoch {epoch + 1})')

            # Remove ticks for cleaner heatmap display
            for ax in [ax2, ax3, ax4, ax5]:
                ax.set_xticks([])
                ax.set_yticks([])

            # Add a single, shared colorbar
            plt.colorbar(im1, cax=cax2)

            plt.tight_layout()
            plt.pause(0.1)

            # Save frame for GIF
            fig.canvas.draw()
            data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            w, h = fig.canvas.get_width_height()
            frame = data.reshape((h, w, 3))
            weight_frames.append(frame)
    
    # Calculate epoch accuracy
    epoch_acc = 100. * correct / total
    train_accuracies.append(epoch_acc)
    print(f'Epoch: {epoch + 1}, Loss: {train_loss/total:.6f}, Accuracy: {epoch_acc:.2f}%')
    
    if epoch_acc >= early_stop_acc:
        print("Early stopping criterion reached.")
        break
    
    if save_checkpoint:
        checkpoint_path = f'{save_folder}/checkpoint_epoch_{epoch}.pth'
        torch.save(model.state_dict(), checkpoint_path)
        print(f'Model checkpoint saved at {checkpoint_path}')

train_time_end = time.time()
total_training_time = train_time_end - train_time_start

if ENABLE_WEIGHT_VIS:
    plt.ioff()
    plt.close()
    
    # Convert frames to PIL Images with optional resizing
    pil_frames = []
    for frame in weight_frames:
        img = Image.fromarray(frame)
        # Optionally resize if the images are too large
        if img.size[0] > 1200:  # If width is greater than 1200px
            new_width = 1200
            new_height = int(img.size[1] * (new_width / img.size[0]))
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        pil_frames.append(img)
    
    gif_path = f'{save_folder}/weight_evolution.gif'
    pil_frames[0].save(
        gif_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=200,  # Duration for each frame in milliseconds
        loop=0,
        optimize=True  # Add optimization to reduce file size
    )
    print(f"Weight evolution saved as GIF to {gif_path}")

# Save final weights
torch.save({
    'fc1_weights': model.fc1.weight.data.cpu(),
    'fc1_bias': model.fc1.bias.data.cpu(),
    'fc2_weights': model.fc2.weight.data.cpu(),
    'fc2_bias': model.fc2.bias.data.cpu(),
}, f'{save_folder}/fc_layers_weights.pth')

torch.save(model.state_dict(), f'{save_folder}/model_{selected_model_name}_full_statedict.pth')

# Final evaluation
accuracy, all_preds, all_targets = evaluate_model(model, test_loader, device)
print(f'Test Accuracy: {accuracy:.2f}%')

# Save final heatmap
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
sns.heatmap(model.fc1.weight.data.cpu().numpy(), ax=axes[0, 0], cmap='viridis')
axes[0, 0].set_title('Final FC1 Weights')
sns.heatmap(np.expand_dims(model.fc1.bias.data.cpu().numpy(), axis=0), 
            ax=axes[0, 1], cmap='viridis', cbar=True, cbar_kws={"shrink": 0.5})
axes[0, 1].set_title('Final FC1 Bias')
sns.heatmap(model.fc2.weight.data.cpu().numpy(), ax=axes[1, 0], cmap='viridis')
axes[1, 0].set_title('Final FC2 Weights')
sns.heatmap(np.expand_dims(model.fc2.bias.data.cpu().numpy(), axis=0), 
            ax=axes[1, 1], cmap='viridis', cbar=True, cbar_kws={"shrink": 0.5})
axes[1, 1].set_title('Final FC2 Bias')
plt.tight_layout()
plt.savefig(f'{save_folder}/final_weights_biases_heatmap.png')
plt.close()

# Save training log
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