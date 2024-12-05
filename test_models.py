import torch
import matplotlib.pyplot as plt
from CrossbarModels.Crossbar_Models_pytorch import (
    jeong_model, dmr_model, gamma_model, solve_passive_model, crosssim_model, IdealModel
)

# Test parameters
input_size = 784  # Number of input nodes
output_size = 64  # Number of output nodes
batch_size = 1  # Number of samples in the batch

# Create random weights (conductance matrix) with positive values
weight = torch.rand(input_size, output_size) * 1e-3  # Random values in range [0, 0.001]

# Calculate parasiticResistance to be three orders below 1/weight
# Adding epsilon to avoid division by zero in extremely small weights
parasiticResistance = 2  # Fixed value for all models

# Create input tensor `x` filled with ones
x = torch.rand(batch_size, input_size) * 0.5


# Compute currents for all models
currents = {
    "DMR Model": dmr_model(weight, x, parasiticResistance),
    "Jeong Model": jeong_model(weight, x, parasiticResistance),
    "Gamma Model": gamma_model(weight, x, parasiticResistance),
    "Passive Model": solve_passive_model(weight, x, parasiticResistance),
    "CrossSim Model": crosssim_model(weight, x, parasiticResistance),
}

# Plot the currents
plt.figure(figsize=(10, 6))
for model_name, current in currents.items():
    # Average current across the batch for plotting
    avg_current = current.mean(dim=0).detach().cpu().numpy()
    plt.plot(avg_current, label=model_name)

plt.title("Currents Across Models")
plt.xlabel("Output Node Index")
plt.ylabel("Current (A)")
plt.legend()
plt.grid()
plt.show()
