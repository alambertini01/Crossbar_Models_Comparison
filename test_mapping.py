import torch

# Define the convert_range function
def convert_range(old_value, old_min, old_max, new_min, new_max):
    """Method to convert values between two ranges.

    Parameters
    ----------
    old_value : object
        Old value(s) to convert. May be a single number, vector or tensor.
    old_min : float
        Minimum old value.
    old_max : float
        Maximum old value.
    new_min : float
        Minimum new value.
    new_max : float
        Maximum new value.

    Returns
    -------
    object
        New value(s).
    """
    return (
        ((old_value - old_min) * (new_max - new_min)) / (old_max - old_min)
    ) + new_min

# Define the mapping function
def map_weights(weight, weight_min, weight_max, r_on, r_off):
    crossbar = weight.clone()
    crossbar = torch.clamp(crossbar, weight_min, weight_max)
    crossbar = convert_range(crossbar, weight_min, weight_max, 1 / r_off, 1 / r_on)
    return crossbar

# Set parameters
input_size = 3
output_size = 2

# Create weight tensor W
W = torch.tensor([[0.2, -0.5],
                  [0.7, 0.3],
                  [-0.1, 0.8]], dtype=torch.float32)

print("Original Weight Matrix W:")
print(W)

# Create input vector x
x = torch.tensor([0.5, 1, 0.9], dtype=torch.float32)

print("\nInput Vector x:")
print(x)

# Compute y = W x
y = W.t().mv(x)  # W.t() is of size (output_size, input_size)
print("\nResult of y = W x:")
print(y)

# Define weight_min, weight_max
weight_min = torch.min(W)
weight_max = torch.max(W)

# Define r_on, r_off
r_on = 1e4  # 10 kΩ
r_off = 1e6  # 1 MΩ

# Map W to crossbar
crossbar = map_weights(W, weight_min, weight_max, r_on, r_off)

print("\nMapped Weight Matrix (Crossbar):")
print(crossbar)

# Compute y_crossbar = crossbar x
y_crossbar = crossbar.t().mv(x)

print("\nResult of y_crossbar = Crossbar x:")
print(y_crossbar)

# Compute mapping constants
k = ((1 / r_on) - (1 / r_off)) / (weight_max - weight_min)
c = ((-weight_min) * k) + (1 / r_off)

print("\nMapping constants:")
print(f"k: {k}")
print(f"c: {c}")

# Recover original y from y_crossbar
x_sum = x.sum()
y_recovered = (y_crossbar - c * x_sum) / k

print("\nRecovered y from y_crossbar:")
print(y_recovered)

print("\nOriginal y:")
print(y)
