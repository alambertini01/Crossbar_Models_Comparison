import torch
from torch.autograd import Variable

def weight_mapping(weights, r_hrs, r_lrs):
    """Map weights to conductances.
    """
    weight_min = torch.min(weights)
    weight_max = torch.max(weights)

    return ((torch.clamp(weights, weight_min, weight_max) - weight_min)
            * (1 / r_lrs - 1 / r_hrs) / (weight_max - weight_min)) + (1 / r_hrs)


def output_unmapping(weights, y_mapped, x, r_hrs, r_lrs):
    """Revert the mapping on the output vector to obtain the original output.
    """
    weight_min = torch.min(weights)
    weight_max = torch.max(weights)
    k = (1 / r_lrs - 1 / r_hrs) / (weight_max - weight_min)
    c = ((-weight_min) * (1 / r_lrs - 1 / r_hrs) / (weight_max - weight_min)) + (1 / r_hrs)
    x_sum = x.sum(dim=1, keepdim=True)
    return (y_mapped - c * x_sum) / k


def quantization(conductances, R_lrs, R_hrs, bits):
    """
    Quantizes the conductances tensor to the nearest level based on given resistance levels and bit precision.

    Parameters:
    conductances (torch.Tensor): Tensor containing the ideal conductances of crossbar devices.
    R_lrs (torch.Tensor): Tensor containing the low resistance values (R_lrs).
    R_hrs (torch.Tensor): Tensor containing the high resistance values (R_hrs).
    bits (int): Number of bits determining the quantization levels (2^bits levels).

    Returns:
    torch.Tensor: Tensor of quantized conductances.
    """
    # Compute the minimum and maximum conductance levels
    G_min = 1.0 / R_hrs
    G_max = 1.0 / R_lrs
    delta_G = G_max - G_min

    # Normalize the conductances between 0 and 1
    norm_G = (conductances - G_min) / delta_G
    norm_G = torch.clamp(norm_G, 0.0, 1.0)

    # Quantize the normalized conductances to the nearest level
    levels = torch.round(norm_G * (2 ** bits - 1))

    # Map the quantized levels back to the conductance values
    quantized_norm_G = levels / (2 ** bits - 1)
    quantized_conductances = G_min + quantized_norm_G * delta_G

    return quantized_conductances
