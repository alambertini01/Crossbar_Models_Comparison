import torch

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
    x_sum = x.sum()
    return (y_mapped - c * x_sum) / k