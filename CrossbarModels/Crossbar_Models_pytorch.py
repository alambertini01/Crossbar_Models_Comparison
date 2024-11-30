import torch
from .Models import memtorch_bindings # type: ignore

# Jeong model implementation
# Computes voltage drops and currents for given weights and inputs
def jeong_model(weight, x, parasiticResistance):
    input_size, output_size = weight.shape
    # Calculate A and B matrices
    A_jeong = parasiticResistance * torch.cumsum(torch.arange(output_size, 0, -1, dtype=torch.float32, device=weight.device), dim=0)
    B_jeong = parasiticResistance * torch.cumsum(torch.arange(input_size, 0, -1, dtype=torch.float32, device=weight.device).flip(0), dim=0).flip(0)
    # Repeat A and B to form matrices
    A_jeong_matrix = A_jeong.repeat(input_size, 1)
    B_jeong_matrix = B_jeong.view(-1, 1).repeat(1, output_size)
    # Calculate voltage drops
    V_a_matrix = x.view(-1, 1).repeat(1, output_size)
    voltage_drops_jeong = (weight * torch.reciprocal(A_jeong_matrix + weight + B_jeong_matrix)) * V_a_matrix
    # Calculate currents
    current_jeong = torch.matmul(x, torch.reciprocal(A_jeong_matrix + weight + B_jeong_matrix))
    return current_jeong

# DMR model implementation
# Calculates voltage drops and currents using diagonal matrices A and B
def dmr_model(weight, x, parasiticResistance):
    input_size, output_size = weight.shape
    G = weight
    # Calculate g_bit and g_word values
    g_bit = g_word = 1 / parasiticResistance
    # Compute average conductances per row and column
    gAverageRow, gAverageCol = G.mean(dim=1), G.mean(dim=0)
    # Calculate a_dmr values
    Summation_bit = torch.sum(torch.arange(input_size, 0, -1, device=weight.device) * gAverageRow)
    Summation1 = torch.arange(input_size, device=weight.device) * torch.cumsum(gAverageRow, dim=0) - torch.cumsum(torch.arange(input_size, device=weight.device) * gAverageRow, dim=0)
    a_dmr = (1 + Summation1 / g_bit) / (1 + Summation_bit / g_bit)
    # Calculate b_dmr values
    Summation_word = torch.sum(torch.arange(1, output_size + 1, device=weight.device) * gAverageCol[:output_size])
    Summation2 = torch.matmul(torch.triu(torch.arange(output_size, dtype=torch.float32, device=weight.device).view(1, -1) - torch.arange(output_size, dtype=torch.float32, device=weight.device).view(-1, 1)), gAverageCol[:output_size])
    b_dmr = (1 + Summation2 / g_word) / (1 + Summation_word / g_word)
    # Create diagonal matrices A and B
    A_dmr, B_dmr = torch.diag(a_dmr), torch.diag(b_dmr)
    # Calculate W_dmr
    W_dmr = A_dmr @ G @ B_dmr
    # Compute voltage drops and currents
    V_a_matrix = x.view(-1, 1).repeat(1, output_size)
    current_dmr = torch.matmul(x, W_dmr)
    voltage_drops_dmr = A_dmr @ V_a_matrix @ B_dmr
    return current_dmr

# Gamma model implementation
# Calculates voltage drops and currents based on alpha and beta factors
def gamma_model(weight, x, parasiticResistance):
    input_size, output_size = weight.shape
    G = weight
    I = G * torch.max(x)
    # Calculate g_bit and g_word values
    g_bit = g_word = 1 / parasiticResistance
    # Compute average conductances per row and column
    gAverageRow, gAverageCol = G.mean(dim=1), G.mean(dim=0)
    G_mean = G.mean()
    # Compute constants A, B, and X
    A, B, X = (input_size - 2) / input_size, 2 / input_size, (input_size * (input_size - 1)) / 2 * G_mean
    g = 1 / parasiticResistance
    # Calculate alpha_gm_0
    Summation_bit = torch.sum(torch.arange(input_size, 0, -1, device=weight.device) * gAverageRow)
    alpha_gm_0 = 1 / (1 + Summation_bit / g_bit)
    # Calculate beta_gm_last
    Summation_word = torch.sum(torch.arange(1, output_size + 1, device=weight.device) * gAverageCol[torch.arange(1, output_size + 1, device=weight.device) - 1])
    beta_gm_last = 1 / (1 + Summation_word / g_word)
    NIR1_n = alpha_gm_0 * beta_gm_last
    # Initialize accumulation matrices
    I_aacc1, I_aacc2, I_betaacc1, I_betaacc2 = (torch.zeros((input_size, output_size), device=weight.device) for _ in range(4))
    # Accumulate currents
    I_aacc1[1:, :] = torch.cumsum(I[:-1, :], dim=0)
    I_aacc2[1:, :] = torch.cumsum(I_aacc1[1:, :], dim=0)
    I_betaacc1[:, :] = torch.cumsum(I.flip(dims=[1]), dim=1).flip(dims=[1])
    I_betaacc2[:, :-1] = torch.cumsum(I_betaacc1.flip(dims=[1])[:, 1:], dim=1).flip(dims=[1])
    # Calculate gamma, gamma_a, and gamma_b
    gamma = torch.clamp((A * X * g * torch.sqrt(NIR1_n) + A * (X ** 2) * torch.sqrt(NIR1_n)) / (g * (g - g * torch.sqrt(NIR1_n) + A * X - A * X * torch.sqrt(NIR1_n) - B * X * torch.sqrt(NIR1_n))), min=0)
    gamma_a = (((input_size - 1) / (input_size + 1)) * (g + (input_size * (input_size + 1) / 2) * G_mean) + (2 / (input_size + 1)) * g * gamma) / (g + ((input_size - 1) / (input_size + 1)) * (input_size * (input_size + 1) / 2) * G_mean)
    gamma_b = (((output_size - 1) / (output_size + 1)) * (g + (output_size * (output_size + 1) / 2) * G_mean) + (2 / (output_size + 1)) * g * gamma) / (g + ((output_size - 1) / (output_size + 1)) * (output_size * (output_size + 1) / 2) * G_mean)
    # Calculate alpha and beta
    alpha_numerator = I[0, :] * g_bit * gamma + I_aacc2 * G[0, :]
    alpha_denominator = I[0, :] * g_bit * gamma + I_aacc2[input_size - 1, :] * G[0, :] * gamma_a
    alpha_gm = alpha_numerator / alpha_denominator
    beta_numerator = I[:, output_size - 1] * g_word * gamma + I_betaacc2.T * G[:, output_size - 1]
    beta_denominator = I[:, output_size - 1] * g_word * gamma + I_betaacc2[:, 0] * G[:, output_size - 1] * gamma_b
    beta_gm = beta_numerator / beta_denominator
    # Compute voltage drops and currents
    V_a_matrix = x.view(-1, 1).repeat(1, output_size)
    voltage_drops_gamma = alpha_gm * V_a_matrix * beta_gm.T
    current_gamma = torch.matmul(x, (alpha_gm * G * beta_gm.T))
    return current_gamma

# Memtorch solve_passive model
# Uses memtorch_bindings to solve passive networks and obtain voltage drops and currents
def solve_passive_model(weight, x, parasiticResistance):
    return memtorch_bindings.solve_passive(
        weight,
        x,
        torch.zeros(weight.shape[0]),
        parasiticResistance,
        parasiticResistance,
        n_input_batches=x.shape[0]
    )

# Ideal model implementation
# Computes currents based on ideal weight and input conditions
def IdealModel(weight, x, parasiticResistance):
    if len(x.shape) == 1:
        x = x.unsqueeze(0)  # Add batch dimension if input is a single sample
    return torch.matmul(x, weight)  # Supports batched input
