import torch
from .Models import memtorch_bindings # type: ignore
from torch.autograd import Variable

# Jeong model implementation
# Computes voltage drops and currents for given weights and inputs
def jeong_model(weight, x, parasiticResistance):
    epsilon = 1e-8  # Small constant to prevent division by zero
    input_size, output_size = weight.shape
    # batch_size = x.shape[0]
    device = weight.device
    dtype = weight.dtype
    # Calculate A_jeong and B_jeong vectors
    A_indices = torch.arange(output_size, 0, -1, dtype=dtype, device=device)
    B_indices = torch.arange(input_size, 0, -1, dtype=dtype, device=device)
    A_jeong = parasiticResistance * torch.cumsum(A_indices, dim=0)  # Shape: (output_size,)
    B_jeong = parasiticResistance * torch.cumsum(B_indices.flip(0), dim=0).flip(0)  # Shape: (input_size,)
    # Compute the reciprocal of weight safely
    reciprocal_weight = torch.reciprocal(weight + epsilon)  # Shape: (input_size, output_size)
    # Compute the denominator with proper broadcasting
    denominator = reciprocal_weight + B_jeong[:, None] + A_jeong[None, :] + epsilon  # Shape: (input_size, output_size)
    # Compute weight over denominator
    # weight_over_denominator = weight / denominator  # Shape: (input_size, output_size)
    # Reshape tensors to include batch dimension and enable broadcasting
    # V_a_matrix: (batch_size, input_size, 1)
    # V_a_matrix = x.unsqueeze(2)  # Adds a dimension at the end
    # Expand weight_over_denominator to match batch dimensions
    # weight_over_denominator: (1, input_size, output_size)
    # weight_over_denominator_expanded = weight_over_denominator.unsqueeze(0)
    # Compute voltage drops
    # voltage_drops_jeong: (batch_size, input_size, output_size)
    # voltage_drops_jeong = weight_over_denominator_expanded * V_a_matrix  # Broadcasting over batch_size
    # Compute currents
    # reciprocal_denominator: (input_size, output_size)
    reciprocal_denominator = torch.reciprocal(denominator)  # Shape: (input_size, output_size)
    # current_jeong: (batch_size, output_size)
    current_jeong = torch.matmul(x, reciprocal_denominator)  # Batch matrix multiplication
    return current_jeong


# DMR model implementation
# Calculates voltage drops and currents using diagonal matrices A and B
def dmr_model(weight, x, parasiticResistance):
    input_size, output_size = weight.shape
    G = weight  # Conductance matrix

    # Calculate g_bit and g_word values
    g_bit = g_word = 1 / parasiticResistance

    # Compute average conductances per row and column
    gAverageRow = G.mean(dim=1)  # Shape: (input_size,)
    gAverageCol = G.mean(dim=0)  # Shape: (output_size,)

    # Calculate a_dmr values
    indices_row = torch.arange(input_size, device=weight.device, dtype=weight.dtype)
    indices_row_rev = torch.arange(input_size, 0, -1, device=weight.device, dtype=weight.dtype)
    Summation_bit = torch.sum(indices_row_rev * gAverageRow)
    Summation1 = indices_row * torch.cumsum(gAverageRow, dim=0) - torch.cumsum(indices_row * gAverageRow, dim=0)
    a_dmr = (1 + Summation1 / g_bit) / (1 + Summation_bit / g_bit)  # Shape: (input_size,)

    # Calculate b_dmr values
    indices_col = torch.arange(output_size, device=weight.device, dtype=weight.dtype)
    indices_col_1 = torch.arange(1, output_size + 1, device=weight.device, dtype=weight.dtype)
    Summation_word = torch.sum(indices_col_1 * gAverageCol)
    difference_matrix = (indices_col.view(1, -1) - indices_col.view(-1, 1)).clamp(min=0)
    Summation2 = torch.matmul(
        torch.triu(difference_matrix),
        gAverageCol
    )
    b_dmr = (1 + Summation2 / g_word) / (1 + Summation_word / g_word)  # Shape: (output_size,)

    # Create diagonal matrices A and B
    A_dmr = torch.diag(a_dmr)  # Shape: (input_size, input_size)
    B_dmr = torch.diag(b_dmr)  # Shape: (output_size, output_size)

    # Calculate W_dmr
    W_dmr = A_dmr @ G @ B_dmr  # Shape: (input_size, output_size)

    # Compute currents
    current_dmr = x @ W_dmr  # Shape: (batch_size, output_size)

    # Calculate voltage drops (not used in output)
    # voltage_drops_dmr represents the voltage drops across the array considering parasitic resistances
    # V_a_matrix = x  # Shape: (batch_size, input_size)
    # voltage_drops_dmr = V_a_matrix @ A_dmr @ G @ B_dmr  # Shape: (batch_size, output_size)

    return current_dmr





def gamma_model(weight, x, parasiticResistance):
    input_size, output_size = weight.shape  # weight: (input_size, output_size)
    batch_size = x.shape[0]  # x: (batch_size, input_size)
    G = weight  # G: (input_size, output_size)
    x_max = torch.max(x, dim=1, keepdim=True)[0].unsqueeze(2)  # x_max: (batch_size, 1, 1)
    I = G.unsqueeze(0) * x_max  # I: (batch_size, input_size, output_size)
    
    # Calculate g_bit and g_word values
    g_bit = g_word = 1 / parasiticResistance
    g = g_bit
    
    # Compute average conductances per row and column
    gAverageRow = G.mean(dim=1)  # (input_size,)
    gAverageCol = G.mean(dim=0)  # (output_size,)
    G_mean = G.mean()
    
    # Compute constants A, B, and X
    A = (input_size - 2) / input_size
    B = 2 / input_size
    X = (input_size * (input_size - 1)) / 2 * G_mean
    
    # Calculate alpha_gm_0
    indices = torch.arange(input_size, 0, -1, device=weight.device)
    Summation_bit = torch.sum(indices * gAverageRow)
    alpha_gm_0 = 1 / (1 + Summation_bit / g_bit)
    
    # Calculate beta_gm_last
    indices_col = torch.arange(1, output_size + 1, device=weight.device)
    Summation_word = torch.sum(indices_col * gAverageCol)
    beta_gm_last = 1 / (1 + Summation_word / g_word)
    
    NIR1_n = alpha_gm_0 * beta_gm_last  # scalar
    
    # Initialize accumulation matrices
    I_aacc1 = torch.zeros((batch_size, input_size, output_size), device=weight.device)
    I_aacc2 = torch.zeros((batch_size, input_size, output_size), device=weight.device)
    I_betaacc1 = torch.zeros((batch_size, input_size, output_size), device=weight.device)
    I_betaacc2 = torch.zeros((batch_size, input_size, output_size), device=weight.device)
    
    # Accumulate currents
    I_aacc1[:, 1:, :] = torch.cumsum(I[:, :-1, :], dim=1)
    I_aacc2[:, 1:, :] = torch.cumsum(I_aacc1[:, 1:, :], dim=1)
    I_betaacc1 = torch.cumsum(I.flip(dims=[2]), dim=2).flip(dims=[2])
    I_betaacc2[:, :, :-1] = torch.cumsum(I_betaacc1[:, :, 1:].flip(dims=[2]), dim=2).flip(dims=[2])
    
    # Calculate gamma, gamma_a, and gamma_b
    gamma = torch.clamp(
        (A * X * g * torch.sqrt(NIR1_n) + A * X**2 * torch.sqrt(NIR1_n)) /
        (g * (g - g * torch.sqrt(NIR1_n) + A * X - A * X * torch.sqrt(NIR1_n) - B * X * torch.sqrt(NIR1_n))),
        min=0
    )
    # gamma is scalar
    
    gamma_a = (
        ((input_size - 1) / (input_size + 1)) * (g + (input_size * (input_size + 1) / 2) * G_mean) +
        (2 / (input_size + 1)) * g * gamma
    ) / (
        g + ((input_size - 1) / (input_size + 1)) * (input_size * (input_size + 1) / 2) * G_mean
    )
    gamma_b = (
        ((output_size - 1) / (output_size + 1)) * (g + (output_size * (output_size + 1) / 2) * G_mean) +
        (2 / (output_size + 1)) * g * gamma
    ) / (
        g + ((output_size - 1) / (output_size + 1)) * (output_size * (output_size + 1) / 2) * G_mean
    )
    # gamma_a and gamma_b are scalars
    
    # Calculate alpha and beta
    G0 = G[0, :].unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, output_size)
    G0_expanded = G0.expand(batch_size, input_size, output_size)  # Shape: (batch_size, input_size, output_size)
    
    # Expand I[:, 0, :] to match dimensions
    I0 = I[:, 0, :].unsqueeze(1)  # Shape: (batch_size, 1, output_size)
    I0_expanded = I0.expand(batch_size, input_size, output_size)
    
    # Calculate alpha_numerator and alpha_denominator
    alpha_numerator = I0_expanded * g_bit * gamma + I_aacc2 * G0_expanded
    alpha_denominator = I0_expanded * g_bit * gamma + I_aacc2[:, -1:, :] * G0[:, :, :] * gamma_a
    alpha_gm = alpha_numerator / alpha_denominator  # Shape: (batch_size, input_size, output_size)
    
    # Calculate beta_numerator and beta_denominator
    G_last_col = G[:, -1].unsqueeze(0).unsqueeze(2)  # Shape: (1, input_size, 1)
    G_last_col_expanded = G_last_col.expand(batch_size, input_size, output_size)
    
    I_last = I[:, :, -1].unsqueeze(2)  # Shape: (batch_size, input_size, 1)
    I_last_expanded = I_last.expand(batch_size, input_size, output_size)
    
    beta_numerator = I_last_expanded * g_word * gamma + I_betaacc2 * G_last_col_expanded
    beta_denominator = I_last_expanded * g_word * gamma + I_betaacc2[:, :, :1] * G_last_col * gamma_b
    beta_gm = beta_numerator / beta_denominator  # Shape: (batch_size, input_size, output_size)
    
    # Compute voltage drops and currents
    V_a_matrix = x.unsqueeze(2)  # Shape: (batch_size, input_size, 1)
    voltage_drops_gamma = alpha_gm * V_a_matrix * beta_gm  # Voltage drops across the array (not used)
    
    # Compute the current
    current_gamma = torch.sum(alpha_gm * G.unsqueeze(0) * beta_gm * V_a_matrix, dim=1)  # Shape: (batch_size, output_size)
    
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

# CrossSim model implementation
# Iteratively solves for voltage and current using parasitic resistance


def crosssim_model(weight, x, parasiticResistance, Verr_th=1e-2, hide_convergence_msg=0):
    """
    Wrapper that implements a convergence loop around the circuit solver using PyTorch.
    Each solver uses successive under-relaxation.
    If the circuit solver fails to find a solution, the relaxation parameter will
    be reduced until the solver converges, or a lower limit on the relaxation parameter
    is reached (returns a ValueError)
    """
    def mvm_parasitics(vector, matrix, parasiticResistance, gamma, Verr_th):
        """
        Calculates the MVM result including parasitic resistance, for a non-interleaved array using PyTorch.
        vector : input tensor of shape (batch_size, input_dim)
        matrix : normalized conductance tensor of shape (input_dim, output_dim)
        """
        # Parasitic resistance
        Rp_in = Rp_out = parasiticResistance

        Niters_max = 1000

        # Initialize error and number of iterations
        Verr = torch.tensor(1e9, device=vector.device, dtype=vector.dtype)
        Niters = 0

        batch_size, input_dim = vector.shape
        output_dim = matrix.shape[1]

        # Initial estimate of device voltages and currents
        dV0 = vector.unsqueeze(2).expand(batch_size, input_dim, output_dim)  # Shape: (batch_size, input_dim, output_dim)
        dV = dV0.clone()  # Initial device voltages
        Ires = dV * matrix.unsqueeze(0)  # Initial currents, shape: (batch_size, input_dim, output_dim)

        # Iteratively calculate parasitics and update device currents
        while Verr > Verr_th and Niters < Niters_max:
            # Calculate parasitic voltage drops
            Isum_col = torch.cumsum(Ires, dim=1)  # Cumulative sum over input_dim
            Isum_row = torch.cumsum(Ires.flip(dims=[2]), dim=2).flip(dims=[2])  # Cumulative sum over output_dim

            Vdrops_col = Rp_out * torch.cumsum(Isum_col.flip(dims=[1]), dim=1).flip(dims=[1])  # Voltage drops along columns
            Vdrops_row = Rp_in * torch.cumsum(Isum_row, dim=2)  # Voltage drops along rows
            Vpar = Vdrops_col + Vdrops_row  # Total parasitic voltage drop

            # Calculate the error for the current estimate of memristor voltages
            VerrMat = dV0 - Vpar - dV  # Error matrix, shape: (batch_size, input_dim, output_dim)

            # Evaluate overall error
            Verr = torch.max(torch.abs(VerrMat))  # Maximum absolute error over all elements
            if Verr < Verr_th:
                break

            # Update memristor voltages for the next iteration
            dV = dV + gamma * VerrMat  # Update device voltages
            Ires = matrix.unsqueeze(0) * dV  # Update currents
            Niters += 1

        # Check for convergence issues
        if Verr > Verr_th:
            raise RuntimeError("Parasitic resistance too high: could not converge!")
        if torch.isnan(Ires).any():
            raise RuntimeError("Nans due to parasitic resistance simulation")

        # Calculate the summed currents on the columns
        Icols = torch.sum(Ires, dim=1)  # Sum over input_dim, resulting shape: (batch_size, output_dim)

        return Icols

    solved, retry = False, False
    input_size, output_size = weight.shape
    initial_gamma = min(0.9, 20 / (input_size + output_size) / parasiticResistance)  # Save the initial gamma
    gamma = initial_gamma

    while not solved:
        solved = True
        try:
            result = mvm_parasitics(
                x,
                weight.clone(),
                parasiticResistance,
                gamma,
                Verr_th
            )
        except RuntimeError:
            solved, retry = False, True
            gamma *= 0.8
            if gamma <= 1e-4:
                raise ValueError("Parasitic MVM solver failed to converge")
    if retry and not hide_convergence_msg:
        print(
            "CrossSim - Initial gamma: {:.5f}, Reduced gamma: {:.5f}".format(
                initial_gamma,
                gamma,
            )
        )

    return result

