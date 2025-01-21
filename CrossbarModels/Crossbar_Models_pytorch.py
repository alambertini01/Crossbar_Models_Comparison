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
    # reciprocal_denominator: (input_size, output_size)
    reciprocal_denominator = torch.reciprocal(denominator)  # Shape: (input_size, output_size)
    # current_jeong: (batch_size, output_size)
    current_jeong = torch.matmul(x, reciprocal_denominator)  # Batch matrix multiplication
    return current_jeong


def dmr_model(weight: torch.Tensor, x: torch.Tensor, parasiticResistance: float):
    """
    PyTorch adaptation of DMRModel_new for batched inputs.
    Computes only the current (no voltage-drop output).
    
    Parameters
    ----------
    weight : torch.Tensor
        Conductance matrix (G) of shape (input_size, output_size).
    x : torch.Tensor
        Batched input voltages of shape (batch_size, input_size).
    parasiticResistance : float
        Parasitic resistance used to compute g_bit and g_word.

    Returns
    -------
    current_dmr : torch.Tensor
        Output currents of shape (batch_size, output_size).
    """
    device = weight.device
    dtype = weight.dtype
    input_size, output_size = weight.shape

    # Parasitic conductances
    g_bit = 1.0 / parasiticResistance
    g_word = g_bit  # same in the original code

    # G is already the conductance matrix
    G = weight

    # Row & column averages
    gAvRow = G.mean(dim=1)  # shape: (input_size,)
    gAvCol = G.mean(dim=0)  # shape: (output_size,)

    # ---------------------------------------------------------------
    # a_dmr calculation (row-based)
    # ---------------------------------------------------------------
    # We need cumulative sums S1, S2 of length (input_size+1)
    # S1[i] = sum(gAvRow[:i]), S2[i] = sum((k*gAvRow[k]) for k=0..i-1)
    cumsum_gAvRow = torch.cumsum(gAvRow, dim=0)
    S1 = torch.cat([torch.zeros(1, device=device, dtype=dtype), cumsum_gAvRow])  # shape: (input_size+1,)

    # j_tensor for indexing must be long
    j_tensor_long = torch.arange(input_size, device=device, dtype=torch.long)
    # We'll also need j_tensor as float for arithmetic
    j_tensor_float = j_tensor_long.to(dtype=dtype)

    cumsum_j_gAvRow = torch.cumsum(j_tensor_float * gAvRow, dim=0)  # shape: (input_size,)
    S2 = torch.cat([torch.zeros(1, device=device, dtype=dtype), cumsum_j_gAvRow])  # shape: (input_size+1,)

    Summation_bit = input_size * S1[input_size] - S2[input_size]
    Summation1 = j_tensor_float * S1[j_tensor_long] - S2[j_tensor_long]
    a_dmr = (1.0 + Summation1 / g_bit) / (1.0 + Summation_bit / g_bit)  # shape: (input_size,)

    # ---------------------------------------------------------------
    # b_dmr calculation (column-based)
    # ---------------------------------------------------------------
    # T1[i] = sum(gAvCol[:i]), T2[i] = sum((k*gAvCol[k]) for k=0..i-1)
    cumsum_gAvCol = torch.cumsum(gAvCol, dim=0)
    T1 = torch.cat([torch.zeros(1, device=device, dtype=dtype), cumsum_gAvCol])  # shape: (output_size+1,)

    j_tensor2_long = torch.arange(output_size, device=device, dtype=torch.long)
    j_tensor2_float = j_tensor2_long.to(dtype=dtype)

    cumsum_j_gAvCol = torch.cumsum(j_tensor2_float * gAvCol, dim=0)
    T2 = torch.cat([torch.zeros(1, device=device, dtype=dtype), cumsum_j_gAvCol])  # shape: (output_size+1,)

    Summation_word = T2[output_size] + T1[output_size]
    Summation2 = (T2[output_size] - T2[j_tensor2_long]) \
                 - j_tensor2_float * (T1[output_size] - T1[j_tensor2_long])
    b_dmr = (1.0 + Summation2 / g_word) / (1.0 + Summation_word / g_word)  # shape: (output_size,)

    # ---------------------------------------------------------------
    # Construct W_dmr and compute currents for the batch
    # ---------------------------------------------------------------
    W_dmr = (a_dmr.unsqueeze(1) * G) * b_dmr.unsqueeze(0)  # shape: (input_size, output_size)
    current_dmr = x @ W_dmr  # shape: (batch_size, output_size)

    return current_dmr



def gamma_model(weight, x, parasiticResistance):
    epsilon = 1e-8  # Small constant to prevent division by zero
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
    alpha_gm_0 = 1 / (1 + Summation_bit / g_bit + epsilon)  # Added epsilon
    
    # Calculate beta_gm_last
    indices_col = torch.arange(1, output_size + 1, device=weight.device)
    Summation_word = torch.sum(indices_col * gAverageCol)
    beta_gm_last = 1 / (1 + Summation_word / g_word + epsilon)  # Added epsilon
    
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
    numerator_gamma = A * X * g * torch.sqrt(NIR1_n) + A * X**2 * torch.sqrt(NIR1_n)
    denominator_gamma = g * (g - g * torch.sqrt(NIR1_n) + A * X - A * X * torch.sqrt(NIR1_n) - B * X * torch.sqrt(NIR1_n)) + epsilon  # Added epsilon
    gamma = torch.clamp(numerator_gamma / denominator_gamma, min=0)  # gamma is scalar
    
    gamma_a = (
        ((input_size - 1) / (input_size + 1)) * (g + (input_size * (input_size + 1) / 2) * G_mean) +
        (2 / (input_size + 1)) * g * gamma
    ) / (
        g + ((input_size - 1) / (input_size + 1)) * (input_size * (input_size + 1) / 2) * G_mean + epsilon  # Added epsilon
    )
    
    gamma_b = (
        ((output_size - 1) / (output_size + 1)) * (g + (output_size * (output_size + 1) / 2) * G_mean) +
        (2 / (output_size + 1)) * g * gamma
    ) / (
        g + ((output_size - 1) / (output_size + 1)) * (output_size * (output_size + 1) / 2) * G_mean + epsilon  # Added epsilon
    )
    
    # Calculate alpha and beta
    G0 = G[0, :].unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, output_size)
    G0_expanded = G0.expand(batch_size, input_size, output_size)  # Shape: (batch_size, input_size, output_size)
    
    # Expand I[:, 0, :] to match dimensions
    I0 = I[:, 0, :].unsqueeze(1)  # Shape: (batch_size, 1, output_size)
    I0_expanded = I0.expand(batch_size, input_size, output_size)
    
    # Calculate alpha_numerator and alpha_denominator
    alpha_numerator = I0_expanded * g_bit * gamma + I_aacc2 * G0_expanded
    alpha_denominator = I0_expanded * g_bit * gamma + I_aacc2[:, -1:, :] * G0[:, :, :] * gamma_a + epsilon  # Added epsilon
    alpha_gm = alpha_numerator / alpha_denominator  # Shape: (batch_size, input_size, output_size)
    
    # Calculate beta_numerator and beta_denominator
    G_last_col = G[:, -1].unsqueeze(0).unsqueeze(2)  # Shape: (1, input_size, 1)
    G_last_col_expanded = G_last_col.expand(batch_size, input_size, output_size)
    
    I_last = I[:, :, -1].unsqueeze(2)  # Shape: (batch_size, input_size, 1)
    I_last_expanded = I_last.expand(batch_size, input_size, output_size)
    
    beta_numerator = I_last_expanded * g_word * gamma + I_betaacc2 * G_last_col_expanded
    beta_denominator = I_last_expanded * g_word * gamma + I_betaacc2[:, :, :1] * G_last_col * gamma_b + epsilon  # Added epsilon
    beta_gm = beta_numerator / beta_denominator  # Shape: (batch_size, input_size, output_size)
    
    # Compute voltage drops and currents
    V_a_matrix = x.unsqueeze(2).repeat(1, 1, output_size)
    voltage_drops_gamma = alpha_gm * V_a_matrix * beta_gm  # Voltage drops across the array (not used)
    
    # Compute the current using broadcasting
    current_gamma = torch.sum(voltage_drops_gamma * G.unsqueeze(0).repeat(batch_size, 1, 1), dim=1)  # Shape: (batch_size, output_size)
    return current_gamma



def alpha_beta_model(weight, x, parasiticResistance):
    """
    PyTorch adaptation of the alpha-beta model for batched inputs.

    Parameters
    ----------
    weight : torch.Tensor
        The conductance matrix of shape (input_size, output_size).
    x : torch.Tensor
        The batched input voltages of shape (batch_size, input_size).
    parasiticResistance : float
        The parasitic resistance value (used to compute parasitic conductances g_bit, g_word).

    Returns
    -------
    current_gamma : torch.Tensor
        Output currents of shape (batch_size, output_size).
    """

    # weight is our G matrix of shape (input_size, output_size)
    # x is our input of shape (batch_size, input_size)

    device = weight.device
    dtype = weight.dtype

    input_size, output_size = weight.shape

    # Parasitic conductances
    g_bit = 1.0 / parasiticResistance
    g_word = 1.0 / parasiticResistance

    # Equivalent of: G = 1.0 / R, but here weight is already G
    G = weight

    # To mimic the original code's "I = G * np.max(Potential)",
    # we'll take the maximum value over the entire batch+input dimension.
    # (This follows the original logic, which took a single max from Potential.)
    max_input = x.max()  # scalar
    I = G * max_input  # shape: (input_size, output_size)

    # ---------------------------------------------------------------------
    # 1) Compute the column-wise cumsums (for alpha)
    #    cumsumI_cols and cumsumK_I_cols both have shape (input_size+1, output_size).
    # ---------------------------------------------------------------------
    cumsumI_cols = torch.zeros(input_size + 1, output_size, device=device, dtype=dtype)
    cumsumK_I_cols = torch.zeros(input_size + 1, output_size, device=device, dtype=dtype)

    cumsumI_cols[1:] = torch.cumsum(I, dim=0)  # cumsum down columns (axis=0)
    # We'll create a row-index vector [0, 1, ..., input_size-1] for weighting
    i2D = torch.arange(input_size, device=device, dtype=dtype).unsqueeze(1)  # shape: (input_size, 1)
    cumsumK_I_cols[1:] = torch.cumsum(i2D * I, dim=0)

    # denomSum = input_size * cumsumI_cols[input_size, :] - cumsumK_I_cols[input_size, :]
    # shape: (output_size,)
    denomSum = input_size * cumsumI_cols[input_size, :] - cumsumK_I_cols[input_size, :]

    # ---------------------------------------------------------------------
    # 2) Compute alpha_gm
    #    Note that topAlpha is broadcast to shape (input_size, output_size),
    #    and botAlpha is (output_size,), so final alpha_gm is (input_size, output_size).
    # ---------------------------------------------------------------------
    # topAlpha = I[0, :] * g_bit + G[0, :] * (i2D * cumsumI_cols[:input_size, :] - cumsumK_I_cols[:input_size, :])
    # Because i2D has shape (input_size,1), (i2D*cumsumI_cols[:...]) is (input_size, output_size).
    topAlpha = (
        I[0, :] * g_bit
        + G[0, :] * (i2D * cumsumI_cols[:input_size, :] - cumsumK_I_cols[:input_size, :])
    )

    # botAlpha = I[0, :] * g_bit + G[0, :] * denomSum  # shape: (output_size,)
    botAlpha = (
        I[0, :] * g_bit
        + G[0, :] * denomSum
    )

    alpha_gm = topAlpha / botAlpha  # broadcasting -> shape: (input_size, output_size)

    # ---------------------------------------------------------------------
    # 3) Compute the row-wise cumsums (for beta)
    #    cumsumI_rows and cumsumP_I_rows both have shape (input_size, output_size+1).
    # ---------------------------------------------------------------------
    cumsumI_rows = torch.zeros(input_size, output_size + 1, device=device, dtype=dtype)
    cumsumP_I_rows = torch.zeros(input_size, output_size + 1, device=device, dtype=dtype)

    cumsumI_rows[:, 1:] = torch.cumsum(I, dim=1)  # cumsum across rows (axis=1)
    # For partialSumBeta, we weight columns by their index j
    j_vals = torch.arange(output_size, device=device, dtype=dtype)  # shape: (output_size,)
    cumsumP_I_rows[:, 1:] = torch.cumsum(j_vals * I, dim=1)  # broadcasting j_vals as needed

    # We need the last column of I and G for the denominator
    I_colEnd = I[:, -1]  # shape: (input_size,)
    G_colEnd = G[:, -1]  # shape: (input_size,)

    cPI_end = cumsumP_I_rows[:, -1]  # shape: (input_size,)
    cI_end = cumsumI_rows[:, -1]     # shape: (input_size,)

    # denomBeta = I_colEnd * g_word + G_colEnd * (cPI_end + cI_end)
    # shape: (input_size,)
    denomBeta = I_colEnd * g_word + G_colEnd * (cPI_end + cI_end)

    # ---------------------------------------------------------------------
    # 4) Compute beta_gm
    #    partialSumBeta is shape (input_size, output_size).
    # ---------------------------------------------------------------------
    j2D = j_vals.unsqueeze(0)  # shape: (1, output_size)
    # partialSumBeta = ( (cPI_end[:, None] - cumsumP_I_rows[:, :output_size])
    #                  - j2D * (cI_end[:, None] - cumsumI_rows[:, :output_size]) )
    partialSumBeta = (
        (cPI_end.unsqueeze(1) - cumsumP_I_rows[:, :output_size])
        - j2D * (cI_end.unsqueeze(1) - cumsumI_rows[:, :output_size])
    )

    # numBeta = I_colEnd[:, None] * g_word + G_colEnd[:, None] * partialSumBeta
    # shape: (input_size, output_size)
    numBeta = I_colEnd.unsqueeze(1) * g_word + G_colEnd.unsqueeze(1) * partialSumBeta

    beta_gm = numBeta / denomBeta.unsqueeze(1)  # shape: (input_size, output_size)

    # ---------------------------------------------------------------------
    # 5) Compute final currents for each batch:
    #    current_gamma = x @ (alpha_gm * G * beta_gm)
    #    => shape: (batch_size, output_size)
    # ---------------------------------------------------------------------
    current_gamma = x @ (alpha_gm * G * beta_gm)

    return current_gamma



# Memtorch solve_passive model
# Uses memtorch_bindings to solve passive networks and obtain voltage drops and currents
def solve_passive_model(weight, x, parasiticResistance):
    return memtorch_bindings.solve_passive(
        weight.double(),
        x.double(),
        torch.zeros(weight.shape[1]).double(),
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

