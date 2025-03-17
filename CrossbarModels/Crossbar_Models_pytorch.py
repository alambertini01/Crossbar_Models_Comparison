import torch
from torch.autograd import Variable
try:
    from .Models import memtorch_bindings # type: ignore
except ImportError as e:
    print(f"Warning: Could not import memtorch_bindings: {e}")

# Jeong model implementation
# Computes voltage drops and currents for given weights and inputs
def jeong_model_mod(weight, x, parasiticResistance, **kwargs):
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


def jeong_model(weight, x, parasiticResistance, R_lrs, R_hrs, k=0.2, epsilon=1e-10):

    device = weight.device
    dtype = weight.dtype
    input_size, output_size = weight.shape
    # -----------------------------
    # Compute the Jeong pre-factors:
    # -----------------------------
    # For the word-line (columns) contribution: create a descending sequence  [output_size, ..., 1]
    weights_wl = torch.arange(output_size, 0, -1, dtype=dtype, device=device)
    A_jeong = parasiticResistance * torch.cumsum(weights_wl, dim=0)  # shape: (output_size,)
    # For the bit-line (rows) contribution: create a descending sequence [input_size, ..., 1]
    # Here, B_jeong is a scalar: parasiticResistance times the sum of the weights.
    weights_bl = torch.arange(input_size, 0, -1, dtype=dtype, device=device)
    B_jeong = parasiticResistance * torch.sum(weights_bl)  # scalar
    # -----------------------------
    # Compute the adaptive resistance average (Rd_avg):
    # -----------------------------
    a = R_lrs**(-k)
    b = R_hrs**(-k)
    Rd_avg = (a * R_lrs + b * R_hrs) / (a + b)  # scalar (or 0D tensor)
    Va = 0.5
    # I_appr: approximated current per column. Note that input_size plays the role of "number of rows".
    # A_jeong is (output_size,), Rd_avg and B_jeong are scalars; broadcasting gives I_appr a shape of (batch_size, output_size).
    I_appr = input_size * Va * torch.reciprocal(A_jeong + Rd_avg + B_jeong + epsilon)
    # I_ideal_avg: the average ideal current per sample.
    I_ideal_avg = input_size * Va / (Rd_avg + epsilon)
    # I_ideal: computed as a dot product between the input potentials and the reciprocal of the weight matrix.
    I_ideal = torch.matmul(x, weight+epsilon)  # shape: (batch_size, output_size)
    # Scale the ideal current with the approximated factors.
    current_jeong = I_ideal * I_appr / I_ideal_avg  # shape: (batch_size, output_size)

    return current_jeong



def dmr_model(weight: torch.Tensor, x: torch.Tensor, parasiticResistance: float, **kwargs):
    """
    PyTorch adaptation of DMRModel_acc for batched inputs.
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



def alpha_beta_model(weight, x, parasiticResistance, return_voltages=False, **kwargs):
    """
    PyTorch adaptation of the alpha-beta model for batched inputs.
    Can return either output currents (2D) or voltage drops (3D) based on flag.

    Parameters
    ----------
    weight : torch.Tensor
        The conductance matrix of shape (input_size, output_size).
    x : torch.Tensor
        The batched input voltages of shape (batch_size, input_size). Assumed to be batch_size=1 and input_size.
    parasiticResistance : float
        The parasitic resistance value (used to compute parasitic conductances g_bit, g_word).
    return_voltages : bool, optional
        If True, returns voltage drops (3D tensor). If False (default), returns output currents (2D tensor).

    Returns
    -------
    current_gamma : torch.Tensor or dV_alpha_beta : torch.Tensor
        Output currents of shape (batch_size, output_size) if return_voltages is False.
        Voltage drops of shape (batch_size, input_size, output_size) if return_voltages is True.
    """

    # weight is our G matrix of shape (input_size, output_size)
    # x is our input of shape (batch_size, input_size). Here we expect x to be (1, input_size)

    device = weight.device
    dtype = weight.dtype

    input_size, output_size = weight.shape

    # Parasitic conductances
    g_bit = 1.0 / parasiticResistance
    g_word = 1.0 / parasiticResistance

    # Equivalent of: G = 1.0 / R, but here weight is already G
    G = weight

    # To mimic the original code's "I = G * np.max(Potential)",
    # we'll take the maximum value from the input x (assuming batch_size=1).
    max_input = x.max()  # scalar if x is (1, input_size) or (input_size,)
    I = G * max_input  # shape: (input_size, output_size)

    # ---------------------------------------------------------------------
    # 1) Compute the column-wise cumsums (for alpha)
    #    cumsumI_cols and cumsumK_I_cols both have shape (input_size+1, output_size).
    # ---------------------------------------------------------------------
    cumsumI_cols = torch.zeros(input_size + 1, output_size, device=device, dtype=dtype)
    cumsumK_I_cols = torch.zeros(input_size + 1, output_size, device=device, dtype=dtype)

    cumsumI_cols[1:] = torch.cumsum(I, dim=0)  # cumsum down columns (axis=0)
    # We'll create a row-index vector [0, 1, ..., input_size-1] for weighting
    i2D = torch.arange(input_size, device=device, dtype=dtype).unsqueeze(1)  # shape: (input_size, 1)
    cumsumK_I_cols[1:] = torch.cumsum(i2D * I, dim=0)

    denomSum = input_size * cumsumI_cols[input_size, :] - cumsumK_I_cols[input_size, :]

    topAlpha = (
        I[0, :] * g_bit
        + G[0, :] * (i2D * cumsumI_cols[:input_size, :] - cumsumK_I_cols[:input_size, :])
    )

    botAlpha = (
        I[0, :] * g_bit
        + G[0, :] * denomSum
    )

    alpha_gm = topAlpha / botAlpha  # broadcasting -> shape: (input_size, output_size)

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

    denomBeta = I_colEnd * g_word + G_colEnd * (cPI_end + cI_end)

    j2D = j_vals.unsqueeze(0)  # shape: (1, output_size)
    partialSumBeta = (
        (cPI_end.unsqueeze(1) - cumsumP_I_rows[:, :output_size])
        - j2D * (cI_end.unsqueeze(1) - cumsumI_rows[:, :output_size])
    )

    numBeta = I_colEnd.unsqueeze(1) * g_word + G_colEnd.unsqueeze(1) * partialSumBeta
    # shape: (input_size, output_size)

    beta_gm = numBeta / denomBeta.unsqueeze(1)  # shape: (input_size, output_size)

    if return_voltages:
        # Compute voltage drops: dV_alpha_beta of shape (batch_size, input_size, output_size)
        dV_alpha_beta = x.unsqueeze(2).expand(-1, input_size, output_size) * (alpha_gm * beta_gm).unsqueeze(0)
        return dV_alpha_beta
    else:

        current_gamma = x @ (alpha_gm * G * beta_gm)
        return current_gamma



def Fused_DMR_Jeong(weight, x, parasiticResistance, R_lrs, R_hrs, k=0.2, epsilon=1e-10):

    # Get currents from both models
    current_jeong = jeong_model(weight, x, parasiticResistance, R_lrs, R_hrs, k, epsilon)
    current_dmr = dmr_model(weight, x, parasiticResistance)

    # Calculate array size for weighting
    input_size, output_size = weight.shape
    array_size = input_size + output_size

    # Define tau for the charging exponential
    tau = 160  # Adjusted tau to get ~63% at array_size 160

    # Calculate weights using charging exponential
    weight_jeong_model = 1.0 - torch.exp(torch.tensor(-array_size / tau))
    weight_dmr_model = 1.0 - weight_jeong_model # or torch.exp(torch.tensor(-array_size / tau))

    # Ensure weights are on the same device as currents if needed, though scalar weights usually broadcast well.
    weight_jeong_model = weight_jeong_model.to(current_jeong.device)
    weight_dmr_model = weight_dmr_model.to(current_dmr.device)

    # Perform weighted average
    fused_current = weight_dmr_model * current_dmr + weight_jeong_model * current_jeong

    return fused_current


# Memtorch solve_passive model
# Uses memtorch_bindings to solve passive networks and obtain voltage drops and currents
def Memtorch_model(weight, x, parasiticResistance, **kwargs):
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
def IdealModel(weight, x, parasiticResistance, **kwargs):
    if len(x.shape) == 1:
        x = x.unsqueeze(0)  # Add batch dimension if input is a single sample
    return torch.matmul(x, weight)  # Supports batched input



# CrossSim model implementation
# Iteratively solves for voltage and current using parasitic resistance
def crosssim_model(weight, x, parasiticResistance, Verr_th=1e-3, hide_convergence_msg=0, **kwargs):
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

        # Use alpha_beta model to get a better initial guess for dV
        # dV_alpha_beta = alpha_beta_model(matrix, vector[0,:].unsqueeze(0), parasiticResistance, return_voltages=True) # use first batch input as representative for alpha-beta and get voltages
        # dV = dV_alpha_beta # Initialize dV with voltage drops from alpha-beta model

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
    initial_gamma = min(0.9, 50 / (input_size + output_size) / parasiticResistance)  # Save the initial gamma
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