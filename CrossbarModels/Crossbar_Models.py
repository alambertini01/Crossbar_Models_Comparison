import numpy as np
import os
import torch
#Import LTSpice functions
from .Functions.Spice_Functions import ngSpice_Sim
from .Functions.Spice_Functions import LTSpice_Sim
from .Functions.Spice_Functions import Create_Structure
from .Functions.Spice_Functions import Create_NonLinear_Structure
#Import NonLinear functions
from .Functions.NonLinear import create_rram_model
# Import Memtorch Solver
from .Models.MemTorch_float import memtorch_bindings as memtorch_bindings_float # type: ignore
from .Models import memtorch_bindings# type: ignore
# Import CrossSim Solver
from .Models.CrossSimModel import CrossSim_Solve

class CrossbarModel:
    def __init__(self, name):
        self.name = name

    def calculate(self, R, parasiticResistance, Potential, **kwargs):
        raise NotImplementedError("Subclasses should implement this method")


######################################################################
#                             IDEAL MODEL                            #
######################################################################

class IdealModel(CrossbarModel):
    def calculate(self, R, parasiticResistance, Potential, **kwargs):

        voltage_ideal = np.tile(Potential.reshape(-1, 1), R.shape[1])
        current_ideal = Potential@np.reciprocal(R)
        return voltage_ideal, current_ideal


######################################################################
#                          ANALYTICAL MODELS                         #
######################################################################

class JeongModel(CrossbarModel):
    def calculate(self, R, parasiticResistance, Potential, **kwargs):
        
        input, output = R.shape
        # Precompute cumulative sums using cumsum
        weights = np.arange(output, 0, -1, dtype=float)
        cumsum_weights = np.cumsum(weights)
        A_jeong = parasiticResistance * cumsum_weights

        cumsum_weights = np.cumsum(weights[::-1])[::-1]
        B_jeong = parasiticResistance * cumsum_weights

        A_jeong_matrix = np.tile(A_jeong, (input, 1))  # Repeat A_jeong across rows (input axis)
        B_jeong_matrix = np.tile(B_jeong.reshape(-1, 1), (1, output))  # Repeat B_jeong across columns (output axis)

        V_a_matrix = np.tile(Potential.reshape(-1, 1), output)

        voltage_drops_jeong = (R*(np.reciprocal(A_jeong_matrix+R+B_jeong_matrix))) * V_a_matrix
        current_jeong = Potential @ (np.reciprocal(A_jeong_matrix+R+B_jeong_matrix))
        
        return voltage_drops_jeong, current_jeong


class JeongModel_avg(CrossbarModel):
    def __init__(self, name, k = 0.95):
        super().__init__(name)
        self.k = k

    def calculate(self, R, parasiticResistance, Potential, R_lrs, MW,**kwargs):
        
        input, output = R.shape
        # Precompute cumulative sums using cumsum
        weights_wl = np.arange(output, 0, -1, dtype=float)
        A_jeong = parasiticResistance * np.cumsum(weights_wl)
        weights_bl = np.arange(input, 0, -1, dtype=float)
        B_jeong = parasiticResistance * np.sum(weights_bl)

        # Calculate Rd_avg (the average resistance)
        R_hrs = R_lrs*MW
        a = R_lrs**(-self.k)
        b = R_hrs**(-self.k)
        Rd_avg = (a * R_lrs + b * R_hrs) / (a + b)

        V_a_matrix = np.tile(Potential.reshape(-1, 1), output)
        Va = np.mean(Potential)

        voltage_drops_jeong = V_a_matrix

        I_appr= input*Va*np.reciprocal(A_jeong+Rd_avg+B_jeong)
        I_ideal_avg = input*Va/Rd_avg
        I_ideal = Potential@np.reciprocal(R)

        current_jeong = I_ideal*I_appr/I_ideal_avg

        return voltage_drops_jeong, current_jeong
    

class JeongModel_adaptive(CrossbarModel):
    def calculate(self, R, parasiticResistance, Potential, R_lrs, MW, Rhrs_percentage, **kwargs):
        
        k= np.interp(Rhrs_percentage, [10, 100], [1.3, 0.4])
        input, output = R.shape
        # Precompute cumulative sums using cumsum
        weights_wl = np.arange(output, 0, -1, dtype=float)
        A_jeong = parasiticResistance * np.cumsum(weights_wl)
        weights_bl = np.arange(input, 0, -1, dtype=float)
        B_jeong = parasiticResistance * np.sum(weights_bl)

        # Calculate Rd_avg (the average resistance)
        R_hrs = R_lrs*MW
        a = R_lrs**(-k)
        b = R_hrs**(-k)
        Rd_avg = (a * R_lrs + b * R_hrs) / (a + b)

        V_a_matrix = np.tile(Potential.reshape(-1, 1), output)
        Va = np.mean(Potential)
        
        voltage_drops_jeong = V_a_matrix

        I_appr= input*Va*np.reciprocal(A_jeong+Rd_avg+B_jeong)
        I_ideal_avg = input*Va/Rd_avg
        I_ideal = Potential@np.reciprocal(R)

        current_jeong = I_ideal*I_appr/I_ideal_avg

        return voltage_drops_jeong, current_jeong
    

class DMRModel(CrossbarModel):
    def calculate(self, R, parasiticResistance, Potential, **kwargs):

        input, output = R.shape
        G = np.reciprocal(R)
        g_bit = 1/parasiticResistance
        g_word = 1/parasiticResistance

        a_dmr = [0]*input
        b_dmr = [0]*output
        
        gAverageRow = G.mean(1)
        gAverageCol = G.mean(0)
        
        for j in range(input):
            Summation_bit = np.sum([ (input+1-(k+1))*gAverageRow[k]  for k in range(input)])
            Summation1 = np.sum([(j-k)*gAverageRow[k] for k in range(j)])
            a_dmr[j] = (1 + Summation1/g_bit)/(1+Summation_bit/g_bit)

        for j in range(output):
            Summation_word = np.sum([k*gAverageCol[k-1] for k in range(1,output+1)])   
            Summation2 = np.sum([(k-j)*gAverageCol[k] for k in range(j,output)])
            b_dmr[j] = (1+ Summation2/g_word)/(1+Summation_word/g_word)
        
        # Creation of the diag matrices A and B with the vector a and b
        A_dmr = np.diag(a_dmr)
        B_dmr = np.diag(b_dmr)
        W_dmr = A_dmr.dot(G).dot(B_dmr)

        V_a_matrix = np.tile(Potential.reshape(-1, 1), output)

        current_dmr = Potential.dot(W_dmr)
        voltage_drops_dmr = A_dmr @ V_a_matrix @ B_dmr

        return voltage_drops_dmr, current_dmr


class DMRModel_acc(CrossbarModel):
    def calculate(self, R, parasiticResistance, Potential, **kwargs):
        # Shape of R:
        input_, output_ = R.shape

        # Conductances
        G = 1.0 / R  # np.reciprocal(R) is fine, but 1.0/R is usually faster
        
        # Parasitic conductances
        g_bit = 1.0 / parasiticResistance
        g_word = g_bit  # same in the original code
        
        # Row and column means
        gAvRow = G.mean(axis=1)  # shape: (input_,)
        gAvCol = G.mean(axis=0)  # shape: (output_,)
        
        S1 = np.insert(np.cumsum(gAvRow), 0, 0.0)               # length: input_+1
        S2 = np.insert(np.cumsum(np.arange(input_)*gAvRow), 0, 0.0)  # length: input_+1
        
        # Summation_bit = sum_{k=0..input_-1} ((input_-k)*gAvRow[k])
        Summation_bit = input_ * S1[input_] - S2[input_]
        
        jArr = np.arange(input_)
        Summation1 = jArr * S1[jArr] - S2[jArr]  # vector of length input_

        a_dmr = (1.0 + Summation1 / g_bit) / (1.0 + Summation_bit / g_bit)

        T1 = np.insert(np.cumsum(gAvCol), 0, 0.0)                 # length: output_+1
        T2 = np.insert(np.cumsum(np.arange(output_)*gAvCol), 0, 0.0)   # length: output_+1

        Summation_word = T2[output_] + T1[output_]

        jArr2 = np.arange(output_)
        Summation2 = (T2[output_] - T2[jArr2]) - jArr2*(T1[output_] - T1[jArr2])

        # b_dmr[j] = (1 + Summation2_j / g_word) / (1 + Summation_word / g_word)
        b_dmr = (1.0 + Summation2 / g_word) / (1.0 + Summation_word / g_word)

        W_dmr = (a_dmr[:, None] * G) * b_dmr
        
        # current_dmr = Potential @ W_dmr
        current_dmr = Potential @ W_dmr  # shape: (output_,)

        Potential2d = Potential[:, None]  # shape: (input_, 1)
        voltage_drops_dmr = (a_dmr[:, None] * Potential2d) * b_dmr  # shape: (input_, output_)

        return voltage_drops_dmr, current_dmr


class alpha_beta(CrossbarModel):
    def calculate(self, R, parasiticResistance, Potential, **kwargs):

        input, output = R.shape
        G = np.reciprocal(R)
        I = G * np.max(Potential)
        # Initialize alpha and beta matrices
        alpha_gm = np.zeros((input, output))
        beta_gm = np.zeros((input, output))
        g_bit = 1/parasiticResistance
        g_word = 1/parasiticResistance

        # Compute alpha values
        for i in range(input):
            for j in range(output):
                num_alpha = I[0, j] * g_bit + sum((i - k) * I[k, j] for k in range(i)) * G[0, j]
                denom_alpha = I[0, j] * g_bit  + sum((input - k) * I[k, j] for k in range(input)) * G[0, j]
                alpha_gm[i, j] = num_alpha / denom_alpha

        # Compute beta values
        for i in range(input):
            for j in range(output):
                num_beta = I[i, output-1] * g_word + sum((k - 1) * I[i, j + k - 1] for k in range(1, output - j + 1)) * G[i, output-1]
                denom_beta = I[i, output-1] * g_word + sum(k * I[i, k - 1] for k in range(1, output + 1)) * G[i, output-1]
                beta_gm[i, j] = num_beta / denom_beta
        
                
        V_a_matrix = np.tile(Potential.reshape(-1, 1), output)
        voltage_drops_gamma = alpha_gm * V_a_matrix * beta_gm
        current_gamma = Potential @ (alpha_gm * G * beta_gm)

        return  voltage_drops_gamma, current_gamma


class alpha_beta_acc(CrossbarModel):
    def calculate(self, R, parasiticResistance, Potential, **kwargs):

        input_size, output_size = R.shape
        # Conductances
        G = np.reciprocal(R)  # shape: (input_size, output_size)
        I = G * np.max(Potential)  # shape: (input_size, output_size)

        # Parasitic conductances
        g_bit = 1.0 / parasiticResistance
        g_word = 1.0 / parasiticResistance
        cumsumI_cols = np.zeros((input_size+1, output_size))
        cumsumK_I_cols = np.zeros((input_size+1, output_size))
        # Compute the cumsums
        cumsumI_cols[1:] = np.cumsum(I, axis=0)               # shape: (input_size, output_size)
        cumsumK_I_cols[1:] = np.cumsum((np.arange(input_size)[:, None] * I), axis=0)
        denomSum = (
            input_size * cumsumI_cols[input_size, :]  # shape: (output_size,)
            - cumsumK_I_cols[input_size, :]
        )  # shape: (output_size,)
        i2D = np.arange(input_size)[:, None]  # shape: (input_size,1)
        topAlpha = (
            I[0, :] * g_bit  # shape: (output_size,)
            + G[0, :] * (i2D * cumsumI_cols[:input_size, :] - cumsumK_I_cols[:input_size, :])
        )
        botAlpha = (
            I[0, :] * g_bit
            + G[0, :] * denomSum
        )
        alpha_gm = topAlpha / botAlpha  # shape: (input_size, output_size)
        cumsumI_rows = np.zeros((input_size, output_size+1))
        cumsumP_I_rows = np.zeros((input_size, output_size+1))

        cumsumI_rows[:, 1:] = np.cumsum(I, axis=1)          # shape: (input_size, output_size)
        cumsumP_I_rows[:, 1:] = np.cumsum((np.arange(output_size) * I), axis=1)
        I_colEnd = I[:, output_size - 1]     # shape: (input_size,)
        G_colEnd = G[:, output_size - 1]     # shape: (input_size,)
        cPI_end  = cumsumP_I_rows[:, output_size]  # shape: (input_size,)
        cI_end   = cumsumI_rows[:, output_size]     # shape: (input_size,)
        denomBeta = I_colEnd * g_word + G_colEnd * (cPI_end + cI_end)  # shape: (input_size,)
        j2D = np.arange(output_size)[None, :]  # shape: (1, output_size)
        partialSumBeta = (
            (cPI_end[:, None] - cumsumP_I_rows[:, :output_size])
            - j2D * (cI_end[:, None] - cumsumI_rows[:, :output_size])
        ) 
        numBeta = I_colEnd[:, None] * g_word + G_colEnd[:, None] * partialSumBeta
        beta_gm = numBeta / denomBeta[:, None]  # shape: (input_size, output_size)
        V_a_matrix = np.tile(Potential.reshape(-1, 1), (1, output_size))  # (input_size, output_size)
        voltage_drops_gamma = alpha_gm * V_a_matrix * beta_gm  # (input_size, output_size)
        current_gamma = Potential @ (alpha_gm * G * beta_gm)  # shape: (output_size,)

        return voltage_drops_gamma, current_gamma
    

######################################################################
#                          ITERATIVE MODELS                          #
######################################################################


class CrossSimModel(CrossbarModel):
    def __init__(self, name, Verr_th=2e-4, hide_convergence_msg=False):
        super().__init__(name)
        self.Verr_th = Verr_th
        self.hide_convergence_msg = hide_convergence_msg

    def calculate(self, R, parasiticResistance, Potential, **kwargs):
        return CrossSim_Solve(np.reciprocal(R.T),parasiticResistance,Potential,self.Verr_th,self.hide_convergence_msg)


class MemtorchModelCpp(CrossbarModel):
    def calculate(self, R, parasiticResistance, Potential, **kwargs):
         # Call the `solve_passive` function (using the cpp implementation)
        
        conductance_matrix = torch.from_numpy(np.reciprocal(R)).to(torch.float32)
        V_WL = torch.from_numpy(Potential).to(torch.float32)
        V_BL = torch.from_numpy(np.zeros(R.shape[1])).to(torch.float32)
        
        voltage_drops = memtorch_bindings_float.solve_passive(
            conductance_matrix,
            V_WL,
            V_BL,
            parasiticResistance,
            parasiticResistance,
            det_readout_currents=False
        )

        # Convert the output to numpy for plotting
        voltage_drops_np = voltage_drops.cpu().detach().numpy()
        # output_currents_np = output_currents.cpu().detach().numpy()
        output_currents_np = np.sum(voltage_drops_np*np.reciprocal(R),axis=0)

        return voltage_drops_np, output_currents_np
    
class MemtorchModelCpp_double(CrossbarModel):
    def calculate(self, R, parasiticResistance, Potential, **kwargs):
         # Call the `solve_passive` function (using the cpp implementation)
        
        conductance_matrix = torch.from_numpy(np.reciprocal(R))
        V_WL = torch.from_numpy(Potential)
        V_BL = torch.from_numpy(np.zeros(R.shape[1]))
        
        voltage_drops = memtorch_bindings.solve_passive(
            conductance_matrix.double(),
            V_WL,
            V_BL,
            parasiticResistance,
            parasiticResistance,
            det_readout_currents=False
        )
        
        # Convert the output to numpy for plotting
        voltage_drops_np = voltage_drops.cpu().detach().numpy()
        # output_currents_np = output_currents.cpu().detach().numpy()
        output_currents_np = np.sum(voltage_drops_np*np.reciprocal(R),axis=0)
        
        return voltage_drops_np, output_currents_np
    

class NgSpiceModel(CrossbarModel):
    def calculate(self, R, parasiticResistance, Potential, **kwargs):
        
        input, output = R.shape
        folder_models = 'LTSpice/Models/'
        if not (os.path.exists(folder_models)):
                os.makedirs(folder_models)
        real = folder_models + 'real.sp'
        folder_results = 'LTSpice/Models/Sim_Data'
        if not (os.path.exists(folder_results)):
                os.makedirs(folder_results)

        voltage_real = np.zeros((input,output))
        Create_Structure(real, Potential, R, input, output, parasiticResistance)
        voltage_real, current_real = ngSpice_Sim(real, folder_results, folder_models, voltage_real,0)

        return voltage_real, current_real

class NgSpiceNonLinearModel(CrossbarModel):
    def calculate(self, R, parasiticResistance, Potential, X, S):
        
        input, output = R.shape
        folder_models = 'LTSpice/Models/'
        if not (os.path.exists(folder_models)):
                os.makedirs(folder_models)
        RealNonLinear = folder_models + 'NonLinear.sp'
        # Create model in current directory
        model_path = folder_models +  'rram_model.sp'
        create_rram_model(model_path)
        folder_results = 'LTSpice/Models/Sim_Data'
        if not (os.path.exists(folder_results)):
                os.makedirs(folder_results)

        voltage_NonLinear = np.zeros((input,output))
        Create_NonLinear_Structure(RealNonLinear, Potential, input, output, parasiticResistance, X, S)
        voltage_NonLinear, current_NonLinear = ngSpice_Sim(RealNonLinear, folder_results, folder_models,voltage_NonLinear,1)
        return voltage_NonLinear, current_NonLinear
        

class LTSpiceModel(CrossbarModel):
    def calculate(self, R, parasiticResistance, Potential, **kwargs):
        
        input, output = R.shape
        folder_models = 'LTSpice/Models/'
        if not (os.path.exists(folder_models)):
                os.makedirs(folder_models)
        real = folder_models + 'real.cir'

        current_matrix = np.zeros((input,output))
        Create_Structure(real, Potential, R, input, output, parasiticResistance)
        current_matrix, current_real = LTSpice_Sim(real,  current_matrix)

        voltage_real = current_matrix * R
        current_real = np.sum(voltage_real*np.reciprocal(R),axis=0)
        return voltage_real, current_real