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
from .Models.MemTorchModel import solve_passive
from .Models import memtorch_bindings # type: ignore
# Import CrossSim Solver
from .Models.CrossSimModel import CrossSim_Solve

class CrossbarModel:
    def __init__(self, name):
        self.name = name

    def calculate(self, R, parasiticResistance, Potential, **kwargs):
        raise NotImplementedError("Subclasses should implement this method")


class JeongModel(CrossbarModel):
    def calculate(self, R, parasiticResistance, Potential, **kwargs):
        
        input, output = R.shape
        # Precompute cumulative sums using cumsum
        weights = np.arange(output, 0, -1, dtype=float)
        cumsum_weights = np.cumsum(weights)
        A_jeong = parasiticResistance * cumsum_weights

        # Precompute the weights: same decreasing weights as before
        weights = np.arange(input, 0, -1, dtype=float)
        cumsum_weights = np.cumsum(weights[::-1])[::-1]
        B_jeong = parasiticResistance * cumsum_weights
        
        A_jeong_matrix = np.full((input, output), A_jeong)
        B_jeong_matrix = np.full((input, output), B_jeong)
        V_a_matrix = np.tile(Potential.reshape(-1, 1), output)

        voltage_drops_jeong = (R*(np.reciprocal(A_jeong_matrix+R+B_jeong_matrix.T))) * V_a_matrix
        current_jeong = Potential @ (np.reciprocal(A_jeong_matrix+R+B_jeong_matrix.T))
        
        return voltage_drops_jeong, current_jeong


class JeongModel_avg(CrossbarModel):
    def calculate(self, R, parasiticResistance, Potential, **kwargs):
        
        input, output = R.shape
        # Precompute cumulative sums using cumsum
        weights = np.arange(output, 0, -1, dtype=float)
        cumsum_weights = np.cumsum(weights)
        A_jeong = parasiticResistance * cumsum_weights

        # Precompute the weights: same decreasing weights as before
        weights = np.arange(input, 0, -1, dtype=float)
        cumsum_weights = np.cumsum(weights[::-1])[::-1]
        B_jeong = parasiticResistance * cumsum_weights

        # Calculate Rd_avg (the average resistance)
        k = 0.9
        R_lrs = np.min(R)
        R_hrs = np.max(R)
        a = R_lrs**(-k)
        b = R_hrs**(-k)
        Rd_avg = (a * R_lrs + b * R_hrs) / (a + b)
        
        A_jeong_matrix = np.full((input, output), A_jeong)
        B_jeong_matrix = np.full((input, output), B_jeong)
        R_avg_matrix = np.full((input, output), Rd_avg)
        V_a_matrix = np.tile(Potential.reshape(-1, 1), output)

        voltage_drops_jeong = (R_avg_matrix*(np.reciprocal(A_jeong_matrix+R_avg_matrix+B_jeong_matrix.T))) * V_a_matrix
        current_jeong = np.sum(voltage_drops_jeong*np.reciprocal(R_avg_matrix),axis=0)
        
        return voltage_drops_jeong, current_jeong
    
class JeongModel_avgv2(CrossbarModel):
    def calculate(self, R, parasiticResistance, Potential, **kwargs):
        
        input, output = R.shape
        # Precompute cumulative sums using cumsum
        weights = np.arange(output, 0, -1, dtype=float)
        cumsum_weights = np.cumsum(weights)
        A_jeong = parasiticResistance * cumsum_weights

        # Precompute the weights: same decreasing weights as before
        weights = np.arange(input, 0, -1, dtype=float)
        cumsum_weights = np.cumsum(weights[::-1])[::-1]
        B_jeong = parasiticResistance * cumsum_weights

        # Calculate Rd_avg (the average resistance)
        k = 0.9
        R_lrs = 1000
        R_hrs = 20000
        a = R_lrs**(-k)
        b = R_hrs**(-k)
        Rd_avg = (a * R_lrs + b * R_hrs) / (a + b)
        
        A_jeong_matrix = np.full((input, output), A_jeong)
        B_jeong_matrix = np.full((input, output), B_jeong)
        R_avg_matrix = np.full((input, output), Rd_avg)
        V_a_matrix = np.tile(Potential.reshape(-1, 1), output)

        voltage_drops_jeong = (R_avg_matrix*(np.reciprocal(A_jeong_matrix+R_avg_matrix+B_jeong_matrix.T))) * V_a_matrix
        current_jeong = np.sum(voltage_drops_jeong*np.reciprocal(R_avg_matrix),axis=0)
        
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

        input, output = R.shape
        G = np.reciprocal(R)
        g_bit = 1/parasiticResistance
        g_word = 1/parasiticResistance

        a_dmr = [0]*input
        b_dmr = [0]*output
        
        gAverageRow = G.mean(1)
        gAverageCol = G.mean(0)

        weights = np.arange(input, 0, -1)  # [input, input-1, ..., 1]
        Summation_bit = np.sum(weights * gAverageRow)
        # For Summation1, we need cumulative sum up to j for each j
        indices = np.arange(input)
        Summation1 = indices * np.cumsum(gAverageRow) - np.cumsum(indices * gAverageRow)
        a_dmr = (1 + Summation1/g_bit)/(1 + Summation_bit/g_bit)

        # Pre-compute weights for Summation_word
        k_arr = np.arange(1, output + 1)
        Summation_word = np.dot(k_arr, gAverageCol[:output])
        # Create triangular matrix directly using triu
        indices = np.arange(output)
        tri_matrix = np.triu(indices - indices[:, None])
        # Compute Summation2 using matrix multiplication
        Summation2 = tri_matrix @ gAverageCol[:output]
        b_dmr = (1 + Summation2/g_word)/(1 + Summation_word/g_word)
        
        # Creation of the diag matrices A and B with the vector a and b
        A_dmr = np.diag(a_dmr)
        B_dmr = np.diag(b_dmr)
        W_dmr = A_dmr.dot(G).dot(B_dmr)

        V_a_matrix = np.tile(Potential.reshape(-1, 1), output)
        current_dmr = Potential.dot(W_dmr)
        voltage_drops_dmr = A_dmr @ V_a_matrix @ B_dmr

        return voltage_drops_dmr, current_dmr
    

class DMRModel_new(CrossbarModel):
    def calculate(self, R, parasiticResistance, Potential, **kwargs):

        input, output = R.shape
        G = np.reciprocal(R)
        g_bit = 1/parasiticResistance
        g_word = 1/parasiticResistance

        a_dmr = [0]*input
        b_dmr = [0]*output
        
        gAverageRow = G.mean(1)
        gAverageCol = G.mean(0)

        weights = np.arange(input, 0, -1)  # [input, input-1, ..., 1]
        Summation_bit = np.sum(weights * gAverageRow)
        # For Summation1, we need cumulative sum up to j for each j
        indices = np.arange(input)
        Summation1 = indices * np.cumsum(gAverageRow) - np.cumsum(indices * gAverageRow)
        a_dmr = (1 + Summation1/g_bit)/(1 + Summation_bit/g_bit)
        
        # Creation of the diag matrices A and B with the vector a and b
        A_dmr = np.diag(a_dmr)
        B_dmr = np.diag(b_dmr)
        W_dmr = A_dmr.dot(G).dot(B_dmr)

        V_a_matrix = np.tile(Potential.reshape(-1, 1), output)
        current_dmr = Potential.dot(W_dmr)
        voltage_drops_dmr = A_dmr @ V_a_matrix @ B_dmr

        return voltage_drops_dmr, current_dmr




class GammaModel(CrossbarModel):
    def calculate(self, R, parasiticResistance, Potential, **kwargs):

        input, output = R.shape
        G = np.reciprocal(R)
        I = G * np.max(Potential)
        # Initialize alpha and beta matrices
        alpha_gm = np.zeros((input, output))
        beta_gm = np.zeros((input, output))
        g_bit = 1/parasiticResistance
        g_word = 1/parasiticResistance

        gAverageRow = G.mean(1)
        gAverageCol = G.mean(0)

        G_mean=G.mean()
        # Calculate A, B, and X based on input
        A = (input - 2) / input
        B = 2 / input
        X = (input * (input -1)) / 2 * G_mean
        g=1/parasiticResistance

        # Calculate a_dmr[0]
        Summation_bit = np.sum([ (input + 1 - (k + 1)) * gAverageRow[k] for k in range(input)])
        alpha_gm_0 = 1 / (1 + Summation_bit / g_bit)
        # Calculate b_dmr[output-1]
        Summation_word = np.sum([k * gAverageCol[k-1] for k in range(1, output + 1)])
        beta_gm_last = 1 / (1 + Summation_word / g_word)

        NIR1_n = alpha_gm_0*beta_gm_last

        # Gamma
        gamma = np.maximum(
            (A * X * g * np.sqrt(NIR1_n) + A*(X**2)*np.sqrt(NIR1_n)) / (g * (g - g * np.sqrt(NIR1_n) + A * X - A * X * np.sqrt(NIR1_n) - B * X * np.sqrt(NIR1_n))),
            0
        )

        gamma_a = (((input - 1) / (input + 1)) * (g + (input * (input + 1) / 2) * G_mean) + 
                (2 / (input + 1)) * g * gamma) / (
                g + ((input - 1) / (input + 1)) * (input * (input + 1) / 2) * G_mean)

        gamma_b = (((output - 1) / (output + 1)) * (g + (output * (output + 1) / 2) * G_mean) + 
                (2 / (output + 1)) * g * gamma) / (
                g + ((output - 1) / (output + 1)) * (output * (output + 1) / 2) * G_mean)

        # Compute alpha values
        for i in range(input):
            for j in range(output):
                num_alpha = I[0, j] * g_bit * gamma + sum((i - k) * I[k, j] for k in range(i)) * G[0, j]
                denom_alpha = I[0, j] * g_bit * gamma + sum((input - k) * I[k, j] for k in range(input)) * G[0, j] * gamma_a
                alpha_gm[i, j] = num_alpha / denom_alpha

        # Compute beta values
        for i in range(input):
            for j in range(output):
                num_beta = I[i, output-1] * g_word * gamma + sum((k - 1) * I[i, j + k - 1] for k in range(1, output - j + 1)) * G[i, output-1]
                denom_beta = I[i, output-1] * g_word * gamma + sum(k * I[i, k - 1] for k in range(1, output + 1)) * G[i, output-1] * gamma_b
                beta_gm[i, j] = num_beta / denom_beta
                
        V_a_matrix = np.tile(Potential.reshape(-1, 1), output)
        voltage_drops_gamma = alpha_gm * V_a_matrix * beta_gm
        current_gamma = Potential @ (alpha_gm * G * beta_gm)

        return  voltage_drops_gamma, current_gamma


class GammaModel_acc(CrossbarModel):
    def calculate(self, R, parasiticResistance, Potential, **kwargs):

        input, output = R.shape
        G = np.reciprocal(R)
        I = G * np.max(Potential)
        # Initialize alpha and beta matrices
        g_bit = g_word = 1/parasiticResistance

        # Initialize the four matrices with zeros
        I_aacc1 = np.zeros((input, output))
        I_aacc2 = np.zeros((input, output))
        I_betaacc1 = np.zeros((input, output))
        I_betaacc2 = np.zeros((input, output))

        I_aacc1[1:, :] = np.cumsum(I[:-1, :], axis=0)
        I_aacc2[1:, :] = np.cumsum(I_aacc1[1:, :], axis=0)
        I_betaacc1[:, :] = np.cumsum(I[:, ::-1], axis=1)[:, ::-1]
        I_betaacc2[:, :-1] = np.cumsum(I_betaacc1[:, :0:-1], axis=1)[:, ::-1]
     
        alpha_numerator = I[0, :] * g_bit  + I_aacc2 * G[0, :]
        alpha_denominator = I[0, :] * g_bit + I_aacc2[input -1, :] * G[0, :] 
        alpha_gm = alpha_numerator / alpha_denominator

        beta_numerator = I[:, output -1] * g_word + I_betaacc2.T * G[:, output -1]
        beta_denominator = I[:, output -1] * g_word  + I_betaacc2[:, 0] * G[:, output -1]
        beta_gm = beta_numerator / beta_denominator

        V_a_matrix = np.tile(Potential.reshape(-1, 1), output)
        voltage_drops_gamma = alpha_gm * V_a_matrix * beta_gm.T
        current_gamma = Potential @ (alpha_gm * G * beta_gm.T)

        return  voltage_drops_gamma, current_gamma


class GammaModel_acc_v2(CrossbarModel):
    def calculate(self, R, parasiticResistance, Potential, **kwargs):

        input, output = R.shape
        G = np.reciprocal(R)
        I = G * np.max(Potential)
        # Initialize alpha and beta matrices
        alpha_gm = np.zeros((input, output))
        beta_gm = np.zeros((input, output))
        g_bit = 1/parasiticResistance
        g_word = 1/parasiticResistance

        gAverageRow = G.mean(1)
        gAverageCol = G.mean(0)

        G_mean=G.mean()
        # Calculate A, B, and X based on input
        A = (input - 2) / input
        B = 2 / input
        X = (input * (input -1)) / 2 * G_mean
        g=1/parasiticResistance

        # Calculate a_dmr[0]
        weights = np.arange(input, 0, -1)  # [input, input-1, ..., 1]
        Summation_bit = np.sum(weights * gAverageRow)
        alpha_gm_0 = 1 / (1 + Summation_bit / g_bit)
        # Calculate b_dmr[output-1]
        k_arr = np.arange(1, output + 1)
        Summation_word = np.sum(k_arr * gAverageCol[k_arr-1])
        beta_gm_last = 1 / (1 + Summation_word / g_word)

        NIR1_n = alpha_gm_0*beta_gm_last

        # Initialize the four matrices with zeros
        I_aacc1 = np.zeros((input, output))
        I_aacc2 = np.zeros((input, output))
        I_betaacc1 = np.zeros((input, output))
        I_betaacc2 = np.zeros((input, output))

        I_aacc1[1:, :] = np.cumsum(I[:-1, :], axis=0)
        I_aacc2[1:, :] = np.cumsum(I_aacc1[1:, :], axis=0)
        I_betaacc1[:, :] = np.cumsum(I[:, ::-1], axis=1)[:, ::-1]
        I_betaacc2[:, :-1] = np.cumsum(I_betaacc1[:, :0:-1], axis=1)[:, ::-1]

        # Gamma
        gamma = np.maximum(
            (A * X * g * np.sqrt(NIR1_n) + A*(X**2)*np.sqrt(NIR1_n)) / (g * (g - g * np.sqrt(NIR1_n) + A * X - A * X * np.sqrt(NIR1_n) - B * X * np.sqrt(NIR1_n))),
            0
        )

        gamma_a = (((input - 1) / (input + 1)) * (g + (input * (input + 1) / 2) * G_mean) + 
                (2 / (input + 1)) * g * gamma) / (
                g + ((input - 1) / (input + 1)) * (input * (input + 1) / 2) * G_mean)

        gamma_b = (((output - 1) / (output + 1)) * (g + (output * (output + 1) / 2) * G_mean) + 
                (2 / (output + 1)) * g * gamma) / (
                g + ((output - 1) / (output + 1)) * (output * (output + 1) / 2) * G_mean)
     
        alpha_numerator = I[0, :] * g_bit * gamma + I_aacc2 * G[0, :]
        alpha_denominator = I[0, :] * g_bit * gamma + I_aacc2[input -1, :] * G[0, :] * gamma_a
        alpha_gm = alpha_numerator / alpha_denominator

        beta_numerator = I[:, output -1] * g_word * gamma + I_betaacc2.T * G[:, output -1]
        beta_denominator = I[:, output -1] * g_word * gamma + I_betaacc2[:, 0] * G[:, output -1] * gamma_b
        beta_gm = beta_numerator / beta_denominator

        V_a_matrix = np.tile(Potential.reshape(-1, 1), output)
        voltage_drops_gamma = alpha_gm * V_a_matrix * beta_gm.T
        current_gamma = Potential @ (alpha_gm * G * beta_gm.T)

        return  voltage_drops_gamma, current_gamma


class CrossSimModel(CrossbarModel):
    def __init__(self, name, Verr_th=2e-4, hide_convergence_msg=False):
        super().__init__(name)
        self.Verr_th = Verr_th
        self.hide_convergence_msg = hide_convergence_msg

    def calculate(self, R, parasiticResistance, Potential, **kwargs):
        return CrossSim_Solve(np.reciprocal(R.T),parasiticResistance,Potential,self.Verr_th,self.hide_convergence_msg)



class IdealModel(CrossbarModel):
    def calculate(self, R, parasiticResistance, Potential, **kwargs):

        input, output = R.shape
        voltage_ideal = np.tile(Potential.reshape(-1, 1), output)
        current_ideal = np.sum(voltage_ideal*np.reciprocal(R),axis=0)
        return voltage_ideal, current_ideal



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



class MemtorchModelCpp(CrossbarModel):
    def calculate(self, R, parasiticResistance, Potential, **kwargs):
         # Call the `solve_passive` function (using the cpp implementation)
        
        conductance_matrix = torch.from_numpy(np.reciprocal(R)).to(torch.float32)
        V_WL = torch.from_numpy(Potential).to(torch.float32)
        V_BL = torch.from_numpy(np.zeros(R.shape[1])).to(torch.float32)
        
        voltage_drops = memtorch_bindings.solve_passive(
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
    


class MemtorchModelPython(CrossbarModel):
    def calculate(self, R, parasiticResistance, Potential, **kwargs):
         # Call the `solve_passive` function (using the Python implementation)
        voltage_drops, output_currents = solve_passive(
            np.reciprocal(np.flip(R,axis=0)),
            Potential,
            np.zeros(R.shape[0]),
            parasiticResistance,
            parasiticResistance,
        )

        # Convert the output to numpy for plotting
        voltage_drops_np = voltage_drops.cpu().detach().numpy()
        output_currents_np = output_currents.cpu().detach().numpy()

        return voltage_drops_np, output_currents_np