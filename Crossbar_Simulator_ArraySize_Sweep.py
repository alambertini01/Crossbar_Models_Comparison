import numpy as np
import matplotlib.pyplot as plt
import time
import datetime
import pytz
#Import NonLinear functions
from CrossbarModels.Functions.NonLinear import resistance_array_to_x
from CrossbarModels.Functions.NonLinear import calculate_resistance
#Import the Crossbar Models
from CrossbarModels.Crossbar_Models import *

# Use this script to compare the performance (time and accuracy) of different parasitc resistance corsbar models


############################ PARAMETERS ##############################

# Dimensions of the crossbar
array_size = np.arange(8,130,8)

# Initialize each model instance
Models = [
    JeongModel("Jeong"),
    IdealModel("Ideal"),
    DMRModel("DMR_old"),
    DMRModel_acc("DMR"),
    GammaModel("Gamma_old"),
    GammaModel_acc("Gamma_acc_v1"),
    GammaModel_acc_v2("Gamma"),
    CrossSimModel("CrossSim"),
    LTSpiceModel("LTSpice"),
    NgSpiceModel("NgSpice"),
    NgSpiceNonLinearModel("NgSpiceNonLinear"),
    MemtorchModelCpp("Memtorch_cpp"),
    MemtorchModelPython("Memtorch_python")
]


# enabled_models = [ "Ideal","DMR_acc","Gamma_acc", "CrossSim","Memtorch_cpp","Memtorch_python","NgSpice"]
# enabled_models = [model.name for model in Models]
enabled_models = [ "Ideal","Jeong","Gamma","DMR","CrossSim","Memtorch_cpp"]

reference_model = "NgSpice"

# Low resistance proggramming value
R_lrs = 1000
Rhrs_percentage=50
# parasitic resistance value
parasiticResistance = np.arange(0.1, 10.1, 0.5)
parasiticResistance = np.array([2])

# Memory window (ratio between Hrs and Lrs)
memoryWindow = np.arange(10, 100.1, 10)
memoryWindow = np.array([20])

# Input voltages parameters
v_On_percentage = 100
population = [1, 0.0]

# Mse type (1=Current, 0=Voltage)
current_mse = 1

# Variability parameters
v_flag = 1
v_size = 5

enabled_models.append(reference_model)
modelSize = len(enabled_models)
# Initialize time measurements
simulation_times = np.zeros((modelSize,np.size(array_size)))



memorySize = np.size(memoryWindow)
parasiticSize = np.size(parasiticResistance)
# Initialize mse array
mse = np.zeros((parasiticSize ,memorySize, np.size(array_size), modelSize))



for d in range(np.size(array_size)):

    input = output = array_size[d]

    ############################ INITIALIZATIONS ############################


    totalIterations = memorySize*parasiticSize*modelSize

    # Tech 4 parameters
    rho=3000
    tox=5
    s0=10.62
    s=rho*tox/R_lrs
    v_ds = 0.35
    v_dx = 0.7

    # Building the Resistance matrices
    R = np.zeros((np.size(parasiticResistance),memorySize, v_size, input, output))
    X = np.zeros((np.size(parasiticResistance),memorySize, v_size, input, output))
    S = np.ones((np.size(parasiticResistance),memorySize, v_size, input, output))

    # Assuming 'input', 'output', and 'v_size' are defined elsewhere
    for m in range(np.size(memoryWindow)):                  # Iterating over memoryWindow
        x_thickness = resistance_array_to_x(np.array([R_lrs*memoryWindow[m]]))
        for z in range(np.size(parasiticResistance)):       # Iterating over parasiticResistance
            for v in range(v_size):                         # iterating over variability 'v'
                Rstate = np.random.choice([1, 0], size=(input, output), p=[Rhrs_percentage/100, 1 - Rhrs_percentage/100])
                # Rstate[5,10]=0
                X[z,m,v] = Rstate*x_thickness+10e-13
                S[z,m,v] = (1-Rstate)*s + Rstate*s0
                if v_flag:
                    # barrier thickness variability for non linear case
                    X[z,m,v] += (np.abs(np.random.randn(input,output)*v_dx/3))*Rstate
                    S[z,m,v] += np.random.randn(input,output)*v_ds/3
                    # Calculate the resistance with the variability R[z, m, v]
                    R[z, m, v] = calculate_resistance(X[z,m,v], S[z,m,v])
                else:
                    R[z,m,v] = Rstate*R_lrs*memoryWindow[m]+(1-Rstate)*R_lrs

    # Generate Potential vector
    Potential = np.random.choice(population, size=input, p=[v_On_percentage / 100, 1 - v_On_percentage / 100])

    reference_index = enabled_models.index(reference_model)

    # Output Currents (for each Parasitic Model)
    output_currents = np.zeros((output, parasiticSize ,memorySize, modelSize))

    # Voltage Drops (for each Parasitic Model)
    V_a_matrix = np.tile(Potential, output)
    voltage_drops = np.zeros((input,output, parasiticSize ,memorySize, modelSize))



    print(str(d)+"th Simulation of crossbar models with array size of: "+str(array_size[d]))

    ############################ MAIN LOOP ############################

    for m in range(memorySize):

        for z in range(np.size(parasiticResistance)):

            for v in range(v_size):

                for index, model in enumerate(Models):
                    if model.name in enabled_models:  # Check if the model is enabled
                        index = enabled_models.index(model.name)
                        NonLinear_params = {'X': X[z,m,v], 'S': S[z,m,v]} if model.name == 'NgSpiceNonLinear' else {}

                        start_time = time.perf_counter()
                        voltage_drops[:, :, z, m, index], output_currents[:, z, m, index] = model.calculate(R[z, m, v], parasiticResistance[z], Potential, **NonLinear_params)
                        # output_currents[:, z, m, index] = np.cumsum(voltage_drops[:,:,z,m, index]*np.reciprocal(R[z,m,v]),axis=0)[input-1,:]
                        end_time = time.perf_counter()
                        simulation_times[index,d] += (end_time - start_time)/totalIterations
                        
                for index, model in enumerate(enabled_models):
                    if current_mse:
                        # Compute Output Current MSE
                        mse[z, m, d,index] += np.mean(np.abs(output_currents[:, z, m, reference_index] - output_currents[:, z, m, index] ) / output_currents[:, z, m, reference_index])*100/v_size
                    else:
                        # Compute Voltage MSE
                        mse[z, m, d,index] += np.mean(np.abs((voltage_drops[:,:,z,m,reference_index] - voltage_drops[:,:,z,m,index]))/voltage_drops[:,:,z,m,reference_index])*100/v_size




###################### Plotting portion ##########################################################################

# plot parameters
colors = ['c','r', 'g', 'b', 'orange', 'purple', 'pink', 'brown', 'lime', 'teal']
markers = ['o', 's', 'D', '^', 'v', 'p']

# Results Folder
end = datetime.datetime.now(pytz.timezone('Europe/Rome'))
folder = 'Results/'+str(end.year)+ str(end.month)+  str(end.day) + '_'+ str(end.hour) +'_'+ str(end.minute)+"_ArraySize_Sweep"
if not (os.path.exists(folder)):
        os.makedirs(folder)

# Different labels based on the used metric
if current_mse:
    error_label = "Normalized Output Current Error (%)"
else:
    error_label = "Normalized Voltage Drops Error (%)"

normalized_simulation_times =np.zeros_like(simulation_times)

ideal_index = enabled_models.index("Ideal")
plot_models = [model for i, model in enumerate(enabled_models) if i != ideal_index]


# Plotting
plt.figure()
for index, model in enumerate(plot_models):
    normalized_simulation_times = simulation_times[index+1,:] / simulation_times[ideal_index,:]
    plt.plot(array_size, normalized_simulation_times, marker=markers[index % len(markers)], color=colors[index % len(colors)], label=model)
# Log scale for y-axis as in the example image
plt.yscale('log')
plt.xlabel("Array Size")
plt.ylabel("Normalized Processing Times relative to the Ideal Model")
plt.title("Normalized Processing Times vs. Array Size")
# Get handles and labels, then reverse them
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles[::-1], labels[::-1])  # Reverses both handles and labels
plt.grid(True, which="both", linestyle='--', linewidth=0.5)
plt.savefig(folder+'/Figure_SimulationTimes_vs_ArraySize.png')
# Show the plot
plt.show()


# Plotting
plt.figure()
for index, model in enumerate(enabled_models):
    if model != reference_model:
        np.mean(mse, axis=(0, 1))[:,index]
        plt.plot(array_size, np.mean(mse, axis=(0, 1))[:,index], marker=markers[index % len(markers)], color=colors[index % len(colors)], label=model)
# Log scale for y-axis as in the example image
plt.yscale('log')
plt.xlabel("Array Size")
plt.ylabel(error_label)
plt.title(error_label+" vs Array Size")
plt.legend()
plt.grid(True, which="both", linestyle='--', linewidth=0.5)
plt.savefig(folder+'/Figure_Error_vs_ArraySize.png')
# Show the plot
plt.show()