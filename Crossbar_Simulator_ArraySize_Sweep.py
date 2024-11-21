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

# Initialize each model instance
Models = [
    JeongModel("Jeong"),
    IdealModel("Ideal"),
    DMRModel("DMR_old"),
    DMRModel_acc("DMR"),
    DMRModel_new("DMR_new"),
    GammaModel("Gamma_old"),
    GammaModel_acc("Gamma_acc_v1"),
    GammaModel_acc_v2("Gamma"),
    CrossSimModel_p1("CrossSim"),
    CrossSimModel_p2("CrossSim2"),
    CrossSimModel_p3("CrossSim3"),
    CrossSimModel("CrossSim4"),
    LTSpiceModel("LTSpice"),
    NgSpiceModel("NgSpice"),
    NgSpiceNonLinearModel("NgSpiceNonLinear"),
    MemtorchModelCpp("Memtorch"),
    MemtorchModelPython("Memtorch_python")
]


# enabled_models = [ "Ideal","DMR_acc","Gamma_acc", "CrossSim","Memtorch_cpp","Memtorch_python","NgSpice"]
# enabled_models = [model.name for model in Models]
enabled_models = ["Ideal", "Jeong","DMR","Gamma","CrossSim"]

reference_model = "CrossSim3"


# Dimensions of the crossbar
array_size = np.arange(8,65,4)

Rhrs_percentage=np.arange(0,100,10)

# parasitic resistance value
parasiticResistance = np.arange(0.1, 5, 0.2)
# parasiticResistance = np.array([2])

# Memory window (ratio between Hrs and Lrs)
memoryWindow = np.arange(10, 100.1, 10)
# memoryWindow = np.array([20])

variabilitySize = 10



# Low resistance proggramming value
R_lrs = 1000

# Input voltages parameters
v_On_percentage = 100
population = [1, 0.0]

# Metric type (1=Current, 0=Voltage)
current_Metric = 1

# Variability parameters
v_flag = 0


enabled_models.append(reference_model)
modelSize = len(enabled_models)
# Initialize time measurements
simulation_times = np.zeros((modelSize,np.size(array_size)))



memorySize = np.size(memoryWindow)
parasiticSize = np.size(parasiticResistance)
sparsitySize = np.size(Rhrs_percentage)
# Initialize Metric array
Current_error = np.zeros((np.size(array_size),parasiticSize ,memorySize, variabilitySize, sparsitySize, modelSize))
Voltage_error = np.zeros((np.size(array_size),parasiticSize ,memorySize, variabilitySize, sparsitySize, modelSize))



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
    R = np.zeros((np.size(parasiticResistance),memorySize, variabilitySize, sparsitySize, input, output))
    X = np.zeros((np.size(parasiticResistance),memorySize, variabilitySize, sparsitySize, input, output))
    S = np.ones((np.size(parasiticResistance),memorySize, variabilitySize, sparsitySize, input, output))

    # Assuming 'input', 'output', and 'variabilitySize' are defined elsewhere
    for m in range(np.size(memoryWindow)):                  # Iterating over memoryWindow
        x_thickness = resistance_array_to_x(np.array([R_lrs*memoryWindow[m]]))
        for z in range(np.size(parasiticResistance)):       # Iterating over parasiticResistance
            for v in range(variabilitySize):                         # iterating over variability 'v'
                for r in range(sparsitySize):
                    Rstate = np.random.choice([1, 0], size=(input, output), p=[Rhrs_percentage[r]/100, 1 - Rhrs_percentage[r]/100])
                    # Rstate[5,10]=1
                    X[z,m,v,r] = Rstate*x_thickness+10e-13
                    S[z,m,v,r] = (1-Rstate)*s + Rstate*s0
                    if v_flag:
                        # barrier thickness variability for non linear case
                        X[z,m,v,r] += (np.abs(np.random.randn(input,output)*v_dx/3))*Rstate
                        S[z,m,v,r] += np.random.randn(input,output)*v_ds/3
                        # Calculate the resistance with the variability R[z, m, v]
                        R[z, m, v,r] = calculate_resistance(X[z,m,v,r], S[z,m,v,r])
                    else:
                        R[z,m,v,r] = Rstate*R_lrs*memoryWindow[m]+(1-Rstate)*R_lrs

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
            for v in range(variabilitySize):
                for r in range(sparsitySize):
                    for index, model in enumerate(Models):
                        if model.name in enabled_models:  # Check if the model is enabled
                            index = enabled_models.index(model.name)
                            NonLinear_params = {'X': X[z,m,v,r], 'S': S[z,m,v,r]} if model.name == 'NgSpiceNonLinear' else {}

                            start_time = time.perf_counter()
                            voltage_drops[:, :, z, m, index], output_currents[:, z, m, index] = model.calculate(R[z, m, v,r], parasiticResistance[z], Potential, **NonLinear_params)
                            # output_currents[:, z, m, index] = np.cumsum(voltage_drops[:,:,z,m, index]*np.reciprocal(R[z,m,v]),axis=0)[input-1,:]
                            end_time = time.perf_counter()
                            simulation_times[index,d] += (end_time - start_time)/totalIterations
                            
                    for index, model in enumerate(enabled_models):
                        # Compute Output Current Metric
                        Current_error[d, z, m, v, r, index] = np.mean(np.abs(output_currents[:, z, m, reference_index] - output_currents[:, z, m, index] ) / output_currents[:, z, m, reference_index])*100
                        # Compute Voltage Metric
                        Voltage_error[d, z, m, v, r, index] = np.mean(np.abs((voltage_drops[:,:,z,m,reference_index] - voltage_drops[:,:,z,m,index]))/voltage_drops[:,:,z,m,reference_index])*100


# Different labels based on the used metric
if current_Metric:
    Metric = Current_error
    error_label = "Normalized Output Current Error (%)"
else:
    Metric = Voltage_error
    error_label = "Normalized Voltage Drops Error (%)"



###################### Plotting portion ##########################################################################

# plot parameters
colors = ['black','c','g', 'r', 'b', 'orange', 'purple', 'pink', 'brown', 'lime', 'teal']
markers = ['o', 's', 'D', '^', 'v', 'p']

# Results Folder
end = datetime.datetime.now(pytz.timezone('Europe/Rome'))
folder = 'Results/'+str(end.year)+ str(end.month)+  str(end.day) + '_'+ str(end.hour) +'_'+ str(end.minute)+"_ArraySize_Sweep"
if not (os.path.exists(folder)):
        os.makedirs(folder)

normalized_simulation_times =np.zeros_like(simulation_times)

ideal_index = enabled_models.index("Ideal")
plot_models = [model for i, model in enumerate(enabled_models) if i != ideal_index]


# Plotting
plt.figure()
for index, model in enumerate(plot_models):
    normalized_simulation_times = simulation_times[index+1,:] / simulation_times[ideal_index,:]
    plt.plot(array_size, normalized_simulation_times, marker=markers[(index+1) % len(markers)], color=colors[(index+1) % len(colors)], label=model)
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


# # Plotting
# plt.figure()
# for index, model in enumerate(enabled_models):
#     if model != reference_model:
#         np.mean(Metric, axis=(0, 1, 3, 3 ,4))[:,index]
#         plt.plot(array_size, np.mean(Metric, axis=( 1, 3, 3 ,4))[:,index], marker=markers[index % len(markers)], color=colors[index % len(colors)], label=model)
# # Log scale for y-axis as in the example image
# plt.yscale('log')
# plt.xlabel("Array Size")
# plt.ylabel(error_label)
# plt.title(error_label+" vs Array Size")
# plt.legend()
# plt.grid(True, which="both", linestyle='--', linewidth=0.5)
# plt.savefig(folder+'/Figure_Error_vs_ArraySize.png')
# # Show the plot
# plt.show()


sum_simulation_times = np.sum(simulation_times,axis=1)
# Take the mean over the first 2 dimensions for Current_error and Voltage_error
current_error_mean = np.mean(Current_error, axis=(0, 1, 2, 3, 4))
voltage_error_mean = np.mean(Voltage_error, axis=(0, 1, 2, 3, 4))
# Compute robustness metrics
# Inverse of standard deviation (higher is better)
robustness_metrics = {
    'Array Size Robustness': np.reciprocal(np.mean(np.std(Metric, axis=0), axis=(0,1,2,3))[:-1]),
    'Parasitic Resistance Robustness': np.reciprocal(np.mean(np.std(Metric, axis=1), axis=(0,1,2,3))[:-1]),
    'Memory Window Robustness': np.reciprocal(np.mean(np.std(Metric, axis=2), axis=(0,1,2,3))[:-1]),
    'Variability Robustness': np.reciprocal(np.mean(np.std(Metric, axis=3), axis=(0,1,2,3))[:-1]),
    'Sparsity Robustness': np.reciprocal(np.mean(np.std(Metric, axis=4), axis=(0,1,2,3))[:-1])
}
# Prepare base metrics
base_metrics = {
    'Current Accuracy': 1 / current_error_mean[:-1],
    'Voltage Accuracy': 1 / voltage_error_mean[:-1],
    'Simulation Speed': 1 / sum_simulation_times[:-1]
}
# Combine metrics, filtering out non-finite values
all_metrics = {**base_metrics, **robustness_metrics}
valid_metrics = {k: v for k, v in all_metrics.items() if np.all(np.isfinite(v))}
# Prepare for plotting
labels = list(valid_metrics.keys())
metrics = np.array(list(valid_metrics.values()))
# Normalize metrics to [0, 1]
metrics_scaled = metrics / metrics.max(axis=1, keepdims=True)
# Compute angles
angles = [n / float(len(labels)) * 2 * np.pi for n in range(len(labels))]
angles += angles[:1]  # Repeat the first angle to close the polygon
# Radar plot setup
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'polar': True})
ax.spines['polar'].set_visible(False)
ax.set_facecolor('#f9f9f9')
ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
ax.set_yticks([])
# Map colors to models
model_colors = colors[:len(enabled_models)]
# Plot for each model
for i, (model_name, color) in enumerate(zip(enabled_models[:-1], model_colors)):
    values = metrics_scaled[:, i].tolist()
    values += values[:1]  # Repeat the first value to close the polygon
    ax.plot(angles, values, label=model_name, color=color, linewidth=2)
    ax.fill(angles, values, color=color, alpha=0.25)
# Customize axis labels
ax.set_xticks(angles[:-1])
ax.set_xticklabels([label.replace(' ', '\n') for label in labels], fontsize=12, fontweight='bold', color='black')
# Title and legend
ax.set_title("Models Performance Comparison\n", size=18, pad=30, color='black')
ax.legend(
    loc='upper right',
    bbox_to_anchor=(1.3, 1.1),
    fontsize=12,
    title="Models",
    title_fontsize=14,
    frameon=True,
    fancybox=True,
    shadow=True,
    borderpad=1.2,
    labelspacing=1.2
)
plt.tight_layout()
plt.savefig(folder + '/Figure_Spider_plot.png')
plt.show()



# Define the data and labels
data_types = ['array_size', 'parasiticResistance', 'memoryWindow', 'variability', 'sparsity']
data = [array_size, parasiticResistance, memoryWindow, np.arange(variabilitySize), Rhrs_percentage]
labels = ['Array Size', 'Parasitic Resistance', 'Memory Window', 'Variability Index', 'Hrs Percentage']
error_labels = [error_label] * len(data_types)
markers = ['o', 's', 'D', '^', 'v', 'p']  # List of markers for each model

# Loop through data and create individual figures for each
for i, (data_type, data_values, label, error_label) in enumerate(zip(data_types, data, labels, error_labels)):
    if len(data_values) > 1:
        # Compute the error for this data type
        error_vs_data = np.mean(Metric, axis=tuple([j for j in range(5) if j != i]))
        # Create a new figure
        fig, ax = plt.subplots(figsize=(12, 7))
        # Plot each model's error
        for j, (model_name, color, marker) in enumerate(zip(enabled_models[:-1], model_colors, markers)):
            ax.plot(data_values, error_vs_data[:, j], label=model_name, color=color, marker=marker, markersize=8)
        # Customize the plot
        ax.set_xlabel(label, fontsize=14)
        ax.set_ylabel(error_label, fontsize=14)
        ax.set_title(f'Model Normalized Error vs {label}', fontsize=16, weight='bold')
        ax.legend(fontsize=12)
        ax.grid(True, which="both", linestyle='--', linewidth=0.5)
        ax.tick_params(axis='both', which='major', labelsize=12)
        # Save each figure
        fig.tight_layout()
        plt.savefig(f"{folder}/Figure_error_vs_{data_type}.png", dpi=300)
        # Show the plot
        plt.show()
