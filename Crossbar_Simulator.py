import random
import datetime
import pytz
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from matplotlib.colors import to_hex
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from math import pi
import time
import tkinter as tk
from tkinter import ttk
import pandas as pd
import openpyxl
from openpyxl.styles import PatternFill
from sklearn.linear_model import LinearRegression
#Import NonLinear functions
from CrossbarModels.Functions.NonLinear import resistance_array_to_x
from CrossbarModels.Functions.NonLinear import calculate_resistance
#Import the Crossbar Models
from CrossbarModels.Crossbar_Models import *

# Use this script to compare the performance (time and accuracy) of different parasitc resistance corsbar models


############################ PARAMETERS ##############################

# Dimensions of the crossbar
input,output = (784,64)

# Initialize each model instance
Models = [
    JeongModel("Jeong"),
    JeongModel_avg("Jeong_avg"),
    JeongModel_avgv2("Jeong_new"),
    IdealModel("Ideal"),
    DMRModel("DMR_old"),
    DMRModel_acc("DMR"),
    DMRModel_new("DMR_new"),
    GammaModel("Gamma_old"),
    GammaModel_acc("Gamma_acc_v1"),
    GammaModel_acc_v2("Gamma"),
    CrossSimModel("CrossSim_ref"),
    CrossSimModel("CrossSim1",Verr_th=0.5),
    CrossSimModel("CrossSim2",Verr_th=1e-1),
    CrossSimModel("CrossSim3",Verr_th=1e-2),
    CrossSimModel("CrossSim4",Verr_th=1e-3),
    CrossSimModel("CrossSim5",Verr_th=1e-4),
    CrossSimModel("CrossSim6",Verr_th=1e-5),
    CrossSimModel("CrossSim7",Verr_th=1e-6),
    CrossSimModel("CrossSim8",Verr_th=1e-7),
    CrossSimModel("CrossSim9",Verr_th=1e-8),
    CrossSimModel("CrossSim10",Verr_th=1e-9),
    CrossSimModel("CrossSim11",Verr_th=1e-10),
    LTSpiceModel("LTSpice"),
    NgSpiceModel("NgSpice"),
    NgSpiceNonLinearModel("NgSpiceNonLinear"),
    MemtorchModelCpp("Memtorch"),
    MemtorchModelPython("Memtorch_python")
]


enabled_models = [ "Ideal","Jeong","Jeong_new","DMR","Gamma","CrossSim3"]
# enabled_models = [model.name for model in Models]
# enabled_models = [ "Ideal","Jeong_avgv2","DMR","Gamma","CrossSim1","CrossSim2","CrossSim3","CrossSim4","CrossSim5","CrossSim6","CrossSim7","CrossSim8","CrossSim9","CrossSim10","Memtorch","NgSpice"]

reference_model = "Memtorch"

# Low resistance proggramming value
R_lrs = 1000
Rhrs_percentage=50
# parasitic resistance value
parasiticResistance = np.arange(0.2, 5, 0.2)
parasiticResistance = np.array([0.01])

# Memory window (ratio between Hrs and Lrs)
memoryWindow = np.arange(5, 101, 5)
memoryWindow = np.array([2])

# Input voltages parameters
v_On_percentage = 10
population = [0.5, 0.0]

# Metric type (2=Current*Times, 1=Current, 0=Voltage)
Metric_type = 1

# Variability parameters
v_flag = 1
v_size = 2





############################ INITIALIZATIONS ############################

enabled_models.append(reference_model)

memorySize = np.size(memoryWindow)
parasiticSize = np.size(parasiticResistance)
modelSize = len(enabled_models)
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


# Generate Potential vector
Potential = np.random.choice(population, size=input, p=[v_On_percentage / 100, 1 - v_On_percentage / 100])

reference_index = enabled_models.index(reference_model)

# Output Currents (for each Parasitic Model)
output_currents = np.zeros((output, parasiticSize ,memorySize, modelSize))

# Voltage Drops (for each Parasitic Model)
V_a_matrix = np.tile(Potential, output)
voltage_drops = np.zeros((input,output, parasiticSize ,memorySize, modelSize))

# Initialize Metric arrays
Current_error = np.zeros((parasiticSize ,memorySize, modelSize-1))
Voltage_error = np.zeros((parasiticSize ,memorySize, modelSize-1))

# Initialize time measurements
simulation_times = np.zeros(modelSize)


# ########## Progress Bar ###########
def update_progress_bars(memory_progress, resistance_progress, variability_progress, m, z, v, memorySize, parasiticResistance, v_size):
    memory_progress['value'] = (m + 1) / memorySize * 100
    resistance_progress['value'] = (z + 1) / np.size(parasiticResistance) * 100
    variability_progress['value'] = (v + 1) / v_size * 100
    root.update_idletasks()
    root.update()            # Update the Tkinter event loop
# Create a tkinter window with 3 progress bars
root = tk.Tk()
root.title("Simulation Progress")
memory_progress = ttk.Progressbar(root, orient="horizontal", length=300, mode="determinate")
memory_progress.pack(pady=10)
memory_progress_label = tk.Label(root, text="Memory Window")
memory_progress_label.pack()
resistance_progress = ttk.Progressbar(root, orient="horizontal", length=300, mode="determinate")
resistance_progress.pack(pady=10)
resistance_progress_label = tk.Label(root, text="Parasitic Resistance")
resistance_progress_label.pack()
variability_progress = ttk.Progressbar(root, orient="horizontal", length=300, mode="determinate")
variability_progress.pack(pady=10)
variability_progress_label = tk.Label(root, text="Variability")
variability_progress_label.pack()
root.update_idletasks()




############################ MAIN LOOP ############################

for m in range(memorySize):
    update_progress_bars(memory_progress, resistance_progress, variability_progress, m, 0, 0, memorySize, parasiticResistance, v_size)
    x_thickness = resistance_array_to_x(np.array([R_lrs*memoryWindow[m]]))

    for z in range(np.size(parasiticResistance)):
        update_progress_bars(memory_progress, resistance_progress, variability_progress, m, z, 0, memorySize, parasiticResistance, v_size)

        for v in range(v_size):
            update_progress_bars(memory_progress, resistance_progress, variability_progress, m, z, v, memorySize, parasiticResistance, v_size)

            # Generate Resistance Matrix based on the parameters
            Rstate = np.random.choice([1, 0], size=(input, output), p=[Rhrs_percentage/100, 1 - Rhrs_percentage/100])
            # Rstate[5,10]=1
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

            # Simulate the crossbar with each model
            for index, model in enumerate(Models):
                if model.name in enabled_models:  # Check if the model is enabled
                    index = enabled_models.index(model.name)
                    NonLinear_params = {'X': X[z,m,v], 'S': S[z,m,v]} if model.name == 'NgSpiceNonLinear' else {'R_lrs': R_lrs, 'MW':memoryWindow[m]}

                    start_time = time.perf_counter()
                    voltage_drops[:, :, z, m, index], output_currents[:, z, m, index] = model.calculate(R[z, m, v], parasiticResistance[z], Potential, **NonLinear_params)
                    # output_currents[:, z, m, index] = np.cumsum(voltage_drops[:,:,z,m, index]*np.reciprocal(R[z,m,v]),axis=0)[input-1,:]
                    end_time = time.perf_counter()
                    simulation_times[index] += (end_time - start_time)/totalIterations
            
            # Calculate the error metrics
            for index, model in enumerate(enabled_models[:-1]):
                # Compute Output Current Metric
                Current_error[z, m, index] += np.mean(np.abs(output_currents[:, z, m, reference_index] - output_currents[:, z, m, index] ) / output_currents[:, z, m, reference_index])*100/v_size
                # Compute Voltage Metric
                Voltage_error[z, m, index] += np.mean(np.abs((voltage_drops[:,:,z,m,reference_index] - voltage_drops[:,:,z,m,index]))/voltage_drops[:,:,z,m,reference_index])*100/v_size

root.destroy()  # Close the window when simulation is complete

if Metric_type==2:
    Metric = Current_error * simulation_times[np.newaxis, np.newaxis, :]
elif Metric_type==1:
    Metric = Current_error
else:
    Metric = Voltage_error

if "Ideal" in enabled_models:
    index = enabled_models.index("Ideal")
    # Divide simulation_times by the element at the found index
    normalized_simulation_times = simulation_times / simulation_times[index]
    if Metric_type==2:
        Metric = np.reciprocal(Current_error[:,:,:-1] * normalized_simulation_times[np.newaxis, np.newaxis, :-1])











###################### Plotting portion ##########################################################################



# Known color mapping
color_mapping = {
    "Jeong": "c",
    "DMR": "g",
    "Gamma": "r",
    "Ng": "pink",
    "CrossSim": "b",
    "Ideal": "black",
    "Memtorch": "orange"
}

# Generate the colors list
colors = [
    color_mapping[next((key for key in color_mapping if model.startswith(key)), None)] 
    if any(model.startswith(key) for key in color_mapping) 
    else "#{:06x}".format(random.randint(0, 0xFFFFFF))  # Generate a random color
    for model in enabled_models
]

markers = ['o', 's', 'D', '^', 'v', 'p']

# Figures Selection
Simulation_times_plot = 1

Absolute_current_plots = 1
Relative_error_plots = 1

Voltage_drops_plot = 1
Voltage_drops_error_plot = 0

Metric_plot = 1
Mean_Metric_plot = 1
Metric_vs_Rpar = 1
Metric_vs_MW = 1
Winning_models_map = 1
print_table = 1
scatter_plot = 1
spider_plot = 1

Resistance_heatmap =0
open_folder = 1


x = np.arange(0, output)
# Results Folder
end = datetime.datetime.now(pytz.timezone('Europe/Rome'))
folder = 'Results/'+str(end.year)+ str(end.month)+  str(end.day) + '_'+ str(end.hour) +'_'+ str(end.minute)+"_"+str(input)+'x'+str(output)
if not (os.path.exists(folder)):
        os.makedirs(folder)
script_dir = os.path.dirname(os.path.abspath(__file__))
folder_path = os.path.join(script_dir, folder)

# Check there are not too many instances of plots
if memorySize>3 or parasiticSize>3:
    Absolute_current_plots = 0
    Relative_error_plots = 0
    Voltage_drops_plot = 0
    Voltage_drops_error_plot = 0
    Resistance_heatmap =0
else:
    Metric_plot=0
    Metric_vs_Rpar=0
    Metric_vs_MW=0
    print_table=0
    Winning_models_map = 0

# Different labels based on the used metric
if Metric_type:
    error_label = "Normalized Output Current Error (%)"
    if Metric_type==2:
        error_label = "Precision over Execution Time [1/s]"
else:
    error_label = "Normalized Voltage Drops Error (%)"
array_size_string = "Array Size: "+ str(input) + 'x' + str(output)




if Simulation_times_plot:
    plt.figure()
    if "Ideal" in enabled_models:
        # Create new lists/arrays without the "Ideal" element
        new_enabled_models = [model for i, model in enumerate(enabled_models) if i != index]
        new_simulation_times = np.delete(normalized_simulation_times, index)
        plt.title('Normalized Processing Time relative to the Ideal Model'+ '\n'+array_size_string)
        plt.ylabel('Normalized Time (log scale)')
        bars = plt.bar(new_enabled_models, new_simulation_times, color=[colors[(i+1) % len(colors)] for i in range(len(enabled_models))])
    else:
        plt.title('Processing Time for Each Model (Log Scale)'+ '\n'+array_size_string)
        plt.ylabel('Time (seconds, log scale)')
        bars = plt.bar(enabled_models, simulation_times, color=[colors[i % len(colors)] for i in range(len(enabled_models))])

    plt.xlabel('Model')
    plt.yscale('log')
    # Optionally rotate x-labels if they overlap
    plt.xticks(rotation=45, ha='right')
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    plt.savefig(folder + '/Figure_Simulation_times.png')
    plt.show()
    print("Simulation Times:",simulation_times)


# Output Currents plot
for z in range(np.size(parasiticResistance)):
    if Relative_error_plots:
        relative_Current = np.zeros(modelSize)
        # First figure: Relative error plots
        figure1, axis1 = plt.subplots(1, memorySize, figsize=(10, 5))  # Create a single row of subplots
        if memorySize == 1:
            axis1 = np.array([axis1])  # Ensure axis1[0] structure for single column case
        for m in range(memorySize):
            for index, model in enumerate(enabled_models):
                if model != reference_model:
                    # Calculate relative currents
                    relative_Current = np.abs(output_currents[:, z, m, reference_index] - output_currents[:, z, m, index] ) / output_currents[:, z, m, reference_index]
                    # Plot relative errors
                    axis1[m].plot(x, relative_Current * 100, color=colors[index % len(colors)], marker='.', label=(model))
                    axis1[m].set_title(f"Relative Error  (MW = {memoryWindow[m]}, Rpar = {parasiticResistance[z]} Ohm)"+ '\n'+array_size_string)
                    axis1[m].set(xlabel='jth bit line', ylabel='Normalized Output Current Error (%)')
                    axis1[m].legend()
        figure1.tight_layout()
        figure1.savefig(folder + f'/Figure_Relative_Current_Error_Rpar={z}.png')
        figure1.show()

    if Absolute_current_plots:
        # Second figure: Absolute current plots 
        figure2, axis2 = plt.subplots(1, memorySize, figsize=(10, 5))  # Create a single row of subplots
        if memorySize == 1:
            axis2 = np.array([axis2])  # Ensure axis2[0] structure for single column case
        for m in range(memorySize):
            for index, model in enumerate(enabled_models):
                axis2[m].plot(x, output_currents[:, z, m, index], color=colors[index % len(colors)], label=(model))
                axis2[m].set_title(f"Output currents  (MW = {memoryWindow[m]}, Rpar = {parasiticResistance[z]} Ohm)"+ '\n'+array_size_string)
                axis2[m].set(xlabel='jth bit line', ylabel='Current [A]')
                axis2[m].legend()
        figure2.tight_layout()
        figure2.savefig(folder + f'/Figure2_Absolute_Current{z}.png')
        figure2.show()


if Voltage_drops_plot:
    for m in range(memorySize):
        for z in range(np.size(parasiticResistance)):
            # Voltage drops comparison (Real vs Jeong vs DMR vs Gamma)
            Vmin = np.min(voltage_drops)
            # Create a figure with subplots side by side
            voltage_fig = plt.figure(figsize=(15, 10))
            for index, model in enumerate(enabled_models):
                # Plot heatmap for each model
                plt.subplot(2, round((modelSize+1)/2), index+1)  # 1 row, 3 columns, 1st subplot
                plt.imshow(voltage_drops[:,:,z,m,index], cmap='hot', interpolation='nearest',vmin=Vmin, vmax=1)
                plt.colorbar(label='Voltage Drop (V)')
                plt.title('Voltage Drop Heatmap ('+ model +')')
                plt.xlabel('Column Index (j)')
                plt.ylabel('Row Index (m)')
            plt.tight_layout()
            plt.savefig(folder + f'/Figure_Voltage_drops_Rpar={parasiticResistance[z]}_MW={memoryWindow[m]}.png')
            voltage_fig.show()


if Voltage_drops_error_plot:
    for m in range(memorySize):
        for z in range(np.size(parasiticResistance)):
            # Voltage drops comparison (Real vs Jeong vs DMR vs Gamma)
            # Vmax = np.max(relative_voltage_drops)
            # Create a figure with subplots side by side
            relative_voltage_fig = plt.figure(figsize=(15, 10))
            for index, model in enumerate(enabled_models):
                if model != reference_model:
                    relative_voltage_drops = np.abs((voltage_drops[:,:,z,m,reference_index] - voltage_drops[:,:,z,m,index]))/voltage_drops[:,:,z,m,reference_index]
                    # Plot heatmap for each model
                    plt.subplot(2, round((modelSize+1)/2), index+1)  # 1 row, 3 columns, 1st subplot
                    plt.imshow(relative_voltage_drops, cmap='hot', interpolation='nearest',vmax = 1 )
                    plt.colorbar(label='Voltage Drop error (V)')
                    plt.title('Voltage Drop error ('+ model +')')
                    plt.xlabel('Column Index (j)')
                    plt.ylabel('Row Index (m)')
            plt.tight_layout()
            plt.savefig(folder + f'/Figure_Relative_Voltage_drops_Rpar={parasiticResistance[z]}_MW={memoryWindow[m]}.png')
            relative_voltage_fig.show()


if Metric_plot:
    # 3D plots of the Metric
    X_plot, Y_plot = np.meshgrid(parasiticResistance, memoryWindow)
    x = X_plot.flatten()
    y = Y_plot.flatten()
    dx = dy = 0.5
    if np.size(parasiticResistance) > 1:
        dx = np.diff(parasiticResistance).min()
    if np.size(memoryWindow) > 1:
        dy = np.diff(memoryWindow).min()
    max_Metric = ((Metric.T).flatten()).max()
    num_models = len(enabled_models) - 1  # Exclude the reference model
    # Calculate optimal rows and columns for the subplots
    cols = int(np.ceil(np.sqrt(num_models)))
    rows = int(np.ceil(num_models / cols))
    # Create the subplots grid
    figure_Metric, axs = plt.subplots(rows, cols, subplot_kw={'projection': '3d'}, figsize=(15, 7))
    axs = np.array(axs).reshape(-1)  # Flatten to make indexing easier
    plot_index = 0
    for index, model in enumerate(enabled_models):
        if model != reference_model:
            dz = (Metric[:, :, index].T).flatten()
            axs[plot_index].bar3d(x, y, np.zeros_like(dz), dx, dy, dz, shade=True, color=colors[plot_index % len(colors)])
            axs[plot_index].set_title(model)
            axs[plot_index].set_xlabel('Parasitic Resistance (Ohm)')
            axs[plot_index].set_ylabel('Memory Window (Ohm)')
            axs[plot_index].set_zlabel(error_label)
            axs[plot_index].set_zlim(0, max_Metric)
            plot_index += 1
    # Hide any unused subplots
    for ax in axs[plot_index:]:
        ax.set_visible(False)
    plt.tight_layout()
    # Save and show the figure
    figure_Metric.savefig(folder + '/Figure_Metric_plot.png')
    plt.show()


if Mean_Metric_plot:
    print("mean Metric (each model):",np.mean(Metric, axis=0).mean(axis=0))
    # Total Metric histogram
    fig_Metric = plt.figure()
    # Prepare the data
    # Plot the histogram
    plt.bar(enabled_models[:-1], np.mean(Metric, axis=0).mean(axis=0), color=[colors[i % len(colors)] for i in range(len(enabled_models) - 1)])
    # Add labels and title
    plt.xlabel('Model')
    plt.ylabel(error_label)
    plt.title('Mean Score throught all simulations)\n' + array_size_string)
    plt.savefig(folder + '/Figure_Mean_Metric_plot.png')
    # Show the plot
    plt.show()


if Metric_vs_Rpar and memorySize==1:
    # Plotting
    plt.figure()
    for index, model in enumerate(enabled_models):
        if model != reference_model:
             plt.plot(parasiticResistance, Metric[:, 0, index], marker=markers[index % len(markers)], color=colors[index % len(colors)], label=model)
    # Log scale for y-axis as in the example image
    plt.xlabel("Line Resistance (Ω)")
    plt.ylabel(error_label)
    plt.title("Model Normalized Error vs Line Resistance\n"+array_size_string)
    plt.legend()
    plt.grid(True, which="both", linestyle='--', linewidth=0.5)
    plt.savefig(folder + '/Figure_error_vs_Rpar_plot.png')
    # Show the plot
    plt.show()

if Metric_vs_MW and parasiticSize==1:
    # Plotting
    plt.figure()
    for index, model in enumerate(enabled_models):
        if model != reference_model:
             plt.plot(memoryWindow, Metric[0, :, index], marker=markers[index % len(markers)], color=colors[index % len(colors)], label=model)
    # Log scale for y-axis as in the example image
    plt.xlabel("Memory Window (On/Off ratio)")
    plt.ylabel(error_label)
    plt.title("Model Normalized Error vs Memory Window\n"+array_size_string)
    plt.legend()
    plt.grid(True, which="both", linestyle='--', linewidth=0.5)
    plt.savefig(folder + '/Figure_error_vs_MW_plot.png')
    # Show the plot
    plt.show()

if print_table:
    model_labels = np.array(enabled_models)
    if Metric_type == 2:
        winning_indices = np.argmax(Metric, axis=-1)
    else:
        winning_indices = np.argmin(Metric, axis=-1)
    winning_models = model_labels[winning_indices]
    winning_df = pd.DataFrame(winning_models, index=parasiticResistance, columns=memoryWindow)
    # First save without formatting
    excel_path = folder_path + '/winning_models.xlsx'
    winning_df.to_excel(excel_path, index=True)
    # Define color mapping
    color_map = {
        'CrossSim': 'ADD8E6',     # light blue
        'Gamma': 'FFB6B6',        # light red
        'Gamma_acc': 'FFB6B6',    # light red
        'Gamma_acc_v2': 'FFB6B6',    # light red
        'Jeong': 'E0FFFF',        # light cyan
        'DMR': '90EE90'  ,        # light green
        'DMR_acc': '90EE90',       # light green
        'DMR_acc_v2': '90EE90'       # light green
    }
    # Open the saved file and add formatting
    wb = openpyxl.load_workbook(excel_path)
    ws = wb.active
    # Apply colors to cells (skip header row and index column)
    for row in range(2, ws.max_row + 1):
        for col in range(2, ws.max_column + 1):
            cell = ws.cell(row=row, column=col)
            model = cell.value
            if model in color_map:
                cell.fill = PatternFill(start_color=color_map[model],
                                      end_color=color_map[model],
                                      fill_type='solid')
    # Save the formatted file
    wb.save(excel_path)
    os.startfile(excel_path)

if Winning_models_map:
    model_labels = np.array(enabled_models)
    if Metric_type == 2:
        winning_indices = np.argmax(Metric, axis=-1)
    else:
        winning_indices = np.argmin(Metric, axis=-1)
    winning_models = model_labels[winning_indices]
    # Define color mapping using the order of enabled_models
    hex_colors = [to_hex(color) for color in colors]
    color_map = {model: hex_colors[i] for i, model in enumerate(enabled_models)}
    # Map winning models to indices
    model_to_idx = {model: idx for idx, model in enumerate(enabled_models)}
    model_indices = np.vectorize(model_to_idx.get)(winning_models)
    winning_models_unique = [model for model in enabled_models if model in np.unique(winning_models)]
    filtered_colors = [color_map[model] for model in winning_models_unique]
    cmap = mcolors.ListedColormap(filtered_colors)
    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(model_indices, cmap=cmap, origin="lower", aspect="auto",
                extent=[memoryWindow[0], memoryWindow[-1], parasiticResistance[0], parasiticResistance[-1]],
                interpolation="nearest")
    # Add axis labels
    plt.xlabel("Memory Window")
    plt.ylabel("Parasitic Resistance")
    plt.title("Winning Models Map")
    # Create a legend using enabled_models order
    legend_patches = [Patch(facecolor=color_map[model], label=model) for model in winning_models_unique]
    plt.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc="upper left", title="Models")
    # Show the plot
    plt.tight_layout()
    plt.savefig(folder + '/Figure_Winning_Models_Map.png')
    plt.show()




if scatter_plot:
    mean_Metric = np.mean(Metric, axis=0).mean(axis=0)
    variance_Metric = np.var(Metric, axis=0).mean(axis=0)  # Compute variance
    if "Ideal" in enabled_models:
        ideal_index = enabled_models.index("Ideal")
        plot_models = [model for i, model in enumerate(enabled_models[:-1]) if i != ideal_index]
        plot_times = np.delete(normalized_simulation_times[:-1], ideal_index)
        plot_Metric = np.delete(mean_Metric, ideal_index)
        plot_variance = np.delete(variance_Metric, ideal_index)  # Adjust variance array
    else:
        plot_models = enabled_models[:-1]
        plot_times = simulation_times[:-1]
        plot_Metric = mean_Metric
        plot_variance = variance_Metric  # Use full variance array
    fig, ax = plt.subplots()
    # Filter data for "CrossSim" models
    crosssim_indices = [i for i, model in enumerate(plot_models) if model.startswith("CrossSim")]
    if len(crosssim_indices) > 1:
        crosssim_times = np.array(plot_times)[crosssim_indices].reshape(-1, 1)
        crosssim_Metric = np.array(plot_Metric)[crosssim_indices]
        # Fit a linear regression model
        reg = LinearRegression().fit(np.log(crosssim_times), np.log(crosssim_Metric))  # Log scale for both time and Metric
        reg_line_x = np.linspace(crosssim_times.min(), crosssim_times.max(), 100).reshape(-1, 1)
        reg_line_y = np.exp(reg.predict(np.log(reg_line_x)))  # Convert the predicted log values back to the original scale
        # Plot regression line
        ax.plot(reg_line_x, reg_line_y, color='blue', linestyle='--', linewidth=1.5, label='CrossSim Regression')
    scatter = ax.scatter(
        plot_times,
        plot_Metric,
        c=[colors[crosssim_indices[1] % len(colors)] if i in crosssim_indices else colors[(i + 1) % len(colors)]
           for i in range(len(plot_models))],
        s=120,
        marker='o',
        edgecolor="black",
        linewidth=0.7
    )
    # Plot variance bars
    for i, (x, y, var) in enumerate(zip(plot_times, plot_Metric, plot_variance)):
        ax.errorbar(x, y, yerr=np.sqrt(var), fmt='o',
                    color=colors[crosssim_indices[1] % len(colors)] if i in crosssim_indices else colors[(i + 1) % len(colors)], capsize=5)
    # # Annotate non-CrossSim models
    # for i, model in enumerate(plot_models):
    #     if not model.startswith("CrossSim"):
    #         ax.annotate(
    #             model, (plot_times[i], plot_Metric[i]),
    #             textcoords="offset points", xytext=(10, 0), ha='left'
    #         )
    ax.grid(True, which="both", linestyle='--', linewidth=0.5, color="gray", alpha=0.7)
    # Legend Handles
    legend_handles = [
        plt.Line2D([0], [0], marker='o', color='w', label=plot_models[i],
                   markerfacecolor=colors[(i + 1) % len(colors)], markersize=10)
        for i in range(len(plot_models)) if i not in crosssim_indices
    ]
    # Add regression line to the legend
    legend_handles.append(plt.Line2D([0], [0], color='blue', linestyle='--', label='CrossSim'))
    ax.legend(handles=legend_handles, title="Models", bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    # Set log scale for x-axis (simulation time)
    ax.set_xscale('log')
    ax.set_xlabel('Normalized Simulation Time (log scale)' if "Ideal" in enabled_models else 'Execution Time (seconds, log scale)')
    ax.set_ylabel(error_label)
    ax.set_title('Scatter Plot' + ' (' + array_size_string +  ')\n' )
    # Add Inset for low-error models
    inset_ax = inset_axes(ax, width="60%", height="60%", loc='upper right')
    threshold = 0.1  # Adjust as needed for "low-error" models
    low_error_indices = [i for i, metric in enumerate(plot_Metric) if metric < threshold]
    inset_times = np.array(plot_times)[low_error_indices].reshape(-1, 1)
    inset_metrics = np.array(plot_Metric)[low_error_indices]
    inset_variance = np.array(plot_variance)[low_error_indices]
    # Linear regression for inset (CrossSim models)
    if len(crosssim_indices) > 1:
        inset_crosssim_times = np.array(plot_times)[crosssim_indices].reshape(-1, 1)
        inset_crosssim_metrics = np.array(plot_Metric)[crosssim_indices]
        inset_reg = LinearRegression().fit(np.log(inset_crosssim_times), np.log(inset_crosssim_metrics))
        inset_reg_line_x = np.linspace(inset_crosssim_times.min(), inset_crosssim_times.max(), 100).reshape(-1, 1)
        inset_reg_line_y = np.exp(inset_reg.predict(np.log(inset_reg_line_x)))
        inset_ax.plot(inset_reg_line_x, inset_reg_line_y, color='blue', linestyle='--', linewidth=1.5, label='Inset Regression')
    inset_ax.scatter(inset_times, inset_metrics, c=[colors[(i + 1) % len(colors)] for i in low_error_indices],
                     s=60, marker='o', edgecolor="black", linewidth=0.7)
    for i, (x, y, var) in enumerate(zip(inset_times, inset_metrics, inset_variance)):
        inset_ax.errorbar(x, y, yerr=np.sqrt(var), fmt='o', color=colors[(low_error_indices[i] + 1) % len(colors)], capsize=3)
    inset_ax.set_xscale('log')
    inset_ax.set_yscale('log')
    inset_ax.grid(True, which="both", linestyle='--', linewidth=0.5, alpha=0.7)
    inset_ax.set_title("Logaritmic Zoom on Low-Error Models", fontsize=10)
    inset_ax.tick_params(axis='both', which='major', labelsize=8)
    # Save and show the plot
    plt.tight_layout()
    plt.savefig(folder + '/Figure_Scatter_SimulationTimes_vs_error_with_inset.png')
    plt.show()


# if spider_plot:
#     # Take the mean over the first 2 dimensions for Current_error and Voltage_error
#     current_error_mean = np.mean(Current_error, axis=(0, 1))
#     voltage_error_mean = np.mean(Voltage_error, axis=(0, 1))
#     # Compute the inverse of all metrics (as smaller is better)
#     inverse_current_error = 1 / current_error_mean
#     inverse_voltage_error = 1 / voltage_error_mean
#     inverse_simulation_times = 1 / simulation_times[:-1]
#     # Handle "Ideal" model
#     if "Ideal" in enabled_models:
#         ideal_index = enabled_models.index("Ideal")
#         # Remove "Ideal" data from metrics
#         inverse_simulation_times = np.delete(inverse_simulation_times, ideal_index)
#         inverse_voltage_error = np.delete(inverse_voltage_error, ideal_index)
#         inverse_current_error = np.delete(inverse_current_error, ideal_index)
#         # Remove "Ideal" from enabled_models
#         enabled_models = [model for i, model in enumerate(enabled_models) if i != ideal_index]
#     # Combine metrics for radar plot
#     metrics = np.vstack([inverse_current_error, inverse_voltage_error, inverse_simulation_times])
#     # Normalize each metric to [0, 1]
#     metrics_min = metrics.min(axis=1, keepdims=True)
#     metrics_max = metrics.max(axis=1, keepdims=True)
#     metrics_scaled = metrics / metrics.max(axis=1, keepdims=True)
#     # Labels and angles for the radar plot
#     labels = ['Current Accuracy', 'Voltage Accuracy', 'Simulation Speed']
#     angles = [n / float(len(labels)) * 2 * pi for n in range(len(labels))]
#     angles += angles[:1]  # Repeat the first angle to close the polygon
#     model_colors = colors[:len(enabled_models)]  # Map colors to models
#     # Radar plot setup
#     fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'polar': True})
#     ax.spines['polar'].set_visible(False)  # Remove default grid circle
#     ax.set_facecolor('#f9f9f9')  # Light gray background
#     ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)  # Faint gridlines
#     ax.set_yticks([])  # Remove radial ticks
#     # Plot for each model
#     for i, (model_name, color) in enumerate(zip(enabled_models[:-1], model_colors[1:])):
#         values = metrics_scaled[:, i].tolist()
#         values += values[:1]  # Repeat the first value to close the polygon
#         ax.plot(angles, values, label=model_name, color=color, linewidth=2)
#         ax.fill(angles, values, color=color, alpha=0.25)
#     # Customize axis labels
#     ax.set_xticks(angles[:-1])
#     ax.set_xticklabels(labels, fontsize=12, fontweight='bold', color='black')
#     # Title
#     ax.set_title("Models Performance Comparison\n"+array_size_string, size=18, pad=30, color='black')
#     # Customize legend
#     ax.legend(
#         loc='upper right',
#         bbox_to_anchor=(1.3, 1.1),
#         fontsize=12,
#         title="Models",
#         title_fontsize=14,
#         frameon=True,
#         fancybox=True,
#         shadow=True,
#         borderpad=1.2,
#         labelspacing=1.2,
#     )
#     plt.tight_layout()
#     plt.savefig(folder + '/Figure_Spider_plot.png')
#     plt.show()



if Resistance_heatmap:
    # Heatmap resistance matrix
    fig, axes = plt.subplots(1, R.shape[2], figsize=(6 * R.shape[2], 4))
    axes = [axes] if R.shape[2] == 1 else axes
    for i, ax in enumerate(axes):
        # Plotting the heatmap using imshow
        cax = ax.imshow(R[0, 0, i], cmap='viridis', aspect='auto')
        ax.set_title(f"Resistance Array Variability Seed {i}")
        ax.set(xlabel='x position', ylabel='y position')
        # Adding the colorbar for each subplot
        fig.colorbar(cax, ax=ax, label='Resistance [R]', fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig.savefig(folder + '/Heatmaps.png')
    plt.show()


if open_folder:
    os.startfile(folder_path)
