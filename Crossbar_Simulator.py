import random as rd 
import datetime
import pytz
import os
import numpy as np
import matplotlib.pyplot as plt
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
input,output = (128,128)

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
    CrossSimModel_p1("CrossSim1"),
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
enabled_models = [ "Ideal","Jeong","DMR","Gamma","CrossSim1","CrossSim2","CrossSim3","CrossSim4"]

reference_model = "Memtorch"

# Low resistance proggramming value
R_lrs = 1000
Rhrs_percentage=70
# parasitic resistance value
parasiticResistance = np.arange(0.1, 5, 0.5)
parasiticResistance = np.array([2])

# Memory window (ratio between Hrs and Lrs)
memoryWindow = np.arange(5, 100, 5)
memoryWindow = np.array([20])

# Input voltages parameters
v_On_percentage = 100
population = [1, 0.0]

# Mse type (1=Current, 0=Voltage)
current_mse = 1

# Variability parameters
v_flag = 1
v_size = 20







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

# Assuming 'input', 'output', and 'v_size' are defined elsewhere
for m in range(np.size(memoryWindow)):                  # Iterating over memoryWindow
    x_thickness = resistance_array_to_x(np.array([R_lrs*memoryWindow[m]]))
    for z in range(np.size(parasiticResistance)):       # Iterating over parasiticResistance
        for v in range(v_size):                         # iterating over variability 'v'
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

# Generate Potential vector
Potential = np.random.choice(population, size=input, p=[v_On_percentage / 100, 1 - v_On_percentage / 100])

reference_index = enabled_models.index(reference_model)

# Output Currents (for each Parasitic Model)
output_currents = np.zeros((output, parasiticSize ,memorySize, modelSize))

# Voltage Drops (for each Parasitic Model)
V_a_matrix = np.tile(Potential, output)
voltage_drops = np.zeros((input,output, parasiticSize ,memorySize, modelSize))

# Initialize mse array
mse = np.zeros((parasiticSize ,memorySize, modelSize))

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

    for z in range(np.size(parasiticResistance)):
        update_progress_bars(memory_progress, resistance_progress, variability_progress, m, z, 0, memorySize, parasiticResistance, v_size)

        for v in range(v_size):
            update_progress_bars(memory_progress, resistance_progress, variability_progress, m, z, v, memorySize, parasiticResistance, v_size)

            for index, model in enumerate(Models):
                if model.name in enabled_models:  # Check if the model is enabled
                    index = enabled_models.index(model.name)
                    NonLinear_params = {'X': X[z,m,v], 'S': S[z,m,v]} if model.name == 'NgSpiceNonLinear' else {}

                    start_time = time.perf_counter()
                    voltage_drops[:, :, z, m, index], output_currents[:, z, m, index] = model.calculate(R[z, m, v], parasiticResistance[z], Potential, **NonLinear_params)
                    # output_currents[:, z, m, index] = np.cumsum(voltage_drops[:,:,z,m, index]*np.reciprocal(R[z,m,v]),axis=0)[input-1,:]
                    end_time = time.perf_counter()
                    simulation_times[index] += (end_time - start_time)/totalIterations
                    
            for index, model in enumerate(enabled_models):
                if current_mse:
                    # Compute Output Current MSE
                    mse[z, m, index] += np.mean(np.abs(output_currents[:, z, m, reference_index] - output_currents[:, z, m, index] ) / output_currents[:, z, m, reference_index])*100/v_size
                else:
                    # Compute Voltage MSE
                    mse[z, m, index] += np.mean(np.abs((voltage_drops[:,:,z,m,reference_index] - voltage_drops[:,:,z,m,index]))/voltage_drops[:,:,z,m,reference_index])*100/v_size

root.destroy()  # Close the window when simulation is complete











###################### Plotting portion ##########################################################################

# plot parameters
colors = ['g', 'r', 'b', 'm', 'c', 'y', 'orange', 'purple', 'pink', 'brown', 'lime', 'teal']
colors = ['black','c', 'g', 'r', 'b', 'b', 'b', 'b', 'orange', 'purple', 'pink', 'brown', 'lime', 'teal']
markers = ['o', 's', 'D', '^', 'v', 'p']

# Figures Selection
Simulation_times_plot = 1

Absolute_current_plots = 1
Relative_error_plots = 1

Voltage_drops_plot = 0
Voltage_drops_error_plot = 0

MSE_plot = 1
Mean_MSE_plot = 1
MSE_vs_Rpar = 1
MSE_vs_MW = 1
print_table = 1
scatter_plot = 1

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
    MSE_plot=0
    MSE_vs_Rpar=0
    MSE_vs_MW=0
    print_table=0

# Different labels based on the used metric
if current_mse:
    error_label = "Normalized Output Current Error (%)"
else:
    error_label = "Normalized Voltage Drops Error (%)"
array_size_string = "Array Size: "+ str(input) + 'x' + str(output)




if Simulation_times_plot:
    plt.figure()
    if "Ideal" in enabled_models:
        index = enabled_models.index("Ideal")
        # Divide simulation_times by the element at the found index
        normalized_simulation_times = simulation_times / simulation_times[index]
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
                plt.subplot(2, round(modelSize/2), index+1)  # 1 row, 3 columns, 1st subplot
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
                    plt.subplot(2, round(modelSize/2), index+1)  # 1 row, 3 columns, 1st subplot
                    plt.imshow(relative_voltage_drops, cmap='hot', interpolation='nearest',vmax = 1 )
                    plt.colorbar(label='Voltage Drop error (V)')
                    plt.title('Voltage Drop error ('+ model +')')
                    plt.xlabel('Column Index (j)')
                    plt.ylabel('Row Index (m)')
            plt.tight_layout()
            plt.savefig(folder + f'/Figure_Relative_Voltage_drops_Rpar={parasiticResistance[z]}_MW={memoryWindow[m]}.png')
            relative_voltage_fig.show()


if MSE_plot:
    # 3D plots of the MSE
    X_plot, Y_plot = np.meshgrid(parasiticResistance, memoryWindow)
    x = X_plot.flatten()
    y = Y_plot.flatten()
    dx = dy = 0.5
    if np.size(parasiticResistance) > 1:
        dx = np.diff(parasiticResistance).min()
    if np.size(memoryWindow) > 1:
        dy = np.diff(memoryWindow).min()
    max_mse = ((mse.T).flatten()).max()
    num_models = len(enabled_models) - 1  # Exclude the reference model
    # Calculate optimal rows and columns for the subplots
    cols = int(np.ceil(np.sqrt(num_models)))
    rows = int(np.ceil(num_models / cols))
    # Create the subplots grid
    figure_mse, axs = plt.subplots(rows, cols, subplot_kw={'projection': '3d'}, figsize=(15, 7))
    axs = np.array(axs).reshape(-1)  # Flatten to make indexing easier
    plot_index = 0
    for index, model in enumerate(enabled_models):
        if model != reference_model:
            dz = (mse[:, :, index].T).flatten()
            axs[plot_index].bar3d(x, y, np.zeros_like(dz), dx, dy, dz, shade=True, color=colors[plot_index % len(colors)])
            axs[plot_index].set_title(model)
            axs[plot_index].set_xlabel('Parasitic Resistance (Ohm)')
            axs[plot_index].set_ylabel('Memory Window (Ohm)')
            axs[plot_index].set_zlabel(error_label)
            axs[plot_index].set_zlim(0, max_mse)
            plot_index += 1
    # Hide any unused subplots
    for ax in axs[plot_index:]:
        ax.set_visible(False)
    plt.tight_layout()
    # Save and show the figure
    figure_mse.savefig(folder + '/Figure_MSE_plot.png')
    plt.show()


if Mean_MSE_plot:
    print("mesn MSE (each model):",np.mean(mse, axis=0).mean(axis=0))
    # Total Mse histogram
    fig_mse = plt.figure()
    # Prepare the data
    # Plot the histogram
    plt.bar(enabled_models[:-1], np.mean(mse, axis=0).mean(axis=0)[:-1], color=[colors[i % len(colors)] for i in range(len(enabled_models) - 1)])
    # Add labels and title
    plt.xlabel('Model')
    plt.ylabel(error_label)
    plt.title('Mean Error (all simulations)\n' + array_size_string)
    plt.savefig(folder + '/Figure_Mean_MSE_plot.png')
    # Show the plot
    plt.show()


if MSE_vs_Rpar and memorySize==1:
    # Plotting
    plt.figure()
    for index, model in enumerate(enabled_models):
        if model != reference_model:
             plt.plot(parasiticResistance, mse[:, 0, index], marker=markers[index % len(markers)], color=colors[index % len(colors)], label=model)
    # Log scale for y-axis as in the example image
    plt.xlabel("Line Resistance (Ω)")
    plt.ylabel(error_label)
    plt.title("Model Normalized Error vs Line Resistance\n"+array_size_string)
    plt.legend()
    plt.grid(True, which="both", linestyle='--', linewidth=0.5)
    plt.savefig(folder + '/Figure_error_vs_Rpar_plot.png')
    # Show the plot
    plt.show()

if MSE_vs_MW and parasiticSize==1:
    # Plotting
    plt.figure()
    for index, model in enumerate(enabled_models):
        if model != reference_model:
             plt.plot(memoryWindow, mse[0, :, index], marker=markers[index % len(markers)], color=colors[index % len(colors)], label=model)
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
    winning_indices = np.argmin(mse[:,:,:-1], axis=-1)
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

    


if scatter_plot:
    mean_mse = np.mean(mse, axis=0).mean(axis=0)
    variance_mse = np.var(mse, axis=0).mean(axis=0)  # Compute variance
    if "Ideal" in enabled_models:
        ideal_index = enabled_models.index("Ideal")
        normalized_simulation_times = simulation_times / simulation_times[ideal_index]
        plot_models = [model for i, model in enumerate(enabled_models) if i != ideal_index]
        plot_times = np.delete(normalized_simulation_times, ideal_index)
        plot_mse = np.delete(mean_mse, ideal_index)
        plot_variance = np.delete(variance_mse, ideal_index)  # Adjust variance array
    else:
        plot_models = enabled_models
        plot_times = simulation_times
        plot_mse = mean_mse
        plot_variance = variance_mse  # Use full variance array
    fig, ax = plt.subplots()
    # Filter data for "CrossSim" models
    crosssim_indices = [i for i, model in enumerate(plot_models) if model.startswith("CrossSim")]
    if crosssim_indices:
        crosssim_times = np.array(plot_times)[crosssim_indices].reshape(-1, 1)
        crosssim_mse = np.array(plot_mse)[crosssim_indices]
        # Fit a linear regression model
        reg = LinearRegression().fit(np.log(crosssim_times), np.log(crosssim_mse))  # Log scale for both time and mse
        reg_line_x = np.linspace(crosssim_times.min(), crosssim_times.max(), 100).reshape(-1, 1)
        reg_line_y = np.exp(reg.predict(np.log(reg_line_x)))  # Convert the predicted log values back to the original scale
        # Plot regression line
        ax.plot(reg_line_x, reg_line_y, color='blue', linestyle='--', linewidth=1.5, label='CrossSim Regression')
    scatter = ax.scatter(plot_times, plot_mse, 
                        c=[colors[(i+1) % len(colors)] for i in range(len(plot_models))],
                        s=120, marker='o', edgecolor="black", linewidth=0.7)
    # Plot variance bars
    for i, (x, y, var) in enumerate(zip(plot_times, plot_mse, plot_variance)):
        ax.errorbar(x, y, yerr=np.sqrt(var), fmt='o', color=colors[(i+1) % len(colors)], capsize=5)
    for i, model in enumerate(plot_models):
        if not model.startswith("CrossSim"):
            ax.annotate(
                model, (plot_times[i], plot_mse[i]),
                textcoords="offset points", xytext=(10, 0), ha='left'  # Adjusted position to the right
            )
    ax.grid(True, which="both", linestyle='--', linewidth=0.5, color="gray", alpha=0.7)
    # Highlight Pareto front
    sorted_indices = np.argsort(plot_times)
    pareto_times = np.array(plot_times)[sorted_indices]
    pareto_mse = np.array(plot_mse)[sorted_indices]
    # Legend Handles
    legend_handles = [
        plt.Line2D([0], [0], marker='o', color='w', label=plot_models[i],
                markerfacecolor=colors[(i+1) % len(colors)], markersize=10)
        for i in range(len(plot_models)) if i not in crosssim_indices
    ]
    # Add regression line to the legend
    legend_handles.append(plt.Line2D([0], [0], color='blue', linestyle='--', label='CrossSim'))
    ax.legend(handles=legend_handles, title="Models", bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    # Set log scale for x-axis (simulation time)
    ax.set_xscale('log')
    ax.set_xlabel('Normalized Simulation Time (log scale)' if "Ideal" in enabled_models else 'Execution Time (seconds, log scale)')
    ax.set_ylabel(error_label)
    ax.set_title('Scatter Plot'+'\n'+array_size_string)
    # Save and show the plot
    plt.tight_layout()
    plt.savefig(folder + '/Figure_Scatter_SimulationTimes_vs_error.png')
    plt.show()




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
