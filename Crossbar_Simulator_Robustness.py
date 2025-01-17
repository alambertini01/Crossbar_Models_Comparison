import os
import random
import numpy as np
import matplotlib.pyplot as plt
import time
import datetime
import pytz

# Import NonLinear functions
from CrossbarModels.Functions.NonLinear import resistance_array_to_x
from CrossbarModels.Functions.NonLinear import calculate_resistance

# Import the Crossbar Models
from CrossbarModels.Crossbar_Models import *

############################ PARAMETERS ##############################

# Models initialization
Models = [
    JeongModel("Jeong"),
    JeongModel_avg("Jeong_avg"),
    JeongModel_avgv2("Jeong_torch"),
    IdealModel("Ideal"),
    DMRModel("DMR_old"),
    DMRModel_acc("DMR"),
    DMRModel_new("DMR_v2"),
    DMRModel_new("DMR_torch"),
    GammaModel("Gamma_torch"),
    GammaModel_acc("Gamma_acc_v1"),
    GammaModel_acc_v2("γ"),
    alpha_beta("alpha_beta_old"),
    alpha_beta_acc("αβ-matrix"),
    CrossSimModel("CrossSim_ref"),
    CrossSimModel("CrossSim", Verr_th=0.3, hide_convergence_msg=True),
    CrossSimModel("CrossSim2", Verr_th=1e-1),
    CrossSimModel("CrossSim3", Verr_th=1e-2),
    CrossSimModel("CrossSim4", Verr_th=1e-3),
    CrossSimModel("CrossSim7", Verr_th=1e-7),
    LTSpiceModel("LTSpice"),
    NgSpiceModel("NgSpice"),
    NgSpiceNonLinearModel("NgSpiceNonLinear"),
    MemtorchModelCpp("Memtorch"),
    MemtorchModelPython("Memtorch_python")
]

# Enabled models
enabled_models = ["Ideal", "Jeong_avg", "DMR", "αβ-matrix"]
reference_model = "CrossSim7"
enabled_models.append(reference_model)

modelSize = len(enabled_models)
show_first_model = False
current_Metric = 1

robustness_flag = True  # Enable/disable 1D-slice robustness computation

# Crossbar dimensions sweep
array_size = np.arange(32, 128, 16)
# Sparsity of the matrix
Rhrs_percentage = np.arange(10, 100, 10)
# Parasitic resistance
parasiticResistance = np.arange(0.1, 5, 0.5)
# Memory window (ratio Hrs/Lrs)
memoryWindow = np.arange(20, 100, 10)
# Number of different variability instances
variabilitySize = 10

# Low resistance programming value
R_lrs = 1000
# Input voltages parameters
v_On_percentage = 100
population = [1, 0.0]
v_flag = 0

# Initialize time measurements
simulation_times = np.zeros((modelSize, np.size(array_size)))

# Initialize Metric arrays
memorySize = np.size(memoryWindow)
parasiticSize = np.size(parasiticResistance)
sparsitySize = np.size(Rhrs_percentage)

Current_error = np.zeros((np.size(array_size), parasiticSize, memorySize, variabilitySize, sparsitySize, modelSize))
Voltage_error = np.zeros_like(Current_error)
Current_error_variance = np.zeros_like(Current_error)
Voltage_error_variance = np.zeros_like(Current_error)

############################ MAIN SIMULATION ############################
for d in range(np.size(array_size)):
    input_size = output_size = array_size[d]
    totalIterations = memorySize * parasiticSize * modelSize

    # NonLinear parameters
    rho, tox, s0 = 3000, 5, 10.62
    s = rho * tox / R_lrs
    v_ds, v_dx = 0.35, 0.7

    # Build arrays for Resistances
    R = np.zeros((parasiticSize, memorySize, variabilitySize, sparsitySize, input_size, output_size))
    X = np.zeros_like(R)
    S = np.ones_like(R)

    # Generate Potential vector
    Potential = np.random.choice(population, size=input_size, p=[v_On_percentage / 100, 1 - v_On_percentage / 100])
    reference_index = enabled_models.index(reference_model)

    # Temporary arrays for each iteration
    output_currents = np.zeros((output_size, parasiticSize, memorySize, modelSize))
    voltage_drops = np.zeros((input_size, output_size, parasiticSize, memorySize, modelSize))

    print(f"{d}th Simulation of crossbar models with array size of: {array_size[d]}")

    # Nested loops
    for m in range(memorySize):
        x_thickness = resistance_array_to_x(np.array([R_lrs * memoryWindow[m]]))  # thickness for Hrs
        for z in range(parasiticSize):
            for v_inst in range(variabilitySize):
                for r in range(sparsitySize):
                    # Generate random Resistances
                    Rstate = np.random.choice(
                        [1, 0],
                        size=(input_size, output_size),
                        p=[Rhrs_percentage[r] / 100, 1 - Rhrs_percentage[r] / 100]
                    )
                    # Update X, S arrays
                    X[z, m, v_inst, r] = Rstate * x_thickness + 1e-12
                    S[z, m, v_inst, r] = (1 - Rstate) * s + Rstate * s0
                    # Apply or skip variability
                    if v_flag:
                        X[z, m, v_inst, r] += np.abs(np.random.randn(input_size, output_size) * v_dx / 3) * Rstate
                        S[z, m, v_inst, r] += np.random.randn(input_size, output_size) * v_ds / 3
                        R[z, m, v_inst, r] = calculate_resistance(X[z, m, v_inst, r], S[z, m, v_inst, r])
                    else:
                        R[z, m, v_inst, r] = Rstate * R_lrs * memoryWindow[m] + (1 - Rstate) * R_lrs

                    # Run each model
                    for model_obj in Models:
                        if model_obj.name in enabled_models:
                            idx = enabled_models.index(model_obj.name)
                            NonLinear_params = (
                                {'X': X[z, m, v_inst], 'S': S[z, m, v_inst]}
                                if model_obj.name == 'NgSpiceNonLinear' else
                                {'R_lrs': R_lrs, 'MW': memoryWindow[m]}
                            )
                            start_time = time.perf_counter()
                            voltage_drops[:, :, z, m, idx], output_currents[:, z, m, idx] = model_obj.calculate(
                                R[z, m, v_inst, r],
                                parasiticResistance[z],
                                Potential,
                                **NonLinear_params
                            )
                            end_time = time.perf_counter()
                            simulation_times[idx, d] += (end_time - start_time) / totalIterations

                    # Compute metrics
                    for idx, model_name in enumerate(enabled_models):
                        # Current
                        curr_diff = np.abs(
                            output_currents[:, z, m, reference_index]
                            - output_currents[:, z, m, idx]
                        ) / output_currents[:, z, m, reference_index]
                        Current_error[d, z, m, v_inst, r, idx] = np.mean(curr_diff) * 100
                        Current_error_variance[d, z, m, v_inst, r, idx] = np.std(curr_diff) * 100
                        # Voltage
                        vol_diff = np.abs(
                            voltage_drops[:, :, z, m, reference_index]
                            - voltage_drops[:, :, z, m, idx]
                        ) / voltage_drops[:, :, z, m, reference_index]
                        Voltage_error[d, z, m, v_inst, r, idx] = np.mean(vol_diff) * 100
                        Voltage_error_variance[d, z, m, v_inst, r, idx] = np.std(vol_diff) * 100

###################### METRIC SELECTION ###############################

if current_Metric:
    Metric = Current_error
    Metric_variance = Current_error_variance
    error_label = "Normalized Output Current Error (%)"
else:
    Metric = Voltage_error
    Metric_variance = Voltage_error_variance
    error_label = "Normalized Voltage Drops Error (%)"

###################### PLOTTING SECTION ###############################
# Known color mapping
color_mapping = {
    "Jeong": "c",
    "DMR": "g",
    "γ": "darkred",
    "αβ-matrix": "r",
    "Ng": "pink",
    "CrossSim": "b",
    "Ideal": "black",
    "Memtorch": "orange"
}

colors = [
    color_mapping[next((key for key in color_mapping if model.startswith(key)), None)]
    if any(model.startswith(key) for key in color_mapping)
    else "#{:06x}".format(random.randint(0, 0xFFFFFF))
    for model in enabled_models
]
model_colors = colors[:len(enabled_models)]

markers = ['o', 's', 'D', '^', 'v', 'p']
plt.rcParams.update({
    'font.size': 14,
    'font.family': 'Arial',
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'legend.fontsize': 14,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14
})

# Create results folder
end = datetime.datetime.now(pytz.timezone('Europe/Rome'))
folder = f"Results/{end.year}{end.month}{end.day}_{end.hour}_{end.minute}_ArraySize_Sweep"
if not os.path.exists(folder):
    os.makedirs(folder)

###################### 1) Processing Time Plot ########################
ideal_index = enabled_models.index("Ideal")
plot_models = [model for i, model in enumerate(enabled_models) if i != ideal_index]

plt.figure()
for idx, model in enumerate(plot_models):
    # Normalized times
    normalized_times = simulation_times[idx + 1, :] / simulation_times[ideal_index, :]
    plt.plot(
        array_size,
        normalized_times,
        marker=markers[(idx + 1) % len(markers)],
        color=colors[(idx + 1) % len(colors)],
        label=model
    )
plt.yscale('log')
plt.xlabel("Array Size")
plt.ylabel("Normalized Processing Times relative to the Ideal Model")
plt.title("Normalized Processing Times vs. Array Size")
handles, labels_ = plt.gca().get_legend_handles_labels()
plt.legend(handles[::-1], labels_[::-1])
plt.grid(True, which="both", linestyle='--', linewidth=0.5)
plt.savefig(folder + '/Figure_SimulationTimes_vs_ArraySize.png')
plt.show()




######################## SPIDER PLOT FUNCTION ########################
def find_nearest_index(array, value):
    """Find the index of the array element closest to `value`."""
    array_np = np.array(array)
    return np.abs(array_np - value).argmin()

def plot_spider_chart(base_metrics, robustness_metrics, param_values_str, save_folder="Results"):
    """
    Plots a radar (spider) chart for the computed metrics.
    param_values_str is a string appended to the saved figure name, 
    e.g. "_as48_pr1.3_mw20_var50_rsp50".
    """
    # Merge metrics
    all_metrics = {**base_metrics, **robustness_metrics}
    valid_metrics = {k: v for k, v in all_metrics.items() if np.all(np.isfinite(v))}
    labels = list(valid_metrics.keys())
    metrics = np.array(list(valid_metrics.values()))

    # Radar plot angles
    angles = [n / float(len(labels)) * 2 * np.pi for n in range(len(labels))]
    angles += angles[:1]

    # Normalize metrics to [0,1]
    if show_first_model:
        metrics_scaled = metrics / metrics.max(axis=1, keepdims=True)
    else:
        metrics_scaled = metrics / metrics[:, 1:].max(axis=1, keepdims=True)

    # Create plot
    fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={'polar': True})
    ax.spines['polar'].set_visible(False)
    ax.set_facecolor('#f9f9f9')
    ax.set_yticks([])
    ax.set_rgrids([0.2, 0.4, 0.6, 0.8, 1.0],
                  labels=['0.2', '0.4', '0.6', '0.8', '1.0'],
                  angle=0,
                  color='gray',
                  alpha=0.2)

    model_colors = colors[:len(enabled_models)]
    for i, (model_name, color) in enumerate(zip(enabled_models[:-1], model_colors)):
        if i == 0 and not show_first_model:
            continue
        vals = metrics_scaled[:, i].tolist()
        vals += vals[:1]
        ax.plot(angles, vals, label=model_name, color=color, linewidth=2)
        ax.fill(angles, vals, color=color, alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([lbl.replace(' ', '\n') for lbl in labels])
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), title="Models")

    plt.title(f"Spider Plot - Params: {param_values_str}", y=1.08)
    fig.tight_layout()
    plt.savefig(f"{save_folder}/SpiderPlot{param_values_str}.png", dpi=300)
    plt.show()

######################## INTERACTIVE LOOP ########################
while True:
    print("\nEnter desired values for each parameter (or 'q' to quit at any time).")
    
    # 1) Array Size
    user_in = input(f"array_size options={array_size}: ")
    if user_in.lower() == 'q': 
        break
    try:
        array_size_val = float(user_in)
    except ValueError:
        print("Invalid input. Please try again.")
        continue
    
    # 2) Parasitic Resistance
    user_in = input(f"parasiticResistance options~[{parasiticResistance[0]}..{parasiticResistance[-1]}]: ")
    if user_in.lower() == 'q':
        break
    try:
        parasiticResistance_val = float(user_in)
    except ValueError:
        print("Invalid input. Please try again.")
        continue

    # 3) Memory Window
    user_in = input(f"memoryWindow options={memoryWindow}: ")
    if user_in.lower() == 'q':
        break
    try:
        memoryWindow_val = float(user_in)
    except ValueError:
        print("Invalid input. Please try again.")
        continue

    # 4) Variability Index
    user_in = input(f"variability index from 0 to {variabilitySize-1}: ")
    if user_in.lower() == 'q':
        break
    try:
        variability_val = float(user_in)
    except ValueError:
        print("Invalid input. Please try again.")
        continue

    # 5) Hrs Percentage
    user_in = input(f"Hrs Percentage options={Rhrs_percentage}: ")
    if user_in.lower() == 'q':
        break
    try:
        Rhrs_percentage_val = float(user_in)
    except ValueError:
        print("Invalid input. Please try again.")
        continue

    # Find nearest indices
    d0 = find_nearest_index(array_size,          array_size_val)
    d1 = find_nearest_index(parasiticResistance, parasiticResistance_val)
    d2 = find_nearest_index(memoryWindow,        memoryWindow_val)
    d3 = find_nearest_index(range(variabilitySize), variability_val)
    d4 = find_nearest_index(Rhrs_percentage,     Rhrs_percentage_val)

    # Now compute base_metrics and (optionally) robustness_metrics for the 1D slices

    # Example: Single-slice "Current Accuracy" for each model
    # (We use the reciprocal of the Metric for demonstration; adapt to your usage)
    # sub_metric: shape (modelSize,)
    sub_metric = Metric[d0, d1, d2, d3, d4, :]  # a single slice across models

    # In this example, let's define:
    # base_metric ~ (1 / sub_metric), just as a placeholder
    base_metrics = {
        'Current Accuracy': 1 / (sub_metric + 1e-12),  # add small epsilon to avoid /0
    }

    # Let's also compute a "robustness" measure across one dimension at a time
    # We'll do a 1D slice approach: for each dimension, fix everything except that dimension
    # In reality, you'd tailor it to your own definition of "robustness".
    arr_slice        = Metric[:,   d1, d2, d3, d4, :]  # vary array_size only
    parasite_slice   = Metric[d0,  :,  d2, d3, d4, :]  # vary parasiticResistance only
    memwin_slice     = Metric[d0,  d1, :,  d3, d4, :]  # vary memoryWindow only
    variab_slice     = Metric[d0,  d1, d2, :,  d4, :]  # vary variability only
    sparsity_slice   = Metric[d0,  d1, d2, d3,  :,  :] # vary Rhrs_percentage only

    # Each slice is shape (N, modelSize), so we can compute std across axis=0 or axis=1
    # We'll do standard deviation across the varying dimension, then take reciprocal
    array_size_robustness        = np.reciprocal(np.std(arr_slice,      axis=0) + 1e-12)
    parasitic_res_robustness     = np.reciprocal(np.std(parasite_slice, axis=0) + 1e-12)
    memory_win_robustness        = np.reciprocal(np.std(memwin_slice,   axis=0) + 1e-12)
    variability_robustness       = np.reciprocal(np.std(variab_slice,   axis=0) + 1e-12)
    sparsity_robustness          = np.reciprocal(np.std(sparsity_slice, axis=0) + 1e-12)

    robustness_metrics = {
        'Array Size Robustness': array_size_robustness,
        'Parasitic Resistance Robustness': parasitic_res_robustness,
        'Memory Window Robustness': memory_win_robustness,
        'Variability Robustness': variability_robustness,
        'Sparsity Robustness': sparsity_robustness
    }

    # Prepare string with parameter values for saving & labeling
    param_values_str = (
        f"_as{array_size[d0]}_pr{parasiticResistance[d1]:.2f}"
        f"_mw{memoryWindow[d2]}_var{d3}_rsp{Rhrs_percentage[d4]}"
    )

    # Plot the new spider chart
    plot_spider_chart(base_metrics, robustness_metrics, param_values_str)
    
    print("Figure saved. Close the plot window to continue or press Ctrl+C to abort.")



###################### 3) Error vs Different Data #####################
data_types = ['array_size', 'parasiticResistance', 'memoryWindow', 'variability', 'sparsity']
data_arrays = [array_size, parasiticResistance, memoryWindow, np.arange(variabilitySize), Rhrs_percentage]
labels = ['Array Size', 'Parasitic Resistance', 'Memory Window', 'Variability Index', 'Hrs Percentage']
error_labels = [error_label] * len(data_types)

for i, (data_type, data_vals, label, err_lbl) in enumerate(zip(data_types, data_arrays, labels, error_labels)):
    if len(data_vals) > 1:
        # Compute mean & variance
        error_vs_data = np.mean(Metric, axis=tuple(j for j in range(5) if j != i))
        variance_vs_data = np.mean(Metric_variance, axis=tuple(j for j in range(5) if j != i))

        fig, ax = plt.subplots(figsize=(12, 7))
        for j, (model_name, color, marker) in enumerate(zip(enabled_models[:-1], model_colors, markers)):
            if j == 0 and not show_first_model:
                continue
            ax.errorbar(
                data_vals,
                error_vs_data[:, j],
                yerr=np.sqrt(variance_vs_data[:, j]),
                label=model_name,
                color=color,
                marker=marker,
                markersize=8,
                capsize=5,
                linestyle='-'
            )
        ax.set_xlabel(label)
        ax.set_ylabel(err_lbl)
        ax.set_title(f'Model Normalized Error vs {label}')
        ax.legend()
        ax.grid(True, which="both", linestyle='--', linewidth=0.5)
        fig.tight_layout()
        plt.savefig(f"{folder}/Figure_error_vs_{data_type}.png", dpi=300)
        plt.show()



