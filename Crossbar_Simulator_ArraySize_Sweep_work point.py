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

# Flag to run one-parameter-at-a-time sweeps around a user-defined work point
partial_sweep = True

# Work point values (used for all parameters except the one being swept)
work_point = {
    'array_size': 64,
    'Rhrs_percentage': 50,
    'parasiticResistance': 2.0,
    'memoryWindow': 40
}

# Full “would-be” sweeps:
array_size_sweep = np.arange(32, 128, 16)
Rhrs_percentage_sweep = np.arange(10, 100, 10)       # Sparsity
parasiticResistance_sweep = np.arange(0.1, 5, 0.1)
memoryWindow_sweep = np.arange(20, 100, 20)

# Number of different variability instances
variabilitySize = 100

# Models initialization
Models = [
    JeongModel("Jeong"),
    JeongModel_avg("Jeong_avg"),
    JeongModel_avgv2("Jeong_torch"),
    IdealModel("Ideal"),
    DMRModel("DMR_old"),
    DMRModel_acc("DMR_acc"),
    DMRModel_new("DMR"),
    DMRModel_new("DMR_torch"),
    GammaModel("Gamma_torch"),
    GammaModel_acc("Gamma_acc_v1"),
    GammaModel_acc_v2("γ"),
    alpha_beta("alpha_beta_old"),
    alpha_beta_acc("alpha-beta"),
    CrossSimModel("CrossSim_ref"),
    CrossSimModel("CrossSim", Verr_th=0.3, hide_convergence_msg=True),
    CrossSimModel("CrossSim2", Verr_th=1e-1),
    CrossSimModel("CrossSim3", Verr_th=1e-2),
    CrossSimModel("CrossSim7", Verr_th=1e-7),
    LTSpiceModel("LTSpice"),
    NgSpiceModel("NgSpice"),
    NgSpiceNonLinearModel("NgSpiceNonLinear"),
    MemtorchModelCpp("Memtorch"),
    MemtorchModelPython("Memtorch_python")
]

# Enabled models
enabled_models = ["Ideal", "Jeong", "DMR", "alpha-beta"]
reference_model = "CrossSim7"
enabled_models.append(reference_model)

modelSize = len(enabled_models)
show_first_model = False
current_Metric = 1

# Fixed parameters
R_lrs = 1000
v_On_percentage = 100
population = [1, 0.0]
v_flag = 0

##################### DEFINE ARRAYS FOR FINAL RESULTS #################
# We keep the same shape as if we had done the full cross product:
#  - First dimension: array_size_sweep
#  - Second dimension: parasiticResistance_sweep
#  - Third dimension: memoryWindow_sweep
#  - Fourth dimension: variabilitySize
#  - Fifth dimension: Rhrs_percentage_sweep
#  - Sixth dimension: modelSize
n_size = len(array_size_sweep)
n_parasitic = len(parasiticResistance_sweep)
n_memwin = len(memoryWindow_sweep)
n_sparsity = len(Rhrs_percentage_sweep)

simulation_times = np.zeros((modelSize, n_size))  # Just an example for storing times vs array_size

# 5D for the error metrics
Current_error = np.zeros((n_size, n_parasitic, n_memwin, variabilitySize, n_sparsity, modelSize))
Voltage_error = np.zeros_like(Current_error)
Current_error_variance = np.zeros_like(Current_error)
Voltage_error_variance = np.zeros_like(Current_error)

##################### HELPER FUNCTION FOR PARTIAL SWEEPS #############
def run_sweep_one_parameter(
    param_name,
    param_values,
    # arrays where results go:
    Current_error,
    Voltage_error,
    Current_error_variance,
    Voltage_error_variance,
    simulation_times
):
    """
    param_name : one of ['array_size', 'Rhrs_percentage', 'parasiticResistance', 'memoryWindow']
    param_values : array of values to sweep for that parameter
    Fills the big arrays in the correct dimension indices, leaving other indices at 0.
    """
    # Nonlinear parameters
    rho, tox, s0 = 3000, 5, 10.62
    s = rho * tox / R_lrs
    v_ds, v_dx = 0.35, 0.7
    
    # We find the dimension index that corresponds to param_name
    #   0 -> array_size_sweep
    #   1 -> parasiticResistance_sweep
    #   2 -> memoryWindow_sweep
    #   4 -> Rhrs_percentage_sweep
    # (variability is dimension 3, modelSize is dimension 5 in the final arrays)
    dims_map = {
        'array_size': 0,
        'parasiticResistance': 1,
        'memoryWindow': 2,
        'Rhrs_percentage': 4
    }
    dim_idx = dims_map[param_name]

    # For each value in param_values, the rest of the parameters are fixed to the work point
    for idx_val, val in enumerate(param_values):
        # Determine the run-time values for each parameter
        if param_name == 'array_size':
            array_size = val
            parasitic_array = np.array([work_point['parasiticResistance']])
            memwin_array = np.array([work_point['memoryWindow']])
            rhrs_array = np.array([work_point['Rhrs_percentage']])
            # We use the dimension for array_size
            d = idx_val  # dimension index for array_size
            # And zeros for other parameter indices
            z, m, r_ = 0, 0, 0

        elif param_name == 'parasiticResistance':
            array_size = work_point['array_size']
            parasitic_array = np.array([val])
            memwin_array = np.array([work_point['memoryWindow']])
            rhrs_array = np.array([work_point['Rhrs_percentage']])
            d, m, r_ = 0, 0, 0
            z = idx_val  # dimension index for parasiticResistance

        elif param_name == 'memoryWindow':
            array_size = work_point['array_size']
            parasitic_array = np.array([work_point['parasiticResistance']])
            memwin_array = np.array([val])
            rhrs_array = np.array([work_point['Rhrs_percentage']])
            d, z, r_ = 0, 0, 0
            m = idx_val  # dimension index for memoryWindow

        else:  # 'Rhrs_percentage'
            array_size = work_point['array_size']
            parasitic_array = np.array([work_point['parasiticResistance']])
            memwin_array = np.array([work_point['memoryWindow']])
            rhrs_array = np.array([val])
            d, z, m = 0, 0, 0
            r_ = idx_val  # dimension index for Rhrs_percentage

        # For each partial run, we still do variability
        input_size = output_size = array_size
        # Only 1 element in the arrays we do not sweep
        memorySize_local = len(memwin_array)
        parasiticSize_local = len(parasitic_array)
        sparsitySize_local = len(rhrs_array)

        # Build arrays for Resistances
        R = np.zeros((parasiticSize_local, memorySize_local, variabilitySize, sparsitySize_local, input_size, output_size))
        X = np.zeros_like(R)
        S = np.ones_like(R)

        # Generate Potential vector
        Potential = np.random.choice(population, size=input_size, p=[v_On_percentage / 100, 1 - v_On_percentage / 100])
        reference_index = enabled_models.index(reference_model)

        # Temporary arrays for each iteration
        output_currents = np.zeros((output_size, parasiticSize_local, memorySize_local, modelSize))
        voltage_drops = np.zeros((input_size, output_size, parasiticSize_local, memorySize_local, modelSize))

        # For counting partial-sweep time
        totalIterations = memorySize_local * parasiticSize_local * modelSize

        print(f"Sweeping {param_name}={val}, (Dim idx={dim_idx}, local idx={idx_val})")

        # Nested loops (but effectively only 1 value for the non-swept parameters)
        for mm in range(memorySize_local):
            x_thickness = resistance_array_to_x(np.array([R_lrs * memwin_array[mm]]))
            for zz in range(parasiticSize_local):
                for v_inst in range(variabilitySize):
                    for rr in range(sparsitySize_local):
                        # Generate random Resistances for the local rhrs
                        Rstate = np.random.choice(
                            [1, 0],
                            size=(input_size, output_size),
                            p=[rhrs_array[rr] / 100, 1 - rhrs_array[rr] / 100]
                        )
                        X[zz, mm, v_inst, rr] = Rstate * x_thickness + 1e-12
                        S[zz, mm, v_inst, rr] = (1 - Rstate) * s + Rstate * s0

                        if v_flag:
                            X[zz, mm, v_inst, rr] += np.abs(np.random.randn(input_size, output_size) * v_dx / 3) * Rstate
                            S[zz, mm, v_inst, rr] += np.random.randn(input_size, output_size) * v_ds / 3
                            R[zz, mm, v_inst, rr] = calculate_resistance(X[zz, mm, v_inst, rr], S[zz, mm, v_inst, rr])
                        else:
                            # memoryWindow used from the single-element array
                            R[zz, mm, v_inst, rr] = (Rstate * R_lrs * memwin_array[mm]
                                                    + (1 - Rstate) * R_lrs)

                        # Run each model
                        for model_obj in Models:
                            if model_obj.name in enabled_models:
                                idx_model = enabled_models.index(model_obj.name)
                                NonLinear_params = (
                                    {'X': X[zz, mm, v_inst], 'S': S[zz, mm, v_inst]}
                                    if model_obj.name == 'NgSpiceNonLinear'
                                    else {'R_lrs': R_lrs, 'MW': memwin_array[mm]}
                                )
                                start_time = time.perf_counter()
                                voltage_drops[:, :, zz, mm, idx_model], output_currents[:, zz, mm, idx_model] = \
                                    model_obj.calculate(
                                        R[zz, mm, v_inst, rr],
                                        parasitic_array[zz],
                                        Potential,
                                        **NonLinear_params
                                    )
                                end_time = time.perf_counter()
                                # Only store time for array_size dimension (as an example)
                                if param_name == 'array_size':
                                    simulation_times[idx_model, d] += (end_time - start_time) / totalIterations

                        # Compute metrics
                        for idx_model, model_name in enumerate(enabled_models):
                            curr_diff = np.abs(
                                output_currents[:, zz, mm, reference_index]
                                - output_currents[:, zz, mm, idx_model]
                            ) / output_currents[:, zz, mm, reference_index]
                            vol_diff = np.abs(
                                voltage_drops[:, :, zz, mm, reference_index]
                                - voltage_drops[:, :, zz, mm, idx_model]
                            ) / voltage_drops[:, :, zz, mm, reference_index]

                            # Insert the results in the global 5D arrays
                            # [d, z, m, v_inst, r_, idx_model] but we set whichever dimension(s) we don't sweep to 0
                            Current_error[d, z, m, v_inst, r_, idx_model] = np.mean(curr_diff) * 100
                            Current_error_variance[d, z, m, v_inst, r_, idx_model] = np.std(curr_diff) * 100

                            Voltage_error[d, z, m, v_inst, r_, idx_model] = np.mean(vol_diff) * 100
                            Voltage_error_variance[d, z, m, v_inst, r_, idx_model] = np.std(vol_diff) * 100


############################ MAIN LOGIC ################################
if partial_sweep:
    # We do four separate sweeps, each for one parameter, while the others remain at the work point.
    run_sweep_one_parameter(
        'array_size',
        array_size_sweep,
        Current_error,
        Voltage_error,
        Current_error_variance,
        Voltage_error_variance,
        simulation_times
    )
    run_sweep_one_parameter(
        'parasiticResistance',
        parasiticResistance_sweep,
        Current_error,
        Voltage_error,
        Current_error_variance,
        Voltage_error_variance,
        simulation_times
    )
    run_sweep_one_parameter(
        'memoryWindow',
        memoryWindow_sweep,
        Current_error,
        Voltage_error,
        Current_error_variance,
        Voltage_error_variance,
        simulation_times
    )
    run_sweep_one_parameter(
        'Rhrs_percentage',
        Rhrs_percentage_sweep,
        Current_error,
        Voltage_error,
        Current_error_variance,
        Voltage_error_variance,
        simulation_times
    )
else:
    # If you ever want the full cross-product back, you can restore it here.
    pass

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
color_mapping = {
    "Jeong": "c",
    "DMR": "g",
    "Gamma": "r",
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
folder = f"Results/{end.year}{end.month}{end.day}_{end.hour}_{end.minute}_PartialSweeps"
if not os.path.exists(folder):
    os.makedirs(folder)

###################### 1) Processing Time Plot (array_size) ###########
# Example: We only stored times for array_size sweeps
ideal_index = enabled_models.index("Ideal")
plot_models = [model for i, model in enumerate(enabled_models) if i != ideal_index]

plt.figure()
for idx, model in enumerate(plot_models):
    # Normalized times
    # We used simulation_times[idx_model, d] in run_sweep_one_parameter for param="array_size"
    # so each row is times vs. array_size for that model
    # The shape: simulation_times is (modelSize, len(array_size_sweep))
    if np.all(simulation_times[ideal_index, :] == 0):
        # Avoid division by zero if the ideal model times are zero
        continue
    normalized_times = np.divide(
        simulation_times[idx + 1, :],
        simulation_times[ideal_index, :],
        out=np.zeros_like(simulation_times[ideal_index, :]),
        where=simulation_times[ideal_index, :] != 0
    )
    plt.plot(
        array_size_sweep,
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

###################### 2) Spider Plot (Radar) ########################
sum_simulation_times = np.sum(simulation_times, axis=1)
current_error_mean = np.mean(Current_error, axis=(0, 1, 2, 3, 4))
voltage_error_mean = np.mean(Voltage_error, axis=(0, 1, 2, 3, 4))

robustness_metrics = {
    'Array Size Robustness': np.reciprocal(np.mean(np.std(Metric, axis=0), axis=(0, 1, 2, 3))[:-1]),
    'Parasitic Resistance Robustness': np.reciprocal(np.mean(np.std(Metric, axis=1), axis=(0, 1, 2, 3))[:-1]),
    'Memory Window Robustness': np.reciprocal(np.mean(np.std(Metric, axis=2), axis=(0, 1, 2, 3))[:-1]),
    'Variability Robustness': np.reciprocal(np.mean(np.std(Metric, axis=3), axis=(0, 1, 2, 3))[:-1]),
    'Sparsity Robustness': np.reciprocal(np.mean(np.std(Metric, axis=4), axis=(0, 1, 2, 3))[:-1])
}

base_metrics = {
    'Current Accuracy': 1 / current_error_mean[:-1],
    'Voltage Accuracy': 1 / voltage_error_mean[:-1],
    'Simulation Speed': 1 / sum_simulation_times[:-1]
}

all_metrics = {**base_metrics, **robustness_metrics}
valid_metrics = {k: v for k, v in all_metrics.items() if np.all(np.isfinite(v))}

labels = list(valid_metrics.keys())
metrics = np.array(list(valid_metrics.values()))
if show_first_model:
    metrics_scaled = metrics / metrics.max(axis=1, keepdims=True)
else:
    metrics_scaled = metrics / metrics[:, 1:].max(axis=1, keepdims=True)

angles = [n / float(len(labels)) * 2 * np.pi for n in range(len(labels))]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={'polar': True})
ax.spines['polar'].set_visible(False)
ax.set_facecolor('#f9f9f9')
ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
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
plt.tight_layout()
plt.savefig(folder + '/Figure_Spider_plot.png')
plt.show()

###################### 3) Error vs Different Data #####################
data_types = ['array_size', 'parasiticResistance', 'memoryWindow', 'variability', 'sparsity']
data_arrays = [
    array_size_sweep,
    parasiticResistance_sweep,
    memoryWindow_sweep,
    np.arange(variabilitySize),
    Rhrs_percentage_sweep
]
labels = ['Array Size', 'Parasitic Resistance', 'Memory Window', 'Variability Index', 'Hrs Percentage']
error_labels = [error_label] * len(data_types)

# Plot only the sections we actually simulated
for i, (data_type, data_vals, label, err_lbl) in enumerate(zip(data_types, data_arrays, labels, error_labels)):
    if len(data_vals) > 1:
        E = np.zeros((len(data_vals), modelSize))
        V = np.zeros((len(data_vals), modelSize))
        for idx_val in range(len(data_vals)):
            # Fix all dimensions to 0 (work point) except the one we sweep (i)
            idx_slices = [0, 0, 0, slice(None), 0]  # (d, z, m, variability, r)
            idx_slices[i] = idx_val
            idx_slices_for_models = tuple(idx_slices + [slice(None)])  # add model dimension

            data_slice = Metric[idx_slices_for_models]         # shape e.g. (variabilitySize, modelSize)
            data_slice_var = Metric_variance[idx_slices_for_models]
            if i != 3:  # if not sweeping variability, average out variability dimension
                E[idx_val] = np.mean(data_slice, axis=0)
                V[idx_val] = np.mean(data_slice_var, axis=0)
            else:       # if sweeping variability, each idx_val is a single variability instance
                E[idx_val] = data_slice
                V[idx_val] = data_slice_var

        fig, ax = plt.subplots(figsize=(12, 7))
        for j, (model_name, color, marker) in enumerate(zip(enabled_models[:-1], model_colors, markers)):
            if j == 0 and not show_first_model: 
                continue
            stdev = np.sqrt(V[:, j])
            ax.errorbar(
                data_vals,
                E[:, j],
                yerr=stdev,
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
