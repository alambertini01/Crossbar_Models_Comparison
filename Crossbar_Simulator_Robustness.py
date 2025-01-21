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
    DMRModel_acc("DMR_v1"),
    DMRModel_new("DMR"),
    DMRModel_new("DMR_torch"),
    GammaModel("Gamma_torch"),
    GammaModel_acc("Gamma_acc_v1"),
    GammaModel_acc_v2("γ"),
    alpha_beta("alpha_beta_old"),
    alpha_beta_acc("αβ-matrix"),
    CrossSimModel("CrossSim"),
    CrossSimModel("CrossSim03", Verr_th=0.3, hide_convergence_msg=True),
    CrossSimModel("CrossSim1", Verr_th=1e-1),
    CrossSimModel("CrossSim2", Verr_th=1e-2),
    CrossSimModel("CrossSim3", Verr_th=1e-3),
    CrossSimModel("CrossSim7", Verr_th=1e-7),
    LTSpiceModel("LTSpice"),
    NgSpiceModel("NgSpice"),
    NgSpiceNonLinearModel("NgSpiceNonLinear"),
    MemtorchModelCpp("Memtorch"),
    MemtorchModelPython("Memtorch_python")
]

# Enabled models
enabled_models = ["Ideal", "Jeong_avg", "DMR", "αβ-matrix"]
reference_model = "CrossSim"
enabled_models.append(reference_model)

modelSize = len(enabled_models)
show_first_model = False
show_reference_model = False
current_Metric = 1

robustness_flag = True  # Enable/disable 1D-slice robustness computation

# Crossbar dimensions sweep
array_size = np.arange(16, 100, 16)
# Sparsity of the matrix
Rhrs_percentage = np.arange(10, 100, 10)
# Parasitic resistance
parasiticResistance = np.arange(0.1, 5, 0.5)
# Memory window (ratio Hrs/Lrs)
memoryWindow = np.arange(20, 100, 10)
# Number of different variability instances
variabilitySize = 5

# Low resistance programming value
R_lrs = 1000
# Input voltages parameters
v_On_percentage = 100
population = [1, 0.0]
v_flag = 1

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
    "αβ": "r",
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

###############################################################################
#             3D PLOTTING (SOLID COLOR PER MODEL) - LEGEND ON EACH FIG        #
###############################################################################
# Explanation:
#   1) We skip the first model if show_first_model=False.
#   2) We skip the reference model if show_reference_model=False.
#   3) We compute z_global_min/max only over the "visible models."
#   4) Each figure plots one surface per visible model, in a single color.
#   5) A legend is placed on EACH figure, anchored outside to the right.
#   6) We place x,y tick labels & axis labels on the floor plane (z=z_global_min).

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches


# Decide which models are "visible"
visible_models = list(range(len(enabled_models)))

if not show_first_model and 0 in visible_models:
    visible_models.remove(0)

if (not show_reference_model) and (reference_model in enabled_models):
    ref_idx = enabled_models.index(reference_model)
    if ref_idx in visible_models:
        visible_models.remove(ref_idx)

if len(visible_models) == 0:
    print("No models to plot! Check your flags/reference_model.")
    # Optionally return or skip the rest
    # return

# Compute the global z-range for just the visible models
visible_data = Metric[..., visible_models]
z_global_min = visible_data.min()
z_global_max = visible_data.max()

# Build a legend (patches) for just the visible models
legend_patches = []
for midx in visible_models:
    legend_patches.append(
        mpatches.Patch(color=model_colors[midx], label=enabled_models[midx])
    )

# Helper: build an integer-based meshgrid
def create_meshgrid_for_3d(x_len, y_len):
    return np.meshgrid(np.arange(x_len), np.arange(y_len), indexing='ij')

# Helper: place X/Y labels & ticks on the "floor" (z=z_global_min) so they don't overlap
def place_3d_labels_and_ticks(ax, x_vals, y_vals, x_label, y_label, zfloor, offset=0.7):
    """
    Removes default axis ticks and places text labels in 3D at z=zfloor,
    'near' the edges. Adjust offset & rotation as needed.
    """
    ax.set_xticks([])
    ax.set_yticks([])

    # Place X ticks along the "front edge" (y=-offset)
    for i, lbl in enumerate(x_vals):
        ax.text(i, -offset, zfloor, lbl, ha='center', va='top', rotation=0, color='black')

    # Place Y ticks along the "left edge" (x=-offset)
    for j, lbl in enumerate(y_vals):
        ax.text(-offset, j, zfloor, lbl, ha='right', va='center', rotation=90, color='black')

    # Axis labels in the "front-left corner"
    ax.text(
        (len(x_vals) - 1) / 2, -2*offset, zfloor,
        x_label, ha='center', va='top', color='black'
    )
    ax.text(
        -2*offset, (len(y_vals) - 1) / 2, zfloor,
        y_label, ha='center', va='center', rotation=90, color='black'
    )

# Dimension labels for each axis
dim_labels_dict = {
    'array_size': [f"{sz}x{sz}" for sz in array_size],
    'parasiticResistance': [f"{p:.2f}" for p in parasiticResistance],
    'memoryWindow': [f"{mw}" for mw in memoryWindow],
    'hrsPercentage': [f"{r}%" for r in Rhrs_percentage],
    'variability': [str(v) for v in range(variabilitySize)],
}

###############################################################################
# FIGURE 1: Array Size vs. Parasitic Resistance
###############################################################################
Z_data_par = np.mean(Metric, axis=(2, 3, 4))
# shape -> (len(array_size), len(parasiticResistance), numModels)

X_par, Y_par = create_meshgrid_for_3d(len(array_size), len(parasiticResistance))
fig1 = plt.figure(figsize=(10, 7))
ax1 = fig1.add_subplot(111, projection='3d')

# Z range
ax1.set_zlim(z_global_min, z_global_max)
ax1.set_zlabel("Current Error (%)")

# Place custom ticks/labels on floor
place_3d_labels_and_ticks(
    ax1,
    x_vals=dim_labels_dict['array_size'],
    y_vals=dim_labels_dict['parasiticResistance'],
    x_label="Array Size",
    y_label="Parasitic R",
    zfloor=z_global_min,
    offset=0.7
)

# Plot each visible model
for m_idx in visible_models:
    surface_data = Z_data_par[..., m_idx]
    ax1.plot_surface(
        X_par, Y_par, surface_data,
        rstride=1, cstride=1,
        color=model_colors[m_idx],
        linewidth=0, antialiased=False,
        alpha=0.9
    )

# Add legend on the right
ax1.legend(handles=legend_patches, loc='upper left', bbox_to_anchor=(1.05, 1.0))
ax1.view_init(elev=25, azim=-60)
fig1.tight_layout()
plt.savefig(f"{folder}/3D_Error_ArraySize_vs_Parasitic.png", dpi=300, bbox_inches='tight')
plt.show()

###############################################################################
# FIGURE 2: Array Size vs. Memory Window
###############################################################################
Z_data_mem = np.mean(Metric, axis=(1, 3, 4))
# shape -> (len(array_size), len(memoryWindow), numModels)

X_mem, Y_mem = create_meshgrid_for_3d(len(array_size), len(memoryWindow))
fig2 = plt.figure(figsize=(10, 7))
ax2 = fig2.add_subplot(111, projection='3d')

ax2.set_zlim(z_global_min, z_global_max)
ax2.set_zlabel("Current Error (%)")

place_3d_labels_and_ticks(
    ax2,
    x_vals=dim_labels_dict['array_size'],
    y_vals=dim_labels_dict['memoryWindow'],
    x_label="Array Size",
    y_label="Memory Window",
    zfloor=z_global_min,
    offset=0.7
)

for m_idx in visible_models:
    surface_data = Z_data_mem[..., m_idx]
    ax2.plot_surface(
        X_mem, Y_mem, surface_data,
        rstride=1, cstride=1,
        color=model_colors[m_idx],
        linewidth=0, antialiased=False,
        alpha=0.9
    )

ax2.legend(handles=legend_patches, loc='upper left', bbox_to_anchor=(1.05, 1.0))
ax2.view_init(elev=25, azim=-60)
fig2.tight_layout()
plt.savefig(f"{folder}/3D_Error_ArraySize_vs_MemoryWindow.png", dpi=300, bbox_inches='tight')
plt.show()

###############################################################################
# FIGURE 3: Array Size vs. HRS Percentage
###############################################################################
Z_data_hrs = np.mean(Metric, axis=(1, 2, 3))
# shape -> (len(array_size), len(Rhrs_percentage), numModels)

X_hrs, Y_hrs = create_meshgrid_for_3d(len(array_size), len(Rhrs_percentage))
fig3 = plt.figure(figsize=(10, 7))
ax3 = fig3.add_subplot(111, projection='3d')

ax3.set_zlim(z_global_min, z_global_max)
ax3.set_zlabel("Current Error (%)")

place_3d_labels_and_ticks(
    ax3,
    x_vals=dim_labels_dict['array_size'],
    y_vals=dim_labels_dict['hrsPercentage'],
    x_label="Array Size",
    y_label="HRS %",
    zfloor=z_global_min,
    offset=0.7
)

for m_idx in visible_models:
    surface_data = Z_data_hrs[..., m_idx]
    ax3.plot_surface(
        X_hrs, Y_hrs, surface_data,
        rstride=1, cstride=1,
        color=model_colors[m_idx],
        linewidth=0, antialiased=False,
        alpha=0.9
    )

ax3.legend(handles=legend_patches, loc='upper left', bbox_to_anchor=(1.05, 1.0))
ax3.view_init(elev=25, azim=-60)
fig3.tight_layout()
plt.savefig(f"{folder}/3D_Error_ArraySize_vs_HRSperc.png", dpi=300, bbox_inches='tight')
plt.show()

###############################################################################
# FIGURE 4: Array Size vs. Variability Index
###############################################################################
Z_data_var = np.mean(Metric, axis=(1, 2, 4))
# shape -> (len(array_size), variabilitySize, numModels)

X_var, Y_var = create_meshgrid_for_3d(len(array_size), variabilitySize)
fig4 = plt.figure(figsize=(10, 7))
ax4 = fig4.add_subplot(111, projection='3d')

ax4.set_zlim(z_global_min, z_global_max)
ax4.set_zlabel("Current Error (%)")

place_3d_labels_and_ticks(
    ax4,
    x_vals=dim_labels_dict['array_size'],
    y_vals=dim_labels_dict['variability'],
    x_label="Array Size",
    y_label="Variability Index",
    zfloor=z_global_min,
    offset=0.7
)

for m_idx in visible_models:
    surface_data = Z_data_var[..., m_idx]
    ax4.plot_surface(
        X_var, Y_var, surface_data,
        rstride=1, cstride=1,
        color=model_colors[m_idx],
        linewidth=0, antialiased=False,
        alpha=0.9
    )

ax4.legend(handles=legend_patches, loc='upper left', bbox_to_anchor=(1.05, 1.0))
ax4.view_init(elev=25, azim=-60)
fig4.tight_layout()
plt.savefig(f"{folder}/3D_Error_ArraySize_vs_Variability.png", dpi=300, bbox_inches='tight')
plt.show()



######################## SPIDER PLOT FUNCTION ########################

def find_nearest_index(array, value):
    """Return index of array element closest to 'value'."""
    arr_np = np.array(array)
    return np.abs(arr_np - value).argmin()

def plot_spider_chart(base_metrics, robustness_metrics, title_suffix="", save_folder="Results"):
    """
    Same logic as your original spider plot. 
    base_metrics, robustness_metrics must each be length (#models) or (#models-1).
    title_suffix is appended to the figure title and filename. 
    """
    # Merge metrics
    all_metrics = {**base_metrics, **robustness_metrics}
    # Filter out non-finite
    valid_metrics = {k: v for k, v in all_metrics.items() if np.all(np.isfinite(v))}

    # Prepare for plotting
    labels = list(valid_metrics.keys())
    metrics = np.array(list(valid_metrics.values()))  # shape: (#metrics, #models or #models-1)

    # Angles for the spider chart
    angles = [n / float(len(labels)) * 2 * np.pi for n in range(len(labels))]
    angles += angles[:1]  # close the polygon

    # Normalization logic (same as your code)
    if show_first_model:
        metrics_scaled = metrics / metrics.max(axis=1, keepdims=True)
    else:
        # Exclude the first column from the maximum, as in your original code
        metrics_scaled = metrics / metrics[:, 1:].max(axis=1, keepdims=True)

    # Create the figure
    fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={'polar': True})
    ax.spines['polar'].set_visible(False)
    ax.set_facecolor('#f9f9f9')
    ax.set_yticks([])
    ax.set_rgrids([0.2, 0.4, 0.6, 0.8, 1.0],
                  labels=['0.2', '0.4', '0.6', '0.8', '1.0'],
                  angle=0,
                  color='gray',
                  alpha=0.2)

    # Plot each model
    model_colors = colors[:len(enabled_models)]
    for i, (model_name, color) in enumerate(zip(enabled_models, model_colors)):
        # Skip the last model if that is your reference (like in your original code).
        # Typically you used `enabled_models[:-1]`. So let's do the same:
        if i == len(enabled_models) - 1:
            break

        # If the user doesn't want to show the first model:
        if i == 0 and not show_first_model:
            continue

        # metrics_scaled has shape (#metrics, #models). The i-th column is the i-th model.
        vals = metrics_scaled[:, i].tolist()
        vals += vals[:1]  # close the polygon
        ax.plot(angles, vals, label=model_name, color=color, linewidth=2)
        ax.fill(angles, vals, color=color, alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([lbl.replace(' ', '\n') for lbl in labels])
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), title="Models")

    plt.title(f"Spider Plot {title_suffix}", y=1.08)
    fig.tight_layout()

    # Save & show
    fig_fn = f"{save_folder}/SpiderPlot{title_suffix}.png"
    plt.savefig(fig_fn, dpi=300)
    plt.show()

####################### MAIN INTERACTIVE LOOP ##########################


robustness_flag = True  # toggle the 1D-slice approach if True

while True:
    print("\nEnter the desired values for each parameter, or type 'q' to quit.\n")

    # 1) Array Size
    user_in = input(f"array_size options={list(array_size)}: ")
    if user_in.lower() == 'q':
        break
    try:
        arr_val = float(user_in)
    except ValueError:
        print("Invalid input.")
        continue

    # 2) Parasitic Resistance
    user_in = input(
        f"parasiticResistance ~ from {parasiticResistance[0]} to {parasiticResistance[-1]}: "
    )
    if user_in.lower() == 'q':
        break
    try:
        pr_val = float(user_in)
    except ValueError:
        print("Invalid input.")
        continue

    # 3) Memory Window
    user_in = input(f"memoryWindow options={list(memoryWindow)}: ")
    if user_in.lower() == 'q':
        break
    try:
        mw_val = float(user_in)
    except ValueError:
        print("Invalid input.")
        continue

    # 4) Variability Index
    user_in = input(f"variability index from 0 to {variabilitySize - 1}: ")
    if user_in.lower() == 'q':
        break
    try:
        var_val = float(user_in)
    except ValueError:
        print("Invalid input.")
        continue

    # 5) Hrs Percentage
    user_in = input(f"Hrs Percentage options={list(Rhrs_percentage)}: ")
    if user_in.lower() == 'q':
        break
    try:
        hrs_val = float(user_in)
    except ValueError:
        print("Invalid input.")
        continue

    # Determine the nearest indices
    d0 = find_nearest_index(array_size, arr_val)
    d1 = find_nearest_index(parasiticResistance, pr_val)
    d2 = find_nearest_index(memoryWindow, mw_val)
    d3 = find_nearest_index(range(variabilitySize), var_val)
    d4 = find_nearest_index(Rhrs_percentage, hrs_val)

    ########## Compute base metrics (Current & Voltage Accuracy, Speed) ##########

    # Original approach: 
    # if not robust_flag => average across all dimensions.
    # if robust_flag => we do a single slice for each dimension.

    if not robustness_flag:
        # This is your original 5D average approach
        current_error_mean = np.mean(Current_error, axis=(0, 1, 2, 3, 4))  # shape (modelSize,)
        voltage_error_mean = np.mean(Voltage_error, axis=(0, 1, 2, 3, 4))
        sum_sim_times      = np.sum(simulation_times, axis=1)  # shape (modelSize,)
    else:
        # 1D-slice approach: fix indices for the rest
        # So current_error_mean is the slice across model dimension
        # which has shape (modelSize,)
        current_error_mean = Current_error[d0, d1, d2, d3, d4, :]
        voltage_error_mean = Voltage_error[d0, d1, d2, d3, d4, :]
        # For the times, you might want a single array_size => d0
        # or you might do something else. Let's replicate the same logic:
        sum_sim_times = simulation_times[:, d0]  # shape (modelSize,)
        # (If you'd rather keep the total sum across dimension, comment out the above line 
        # and do sum_simulation_times = np.sum(simulation_times, axis=1))

    # We skip the last index (reference model) to match your original "[:-1]" usage
    # because you do that in base_metrics below
    base_metrics = {
        'Current Accuracy': 1.0 / current_error_mean[:-1],  # invert the error => "accuracy"
        'Voltage Accuracy': 1.0 / voltage_error_mean[:-1],
        'Simulation Speed': 1.0 / sum_sim_times[:-1]
    }

    ########## Compute robustness metrics ##########
    if not robustness_flag:
        # Original multi-dimensional averaging approach
        arr_rob   = np.reciprocal(np.mean(np.std(Current_error, axis=0), axis=(0,1,2,3))[:-1])
        pr_rob    = np.reciprocal(np.mean(np.std(Current_error, axis=1), axis=(0,1,2,3))[:-1])
        mw_rob    = np.reciprocal(np.mean(np.std(Current_error, axis=2), axis=(0,1,2,3))[:-1])
        var_rob   = np.reciprocal(np.mean(np.std(Current_error, axis=3), axis=(0,1,2,3))[:-1])
        spars_rob = np.reciprocal(np.mean(np.std(Current_error, axis=4), axis=(0,1,2,3))[:-1])

        robustness_metrics = {
            'Array Size Robustness': arr_rob,
            'Parasitic Resistance Robustness': pr_rob,
            'Memory Window Robustness': mw_rob,
            'Variability Robustness': var_rob,
            'Sparsity Robustness': spars_rob
        }
    else:
        # 1D-slice approach for each dimension (fix the others)
        # We'll do it for Current_error (or Voltage_error) as in the example
        arr_slice   = Current_error[:,  d1, d2, d3, d4, :]  # vary array_size
        pr_slice    = Current_error[d0, :,  d2, d3, d4, :]  # vary pr
        mw_slice    = Current_error[d0,  d1, :,  d3, d4, :] # vary mw
        var_slice   = Current_error[d0,  d1, d2, :,  d4, :] # vary var
        hrs_slice   = Current_error[d0,  d1, d2, d3, :,  :] # vary hrs

        arr_rob   = np.reciprocal(np.std(arr_slice, axis=0) + 1e-12)[:-1]
        pr_rob    = np.reciprocal(np.std(pr_slice,  axis=0) + 1e-12)[:-1]
        mw_rob    = np.reciprocal(np.std(mw_slice,  axis=0) + 1e-12)[:-1]
        var_rob   = np.reciprocal(np.std(var_slice, axis=0) + 1e-12)[:-1]
        hrs_rob   = np.reciprocal(np.std(hrs_slice, axis=0) + 1e-12)[:-1]

        robustness_metrics = {
            'Array Size Robustness': arr_rob,
            'Parasitic Resistance Robustness': pr_rob,
            'Memory Window Robustness': mw_rob,
            'Variability Robustness': var_rob,
            'Sparsity Robustness': hrs_rob
        }

    ########## Create a suffix for the figure name ##########
    title_suffix = (
        f"_as{array_size[d0]}_pr{parasiticResistance[d1]:.2f}"
        f"_mw{memoryWindow[d2]}_var{d3}_hrs{Rhrs_percentage[d4]}"
    )

    ########## Plot the Spider Chart ##########
    plot_spider_chart(base_metrics, robustness_metrics, title_suffix, folder)

    print("Figure saved. Close it to enter new parameters or press Ctrl+C to stop.")


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



