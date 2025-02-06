import os
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import textwrap
import time
import datetime
import pytz

# Import NonLinear functions
from CrossbarModels.Functions.NonLinear import resistance_array_to_x
from CrossbarModels.Functions.NonLinear import calculate_resistance

# Import the Crossbar Models
from CrossbarModels.Crossbar_Models import *

############################ PARAMETERS ##############################

# Initialize each model instance
Models = [
    JeongModel("Jeongv1"),
    JeongModel_avg("Jeong"),
    JeongModel_avg("jeong_avg76",k=0.76),
    JeongModel_avg("jeong_avg8",k=0.8),
    JeongModel_avg("jeong_avg9",k=0.9),
    JeongModel_avg("jeong_avg92",k=0.92),
    JeongModel_avg("jeong_avg95",k=0.95),
    JeongModel_avg("Jeong_torch"),
    IdealModel("Ideal"),
    DMRModel("DMR_old"),
    DMRModel_acc("DMR"),
    DMRModel_acc("DMR_torch"),
    alpha_beta("αβ-matrix_old"),
    alpha_beta_acc("αβ-matrix"),
    alpha_beta_acc("αβ-matrix_torch"),
    CrossSimModel("CrossSim"),
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
    CrossSimModel("CrossSim_torch"),
    LTSpiceModel("LTSpice"),
    NgSpiceModel("NgSpice"),
    NgSpiceNonLinearModel("NgSpiceNonLinear"),
    MemtorchModelCpp("Memtorch_float"),
    MemtorchModelCpp_double("Memtorch"),
]

# Enabled models
enabled_models = ["Ideal", "Jeong", "DMR", "αβ-matrix"]
reference_model = "CrossSim"
enabled_models.append(reference_model)

modelSize = len(enabled_models)
show_first_model = False
show_reference_model = False
current_Metric = 1

work_point_robustness = False  # Toggle between modes


# Crossbar dimensions sweep
array_size = np.arange(100, 150, 10)
# Sparsity of the matrix
Rhrs_percentage = np.arange(10, 100, 10)
# Parasitic resistance
parasiticResistance = np.arange(0.1, 5, 0.5)
# Memory window (ratio Hrs/Lrs)
memoryWindow = np.arange(10, 100, 20)
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




###############################################################################
#                         SPIDER PLOT & ROBUSTNESS CODE                       #
###############################################################################
####################### SPIDER PLOT FUNCTION ##############################

def find_nearest_index(array, value):
    """Return index of array element closest to 'value'."""
    arr_np = np.array(array)
    return np.abs(arr_np - value).argmin()

def plot_spider_chart(base_metrics,
                      robustness_metrics,
                      enabled_models,
                      model_colors,
                      markers,
                      show_first_model=False,
                      reference_model=None,
                      param_dict=None,
                      is_work_point=False,
                      save_suffix="",
                      folder="Results"):
    ######################################################################
    # 1) MERGE METRICS + RENAME
    all_metrics = {}

    # Add remaining base metrics
    for k, v in base_metrics.items():
        if k not in all_metrics:
            all_metrics[k] = v

    # Add robustness metrics
    for k, v in robustness_metrics.items():
        all_metrics[k] = v

    # Filter out invalid (NaN/Inf) arrays
    valid_metrics = {
        k: v for k, v in all_metrics.items()
        if (v is not None) and np.all(np.isfinite(v))
    }
    if len(valid_metrics) == 0:
        print("No valid metrics to plot. Exiting radar.")
        return

    labels = list(valid_metrics.keys())                # e.g. ["Simulation Speed", "Current Accuracy", ...]
    metrics = np.array(list(valid_metrics.values()))   # shape => (N, M): N metrics x M models
    num_metrics = len(labels)
    num_models = metrics.shape[1]

    if num_metrics < 2:
        print("Radar chart requires at least 2 metrics. Exiting.")
        return

    ######################################################################
    # 2) NORMALIZE DATA [0..1]
    if show_first_model:
        max_vals = metrics.max(axis=1, keepdims=True) + 1e-12
    else:
        if num_models > 1:
            max_vals = metrics[:, 1:].max(axis=1, keepdims=True) + 1e-12
        else:
            max_vals = metrics.max(axis=1, keepdims=True) + 1e-12

    metrics_scaled = metrics / max_vals

    ######################################################################
    # 3) RADAR PLOT SETUP
    angles_base = [n / float(num_metrics) * 2 * np.pi for n in range(num_metrics)]
    angles_poly = angles_base + angles_base[:1]

    # Create the polar subplot and get both the figure and axis objects
    fig, ax = plt.subplots(figsize=(4.5, 4.5), subplot_kw={'polar': True})
    ax.set_facecolor('#f9f9f9')
    ax.spines['polar'].set_visible(False)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Hide default radial ticks
    ax.set_yticks([])

    ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

    # Set radial ticks as percentages
    radial_ticks = [0.2, 0.4, 0.6, 0.8, 1.0]
    radial_tick_labels = [f"{int(rt * 100)}%" for rt in radial_ticks]
    lines, lbls = ax.set_rgrids(
        radial_ticks,
        labels=radial_tick_labels,
        angle=0,
        color='gray',
        alpha=0.3
    )
    # Extend radial limit to 105%
    ax.set_ylim(0, 1.05)
    for lbl in lbls:
        lbl.set_fontsize(16)

    ######################################################################
    # 4) METRIC LABELS AROUND THE CIRCLE
    def two_line_label(text, width=12):
        wrapped = textwrap.wrap(text, width=width)
        if len(wrapped) <= 3:
            return "\n".join(wrapped)
        else:
            # Combine all lines beyond the 2nd index into a single third line
            return wrapped[0] + "\n" + wrapped[1] + "\n" + " ".join(wrapped[2:])
        
    wrapped_labels = [two_line_label(lbl, width=12) for lbl in labels]

    ax.set_xticks(angles_base)
    ax.set_xticklabels(wrapped_labels, fontsize=14)
    # Increase distance of labels from the circle
    ax.tick_params(axis='x', pad=20)

    ######################################################################
    # 5) PLOT EACH MODEL’S POLYGON
    legend_handles = []
    for i, model_name in enumerate(enabled_models):
        # Skip first model if user says so
        if i == 0 and not show_first_model:
            continue
        # Skip reference model if needed
        if reference_model and model_name == reference_model:
            continue
        if i >= num_models:
            break

        vals = metrics_scaled[:, i].tolist()
        vals_poly = vals + vals[:1]

        color = model_colors[i]
        marker_style = markers[i % len(markers)]

        ax.plot(
            angles_poly, vals_poly,
            label=model_name,
            color=color,
            linewidth=2,
            marker=marker_style,
            markersize=6
        )
        ax.fill(angles_poly, vals_poly, color=color, alpha=0.25)

        legend_line = mlines.Line2D(
            [], [], color=color,
            marker=marker_style,
            markersize=6,
            linewidth=2,
            label=model_name
        )
        legend_handles.append(legend_line)

    ######################################################################
    # 6) PLACE THE LEGEND ABOVE THE PLOT
    fig.legend(
        handles=legend_handles,
        loc='upper center',
        ncol=len(legend_handles),
        # title="Models",
        fontsize=16,
        title_fontsize=14,
        frameon=False,
        handletextpad=0.5,
        columnspacing=1.0,
        bbox_to_anchor=(0.5, 1)
    )

    if is_work_point and param_dict is not None:
        info_str = "Work Point:\n"
        for k, v in param_dict.items():
            info_str += f"  {k}: {v}\n"
        plt.gcf().text(
            0.02, 0.95, info_str,
            fontsize=9,
            va='bottom',
            ha='right',
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8)
        )

    # Adjust layout to ensure the radar chart remains centered and space is reserved for the legend.
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.savefig(f"{folder}/Figure_Spider_plot{save_suffix}.png", dpi=300)
    plt.show()


####################### MAIN SPIDER PLOT LOGIC ##############################

if work_point_robustness:
    ########## INTERACTIVE WORK POINT MODE ##########
    while True:
        print("\nEnter parameter values (or 'q' to quit):")
        
        # Get user inputs
        inputs = {}
        try:
            inputs['array_size'] = float(input(f"Array size ({array_size.tolist()}): "))
            inputs['parasitic'] = float(input(f"Parasitic R ({parasiticResistance.tolist()}): "))
            inputs['memory_window'] = float(input(f"Memory window ({memoryWindow.tolist()}): "))
            inputs['variability'] = float(input(f"Variability index (0-{variabilitySize-1}): "))
            inputs['hrs_percent'] = float(input(f"HRS % ({Rhrs_percentage.tolist()}): "))
        except (ValueError, KeyboardInterrupt):
            break
            
        # Find nearest indices
        indices = {
            'd0': find_nearest_index(array_size, inputs['array_size']),
            'd1': find_nearest_index(parasiticResistance, inputs['parasitic']),
            'd2': find_nearest_index(memoryWindow, inputs['memory_window']),
            'd3': int(np.clip(inputs['variability'], 0, variabilitySize-1)),
            'd4': find_nearest_index(Rhrs_percentage, inputs['hrs_percent'])
        }

        ########## Compute Metrics at Work Point ##########
        # Base metrics
        current_error = Current_error[indices['d0'], indices['d1'], indices['d2'], 
                                      indices['d3'], indices['d4'], :]
        voltage_error = Voltage_error[indices['d0'], indices['d1'], indices['d2'],
                                      indices['d3'], indices['d4'], :]
        sim_times = simulation_times[:, indices['d0']]
        
        base_metrics = {
            'Current Accuracy': 1/(current_error[:-1] + 1e-12),
            # 'Voltage Accuracy': 1/(voltage_error[:-1] + 1e-12),
            'Speed': 1/(sim_times[:-1] + 1e-12)
        }

        ########## Compute Robustness Metrics ##########
        robustness_metrics = {}
        param_slices = {
            'Array Size': Current_error[:, indices['d1'], indices['d2'], 
                                       indices['d3'], indices['d4'], :],
            'Parasitic Resistance': Current_error[indices['d0'], :, indices['d2'], 
                                        indices['d3'], indices['d4'], :],
            'Memory Window': Current_error[indices['d0'], indices['d1'], :, 
                                          indices['d3'], indices['d4'], :],
            'Variability': Current_error[indices['d0'], indices['d1'], indices['d2'], 
                                        :, indices['d4'], :],
            'Sparsity': Current_error[indices['d0'], indices['d1'], indices['d2'], 
                                  indices['d3'], :, :]
        }
        
        for param_name, slice_data in param_slices.items():
            robustness = 1/(np.std(slice_data, axis=0) + 1e-12)[:-1]  # Exclude reference
            robustness_metrics[f"{param_name} Robustness"] = robustness

        ########## Generate Title Suffix ##########
        title_suffix = (f"_AS{int(inputs['array_size'])}_PR{inputs['parasitic']:.1f}_"
                      f"MW{int(inputs['memory_window'])}_VAR{int(inputs['variability'])}_"
                      f"HRS{int(inputs['hrs_percent'])}%")

        plot_spider_chart(
            base_metrics=base_metrics,
            robustness_metrics=robustness_metrics,
            enabled_models=enabled_models,
            model_colors=model_colors,
            markers=markers,
            show_first_model=show_first_model,
            reference_model=reference_model,
            param_dict=inputs,
            is_work_point=True,
            save_suffix=title_suffix,
            folder=folder,
        )

else:
    ########## AVERAGED METRICS MODE ##########
    # Compute base metrics
    current_error_avg = np.mean(Current_error, axis=(0,1,2,3,4))
    voltage_error_avg = np.mean(Voltage_error, axis=(0,1,2,3,4))
    total_sim_times = np.sum(simulation_times, axis=1)
    
    base_metrics = {
        'Current Accuracy': 1/(current_error_avg[:-1] + 1e-12),
        # 'Voltage Accuracy': 1/(voltage_error_avg[:-1] + 1e-12),
        'Simulation Speed': 1/(total_sim_times[:-1] + 1e-12)
    }

    ########## Compute Robustness Metrics ##########
    robustness_metrics = {}
    param_dims = {
        'Array Size': 0,
        'Parasitic Resistance': 1,
        'Memory Window': 2,
        'Variability': 3,
        'Sparsity': 4
    }
    
    for param_name, dim in param_dims.items():
        std_dev = np.std(Current_error, axis=dim)
        robustness = 1/(np.mean(std_dev, axis=tuple(range(4))) + 1e-12)[:-1]
        robustness_metrics[f"{param_name} Robustness"] = robustness

    ########## Plot ##########
    plot_spider_chart(
    base_metrics=base_metrics,
    robustness_metrics=robustness_metrics,
    enabled_models=enabled_models,
    model_colors=model_colors,
    markers=markers,
    show_first_model=show_first_model,
    reference_model=reference_model,
    is_work_point=False,
    save_suffix="_AverageMetrics",
    folder=folder,
    )





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
#                                3D PLOTTING                                  #
###############################################################################

# Helper functions (repeated here for convenience):
def create_meshgrid_for_3d(x_len, y_len):
    """Create integer-based meshgrids for X, Y of shape (x_len, y_len)."""
    return np.meshgrid(np.arange(x_len), np.arange(y_len), indexing='ij')

def place_3d_labels_and_ticks(ax, x_vals, y_vals, x_label, y_label, zfloor, offset=0.7):
    """
    Removes default axis ticks and places text labels in 3D at z=zfloor,
    near the edges. Adjust offset & rotation as needed.
    """
    ax.set_xticks([])
    ax.set_yticks([])

    # Place X ticks along the "front edge" (y = -offset)
    for i, lbl in enumerate(x_vals):
        ax.text(i, -offset, zfloor, lbl, ha='center', va='top', rotation=0, color='black')

    # Place Y ticks along the "left edge" (x = -offset)
    for j, lbl in enumerate(y_vals):
        ax.text(-offset, j, zfloor, lbl, ha='right', va='center', rotation=90, color='black')

    # Axis labels in the "front-left corner"
    ax.text((len(x_vals) - 1) / 2, -2 * offset, zfloor, x_label, 
            ha='center', va='top', color='black')
    ax.text(-2 * offset, (len(y_vals) - 1) / 2, zfloor, y_label, 
            ha='center', va='center', rotation=90, color='black')

def get_surface_colors(z_data, base_color_rgba, zmin, zmax, dark_factor=0.4):
    """
    Returns an (n,m,4) array of RGBA values that interpolates from a darker
    variant of `base_color_rgba` (at z=zmin) up to the base color (at z=zmax).
    """
    n, m = z_data.shape
    facecolors = np.zeros((n, m, 4))
    # Darker version of the base color
    dark_rgba = np.array([base_color_rgba[0] * dark_factor,
                          base_color_rgba[1] * dark_factor,
                          base_color_rgba[2] * dark_factor,
                          1.0])
    base_arr = np.array(base_color_rgba)

    for i in range(n):
        for j in range(m):
            # Normalize z between 0 and 1
            ratio = (z_data[i, j] - zmin) / max((zmax - zmin), 1e-12)
            ratio = np.clip(ratio, 0, 1)
            # Linear interpolation between dark_rgba and base_color
            facecolors[i, j] = dark_rgba + ratio * (base_arr - dark_rgba)

    return facecolors

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
    # If there are no models to plot, you can skip or return
    # return

# Build a legend (patches) for just the visible models
legend_patches = [
    mpatches.Patch(color=model_colors[midx], label=enabled_models[midx])
    for midx in visible_models
]

# Compute global z-range from the visible models only
visible_data = Metric[..., visible_models]
z_global_min = visible_data.min()
z_global_max = visible_data.max()

# Dimension labels for each axis
dim_labels_dict = {
    'array_size'        : [f"{sz}x{sz}" for sz in array_size],
    'parasiticResistance': [f"{p:.2f}" for p in parasiticResistance],
    'memoryWindow'      : [f"{mw}" for mw in memoryWindow],
    'hrsPercentage'     : [f"{r}%" for r in Rhrs_percentage],
    'variability'       : [str(v) for v in range(variabilitySize)],
}

###############################################################################
# Define each 3D plot configuration in a list of dicts
plot_configs = [
    {
        'title'   : "3D_Error_ArraySize_vs_Parasitic",
        'Z_data'  : np.mean(Metric, axis=(2, 3, 4)),  # shape (len(array_size), len(parasiticResistance), #models)
        'x_vals'  : array_size,
        'y_vals'  : parasiticResistance,
        'x_label' : "Array Size",
        'y_label' : "Parasitic R",
        'x_ticks' : dim_labels_dict['array_size'],
        'y_ticks' : dim_labels_dict['parasiticResistance']
    },
    {
        'title'   : "3D_Error_ArraySize_vs_MemoryWindow",
        'Z_data'  : np.mean(Metric, axis=(1, 3, 4)),  # shape (len(array_size), len(memoryWindow), #models)
        'x_vals'  : array_size,
        'y_vals'  : memoryWindow,
        'x_label' : "Array Size",
        'y_label' : "Memory Window",
        'x_ticks' : dim_labels_dict['array_size'],
        'y_ticks' : dim_labels_dict['memoryWindow']
    },
    {
        'title'   : "3D_Error_ArraySize_vs_HRSperc",
        'Z_data'  : np.mean(Metric, axis=(1, 2, 3)),  # shape (len(array_size), len(Rhrs_percentage), #models)
        'x_vals'  : array_size,
        'y_vals'  : Rhrs_percentage,
        'x_label' : "Array Size",
        'y_label' : "HRS %",
        'x_ticks' : dim_labels_dict['array_size'],
        'y_ticks' : dim_labels_dict['hrsPercentage']
    },
    {
        'title'   : "3D_Error_ArraySize_vs_Variability",
        'Z_data'  : np.mean(Metric, axis=(1, 2, 4)),  # shape (len(array_size), variabilitySize, #models)
        'x_vals'  : array_size,
        'y_vals'  : np.arange(variabilitySize),
        'x_label' : "Array Size",
        'y_label' : "Variability Index",
        'x_ticks' : dim_labels_dict['array_size'],
        'y_ticks' : dim_labels_dict['variability']
    }
]

###############################################################################
# Main Loop: Generate each 3D plot using the above configurations
for config in plot_configs:
    # Prepare data
    Z_data = config['Z_data']  # shape -> (Nx, Ny, #models)
    Nx, Ny, _ = Z_data.shape
    
    # Create the integer-based meshgrids
    X, Y = create_meshgrid_for_3d(Nx, Ny)

    # Create the figure and axis
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Set Z-limits and label
    ax.set_zlim(z_global_min, z_global_max)
    ax.set_zlabel(error_label)

    # Place custom ticks/labels on the "floor"
    place_3d_labels_and_ticks(
        ax,
        x_vals=config['x_ticks'],
        y_vals=config['y_ticks'],
        x_label=config['x_label'],
        y_label=config['y_label'],
        zfloor=z_global_min,
        offset=0.7
    )

    # Plot each visible model with shading
    for m_idx in visible_models:
        surface_data = Z_data[..., m_idx]
        
        # Convert base color name to RGBA
        base_rgba = mcolors.to_rgba(model_colors[m_idx])
        
        # Build a color array that maps z-values to a lighter/darker shade
        facecolors = get_surface_colors(surface_data, base_rgba, z_global_min, z_global_max)

        ax.plot_surface(
            X, Y, surface_data,
            rstride=1, cstride=1,
            facecolors=facecolors,
            shade=False,  # We handle shading via facecolors
            linewidth=0,
            antialiased=False,
            alpha=1.0
        )

    # Add legend on the right
    ax.legend(handles=legend_patches, loc='upper left', bbox_to_anchor=(1.05, 1.0))

    # Adjust view angle
    ax.view_init(elev=25, azim=-60)
    fig.tight_layout()

    # Save and show
    save_name = f"{folder}/{config['title']}.png"
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    plt.show()



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



