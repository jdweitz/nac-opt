# %% [markdown]
# # Neural Architecture Codesign (NAC) with Resource Utilization and Latency Estimation for ML on FPGA (rule4ml) Tutorial

# %% [markdown]
# In this tutorial, you will be able to run a search to optimize a model with respect to performance and hardware metrics. This consists of 3 stages: global search, local search, and model synthesis.

# %% [markdown]
# ### Setup

# %%
# Create an env and install requirements

!python -m venv nac_rule4ml_env

# %%
%pip install -r requirements.txt

# %% [markdown]
# ## Global Search
# In the global search, the user provides a dataset, model, and search space (defaults are provided). It is recommended to create a search space in the same structure as defined below. Global search can then be ran with user defined metrics to optimize for, i.e. accuracy, BOPs, average hardware utilization, clock cycles

# %% [markdown]
# ### Dataset loading & Preprocessing (notebook)

# %%
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

def load_and_preprocess_data(resize_val=8, subset_size=None):
    """
    Loads the MNIST dataset, preprocesses it by resizing, normalizing, and flattening the images,
    and converts the labels to one-hot encoding. Optionally, a subset of the data can be used.
    
    Parameters:
        resize_val (int): The target height and width for resizing the images.
        subset_size (int or None): If specified, the number of samples to use from the training and validation sets.
    
    Returns:
        x_train, y_train, x_val, y_val: Preprocessed training and validation data.
    """
    # Load MNIST dataset
    (x_train_full, y_train_full), (x_val_full, y_val_full) = mnist.load_data()

    # Expand dims: from (num_samples, 28, 28) to (num_samples, 28, 28, 1)
    x_train_full = x_train_full[..., None]
    x_val_full = x_val_full[..., None]

    # Resize images
    x_train_full = tf.image.resize(x_train_full, [resize_val, resize_val]).numpy()
    x_val_full = tf.image.resize(x_val_full, [resize_val, resize_val]).numpy()

    # Normalize pixel values to [0, 1]
    x_train_full = x_train_full.astype("float32") / 255.0
    x_val_full = x_val_full.astype("float32") / 255.0

    # Flatten images: reshape (num_samples, resize_val, resize_val, 1) to (num_samples, resize_val*resize_val)
    squared_resize = resize_val ** 2
    x_train_full = x_train_full.reshape(-1, squared_resize)
    x_val_full = x_val_full.reshape(-1, squared_resize)

    # Convert labels to one-hot encoding (10 classes)
    num_classes = 10
    y_train_full = to_categorical(y_train_full, num_classes)
    y_val_full = to_categorical(y_val_full, num_classes)

    # Subset if specified
    if subset_size is not None:
        x_train = x_train_full[:subset_size]
        y_train = y_train_full[:subset_size]
        x_val = x_val_full[:subset_size]
        y_val = y_val_full[:subset_size]
    else:
        x_train, y_train, x_val, y_val = x_train_full, y_train_full, x_val_full, y_val_full

    print("x_train shape:", x_train.shape)
    return x_train, y_train, x_val, y_val

# %%
def visualize_sample(x_train, index=0, resize_val=8):
    """
    Visualizes a sample image from the training set.
    
    Parameters:
        x_train (np.array): The training data containing flattened images.
        index (int): The index of the image to visualize.
        resize_val (int): The height and width used in preprocessing.
    """
    # Reshape the flattened image back to its 2D form
    image = x_train[index].reshape(resize_val, resize_val)
    plt.imshow(image, cmap="gray")
    plt.title("Sample Image")
    plt.axis("off")
    plt.show()

x_train, y_train, x_val, y_val = load_and_preprocess_data(resize_val=8, subset_size=60000)
visualize_sample(x_train, index=0, resize_val=8)

# %% [markdown]
# ### Dataset loading & Preprocessing (terminal)

# %%
# To run at command line, as opposed to using the 2 cells above

!python mnist_preprocess.py --resize 8 --subset_size 60000 --index 0

# %%
from IPython.display import Image
Image("sample_image.png")

# %% [markdown]
# ### Define model and search space

# %%
from model_builder import load_search_space, create_model_from_config

# Load the search space from yaml
search_space = load_search_space("search_space.yaml")

# Default config by selecting the first option for each hyperparameter in yaml
default_config = {key: value[0] for key, value in search_space.items()}
print("Default configuration:", default_config)

# Build model
model = create_model_from_config(default_config, input_size=x_train.shape[1], num_classes=10) # error in the mnist_preprocess.py about the naming of x_train, maybe something to do with the subset size?

model.summary()

# Check
history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=1,
    batch_size=128,
    verbose=1
)

# %% [markdown]
# ### Run global search

# %%
!python global.py --n_trials 50 --epochs 2 --search_space search_space.yaml --board zcu102 --precision "ap_fixed<8,3>" --reuse_factor 1 --strategy Latency --objectives "accuracy,avg_hw,clock_cycles,BOPs"

# Options to select from:
# board: zcu102, pynq-z2. alveo-u200
# reuse factor: 1, 2, 4, 8, 16, 32, 64
# precision: <2,1>, <8,3>, <16,6>; ex. ap_fixed<8,3>
# strategy: Latency, Resource
# objectives: "accuracy, BOPs, avg_hw, LUT (%), BRAM (%), DSP (%), FF (%), clock_cycles"

# Example of using all objectives below
# !python global.py --n_trials 20 --epochs 1 --search_space search_space.yaml --board zcu102 --precision "ap_fixed<8,3>" --reuse_factor 1 --strategy Latency --objectives "accuracy,BOPs,avg_hw,LUT (%),BRAM (%),DSP (%),FF (%), clock_cycles"

# %% [markdown]
# ### Plotting Results

# %%
!python plot_results.py --results_file global_search_results.txt --objectives "accuracy,avg_hw,clock_cycles,BOPs" --maximize "True,False,False,False"

# Define the objectives based on those chosen for the global search
# Pareto points selected if the point cannot be improved without worsening another

# %%
from IPython.display import Image
Image("pareto_fronts.png")

# %% [markdown]
# ### 3D Plot

# %%
def pareto_front_indices_general(df, objectives, maximize_flags):
    """
    Returns indices of non-dominated (Pareto optimal) points for a set of objectives.
    
    Parameters:
        df (pd.DataFrame): DataFrame with objective columns.
        objectives (list of str): List of objective names to compare.
        maximize_flags (list of bool): List of booleans indicating if each objective should be maximized.
        
    Returns:
        list: Indices of non-dominated points.
    """
    # Create a list of transformed objective arrays. If an objective is to be maximized,
    # we flip its sign so that all objectives can be treated as minimization.
    f_list = []
    for obj, maximize in zip(objectives, maximize_flags):
        f_list.append(-df[obj] if maximize else df[obj])
    
    n = len(df)
    indices = []
    
    # Loop through each point in the df, get pareto front
    for i in range(n):
        current = [f.iloc[i] for f in f_list]
        dominated = False
        for j in range(n):
            if i == j:
                continue
            other = [f.iloc[j] for f in f_list]
            # Check if other dominates current:
            if all(o <= c for o, c in zip(other, current)) and any(o < c for o, c in zip(other, current)):
                dominated = True
                break
        if not dominated:
            indices.append(i)
    
    return indices

# %%
import plotly.graph_objects as go

def plot_3d_with_heatmap(df, objectives_info):
    """
    Plots a 3D scatter plot using the first three objectives as axes
    and the fourth objective as a heat map (color).

    Parameters:
        df (pd.DataFrame): DataFrame containing the objective data.
        objectives_info (list of tuples): A list of four tuples (name, maximize),
            where the first three correspond to the x, y, and z dimensions,
            and the fourth is used for the color mapping.
    """
    if len(objectives_info) != 4:
        raise ValueError("Exactly 4 objectives must be provided.")
    
    # Unpack objective names and maximize flags.
    obj1, obj2, obj3, obj4 = [info[0] for info in objectives_info]
    max1, max2, max3, _ = [info[1] for info in objectives_info]
    
    # Use the generalized Pareto front function for the first three objectives.
    pareto_indices = pareto_front_indices_general(df, [obj1, obj2, obj3], [max1, max2, max3])
    pareto_points = df.iloc[pareto_indices]
    
    fig = go.Figure()
    
    # Trace for all trials, colored by the 4th objective.
    fig.add_trace(go.Scatter3d(
        x=df[obj1],
        y=df[obj2],
        z=df[obj3],
        mode="markers",
        marker=dict(
            size=5,
            color=df[obj4],
            colorscale="Viridis",
            opacity=0.6,
            colorbar=dict(title=obj4)
        ),
        name="All Trials"
    ))
    
    # Trace for Pareto front points.
    fig.add_trace(go.Scatter3d(
        x=pareto_points[obj1],
        y=pareto_points[obj2],
        z=pareto_points[obj3],
        mode="markers",
        marker=dict(
            size=8,
            color=pareto_points[obj4],
            colorscale="Viridis",
            symbol="diamond",
            colorbar=dict(title=obj4)
        ),
        name="Pareto Front (3D)"
    ))
    
    # Update layout with axis titles and overall title.
    fig.update_layout(
        title=f"3D Pareto Front: {obj1} vs {obj2} vs {obj3} with {obj4} as color",
        scene=dict(
            xaxis_title=obj1,
            yaxis_title=obj2,
            zaxis_title=obj3
        )
    )
    
    fig.show()

# %%
from plot_results import parse_results

# Define the objective names in order of appearance based on global search
objective_names = ["accuracy", "avg_hw", "clock_cycles", "BOPs"]

# Parse
df = parse_results("global_search_results.txt", objective_names)
print("Parsed results:")
print(df)

objectives_info = [
    ("accuracy", True),    # maximize
    ("avg_hw", False),   # minimize
    ("clock_cycles", False),  # minimize
    ("BOPs", False)        # used as heat map (color)
]

# %%
plot_3d_with_heatmap(df, objectives_info)

# %%
import pandas as pd

# Filter for rows with accuracy >= 50%
df_filtered = df[df['accuracy'] >= 0.5]

# Using nsmallest/nlargest
lowest_latency = df_filtered.nsmallest(1, 'clock_cycles')
lowest_resource = df_filtered.nsmallest(1, 'avg_hw')
highest_accuracy = df_filtered.nlargest(1, 'accuracy')

# Combine the rows (drop_duplicates in case one row satisfies multiple conditions)
result = pd.concat([lowest_latency, lowest_resource, highest_accuracy]).drop_duplicates()
print(result)

# %% [markdown]
# ## Local Search
# X models are selected from the global search based on the metrics provided by the user. The models are then passed through a local search, performing quantization-aware-training at various bit precisions and iterative magnitude pruning, optimizing for accuracy, to compress the models. 

# %% [markdown]
# ### Pruning with QAT

# %%
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow import keras
from qkeras import QDense, QActivation, quantizers
from qkeras.qlayers import Clip
from tensorflow.keras.utils import get_custom_objects
import numpy as np
import os

# Create a directory to save the models
save_dir = "best_models"
os.makedirs(save_dir, exist_ok=True)

# File to log iteration details
log_filename = "pruning_iterations_log.txt"
with open(log_filename, "w") as log_file:
    log_file.write("Precision,Iteration,Sparsity,Accuracy\n")

def create_qat_model(input_dim, num_classes, total_bits, int_bits):
    weight_quantizer = quantizers.quantized_bits(total_bits, int_bits, 1)
    bias_quantizer   = quantizers.quantized_bits(total_bits, int_bits, 1)
    activation_quantizer = quantizers.quantized_relu(total_bits, int_bits)
    
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        QDense(32,
               kernel_quantizer=weight_quantizer,
               bias_quantizer=bias_quantizer,
               name='q_dense_1'),
        QActivation(activation=activation_quantizer, name='q_activation_1'),
        QDense(num_classes,
               kernel_quantizer=weight_quantizer,
               bias_quantizer=bias_quantizer,
               name='q_dense_2'),
        keras.layers.Softmax()
    ])
    return model

def clone_model_with_custom_objects(model):
    custom_objects = {
        "QDense": QDense,
        "QActivation": QActivation,
        "quantized_bits": quantizers.quantized_bits,
        "quantized_relu": quantizers.quantized_relu,
        "Clip": Clip
    }
    get_custom_objects().update(custom_objects)
    return tf.keras.models.clone_model(model)

def rewind_weights(model, original_weights):
    """Rewind surviving weights (nonzero) to their original initialization."""
    current_weights = model.get_weights()
    new_weights = []
    for orig, curr in zip(original_weights, current_weights):
        mask = np.where(curr != 0, 1.0, 0.0)
        new_weights.append(orig * mask)
    model.set_weights(new_weights)
    return model

# x_train, y_train, x_val, y_val are already defined
precision_pairs = [(2, 1), (8, 3), (16, 6)]
results = {}

for total_bits, int_bits in precision_pairs:
    print(f"\n=== Running for precision <{total_bits},{int_bits}> ===")
    
    # Create and clone model with the desired precision.
    model = create_qat_model(input_dim=x_train.shape[1], num_classes=10,
                             total_bits=total_bits, int_bits=int_bits)
    model = clone_model_with_custom_objects(model)
    
    # Save initial weights for lottery ticket rewinding.
    original_weights = model.get_weights()
    
    # Compile the base model.
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    num_iterations = 5
    best_val_accuracy = 0.0
    best_weights = None
    
    for iteration in range(num_iterations):
        # Compute target sparsity: prune 20% of the remaining weights each iteration.
        # target_sparsity = 1 - (0.8)^(iteration+1)
        target_sparsity = 1 - (0.8 ** (iteration + 1))
        print(f"\nIteration {iteration+1}/{num_iterations} - Target sparsity: {target_sparsity:.4f}")
        
        # Set up the pruning schedule.
        pruning_params = {
            'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(
                target_sparsity=target_sparsity,
                begin_step=0,
                frequency=100
            )
        }
        
        # Wrap the model with pruning.
        pruned_model = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)
        pruned_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        # Use the pruning update callback.
        callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]
        
        # Train for several epochs (adjust epochs as needed).
        pruned_model.fit(x_train, y_train,
                         validation_data=(x_val, y_val),
                         epochs=1,
                         batch_size=128,
                         callbacks=callbacks,
                         verbose=1)
        
        # Strip pruning wrappers.
        model_stripped = tfmot.sparsity.keras.strip_pruning(pruned_model)
        model_stripped.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        # Evaluate the pruned model BEFORE rewinding.
        score = model_stripped.evaluate(x_val, y_val, verbose=0)
        val_loss, val_acc = score
        print(f"Iteration {iteration+1} evaluation BEFORE rewinding (loss, accuracy): {score}")
        
        # Log the details for this iteration (using pre-rewind accuracy).
        with open(log_filename, "a") as log_file:
            log_file.write(f"<{total_bits},{int_bits}>, {iteration+1}, {target_sparsity:.4f}, {val_acc:.4f}\n")
        
        # Save best weights if validation accuracy improves.
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            best_weights = model_stripped.get_weights()
            print(f"--> New best model found with accuracy: {best_val_accuracy:.4f}")
        
        # Now perform the rewinding step for the lottery ticket experiment.
        model = rewind_weights(model_stripped, original_weights)
    
    # At the end of iterations, if best weights were found, restore and save the best model.
    if best_weights is not None:
        model.set_weights(best_weights)
        model.save(f"{save_dir}/best_model_{total_bits}_{int_bits}.h5")
        print(f"Best model for precision <{total_bits},{int_bits}> saved.")
    else:
        print("No improvement during pruning iterations.")
    
    final_score = model.evaluate(x_val, y_val, verbose=1)
    results[f"<{total_bits},{int_bits}>"] = final_score

print("\nOverall results per precision:")
for prec, score in results.items():
    print(f"{prec}: {score}")

print(f"\nIteration details logged to {log_filename}")

# %%
import re
import matplotlib.pyplot as plt
from collections import defaultdict

results_file = "pruning_iterations_log.txt"
results = defaultdict(list)

# Regex to parse each line.
pattern = r'^(<[^>]+>),\s*(\d+),\s*([\d.]+),\s*([\d.]+)$'

with open(results_file, "r") as f:
    header = f.readline()  # skip header
    for line in f:
        line = line.strip()
        if not line:
            continue
        m = re.match(pattern, line)
        if m:
            precision = m.group(1)
            iteration = int(m.group(2))
            target_sparsity = float(m.group(3))
            val_acc = float(m.group(4))
            results[precision].append((iteration, target_sparsity, val_acc))
        else:
            print("Line didn't match:", line)

# Debug print the parsed results.
print("Parsed results:", dict(results))

# Define a custom sort key to sort by the first number inside the angle brackets.
def precision_sort_key(prec_str):
    m = re.match(r'<(\d+),', prec_str)
    return int(m.group(1)) if m else float('inf')

# Plot the results.
plt.figure(figsize=(10, 6))
# Sort the dictionary keys using our custom sort key.
for precision in sorted(results.keys(), key=precision_sort_key):
    data = results[precision]
    # Sort data by iteration.
    data.sort(key=lambda x: x[0])
    iterations, sparsity_values, accuracy_values = zip(*data)
    plt.plot(sparsity_values, accuracy_values, marker='o', linestyle='-', label=precision)

plt.xlabel("Target Sparsity")
plt.ylabel("Validation Accuracy")
plt.title("Target Sparsity vs. Validation Accuracy for Different Precisions")
plt.legend(title="Precision")
plt.grid(True, linestyle="--", alpha=0.6)

# Flip (invert) the x-axis so that Sparsity is reversed.
plt.gca().invert_xaxis()

plt.show()

# %%
def estimate_bops(precision, target_sparsity, baseline=1e6): # May need to alter
    """
    Estimates BOPs as: baseline * (total_bits/32)^2 * (1 - target_sparsity).
    """
    m = re.match(r'<(\d+),', precision)
    if m:
        total_bits = int(m.group(1))
        quant_scale = (total_bits / 32) ** 2
        effective_bops = baseline * quant_scale * (1 - target_sparsity)
        return effective_bops
    else:
        return baseline * (1 - target_sparsity)


# Plot BOPs vs accuracy

plt.figure(figsize=(10, 6))
for precision in sorted(results.keys(), key=precision_sort_key):
    data = results[precision]
    data.sort(key=lambda x: x[0])
    iterations, sparsity_values, accuracy_values = zip(*data)
    # Compute estimated BOPs for each iteration
    bops_values = [estimate_bops(precision, sparsity) for sparsity in sparsity_values]
    plt.plot(bops_values, accuracy_values, marker='o', linestyle='-', label=precision)

plt.xlabel("Estimated BOPs")
plt.ylabel("Validation Accuracy")
plt.title("Estimated BOPs vs. Validation Accuracy for Different Precisions")
plt.legend(title="Precision")
plt.grid(True, linestyle="--", alpha=0.6)
plt.xscale("log")
plt.show()

# %% [markdown]
# ## Model Synthesis
# The selected models can then be synthesized for FPGA deployment. The resulting resources can be compared to the rule4ml estimation as well.

# %% [markdown]
# ### Synthesize

# %%
# Select 1 or X models and run hardware estimation with rule4ml

# Define model
# Will be made such that you can get a model from the local search results and load in the weights and then synthesize that model, for now using the default

from model_builder import load_search_space, create_model_from_config

# Load the search space from yaml
search_space = load_search_space("search_space.yaml")

# Default configuration by selecting the first option for each hyperparameter in yaml
default_config = {key: value[1] for key, value in search_space.items()} # value[0] corresponds to the default config, try value[1]
print("Configuration:", default_config)

# Build model
model = create_model_from_config(default_config, input_size=x_train.shape[1], num_classes=10)

model.summary()

# Quick check
history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=10,
    batch_size=128,
    verbose=1
)

# To fix: adjust model_builder.py so that there is no 3rd layer when 2 num_layers is selected

# %%
hls_simple_config = {
    "model": {
        "precision": "ap_fixed<8,3>",
        "reuse_factor": 1,
        "strategy": "Latency"
    },
    "board": "alveo-u200"
}

# Options to select from:
# board: zcu102, pynq-z2, alveo-u200 (alveo-u200 > zcu102 > pynq-z2)
# reuse factor: 1-64
# precision: ap_fixed<8,3>
# strategy: Latency, Resource
# objectives: "accuracy, BOPs, avg_hw, LUT (%), BRAM (%), DSP (%), FF (%), clock_cycles"

# !python global.py --n_trials 50 --epochs 2 --search_space search_space.yaml --board zcu102 --precision "ap_fixed<8,3>" --reuse_factor 1 --strategy Latency --objectives "accuracy,avg_hw,clock_cycles,BOPs"

# %%
# Estimate resources with rule4ml

# Error with tensorflow

from rule4ml.models.estimators import MultiModelEstimator

import pandas as pd
pd.options.display.max_columns = None
pd.options.display.max_rows = None

global_estimator = MultiModelEstimator()
global_estimator.load_default_models()

# pred_df = global_estimator.predict([model], [hls_simple_config])

prediction_df = global_estimator.predict(model)

# pred_df

if not prediction_df.empty:
    prediction_df = prediction_df.groupby(
        ["Model", "Board", "Strategy", "Precision", "Reuse Factor"], observed=True
    ).mean()

prediction_df

# %%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, BatchNormalization
import hls4ml

# Load the search space from yaml
search_space = load_search_space("search_space.yaml")

# Default configuration by selecting the first option for each hyperparameter in yaml
default_config = {key: value[1] for key, value in search_space.items()} # value[0] corresponds to the default config, try value[1]
print("Configuration:", default_config)

# Build model
model = create_model_from_config(default_config, input_size=x_train.shape[1], num_classes=10)

model.summary()

# Quick check
history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=10,
    batch_size=128,
    verbose=1
)

# model = Sequential([
#     Dense(4, input_shape=(64,), name='dense_1'),
#     Dense(4, activation='relu', name='dense_2'),
#     Dense(2, activation='relu', name='dense_3'),
#     Dense(10, name='output')
# ])

# model.compile(optimizer='adam', loss='categorical_crossentropy')

# model.summary()

config = hls4ml.utils.config_from_keras_model(model, granularity='name', default_precision='fixed<8,3>', backend='Vitis')
print("Configuration:")
print(config)

# Oh I haven't added the precision or resuse factor or even the strategy

# %%
# rename the output_dir based on the name of the model

hls_model = hls4ml.converters.convert_from_keras_model(
    model, hls_config=config, backend='Vitis', output_dir='model_2_pynq-z2/hls4ml_prj', part='xc7z020clg400-1'
)

# Supported board from hls4ml documentation, VivadoAccelerator

# pynq-z2 (part: xc7z020clg400-1)

# zcu102 (part: xczu9eg-ffvb1156-2-e)

# alveo-u50 (part: xcu50-fsvh2104-2-e) # not supported by rule4ml

# alveo-u250 (part: xcu250-figd2104-2L-e) # not supported by rule4ml

# alveo-u200 (part: xcu200-fsgd2104-2-e)

# alveo-u280 (part: xcu280-fsvh2892-2L-e) # not supported by rule4ml

# %%
hls_model.compile()

# %% [markdown]
# ### Comparison of rule4ml estimates and hls4ml result

# %% [markdown]
# #### hls4ml

# %% [markdown]
# From hls4ml_prj/myproject_prj/solution1/syn/report/myproject_csynth.rpt:

# %%
# ================================================================
# == Performance Estimates
# ================================================================
# + Timing: 
#     * Summary: 
#     +--------+---------+----------+------------+
#     |  Clock |  Target | Estimated| Uncertainty|
#     +--------+---------+----------+------------+
#     |ap_clk  | 5.00 ns | 7.289 ns |   1.35 ns  |
#     +--------+---------+----------+------------+

# + Latency: 
#     * Summary: 
#     +---------+---------+----------+----------+-----+-----+----------+
#     |  Latency (cycles) |  Latency (absolute) |  Interval | Pipeline |
#     |   min   |   max   |    min   |    max   | min | max |   Type   |
#     +---------+---------+----------+----------+-----+-----+----------+
#     |       79|       79| 0.576 us | 0.576 us |   32|   32| function |
#     +---------+---------+----------+----------+-----+-----+----------+

# ================================================================
# == Utilization Estimates
# ================================================================
# * Summary: 
# +-----------------+---------+-----+--------+-------+-----+
# |       Name      | BRAM_18K| DSP |   FF   |  LUT  | URAM|
# +-----------------+---------+-----+--------+-------+-----+
# |DSP              |        -|    -|       -|      -|    -|
# |Expression       |        -|    -|       0|      4|    -|
# |FIFO             |        -|    -|       -|      -|    -|
# |Instance         |        2|   96|    8784|  11764|    -|
# |Memory           |        -|    -|       -|      -|    -|
# |Multiplexer      |        -|    -|       -|    167|    -|
# |Register         |        -|    -|     268|      -|    -|
# +-----------------+---------+-----+--------+-------+-----+
# |Total            |        2|   96|    9052|  11935|    0|
# +-----------------+---------+-----+--------+-------+-----+
# |Available        |      280|  220|  106400|  53200|    0|
# +-----------------+---------+-----+--------+-------+-----+
# |Utilization (%)  |    ~0   |   43|       8|     22|    0|
# +-----------------+---------+-----+--------+-------+-----+

# %% [markdown]
# From hls4ml_prj/vivado_synth.rpt:

# %%
# 1. Slice Logic
# --------------

# +----------------------------+------+-------+-----------+-------+
# |          Site Type         | Used | Fixed | Available | Util% |
# +----------------------------+------+-------+-----------+-------+
# | Slice LUTs*                | 5755 |     0 |     53200 | 10.82 |
# |   LUT as Logic             | 5730 |     0 |     53200 | 10.77 |
# |   LUT as Memory            |   25 |     0 |     17400 |  0.14 |
# |     LUT as Distributed RAM |    0 |     0 |           |       |
# |     LUT as Shift Register  |   25 |     0 |           |       |
# | Slice Registers            | 6526 |     0 |    106400 |  6.13 |
# |   Register as Flip Flop    | 6526 |     0 |    106400 |  6.13 |
# |   Register as Latch        |    0 |     0 |    106400 |  0.00 |
# | F7 Muxes                   |    0 |     0 |     26600 |  0.00 |
# | F8 Muxes                   |    0 |     0 |     13300 |  0.00 |
# +----------------------------+------+-------+-----------+-------+

# 2. Memory
# ---------

# +-------------------+------+-------+-----------+-------+
# |     Site Type     | Used | Fixed | Available | Util% |
# +-------------------+------+-------+-----------+-------+
# | Block RAM Tile    |    1 |     0 |       140 |  0.71 |
# |   RAMB36/FIFO*    |    0 |     0 |       140 |  0.00 |
# |   RAMB18          |    2 |     0 |       280 |  0.71 |
# |     RAMB18E1 only |    2 |       |           |       |
# +-------------------+------+-------+-----------+-------+


# 3. DSP
# ------

# +----------------+------+-------+-----------+-------+
# |    Site Type   | Used | Fixed | Available | Util% |
# +----------------+------+-------+-----------+-------+
# | DSPs           |   94 |     0 |       220 | 42.73 |
# |   DSP48E1 only |   94 |       |           |       |
# +----------------+------+-------+-----------+-------+

# %% [markdown]
# #### rule4ml

# %%
# Select 1 or X models and run hardware estimation with rule4ml

# Define model
# Will be made such that you can get a model from the local search results and load in the weights and then synthesize that model, for now using the default

from model_builder import load_search_space, create_model_from_config

# Load the search space from yaml
search_space = load_search_space("search_space.yaml")

# Default configuration by selecting the first option for each hyperparameter in yaml
default_config = {key: value[1] for key, value in search_space.items()} # value[0] corresponds to the default config, try value[1]
print("Configuration:", default_config)

# Build model
model = create_model_from_config(default_config, input_size=x_train.shape[1], num_classes=10)

model.summary()

# Quick check
history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=1,
    batch_size=128,
    verbose=1
)

# To fix: adjust model_builder.py so that there is no 3rd layer when 2 num_layers is selected

# %%
# Estimate resources with rule4ml

# Error with tensorflow

from rule4ml.models.estimators import MultiModelEstimator

import pandas as pd
pd.options.display.max_columns = None
pd.options.display.max_rows = None

global_estimator = MultiModelEstimator()
global_estimator.load_default_models()

hls_simple_config = {
    "model": {
        "precision": "ap_fixed<8,3>",
        "reuse_factor": 1,
        "strategy": "Latency"
    },
    "board": "pynq-z2"
}

prediction_df = global_estimator.predict([model], [hls_simple_config])

# prediction_df = global_estimator.predict(model)

if not prediction_df.empty:
    prediction_df = prediction_df.groupby(
        ["Model", "Board", "Strategy", "Precision", "Reuse Factor"], observed=True
    ).mean()

prediction_df

# %%



