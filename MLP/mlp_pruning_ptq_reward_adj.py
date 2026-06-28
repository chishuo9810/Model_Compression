# mlp_reward_exp.py is the first version of the method I made

import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Dense, Flatten, Softmax, Input, ReLU
from tensorflow_model_optimization.quantization.keras import vitis_quantize

from analysis import ana, select_best_ratio_from_csv, sensitivity_score, normalize_sensitivity
from DataGenerator_with_test import training_data


# ============================================================
# Parameters
# ============================================================

CALIBRATION_SAMPLES = 1024

LR_BASE = 0.05
LR_MIN = 0.01
LR_MAX = 0.2

BETA = 5.0
ALPHA = 0.2

target_acc = 0.9
target_loss = 0.2

MODEL_PATH = "./mlp_model/MLP_10layers_001full_final.h5"
DENSE_PRUNE_LOG = "dense_prune_log.csv"

USE_BEST_RATIO_AS_INIT = True

DEFAULT_INIT_RATIO = 0.5

MIN_PRUNE_RATIO = 0.1
MAX_PRUNE_RATIO = 0.9
PRUNE_METHOD = 'L1-norm' # L1-norm, L2-norm, FPGM, Random
NUM_ITERATIONS = 50


# ============================================================
# Utility functions
# ============================================================
# for ana function
def eval(model):
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-5),
        loss='sparse_categorical_crossentropy',
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='sparse_acc')]
    )
    _, acc = model.evaluate(sample_x, sample_y, verbose=0)
    return acc

# for compile
def compile_for_eval(model, learning_rate=1e-5):
     model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="sparse_acc"),
            tf.keras.metrics.SparseCategoricalCrossentropy(
                name="classification_ce",
                from_logits=False
            )
        ],
    )

# return the index to keep according to the ratio
def keep(prune_method, w, ratio):
    if prune_method == "L1-norm":
        score = np.linalg.norm(w, ord=1, axis=0)

        num_neurons = len(score)
        num_keep = int(num_neurons * (1.0 - ratio))

        # keep at least one neuron
        num_keep = max(1, num_keep)

        keep_idx = np.argsort(score)[-num_keep:]
        keep_idx = np.sort(keep_idx)

        return keep_idx
    elif prune_method == "L2-norm":
        score = np.linalg.norm(w, ord=2, axis=0)

        keep_idx = np.argsort(score)[-num_keep:]
        keep_idx = np.sort(keep_idx)

        return keep_idx

    elif prune_method == "FPGM":
        # FPGM: Filter Pruning via Geometric Median
        # 在 Dense layer 中，可以把每個 neuron 的 weight vector 當成一個 filter
        #
        # w shape: (input_dim, num_neurons)
        # neurons shape: (num_neurons, input_dim)
        neurons = w.T

        # 計算每個 neuron 跟其他 neuron 的距離
        # dist_matrix[i, j] = neuron i 跟 neuron j 的 L2 distance
        diff = neurons[:, np.newaxis, :] - neurons[np.newaxis, :, :]
        dist_matrix = np.linalg.norm(diff, ord=2, axis=2)

        # FPGM 的想法：
        # 如果某個 neuron 跟很多其他 neuron 很接近，
        # 代表它比較 redundant，較適合被 prune。
        #
        # 所以 distance_sum 越小，越接近 geometric median，越不重要。
        # distance_sum 越大，越獨特，越應該保留。
        score = np.sum(dist_matrix, axis=1)

        keep_idx = np.argsort(score)[-num_keep:]
        keep_idx = np.sort(keep_idx)

        return keep_idx

    elif prune_method == "Random":
        rng = np.random.default_rng(seed)

        keep_idx = rng.choice(
            num_neurons,
            size=num_keep,
            replace=False
        )

        keep_idx = np.sort(keep_idx)

        return keep_idx

    else:
        raise ValueError(f"Unsupported prune method: {prune_method}")

# build mlp model
def build_dynamic_mlp(input_shape, hidden_units, num_classes):
    inputs = Input(shape=input_shape)
    x = Flatten(name="flatten")(inputs)

    for i, units in enumerate(hidden_units):
        x = Dense(units, name=f"dense{i}")(x)
        x = ReLU(max_value=6, name=f"relu{i}")(x)

    x = Dense(num_classes, name=f"dense{len(hidden_units)}")(x)
    outputs = Softmax(name="softmax")(x)

    return tf.keras.Model(inputs, outputs)

# set the pruning weight to the built model, do not prune the output class
def set_pruned_weights(slim_model, original_weights, original_biases, keep_indices):
    """
    For example：
        Dense0 -> Dense1 -> Dense2 -> ... -> OutputDense
    Example:
        keep_indices[0] = neuron keep in dense0
        keep_indices[1] = neuron keep in dense1
        keep_indices[2] = neuron keep in dense2
    """
    slim_dense_layers = [
        layer for layer in slim_model.layers
        if isinstance(layer, tf.keras.layers.Dense)
    ]

    num_hidden = len(keep_indices)
    num_dense = len(slim_dense_layers)

    if num_dense != len(original_weights):
        raise ValueError(
            f"Dense layer count mismatch: slim_model has {num_dense}, "
            f"but original has {len(original_weights)}"
        )

    for i in range(num_dense):
        w = original_weights[i]
        b = original_biases[i]

        if i == 0:
            # first layer hidden Dense:
            # input feature don't need to be pruned, only prunes output neuron
            curr_keep = keep_indices[i]

            w_new = w[:, curr_keep]
            b_new = b[curr_keep]

        elif i < num_hidden:
            # middle hidden Dense:
            # pruned row according to the previous neuron
            # pruned column according to the ratio of the keep neuron
            prev_keep = keep_indices[i - 1]
            curr_keep = keep_indices[i]

            w_new = w[prev_keep, :][:, curr_keep]
            b_new = b[curr_keep]

        else:
            # last output Dense:
            # do not prune output class, only prune input row
            prev_keep = keep_indices[i - 1]

            w_new = w[prev_keep, :]
            b_new = b

        slim_dense_layers[i].set_weights([w_new, b_new])
def build_balanced_calib_from_generator(
    train_gen,
    calib_samples=1000,
    pool_multiplier=10,
    seed=42
):
    rng = np.random.default_rng(seed)
    xs = []
    ys = []
    seen = 0
    pool_size = calib_samples * pool_multiplier

    if hasattr(train_gen, "on_epoch_end"):
        train_gen.on_epoch_end()

    for batch_x, batch_y in train_gen:
        xs.append(batch_x)
        ys.append(batch_y)
        seen += batch_x.shape[0]

        if seen >= pool_size:
            break
    pool_x = np.concatenate(xs, axis=0)
    pool_y = np.concatenate(ys, axis=0)

    pool_label = pool_y.astype(int)

    classes = np.unique(pool_label)
    samples_per_class = calib_samples // len(classes)

    selected_indices = []

    for c in classes:
        class_indices = np.where(pool_label == c)[0]

        n = min(samples_per_class, len(class_indices))

        chosen = rng.choice(
            class_indices,
            size=n,
            replace=False
        )

        selected_indices.extend(chosen)

    selected_indices = np.array(selected_indices)
    
    if len(selected_indices) < calib_samples:
        remaining = np.setdiff1d(np.arange(len(pool_x)), selected_indices)
        need = calib_samples - len(selected_indices)

        extra = rng.choice(
            remaining,
            size=need,
            replace=False
        )

        selected_indices = np.concatenate([selected_indices, extra])

    selected_indices = selected_indices[:calib_samples]

    rng.shuffle(selected_indices)

    calib_x = pool_x[selected_indices].astype(np.float32)
    calib_y = pool_y[selected_indices]

    return calib_x, calib_y

def update_prune_ratios(
    prune_ratios,
    Si,
    delta_M,
    lr_t,
    min_ratio=0.1,
    max_ratio=0.9
):
    new_ratios = {}

    for layer_idx, p in prune_ratios.items():
        sensitivity = Si.get(layer_idx, 1.0)

        if delta_M < 0:
            new_p = p - lr_t * sensitivity * p # the more sensitivity, the less pruning ratio
        else:
            new_p = p + lr_t * (1.0 - sensitivity) * (max_ratio - p) # the less sensitivity, the more pruning ratio

        new_p = max(min_ratio, min(max_ratio, new_p))
        new_ratios[layer_idx] = new_p

    return new_ratios

def build_keep_indices(weights, prune_ratios, num_hidden_dense):
    keep_indices = []

    for i in range(num_hidden_dense):
        keep_idx = keep(PRUNE_METHOD, weights[i], prune_ratios[i])
        keep_indices.append(keep_idx)

    return keep_indices


def print_prune_ratios(prune_ratios, prefix=""):
    for layer_idx, ratio in prune_ratios.items():
        print(f"{prefix}layer {layer_idx} pruning ratio = {ratio:.4f}")


# ============================================================
# Main flow
# ============================================================

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
model.summary()

print("\nLoading datasets...")
train_data, val_data, test_data = training_data(type="full")


# ============================================================
# Prepare calibration data
# ============================================================

print("\nPreparing calibration data...")

xs = []
seen = 0
calib_x, calib_y = build_balanced_calib_from_generator(
    train_data,
    calib_samples=CALIBRATION_SAMPLES,
    pool_multiplier=10,
    seed=42
)

print("Calibration data shape:", calib_x.shape)


# ============================================================
# Original model evaluation
# ============================================================

print("\nEvaluating original model...")
sample_x, sample_y = next(iter(val_data))
compile_for_eval(model)
original_loss, original_acc, original_ce = model.evaluate(test_data)

print(f"Original loss = {original_ce:.4f}")
print(f"Original acc  = {original_acc:.4f}")


# ============================================================
# Automatically detect Dense layers
# ============================================================

dense_layers = [
    layer for layer in model.layers
    if isinstance(layer, tf.keras.layers.Dense)
]

if len(dense_layers) < 2:
    raise ValueError(
        "This script expects at least two Dense layers: "
        "one hidden Dense layer and one output Dense layer."
    )

hidden_dense_layers = dense_layers[:-1]
output_dense_layer = dense_layers[-1]

num_hidden_dense = len(hidden_dense_layers)
num_classes = output_dense_layer.units
input_shape = model.input_shape[1:]

print("\nDetected Dense structure:")
print(f"Input shape: {input_shape}")
print(f"Total Dense layers: {len(dense_layers)}")
print(f"Hidden Dense layers to prune: {num_hidden_dense}")
print(f"Output classes: {num_classes}")

for i, layer in enumerate(dense_layers):
    print(f"Dense layer {i}: name={layer.name}, units={layer.units}")


# ============================================================
# Save original Dense weights
# ============================================================

weights = []
biases = []

for i, layer in enumerate(dense_layers):
    layer_weights = layer.get_weights()

    if len(layer_weights) != 2:
        raise ValueError(
            f"Dense layer {i} does not have both weight and bias. "
            "This script currently assumes use_bias=True."
        )

    w, b = layer_weights
    weights.append(w)
    biases.append(b)

    print(f"Dense {i}: weight shape={w.shape}, bias shape={b.shape}")


# ============================================================
# Analysis and load best ratio and sensitivity
# ============================================================

print("\nLoading best ratio and sensitivity...")
ana(eval, model, 'L1-norm', 'dense_prune_log.csv')
best_ratio = select_best_ratio_from_csv(
    DENSE_PRUNE_LOG,
    acc_threshold=target_acc
)

best_ratio = {int(k): float(v) for k, v in best_ratio.items()}

si = sensitivity_score(DENSE_PRUNE_LOG, original_acc)
Si = normalize_sensitivity(si, num_hidden_dense)

print("\nNormalized sensitivity:")
for layer_idx, value in Si.items():
    print(f"layer {layer_idx}: Si = {value:.4f}")


# ============================================================
# Initialize pruning ratios
# ============================================================

prune_ratios = {}

for i in range(num_hidden_dense):
    if USE_BEST_RATIO_AS_INIT:
        prune_ratios[i] = best_ratio.get(i, DEFAULT_INIT_RATIO)
    else:
        prune_ratios[i] = DEFAULT_INIT_RATIO

    prune_ratios[i] = max(
        MIN_PRUNE_RATIO,
        min(MAX_PRUNE_RATIO, prune_ratios[i])
    )

print("\nInitial pruning ratios:")
print_prune_ratios(prune_ratios)


# ============================================================
# Iterative pruning ratio adjustment + PTQ
# ============================================================

best_ptq_acc = -1.0
best_ptq_loss = None
best_ptq_model = None
best_slim_model = None
best_prune_ratios = None

ptq_model = None
slim_model = None
flag = 0
temp = 0
for iteration in range(NUM_ITERATIONS):
    print("\n============================================================")
    print(f"Iteration {iteration + 1}/{NUM_ITERATIONS}")
    print("============================================================")

    # --------------------------------------------------------
    # 1. Generate keep indices
    # --------------------------------------------------------
    keep_indices = build_keep_indices(
        weights=weights,
        prune_ratios=prune_ratios,
        num_hidden_dense=num_hidden_dense
    )

    hidden_units = [len(k) for k in keep_indices]

    print("Current hidden units:", hidden_units)
    print("Current pruning ratios:")
    print_prune_ratios(prune_ratios, prefix="  ")

    # --------------------------------------------------------
    # 2. Reconstruct slim model
    # --------------------------------------------------------
    slim_model = build_dynamic_mlp(
        input_shape=input_shape,
        hidden_units=hidden_units,
        num_classes=num_classes
    )

    # --------------------------------------------------------
    # 3. Transfer pruned weights
    # --------------------------------------------------------
    set_pruned_weights(
        slim_model=slim_model,
        original_weights=weights,
        original_biases=biases,
        keep_indices=keep_indices
    )

    # --------------------------------------------------------
    # 4. PTQ
    # --------------------------------------------------------
    print("\nRunning PTQ...")

    quantizer = vitis_quantize.VitisQuantizer(
        slim_model,
        quantize_strategy="pof2s"
    )

    ptq_model = quantizer.quantize_model(
        calib_dataset=calib_x
    )

    compile_for_eval(ptq_model)

    ptq_loss, ptq_acc, ptq_ce = ptq_model.evaluate(test_data)

    print(f"PTQ loss = {ptq_ce:.4f}")
    print(f"PTQ acc  = {ptq_acc:.4f}")

    # --------------------------------------------------------
    # 5. Save best result
    # --------------------------------------------------------
    if ptq_acc > best_ptq_acc:
        best_ptq_acc = ptq_acc
        best_ptq_loss = ptq_ce
        best_ptq_model = ptq_model
        best_slim_model = slim_model
        best_prune_ratios = prune_ratios.copy()

        print("New best PTQ model found.")

    # --------------------------------------------------------
    # 6. Update pruning ratios
    # --------------------------------------------------------
    if iteration != NUM_ITERATIONS - 1:
        M = ptq_acc - ptq_ce * ALPHA
        M_target = target_acc - target_loss * ALPHA
        delta_M = M - M_target

        lr_t = LR_BASE * (1.0 + BETA * abs(delta_M))
        lr_t = max(LR_MIN, min(LR_MAX, lr_t))
        if temp == delta_M:
            if flag == 3:
                break;
            else:
                flag += 1
        else:
            temp = delta_M

        prune_ratios = update_prune_ratios(
            prune_ratios=prune_ratios,
            Si=Si,
            delta_M=delta_M,
            lr_t=lr_t,
            min_ratio=MIN_PRUNE_RATIO,
            max_ratio=MAX_PRUNE_RATIO
        )

        print("\nAdjustment info:")
        print(f"M        = {M:.4f}")
        print(f"M_target = {M_target:.4f}")
        print(f"delta_M  = {delta_M:.4f}")
        print(f"lr_t     = {lr_t:.4f}")

        print("\nUpdated pruning ratios:")
        print_prune_ratios(prune_ratios, prefix="  ")


# ============================================================
# Final result
# ============================================================

print("\n============================================================")
print("Final Result")
print("============================================================")

print("\nLast iteration pruning ratios:")
print_prune_ratios(prune_ratios, prefix="  ")

print("\nBest pruning ratios:")
print_prune_ratios(best_prune_ratios, prefix="  ")

print(f"\nBest PTQ loss = {best_ptq_loss:.4f}")
print(f"Best PTQ acc  = {best_ptq_acc:.4f}")


# ============================================================
# Save models
# ============================================================

print("\nSaving models...")

if best_slim_model is not None:
    best_slim_model.save("best_dynamic_slim_model.h5")
    print("Saved: best_dynamic_slim_model.h5")

if best_ptq_model is not None:
    best_ptq_model.save("best_dynamic_ptq_model.h5")
    print("Saved: best_dynamic_ptq_model.h5")

print("\nDone.")
