import os
import numpy as np
import tensorflow as tf
import pdb

from tensorflow_model_optimization.quantization.keras import vitis_quantize
from analysis import ana, select_best_ratio_from_csv, sensitivity_score, get_filter_to_keep, normalize_sensitivity
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

MODEL_PATH = "./cnn_model/CNN_0011_full_full_final.h5"
CNN_PRUNE_LOG = "cnn_prune_log.csv"

USE_BEST_RATIO_AS_INIT = True
DEFAULT_INIT_RATIO = 0.5

MIN_PRUNE_RATIO = 0.1
MAX_PRUNE_RATIO = 0.9

NUM_ITERATIONS = 50

PRUNE_METHOD = "FPGM" # L1-norm, L2-norm, FPGM, random
QUANTIZE_STRATEGY = "pof2s"

SAVE_SLIM_MODEL_PATH = "./proposed_model/best_cnn0011_slim_model.h5"
SAVE_PTQ_MODEL_PATH = "./proposed_model/best_cnn0011_ptq_model.h5"


# ============================================================
# Utility functions
# ============================================================

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


def eval_for_analysis(model):
    compile_for_eval(model)
    _, acc, _ = model.evaluate(sample_x, sample_y, verbose=0)
    return float(acc)

def update_prune_ratios(
    prune_ratios,
    Si,
    delta_M,
    lr_t,
    min_ratio=0.1,
    max_ratio=0.9,
):
    new_ratios = {}

    for layer_idx, p in prune_ratios.items():
        sensitivity = Si.get(layer_idx, 1.0)

        if delta_M < 0:
            # Bad reward: reduce pruning. More sensitive layers reduce more.
            new_p = p - lr_t * sensitivity * p
        else:
            # Good reward: increase pruning. Less sensitive layers increase more.
            new_p = p + lr_t * (1.0 - sensitivity) * (max_ratio - p)

        new_p = max(min_ratio, min(max_ratio, new_p))
        new_ratios[layer_idx] = float(new_p)

    return new_ratios

def print_prune_ratios(prune_ratios, prefix=""):
    for layer_idx, ratio in prune_ratios.items():
        print(f"{prefix}Conv2D layer {layer_idx} pruning ratio = {ratio:.4f}")

def build_keep_indices(conv_weights, prune_ratios, num_conv_layers):
    keep_indices = []

    for i in range(num_conv_layers):
        keep_idx = get_filter_to_keep(
            prune_method=PRUNE_METHOD,
            w=conv_weights[i],
            ratio=prune_ratios[i],
        )
        keep_indices.append(keep_idx)

    return keep_indices

def _get_layer_map(model):
    return {layer.name: layer for layer in model.layers}


def build_slim_cnn_by_cloning(original_model, keep_indices):
    """
    Clone the original CNN, but replace each Conv2D filters count with
    len(keep_indices[i]). Other layer configs are kept unchanged.

    This works for common CNNs such as:
        Conv2D -> ReLU/BN/Pool -> Conv2D -> ... -> Flatten -> Dense -> Dense

    The Dense input dimension after Flatten is inferred automatically from
    the reduced channel count.
    """
    conv_counter = {"idx": 0}

    def clone_function(layer):
        if isinstance(layer, tf.keras.layers.Conv2D):
            idx = conv_counter["idx"]
            config = layer.get_config()
            config["filters"] = int(len(keep_indices[idx]))
            conv_counter["idx"] += 1
            return layer.__class__.from_config(config)

        return layer.__class__.from_config(layer.get_config())

    slim_model = tf.keras.models.clone_model(
        original_model,
        clone_function=clone_function,
    )

    return slim_model


def _subset_batchnorm_weights(weights, keep_idx):
    """
    BatchNorm weights usually are [gamma, beta, moving_mean, moving_var].
    Some configs may not have gamma/beta, so subset every 1D vector whose
    length matches the original channel count.
    """
    new_weights = []
    for arr in weights:
        if arr.ndim == 1 and arr.shape[0] >= np.max(keep_idx) + 1:
            new_weights.append(arr[keep_idx])
        else:
            new_weights.append(arr)
    return new_weights


def _flatten_keep_rows(original_feature_shape, keep_idx):
    """
    Keras Flatten with channels_last flattens (H, W, C) with C as fastest axis.
    Return row indices in the original Dense kernel that correspond to kept
    channels.
    """
    if len(original_feature_shape) != 3:
        raise ValueError(
            "Only channels_last 2D CNN flatten shape (H, W, C) is supported. "
            f"Got feature shape: {original_feature_shape}"
        )

    h, w, c = [int(v) for v in original_feature_shape]
    rows = []

    for ih in range(h):
        for iw in range(w):
            base = (ih * w + iw) * c
            rows.extend(base + keep_idx)

    return np.array(rows, dtype=np.int64)


def set_slim_cnn_weights(slim_model, original_model, keep_indices):
    """
    Transfer weights from original CNN to slim CNN after Conv2D structural pruning.

    Rules:
    - Conv2D i: keep output filters keep_indices[i]
    - Conv2D i>0: also keep input channels according to keep_indices[i-1]
    - BatchNorm after a Conv2D: keep channels according to the latest Conv2D
    - First Dense after Flatten: keep rows corresponding to final Conv2D channels
    - Later Dense layers: copy directly
    """
    original_layers = _get_layer_map(original_model)

    conv_seen = 0
    last_conv_keep = None
    last_original_conv_filters = None
    first_dense_after_flatten_done = False

    # These are needed to map Flatten -> Dense rows.
    original_feature_shape = None
    final_keep_for_flatten = keep_indices[-1] if len(keep_indices) > 0 else None

    for layer in original_model.layers:
        if isinstance(layer, tf.keras.layers.Flatten):
            original_feature_shape = layer.input_shape[1:]
            break

    for slim_layer in slim_model.layers:
        if slim_layer.name not in original_layers:
            continue

        orig_layer = original_layers[slim_layer.name]
        orig_weights = orig_layer.get_weights()

        if len(orig_weights) == 0:
            continue

        if isinstance(orig_layer, tf.keras.layers.Conv2D):
            curr_keep = keep_indices[conv_seen]

            w = orig_weights[0]
            new_w = w

            if conv_seen > 0:
                prev_keep = keep_indices[conv_seen - 1]
                new_w = new_w[:, :, prev_keep, :]

            new_w = new_w[:, :, :, curr_keep]

            new_weights = [new_w]

            if orig_layer.use_bias and len(orig_weights) > 1:
                new_b = orig_weights[1][curr_keep]
                new_weights.append(new_b)

            slim_layer.set_weights(new_weights)

            last_conv_keep = curr_keep
            last_original_conv_filters = w.shape[-1]
            conv_seen += 1

        elif isinstance(orig_layer, tf.keras.layers.BatchNormalization):
            # Only prune BN when its channel dimension matches the previous Conv2D.
            if (
                last_conv_keep is not None
                and len(orig_weights) > 0
                and orig_weights[0].ndim == 1
                and orig_weights[0].shape[0] == last_original_conv_filters
            ):
                new_weights = _subset_batchnorm_weights(orig_weights, last_conv_keep)
                slim_layer.set_weights(new_weights)
            else:
                slim_layer.set_weights(orig_weights)

        elif isinstance(orig_layer, tf.keras.layers.Dense):
            w = orig_weights[0]
            b = orig_weights[1] if len(orig_weights) > 1 else None

            if not first_dense_after_flatten_done and original_feature_shape is not None:
                row_idx = _flatten_keep_rows(original_feature_shape, final_keep_for_flatten)
                new_w = w[row_idx, :]

                if b is not None:
                    slim_layer.set_weights([new_w, b])
                else:
                    slim_layer.set_weights([new_w])

                first_dense_after_flatten_done = True
            else:
                slim_layer.set_weights(orig_weights)

        else:
            # For other weighted layers. Most CNNs here will not have any.
            try:
                slim_layer.set_weights(orig_weights)
            except ValueError as e:
                print(
                    f"Warning: skip weight transfer for layer {slim_layer.name}. "
                    f"Reason: {e}"
                )


def make_slim_cnn(original_model, keep_indices):
    slim_model = build_slim_cnn_by_cloning(original_model, keep_indices)
    set_slim_cnn_weights(slim_model, original_model, keep_indices)
    return slim_model

def compute_delta_M(
    ptq_acc,
    ptq_loss,
    target_acc,
    baseline_ptq_loss,
    alpha=0.2,
    loss_cap=1.0
):
    acc_margin = ptq_acc - target_acc

    # punish the loss that are worse than no-prune PTQ
    loss_increase = max(ptq_loss - baseline_ptq_loss, 0.0)

    # avoid extreme loss lead the reward
    loss_increase = min(loss_increase, loss_cap)
    delta_M = acc_margin - alpha * loss_increase
    return delta_M

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

def get_clip_bounds(x_train, lower_p=0.1, upper_p=99.9):
    lower = np.percentile(x_train, lower_p)
    upper = np.percentile(x_train, upper_p)
    return lower, upper

def apply_clip(x, lower, upper):
    return np.clip(x, lower, upper).astype(np.float32)
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
# pdb.set_trace()
calib_x, calib_y = build_balanced_calib_from_generator(
    train_data,
    calib_samples=CALIBRATION_SAMPLES,
    pool_multiplier=10,
    seed=42
)
'''
# no obvious enhance
lower, upper = get_clip_bounds(calib_x, lower_p=0.1, upper_p=99.9)
calib_x = apply_clip(calib_x, lower, upper)
'''
print("Calibration data shape:", calib_x.shape)


# ============================================================
# Original model evaluation
# ============================================================

print("\nEvaluating original model...")
sample_x, sample_y = next(iter(val_data))
compile_for_eval(model)
original_loss, original_acc ,original_ce= model.evaluate(test_data)

original_loss = float(original_loss)
original_acc = float(original_acc)
original_ce = float(original_ce)

print(f"Original loss = {original_ce:.4f}")
print(f"Original acc  = {original_acc:.4f}")


# ============================================================
# Automatically detect Conv2D layers
# ============================================================

conv_layers = [
    layer for layer in model.layers
    if isinstance(layer, tf.keras.layers.Conv2D)
]

dense_layers = [
    layer for layer in model.layers
    if isinstance(layer, tf.keras.layers.Dense)
]

if len(conv_layers) == 0:
    raise ValueError("No Conv2D layer found. This script is for CNN Conv2D pruning.")

if len(dense_layers) == 0:
    raise ValueError("No Dense output layer found. This script expects a CNN classifier.")

num_conv_layers = len(conv_layers)

print("\nDetected CNN structure:")
print(f"Input shape: {model.input_shape[1:]}")
print(f"Conv2D layers to prune: {num_conv_layers}")
print(f"Dense layers: {len(dense_layers)}")

for i, layer in enumerate(conv_layers):
    print(
        f"Conv2D layer {i}: name={layer.name}, "
        f"filters={layer.filters}, kernel_size={layer.kernel_size}"
    )

for i, layer in enumerate(dense_layers):
    print(f"Dense layer {i}: name={layer.name}, units={layer.units}")


# ============================================================
# Save original Conv2D weights
# ============================================================

conv_weights = []

for i, layer in enumerate(conv_layers):
    layer_weights = layer.get_weights()

    if len(layer_weights) == 0:
        raise ValueError(f"Conv2D layer {i} has no weights.")

    w = layer_weights[0]
    conv_weights.append(w)
    print(f"Conv2D {i}: kernel shape={w.shape}")


# ============================================================
# Analysis and load best ratio and sensitivity
# ============================================================

print("\nRunning Conv2D analysis...")
ana(
    eval_s=eval_for_analysis,
    model=model,
    prune_method=PRUNE_METHOD,
    output_file=CNN_PRUNE_LOG,
    target_layer_type="conv2d",
)

print("\nLoading best ratio and sensitivity...")
best_ratio = select_best_ratio_from_csv(
    CNN_PRUNE_LOG,
    acc_threshold=target_acc,
)
best_ratio = {int(k): float(v) for k, v in best_ratio.items()}

si = sensitivity_score(CNN_PRUNE_LOG, original_acc)
Si = normalize_sensitivity(si, num_conv_layers)

print("\nNormalized sensitivity:")
for layer_idx, value in Si.items():
    print(f"Conv2D layer {layer_idx}: Si = {value:.4f}")


# ============================================================
# Initialize pruning ratios
# ============================================================

prune_ratios = {}

for i in range(num_conv_layers):
    if USE_BEST_RATIO_AS_INIT:
        prune_ratios[i] = best_ratio.get(i, DEFAULT_INIT_RATIO)
    else:
        prune_ratios[i] = DEFAULT_INIT_RATIO

    prune_ratios[i] = max(
        MIN_PRUNE_RATIO,
        min(MAX_PRUNE_RATIO, prune_ratios[i]),
    )

print("\nInitial pruning ratios:")
print_prune_ratios(prune_ratios)


# ============================================================
# Iterative pruning ratio adjustment + PTQ
# ============================================================

best_ptq_acc = -1.0
best_ptq_loss = None
best_ptq_model = None
best_ptq_ce = None
best_slim_model = None
best_prune_ratios = None

ptq_model = None
slim_model = None
# baseline acc (without pruning)
quantizer = vitis_quantize.VitisQuantizer(
    model,
    quantize_strategy=QUANTIZE_STRATEGY,
)
ptq_model = quantizer.quantize_model(
    calib_dataset=calib_x,
)
compile_for_eval(ptq_model)
baseline_loss, baseline_acc, baseline_ce = ptq_model.evaluate(test_data)
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
        conv_weights=conv_weights,
        prune_ratios=prune_ratios,
        num_conv_layers=num_conv_layers,
    )

    current_filters = [len(k) for k in keep_indices]

    print("Current Conv2D filters:", current_filters)
    print("Current pruning ratios:")
    print_prune_ratios(prune_ratios, prefix="  ")

    # --------------------------------------------------------
    # 2. Reconstruct slim CNN and transfer pruned weights
    # --------------------------------------------------------
    print("\nBuilding slim CNN...")
    slim_model = make_slim_cnn(model, keep_indices)
    slim_model.summary()

    # --------------------------------------------------------
    # 3. PTQ
    # --------------------------------------------------------
    print("\nRunning PTQ...")

    quantizer = vitis_quantize.VitisQuantizer(
        slim_model,
        quantize_strategy=QUANTIZE_STRATEGY,
    )

    ptq_model = quantizer.quantize_model(
        calib_dataset=calib_x,
    )

    compile_for_eval(ptq_model)
    ptq_loss, ptq_acc, ptq_ce = ptq_model.evaluate(test_data)

    ptq_loss = float(ptq_loss)
    ptq_acc = float(ptq_acc)
    ptq_ce = float(ptq_ce)
    print(f"PTQ loss = {ptq_ce:.4f}")
    print(f"PTQ acc  = {ptq_acc:.4f}")
    
    # --------------------------------------------------------
    # 4. Save best result
    # --------------------------------------------------------
    if ptq_acc > best_ptq_acc:
        best_ptq_acc = ptq_acc
        best_ptq_loss = ptq_loss
        best_ptq_ce = ptq_ce
        best_ptq_model = ptq_model
        best_slim_model = slim_model
        best_prune_ratios = prune_ratios.copy()

        print("New best PTQ model found.")

    # --------------------------------------------------------
    # 5. Update pruning ratios
    # --------------------------------------------------------
    if iteration != NUM_ITERATIONS - 1:
        '''
        M = ptq_acc - ptq_loss * ALPHA
        M_target = target_acc - target_loss * ALPHA
        delta_M = M - M_target
        '''
        delta_M = compute_delta_M(ptq_acc, ptq_ce, baseline_acc, baseline_ce, ALPHA, 1)
        
        if temp == delta_M:
            if flag == 3:
                break
            else:
                flag += 1
        else:
            flag = 0
        temp = delta_M 
        lr_t = LR_BASE * (1.0 + BETA * abs(delta_M))
        lr_t = max(LR_MIN, min(LR_MAX, lr_t))

        prune_ratios = update_prune_ratios(
            prune_ratios=prune_ratios,
            Si=Si,
            delta_M=delta_M,
            lr_t=lr_t,
            min_ratio=MIN_PRUNE_RATIO,
            max_ratio=MAX_PRUNE_RATIO,
        )

        print("\nAdjustment info:")
        #print(f"M        = {M:.4f}")
        #print(f"M_target = {M_target:.4f}")
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

print(f"\nBest PTQ loss = {best_ptq_ce:.4f}")
print(f"Best PTQ acc  = {best_ptq_acc:.4f}")


# ============================================================
# Save models
# ============================================================

print("\nSaving models...")

if best_slim_model is not None:
    best_slim_model.save(SAVE_SLIM_MODEL_PATH)
    print(f"Saved: {SAVE_SLIM_MODEL_PATH}")

if best_ptq_model is not None:
    best_ptq_model.save(SAVE_PTQ_MODEL_PATH)
    print(f"Saved: {SAVE_PTQ_MODEL_PATH}")

print("\nDone.")

