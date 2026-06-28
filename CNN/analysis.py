import copy
import numpy as np
import pandas as pd
import tensorflow as tf
import pdb

def _is_prunable_layer(layer):
    return isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.Dense))


def _layer_kind(layer):
    if isinstance(layer, tf.keras.layers.Conv2D):
        return "conv2d"
    if isinstance(layer, tf.keras.layers.Dense):
        return "dense"
    return "unknown"


def get_filter_to_keep(prune_method="L1-norm", w=None, ratio=0.5):
    """
    Conv2D kernel shape: (kh, kw, in_channels, out_filters)
    Prune output filters by L1 norm.
    """
    print(f'method : {prune_method}')
    if w is None:
        raise ValueError("w cannot be None")

    if prune_method == "L1-norm":
        score = np.sum(np.abs(w), axis=(0, 1, 2))
        num_keep = int(len(score) * (1.0 - ratio))
        num_keep = max(1, num_keep)
        keep_idx = np.argsort(score)[-num_keep:]
        return np.sort(keep_idx)
    elif prune_method == "L2-norm":
        score = np.sqrt(np.sum(w ** 2, axis=(0, 1, 2)))

        num_keep = int(len(score) * (1.0 - ratio))
        num_keep = max(1, num_keep)

        keep_idx = np.argsort(score)[-num_keep:]
        return np.sort(keep_idx)
    elif prune_method == "FPGM":
        # w shape: (kh, kw, in_channels, out_filters)
        kh, kw, in_channels, out_filters = w.shape

        # reshape to: (out_filters, kh * kw * in_channels)
        filters = np.transpose(w, (3, 0, 1, 2))
        filters = filters.reshape(out_filters, -1)

        # pairwise Euclidean distance
        # dist_matrix[i, j] = distance between filter i and filter j
        diff = filters[:, np.newaxis, :] - filters[np.newaxis, :, :]
        dist_matrix = np.linalg.norm(diff, axis=2)

        # FPGM score:
        # smaller total distance means the filter is closer to geometric median,
        # so it is more redundant.
        score = np.sum(dist_matrix, axis=1)

        num_keep = int(out_filters * (1.0 - ratio))
        num_keep = max(1, num_keep)

        # keep filters with larger distance score
        keep_idx = np.argsort(score)[-num_keep:]
        return np.sort(keep_idx)
    elif prune_method == "random":
        out_filters = w.shape[-1]

        num_keep = int(out_filters * (1.0 - ratio))
        num_keep = max(1, num_keep)

        keep_idx = np.random.choice(out_filters, size=num_keep, replace=False)
        return np.sort(keep_idx)

    raise ValueError(f"Unsupported prune_method: {prune_method}")


def get_neuron_to_keep(prune_method="L1-norm", w=None, ratio=0.5):
    """
    Dense kernel shape: (in_features, out_units)
    Prune output neurons by L1 norm.
    """
    if w is None:
        raise ValueError("w cannot be None")

    if prune_method == "L1-norm":
        score = np.linalg.norm(w, ord=1, axis=0)
        num_keep = int(len(score) * (1.0 - ratio))
        num_keep = max(1, num_keep)
        keep_idx = np.argsort(score)[-num_keep:]
        return np.sort(keep_idx)

    raise ValueError(f"Unsupported prune_method: {prune_method}")


def fake_prune_conv2d_layer_weights(weights, keep_idx):
    """
    Fake prune Conv2D output filters by setting removed filters to zero.
    This does not change model shape, so it is safe for per-layer analysis.
    """
    new_weights = copy.deepcopy(weights)
    kernel = new_weights[0]

    all_idx = np.arange(kernel.shape[-1])
    remove_idx = np.setdiff1d(all_idx, keep_idx)

    kernel[:, :, :, remove_idx] = 0
    new_weights[0] = kernel

    if len(new_weights) > 1 and new_weights[1] is not None:
        bias = new_weights[1]
        bias[remove_idx] = 0
        new_weights[1] = bias

    return new_weights


def fake_prune_dense_layer_weights(weights, keep_idx):
    """
    Fake prune Dense output neurons by setting removed neurons to zero.
    This keeps the original model shape.
    """
    new_weights = copy.deepcopy(weights)
    kernel = new_weights[0]

    all_idx = np.arange(kernel.shape[1])
    remove_idx = np.setdiff1d(all_idx, keep_idx)

    kernel[:, remove_idx] = 0
    new_weights[0] = kernel

    if len(new_weights) > 1 and new_weights[1] is not None:
        bias = new_weights[1]
        bias[remove_idx] = 0
        new_weights[1] = bias

    return new_weights


def ana(
    eval_s,
    model,
    prune_method="L1-norm",
    output_file="cnn_prune_log.csv",
    target_layer_type="conv2d",
    ratios=None,
    skip_last_dense=True,
):
    """
    Per-layer fake-pruning analysis.

    Parameters
    ----------
    eval_s : callable
        eval_s(model) -> accuracy
    model : tf.keras.Model
        Original model.
    prune_method : str
        Currently supports "L1-norm".
    output_file : str
        CSV output path.
    target_layer_type : str
        "conv2d", "dense", or "all".
        For CNN pruning, use "conv2d".
    ratios : iterable
        Pruning ratios to test. Default: 0.1 ... 0.9.
    skip_last_dense : bool
        If target_layer_type is "dense" or "all", skip the final Dense layer,
        which is usually the classifier output layer.
    """
    if ratios is None:
        ratios = np.arange(0.1, 1.0, 0.1)

    if target_layer_type not in {"conv2d", "dense", "all"}:
        raise ValueError("target_layer_type must be 'conv2d', 'dense', or 'all'")

    print(f"Start analysis: target_layer_type={target_layer_type}")

    candidate_layers = []
    dense_layers = [l for l in model.layers if isinstance(l, tf.keras.layers.Dense)]
    last_dense = dense_layers[-1] if len(dense_layers) > 0 else None

    for layer in model.layers:
        kind = _layer_kind(layer)

        if target_layer_type != "all" and kind != target_layer_type:
            continue

        if kind == "unknown":
            continue

        if skip_last_dense and layer is last_dense:
            continue

        candidate_layers.append(layer)

    if len(candidate_layers) == 0:
        raise ValueError(
            f"No target layers found for target_layer_type={target_layer_type}."
        )

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("layer_index,layer_name,layer_type,ratio,acc\n")

        for layer_idx, layer in enumerate(candidate_layers):
            original_weights = layer.get_weights()

            if len(original_weights) == 0:
                print(f"Skip layer {layer.name}: no weights")
                continue

            kernel = original_weights[0]
            kind = _layer_kind(layer)

            for ratio in ratios:
                ratio = round(float(ratio), 1)

                test_model = tf.keras.models.clone_model(model)
                test_model.set_weights(model.get_weights())

                test_candidate_layers = []
                test_dense_layers = [
                    l for l in test_model.layers
                    if isinstance(l, tf.keras.layers.Dense)
                ]
                test_last_dense = test_dense_layers[-1] if len(test_dense_layers) > 0 else None

                for test_layer in test_model.layers:
                    test_kind = _layer_kind(test_layer)

                    if target_layer_type != "all" and test_kind != target_layer_type:
                        continue
                    if test_kind == "unknown":
                        continue
                    if skip_last_dense and test_layer is test_last_dense:
                        continue

                    test_candidate_layers.append(test_layer)

                if kind == "conv2d":
                    keep_idx = get_filter_to_keep(
                        prune_method=prune_method,
                        w=kernel,
                        ratio=ratio,
                    )
                    pruned_weights = fake_prune_conv2d_layer_weights(
                        original_weights,
                        keep_idx,
                    )
                elif kind == "dense":
                    keep_idx = get_neuron_to_keep(
                        prune_method=prune_method,
                        w=kernel,
                        ratio=ratio,
                    )
                    pruned_weights = fake_prune_dense_layer_weights(
                        original_weights,
                        keep_idx,
                    )
                else:
                    raise ValueError(f"Unsupported layer type: {kind}")

                test_candidate_layers[layer_idx].set_weights(pruned_weights)

                acc = eval_s(test_model)

                print(
                    f"layer={layer_idx}, name={layer.name}, "
                    f"type={kind}, ratio={ratio}, acc={acc}"
                )

                f.write(f"{layer_idx},{layer.name},{kind},{ratio},{acc}\n")
                f.flush()

    return model


def sensitivity_score(csv_path, baseline_acc=None):
    """
    Calculate sensitivity from ana() output.

    score_i = sum(ratio * max(baseline_acc - acc, 0)) / sum(ratio)

    Larger score means the layer is more sensitive to pruning.
    """
    df = pd.read_csv(csv_path)
    sensitivity = {}

    for layer_idx in df["layer_index"].unique():
        sub = df[df["layer_index"] == layer_idx].copy()
        sub = sub.sort_values("ratio")

        ratios = sub["ratio"].values.astype(float)
        accs = sub["acc"].values.astype(float)

        if baseline_acc is None:
            base_acc = float(accs[0])
        else:
            base_acc = float(baseline_acc)

        drops = np.maximum(base_acc - accs, 0.0)
        score = np.sum(ratios * drops) / np.sum(ratios)
        sensitivity[int(layer_idx)] = float(score)

    return sensitivity

def normalize_sensitivity(si, num_conv_layers):
    if si is None or len(si) == 0:
        print("Warning: empty sensitivity score. All Conv2D layers are set to 1.0.")
        return {i: 1.0 for i in range(num_conv_layers)}

    si = {int(k): float(v) for k, v in si.items()}
    max_val = max(si.values())

    if max_val == 0:
        print("Warning: max sensitivity is 0. All Conv2D layers are set to 1.0.")
        return {i: 1.0 for i in range(num_conv_layers)}

    Si = {}
    for i in range(num_conv_layers):
        Si[i] = si.get(i, max_val) / max_val

    return Si

def select_best_ratio_from_csv(csv_path, acc_threshold=0.9):
    """
    For each layer, select the largest pruning ratio whose analysis accuracy
    is still >= acc_threshold.
    """
    df = pd.read_csv(csv_path)
    best_ratio = {}

    for layer_idx in df["layer_index"].unique():
        sub = df[df["layer_index"] == layer_idx].copy()
        good = sub[sub["acc"] >= acc_threshold]

        if len(good) == 0:
            best_ratio[int(layer_idx)] = 0.0
        else:
            best = good.sort_values("ratio", ascending=False).iloc[0]
            best_ratio[int(layer_idx)] = float(best["ratio"])

    return best_ratio

