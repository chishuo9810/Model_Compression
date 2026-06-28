import tensorflow as tf
import pdb
import copy
import numpy as np
import pandas as pd
def sensitivity_score(csv_path, baseline_acc=None): 
    df = pd.read_csv(csv_path)
    sensitivity = {}
				 
    for layer_idx in df["layer_index"].unique():
        sub = df[df["layer_index"] == layer_idx].copy()
        ratios = sub["ratio"].values
        accs = sub["acc"].values

	# ensure that baseline
        if baseline_acc is None:
            base_acc = accs[0]
        else:
            base_acc = baseline_acc

	# accuracy drop
        drops = np.maximum(base_acc - accs, 0.0)
        print(drops)
	# layer sensitivity score
        score = np.sum(ratios * drops) / np.sum(ratios)
        sensitivity[layer_idx] = float(score)
    return sensitivity

def normalize_sensitivity(si, num_hidden_dense):
    if si is None or len(si) == 0:
        print("Warning: empty sensitivity score. All layers are set to 1.0.")
        return {i: 1.0 for i in range(num_hidden_dense)}

    # 確�~] key �~X� int
    si = {int(k): float(v) for k, v in si.items()}

    max_val = max(si.values())

    if max_val == 0:
        print("Warning: max sensitivity is 0. All layers are set to 1.0.")
        return {i: 1.0 for i in range(num_hidden_dense)}

    Si = {}

    for i in range(num_hidden_dense):
        if i in si:
            Si[i] = si[i] / max_val
        else:
            Si[i] = 1.0

    return Si

	
def select_best_ratio_from_csv(csv_path, acc_threshold=0.9):
    df = pd.read_csv(csv_path)

    best_ratio = {}

    for layer_idx in df["layer_index"].unique():
        sub = df[df["layer_index"] == layer_idx]

        good = sub[sub["acc"] >= acc_threshold]

        if len(good) == 0:
            best_ratio[layer_idx] = 0.0
        else:
            best = good.sort_values("ratio", ascending=False).iloc[0]
            best_ratio[layer_idx] = float(best["ratio"])

    return best_ratio


def get_neuron_to_keep(prune_method='L1-norm', w=None, ratio=0.5):
    if prune_method == 'L1-norm':
        score = np.linalg.norm(w, ord=1, axis=0)
        num_keep = int(len(score) * (1-ratio))
        num_keep = max(1, num_keep)
        keep_idx = np.argsort(score)[-num_keep:]
        return np.sort(keep_idx)
    else:
        raise ValueError(f"Unsupported prune_method : {prune_method}")

def fake_prune_dense_layer_weights(weights, keep_idx):
    new_weights = copy.deepcopy(weights)

    kernel = new_weights[0]
    bias = new_weights[1]

    all_idx = np.arange(kernel.shape[1])
    remove_idx = np.setdiff1d(all_idx, keep_idx)

    kernel[:, remove_idx] = 0
    bias[remove_idx] = 0

    new_weights[0] = kernel
    new_weights[1] = bias
    return new_weights


def ana(eval_s, model, prune_method='L1-norm', output_file='dense_prune_log.csv'):
    print("Start analysis Dense layer")
    dense_layers = [l for l in model.layers if isinstance(l, tf.keras.layers.Dense)]
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("layer_index,layer_name,ratio,acc\n")

        for layer_idx, layer in enumerate(dense_layers[:-1]):
            original_weights = layer.get_weights()


            kernel = original_weights[0]

            for ratio in np.arange(0.1, 1.0, 0.1):
                ratio = round(float(ratio), 1)

                # copy original model
                test_model = tf.keras.models.clone_model(model)
                test_model.set_weights(model.get_weights())

                test_dense_layers = [
                    l for l in test_model.layers
                    if isinstance(l, tf.keras.layers.Dense)
                ]

                keep_idx = get_neuron_to_keep(
                    prune_method=prune_method,
                    w=kernel,
                    ratio=ratio
                )

                pruned_weights = fake_prune_dense_layer_weights(original_weights, keep_idx)
                test_dense_layers[layer_idx].set_weights(pruned_weights)

                acc = eval_s(test_model)

                print(f"layer={layer_idx}, name={layer.name}, ratio={ratio}, acc={acc}")

                f.write(f"{layer_idx},{layer.name},{ratio},{acc}\n")
                f.flush()

    return model

