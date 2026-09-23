"""
Calculate block influence scores for each layer in the encoder of Qwen2-Audio, Audio Flamingo 3, and Voxtral.
The block influence of a layer is computed by removing that layer and measuring the change in the adapter's output hidden state.
The sum of the block influence scores across all calibration samples is returned for each layer.
The encoder block influence scores can be used to identify the least important layers for pruning.
Run with:  python -m sbi.encoder_bi --model qwen2audio --calibration_data_file (optional)
"""
import json
import os
import torch
import matplotlib.pyplot as plt
from shortGPT.compute_bi import block_influence
from sbi.decoder_bi import model_config
from calibration.transcribe_calibration import resolve_audio_path
from evaluate.evaluate_model import load_model
import argparse


def plot_importances(importances, filename="importances.png", title="Layer Importances"):
    plt.figure(figsize=(7, 5))
    plt.bar(range(len(importances)), importances)
    plt.title(title)
    plt.xlabel("Layer Index")
    plt.ylabel("Mean Block Influence (log scale)")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def enc_kwargs_qwen(inputs):
    return {"input_features": inputs["input_features"]}


def enc_kwargs_af3(inputs):
    # AF3's encoder dereferences input_features_mask unconditionally
    return {"input_features": inputs["input_features"],
            "input_features_mask": inputs["input_features_mask"]}

def enc_kwargs_voxtral(inputs):
    return {"input_features": inputs["input_features"]}


ENC_KWARGS_FNS = {
    "qwen2audio": enc_kwargs_qwen,
    "audioflamingo3": enc_kwargs_af3,
    "voxtral": enc_kwargs_voxtral,
}

def accumulate_bi_enc(base_hidden_state, pruned_states, importances, token_counts):
    """hidden_states: tuple of (num_layers) tensors. importances: list to add into, in place.
    For each layer, compute block influence and add to the corresponding entry in importances.
    """
    for i, h in enumerate(pruned_states):
        bi = block_influence(base_hidden_state, pruned_states[i])
        importances[i] += bi.sum().item()
        token_counts[i] += bi.numel()

def encoder_bi(model_name, calibration_data_file, max_calibration_samples=None,
               calibration_dataset_path=None):
    """
    Evaluate the adapter-output block influence of each encoder layer for the
    given model (qwen2audio, audioflamingo3, or voxtral) using a set of
    calibration rows.
    Returns:
        mean_enc_importances: A list of mean block influence scores for each encoder layer.
    """
    cfg = model_config[model_name]
    model, processor = load_model(model_name)
    enc = model.model.audio_tower
    proj = model.model.multi_modal_projector

    encoder_importances = [0.0 for _ in range(len(enc.layers))]
    enc_tok_count = [0 for _ in range(len(enc.layers))]

    original_layers = enc.layers 

    with open(calibration_data_file, "r") as f:
        for i, line in enumerate(f):
            if max_calibration_samples is not None and i >= max_calibration_samples:
                break
            data = json.loads(line)
            audio_path = resolve_audio_path(data["audio_path"], calibration_dataset_path, data.get("corpus"))

            inputs = cfg["build_input_fn"](processor, data, audio_path).to(model.device, model.dtype)

            enc_kwargs = ENC_KWARGS_FNS[model_name](inputs)

            hidden_states = []
            for j in range(len(original_layers) + 1):
                enc.layers = torch.nn.ModuleList(
                    [l for k, l in enumerate(original_layers) if k != j])
                with torch.inference_mode():
                    encoder_out = enc(**enc_kwargs).last_hidden_state
                    if model_name == "voxtral":
                        encoder_out = encoder_out.reshape(-1, model.config.audio_config.intermediate_size)
                    hidden_states.append(proj(encoder_out))
            enc.layers = original_layers             # <-- restore

            accumulate_bi_enc(hidden_states[-1], hidden_states[:-1],
                              encoder_importances, enc_tok_count)

    mean_enc_importances = [imp / count if count > 0 else 0.0 for imp, count in zip(encoder_importances, enc_tok_count)]
    return mean_enc_importances 

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["qwen2audio", "audioflamingo3", "voxtral"], default="qwen2audio")
    p.add_argument("--calibration_data_file", type=str, default=None)
    args = p.parse_args()
    
    calibration_dir = os.getenv("CALIBRATION_DATA_DIR")
    calibration_dataset_path = os.getenv("CALIBRATION_DATASET_PATH")
    max_calibration_samples = int(os.getenv("CALIBRATION_SIZE", "100"))
    bi_score_dir = os.getenv("BI_SCORES_DIR")

    if not bi_score_dir:
        raise RuntimeError("BI_SCORES_DIR must be set in the environment")
    if not calibration_dir:
        raise RuntimeError("CALIBRATION_DATA_DIR must be set in the environment")
    if not calibration_dataset_path:
        raise RuntimeError("CALIBRATION_DATASET_PATH must be set in the environment")

    if args.calibration_data_file is not None:
        calibration_file = args.calibration_data_file
    else:
        calibration_file = os.path.join(calibration_dir, f"calibration_data_{args.model}.jsonl")

    encoder_importances = encoder_bi(args.model, calibration_file, max_calibration_samples, calibration_dataset_path)

    os.makedirs(bi_score_dir, exist_ok=True)
    out_file = os.path.join(bi_score_dir, f"{args.model}_encoder_bi_{max_calibration_samples}.jsonl")
    plot_file = os.path.join(bi_score_dir, f"{args.model}_encoder_bi_{max_calibration_samples}.png")

    with open(out_file, "w") as f:
        data = {"encoder_bi": encoder_importances}
        json.dump(data, f)

    plot_importances(encoder_importances, filename=plot_file)