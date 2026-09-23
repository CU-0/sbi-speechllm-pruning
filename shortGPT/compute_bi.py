# Calculate block influence for Qwen2-Audio-7B-Instruct, Voxtral-Mini-3B-2507,
# or Audio Flamingo 3, encoder and decoder. Select the model via --model.
# run with:  python -m shortGPT.compute_bi --model qwen2audio

import argparse
import os
import json
import types
import librosa
import torch
import matplotlib.pyplot as plt

from evaluate.evaluate_model import load_model
from calibration.transcribe_calibration import resolve_audio_path

REPO_ROOT = os.getenv("REPO_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CALIBRATION_DATA_DIR = os.getenv("CALIBRATION_DATA_DIR", os.path.join(REPO_ROOT, "calibration"))
BI_SCORES_DIR = os.getenv("BI_SCORES_DIR", os.path.join(REPO_ROOT, "bi_scores"))




def block_influence(hidden_input, hidden_output):
    """
    Return a per-token distance score: for each token position, take the cosine similarity between its input vector and output vector, 
    then convert to 1 - similarity (so redundant layers, which barely change the vector, score close to 0).

    Args:
        hidden_input (torch.Tensor): The input tensor from the layer's hidden state. (batch, seq, d_model) d_model is vector representation dimension
        hidden_output (torch.Tensor): The output tensor from the layer's hidden state. (batch, seq, d_model)
    Returns:
        Block influence score (torch.Tensor): A tensor of shape (batch, seq) containing the block influence scores for each token position.
    """
    hidden_input_norm = torch.nn.functional.normalize(hidden_input, p=2, dim=-1)
    hidden_output_norm = torch.nn.functional.normalize(hidden_output, p=2, dim=-1)
    cosine_similarity = torch.sum(hidden_input_norm * hidden_output_norm, dim=-1).nan_to_num(nan=0.5)
    distance_score = 1 - cosine_similarity
    return distance_score  # (batch, seq)


def accumulate_bi(hidden_states, importances, token_counts):
    """hidden_states: tuple of (num_layers+1) tensors. importances: list to add into, in place.
    For each layer, compute block influence and add to the corresponding entry in importances.
    """
    num_layers = len(hidden_states) - 1
    for i in range(num_layers):
        bi = block_influence(hidden_states[i], hidden_states[i + 1])
        importances[i] += bi.sum().item()
        token_counts[i] += bi.numel()


# -------------------------------------------------------------- input builders


def build_input_qwen(processor, data, audio_path):
    conversation = [
        {"role": "user", "content": [
            {"type": "audio", "audio_url": "placeholder"},
            {"type": "text", "text": data["prompt"]},
        ]},
    ]
    text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audio, sr = librosa.load(audio_path, sr=processor.feature_extractor.sampling_rate)
    inputs = processor(
        text=text_prompt,
        audio=[audio],
        sampling_rate=sr,
        return_tensors="pt",
        padding=True
    )
    return inputs


def build_input_af3(processor, data, audio_path):
    conversation = [
        {"role": "user", "content": [
            {"type": "audio", "path": audio_path},
            {"type": "text", "text": data["prompt"]},
        ]},
    ]
    inputs = processor.apply_chat_template(
        conversation, tokenize=True, add_generation_prompt=True, return_dict=True,
    )
    return inputs


def build_input_voxtral(processor, data, audio_path):
    conversation = [
        {"role": "user", "content": [
            {"type": "audio", "path": audio_path},
            {"type": "text", "text": data["prompt"]},
        ]},
    ]
    inputs = processor.apply_chat_template(
        conversation, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    )
    return inputs


# ============================================================ Qwen2-Audio

def eval_importance_qwen(model, processor, calibration_data_file, max_calibration_samples=None, calibration_dataset_path=None):
    """
    Evaluate the block influence of each layer in the encoder and decoder of Qwen2-Audio
    using a set of calibration rows.
    Returns:
        encoder_importances: A list of summed block influence scores for each encoder layer.
        decoder_importances: A list of summed block influence scores for each decoder layer.
    """
    enc_layers = model.config.audio_config.encoder_layers
    dec_layers = model.config.text_config.num_hidden_layers

    # --- Forward hook setup ---
    # Qwen2-Audio's audio_tower pools its output after the layer stack, so
    # output_hidden_states=True skips straight from "before layer 31" to
    # "after pooling" -- the raw un-pooled output of layer 31 is only
    # obtainable via a forward hook.
    captured = {}

    def capture_last_encoder_layer(module, layer_input, layer_output):
        captured["layer31_raw"] = layer_output[0] if isinstance(layer_output, tuple) else layer_output

    hook_handle = model.model.audio_tower.layers[enc_layers - 1].register_forward_hook(
        capture_last_encoder_layer
    )

    encoder_importances = [0.0 for _ in range(enc_layers)]
    decoder_importances = [0.0 for _ in range(dec_layers)]
    enc_tok_count = [0.0 for _ in range(enc_layers)]
    dec_tok_count = [0.0 for _ in range(dec_layers)]

    with torch.inference_mode():
        with open(calibration_data_file, "r") as f:
            for i, line in enumerate(f):
                if max_calibration_samples is not None and i >= max_calibration_samples:
                    break
                data = json.loads(line)
                conversation = [
                    {"role": "user", "content": [
                        {"type": "audio", "audio_url": "placeholder"},
                        {"type": "text", "text": data["prompt"]},
                    ]},
                ]
                text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
                audio_path = resolve_audio_path(data["audio_path"], calibration_dataset_path, data.get("corpus"))
                audios = [
                    librosa.load(audio_path, sr=processor.feature_extractor.sampling_rate)[0]
                ]

                inputs = processor(
                    text=text_prompt, audio=audios, sampling_rate=16000,
                    return_tensors="pt", padding=True,
                ).to(model.device)

                encoder_out = model.model.audio_tower(
                    input_features=inputs["input_features"],
                    output_hidden_states=True,
                )
                decoder_out = model(**inputs, output_hidden_states=True)

                encoder_states = list(encoder_out.hidden_states)
                encoder_states[enc_layers] = captured["layer31_raw"]
                decoder_states = decoder_out.hidden_states

                accumulate_bi(encoder_states, encoder_importances, enc_tok_count)
                accumulate_bi(decoder_states, decoder_importances, dec_tok_count)

    mean_enc_importances = [imp / count for imp, count in zip(encoder_importances, enc_tok_count)]
    mean_dec_importances = [imp / count for imp, count in zip(decoder_importances, dec_tok_count)]

    hook_handle.remove()
    return mean_enc_importances, mean_dec_importances


# ============================================================ Audio Flamingo 3 and Voxtral

def _capture_encoder_layers(audio_tower, captured):
    """Hook every encoder layer so the whole stack of hidden states is available.

    Needed when the audio tower will not hand them over: AF3 relies on an
    auto-capture decorator with ambiguous tuple semantics, and VoxtralEncoder
    returns BaseModelOutputWithPooling(last_hidden_state=...) only - it never
    populates hidden_states at all.
    """
    def make_hook(layer_idx):
        def hook(module, layer_input, layer_output):
            if layer_idx == 0:
                captured.append(layer_input[0] if isinstance(layer_input, tuple) else layer_input)
            captured.append(layer_output[0] if isinstance(layer_output, tuple) else layer_output)
        return hook

    return [layer.register_forward_hook(make_hook(idx))
            for idx, layer in enumerate(audio_tower.layers)]


def _eval_importance_hooked(model, processor, build_inputs, calibration_data_file,
                            max_calibration_samples=None, calibration_dataset_path=None):
    enc_layers = len(model.model.audio_tower.layers)
    dec_layers = model.config.text_config.num_hidden_layers

    captured = []
    hook_handles = _capture_encoder_layers(model.model.audio_tower, captured)

    encoder_importances = [0.0 for _ in range(enc_layers)]
    decoder_importances = [0.0 for _ in range(dec_layers)]
    enc_tok_count = [0.0 for _ in range(enc_layers)]
    dec_tok_count = [0.0 for _ in range(dec_layers)]

    try:
        with torch.inference_mode():
            with open(calibration_data_file, "r") as f:
                for i, line in enumerate(f):
                    if max_calibration_samples is not None and i >= max_calibration_samples:
                        break
                    data = json.loads(line)
                    audio_path = resolve_audio_path(data["audio_path"], calibration_dataset_path, data.get("corpus"))

                    inputs = build_inputs(processor, data, audio_path).to(model.device)
                    if "input_features" in inputs:
                        inputs["input_features"] = inputs["input_features"].to(model.dtype)

                    captured.clear()
                    decoder_out = model(**inputs, output_hidden_states=True)

                    accumulate_bi(captured, encoder_importances, enc_tok_count)
                    accumulate_bi(decoder_out.hidden_states, decoder_importances, dec_tok_count)
    finally:
        for h in hook_handles:
            h.remove()

    return ([imp / c for imp, c in zip(encoder_importances, enc_tok_count)],
            [imp / c for imp, c in zip(decoder_importances, dec_tok_count)])


def eval_importance_voxtral(model, processor, calibration_data_file,
                            max_calibration_samples=None, calibration_dataset_path=None):
    return _eval_importance_hooked(model, processor, build_input_voxtral,
                                   calibration_data_file, max_calibration_samples,
                                   calibration_dataset_path)

def eval_importance_af3(model, processor, calibration_data_file,
                            max_calibration_samples=None, calibration_dataset_path=None):
    return _eval_importance_hooked(model, processor, build_input_af3,
                                   calibration_data_file, max_calibration_samples,
                                   calibration_dataset_path)


# ============================================================ shared

def plot_importances(encoder_importances, decoder_importances, filename="importances.png"):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.bar(range(len(encoder_importances)), encoder_importances)
    plt.title("Encoder Layer Importances")
    plt.xlabel("Layer Index")
    plt.ylabel("Mean Block Influence (log scale)")
    plt.yscale("log")

    plt.subplot(1, 2, 2)
    plt.bar(range(len(decoder_importances)), decoder_importances)
    plt.title("Decoder Layer Importances")
    plt.xlabel("Layer Index")
    plt.ylabel("Mean Block Influence (log scale)")
    plt.yscale("log")

    plt.tight_layout()
    plt.savefig(filename)

eval_importance = {
    "qwen2audio": eval_importance_qwen,
    "voxtral": eval_importance_voxtral,
    "audioflamingo3": eval_importance_af3,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", choices=["qwen2audio", "voxtral", "audioflamingo3"], default="qwen2audio",
        help="Which audio LLM to compute layer importances for.",
    )
    args = parser.parse_args()

    calibration_size = int(os.getenv("CALIBRATION_SIZE", "100"))
    calibration_data_file = ""
    calibration_dataset_path = os.getenv("CALIBRATION_DATASET_PATH", None)
    model, processor = load_model(args.model)


    calibration_data_file = os.path.join(CALIBRATION_DATA_DIR, f"calibration_data_{args.model}.jsonl")

    encoder_importances, decoder_importances = eval_importance[args.model](
        model, processor, calibration_data_file, calibration_size, calibration_dataset_path
    )
    out_file = os.path.join(BI_SCORES_DIR, f"{args.model}_layer_bi_{calibration_size}.jsonl")
    plot_file = os.path.join(BI_SCORES_DIR, f"{args.model}-importances_{calibration_size}.png")

    with open(out_file, "w") as f:
        data = {"encoder_bi": encoder_importances, "decoder_bi": decoder_importances}
        json.dump(data, f)

    plot_importances(encoder_importances, decoder_importances, filename=plot_file)