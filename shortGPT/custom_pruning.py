"""
Targeted ablation sweep: prune exact encoder/decoder layer indices
specified, rather than picking the lowest-BI n_enc/n_dec layers. Useful for
isolating whether a performance drop comes from a *specific* layer or just
from removing *that many* layers.

Functions similarly to shortGPT.sweep, the only difference is how layers are
selected for removal: explicit index lists here instead of BI-score-based
selection.

Run with:  python -m shortGPT.custom_pruning --dataset mmsu
           python -m shortGPT.custom_pruning --model voxtral --dataset asr --n_total 100
"""

import json
import os
import argparse
import torch
from transformers import AutoProcessor
from shortGPT.prune import remove_layers
from shortGPT.sweep import RESULTS_DIR as SWEEP_RESULTS_DIR, load_eval_data, run_candidate_eval
from evaluate.evaluate_model import MODEL_CONFIGS, load_model


# Each candidate: (label, encoder_indices_to_remove, decoder_indices_to_remove).
# label is used to name the output file / model_name, so make it descriptive
# of what the candidate is testing.
CANDIDATES = [
    ("enc6_dec3", [7, 12, 17, 13, 16, 9], [3, 1, 4])
]


def prune_candidate_by_indices(model, enc_indices, dec_indices):
    """
    Prunes the exact encoder_indices and decoder_indices given, in place,
    syncing config the same way prune_candidate() does in sweep.py.
    Returns (removed_encoder_indices, removed_decoder_indices).
    """
    removed_enc, removed_dec = [], []

    if enc_indices:
        removed_enc = sorted(set(enc_indices))
        model.model.audio_tower.layers = remove_layers(
            model.model.audio_tower.layers, removed_enc
        )
        model.config.audio_config.encoder_layers = len(model.model.audio_tower.layers)

    if dec_indices:
        removed_dec = sorted(set(dec_indices))
        model.model.language_model.layers = remove_layers(
            model.model.language_model.layers, removed_dec
        )
        model.config.text_config.num_hidden_layers = len(model.model.language_model.layers)
        model.config.text_config.layer_types = [
            lt for i, lt in enumerate(model.config.text_config.layer_types) if i not in removed_dec
        ]

    return removed_enc, removed_dec


def run_indexed_candidate(label, enc_indices, dec_indices, processor, ds, dataset_name, results_file, results_dir, model_name):
    model, _ = load_model(model_name, load_processor=False)
    removed_enc, removed_dec = prune_candidate_by_indices(model, enc_indices, dec_indices)

    eval_output_file = f"{results_dir}/{dataset_name}_results_{label}.jsonl"
    print(f"evaluating: {label} (enc={removed_enc}, dec={removed_dec}) on {dataset_name}")

    record = {
        "model_name": f"{model_name}_pruned-{label}",
        "arch": model_name,
        "dataset": dataset_name,
        "label": label,
        "n_enc_removed": len(removed_enc),
        "n_dec_removed": len(removed_dec),
        "removed_encoder_indices": removed_enc,
        "removed_decoder_indices": removed_dec,
        "results_file": eval_output_file,
    }
    record.update(run_candidate_eval(model, processor, ds, dataset_name, eval_output_file, model_name))

    with open(results_file, "a") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()

    del model
    torch.cuda.empty_cache()

    print(f"Done: {label} -> results in {eval_output_file}")
    return record


def run_indexed_sweep(model_name, dataset_name, n_total=None):
    processor = AutoProcessor.from_pretrained(MODEL_CONFIGS[model_name]["checkpoint"])

    ds, total = load_eval_data(dataset_name, n_total)
    print("Total samples in subset:", total)

    results_dir = os.path.join(SWEEP_RESULTS_DIR, f"{model_name}_custom_pruning")
    os.makedirs(results_dir, exist_ok=True)
    results_file = f"{results_dir}/performance_results_{dataset_name}.jsonl"

    for label, enc_indices, dec_indices in CANDIDATES:
        run_indexed_candidate(label, enc_indices, dec_indices, processor, ds, dataset_name, results_file, results_dir, model_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", choices=list(MODEL_CONFIGS.keys()), default="qwen2audio", help="Specify the model to use.")
    parser.add_argument("--dataset", type=str, default="mmsu", choices=["asr", "obqa", "mmsu"])
    parser.add_argument("--n_total", type=int, default=None,
                         help="Total eval sample size (default: same per-dataset defaults as shortGPT.sweep, e.g. 500 for mmsu/obqa, 100 for asr). Stratified across task_key if the dataset has one, else random.")
    args = parser.parse_args()

    run_indexed_sweep(args.model, args.dataset, n_total=args.n_total)
