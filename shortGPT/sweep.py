"""
Sweep over a set of (n_enc, n_dec) pruning candidates using the original
ShortGPT baseline score: one BI file with both "encoder_bi" and
"decoder_bi" keys (as written by shortGPT.compute_bi), applied to both
components. Prune -> eval on --dataset -> lightweight result row.

sbi.sweep imports and calls run_performance_sweep() from this module
for its own two-BI-file sweep (which internally uses load_eval_data()/
run_candidate_eval() below), and shortGPT.custom_pruning imports
load_eval_data()/run_candidate_eval() directly for its explicit-index-list
ablation -- so every --dataset choice (including asr) is supported
identically everywhere prune -> eval happens.

Run with:  python -m shortGPT.sweep -m audioflamingo3 --dataset mmsu --bi_file <path_to_bi_file>
           python -m shortGPT.sweep --dataset obqa --bi_file <path_to_bi_file>
"""

import json
import os
import argparse
import torch
from transformers import AutoProcessor
import pandas as pd

from shortGPT.prune import remove_layers
from evaluate.evaluate_model import (
    DATASET_CONFIGS,
    MODEL_CONFIGS,
    load_eval_dataset,
    load_model,
    run_eval,
    sample_ds,
)
from evaluate.asr_eval import evaluate_asr, compute_wer


REPO_ROOT = os.getenv("REPO_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.getenv("RESULTS_DIR") or os.path.join(REPO_ROOT, "results")
ASR_DATASET_PATH = os.getenv("ASR_DATASET_PATH")  # used for ASR eval in performance sweep

CANDIDATES = [ #(encoder, decoder)
    (0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9), (0, 10),
    (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (8, 0), (9, 0), (10, 0)
]

DEFAULT_N_TOTAL = {"asr": 100, "mmsu": 500, "obqa": 500}


def load_bi_scores(bi_file, key):
    """
    Loads precomputed mean BI scores (a single JSON object with "encoder_bi"
    and "decoder_bi" keys -- note the .jsonl extension is misleading, it's
    one JSON object, not one-object-per-line).
    """
    with open(bi_file, "r") as f:
        data = json.load(f)
    return data[key]


def prune_candidate(model, n_enc, n_dec, encoder_importances, decoder_importances):
    """
    Prunes n_enc encoder layers and n_dec decoder layers (lowest-BI first)
    in place, syncing config so the model reloads correctly if ever saved.
    Deterministic given the same importances -- same (n_enc, n_dec) always
    removes the same indices.

    Returns (removed_encoder_indices, removed_decoder_indices).
    """
    assert len(encoder_importances) == len(model.model.audio_tower.layers), \
        f"encoder BI has {len(encoder_importances)} entries, model has {len(model.model.audio_tower.layers)} layers"
    assert len(decoder_importances) == len(model.model.language_model.layers), \
        f"decoder BI has {len(decoder_importances)} entries, model has {len(model.model.language_model.layers)} layers"
    enc_to_remove = []
    dec_to_remove = []

    if n_enc > 0:
        enc_to_remove = sorted(range(len(encoder_importances)), key=lambda i: encoder_importances[i])[:n_enc]
        model.model.audio_tower.layers = remove_layers(
            model.model.audio_tower.layers, enc_to_remove
        )
        model.config.audio_config.encoder_layers = len(model.model.audio_tower.layers)

    if n_dec > 0:
        dec_to_remove = sorted(range(len(decoder_importances)), key=lambda i: decoder_importances[i])[:n_dec]
        model.model.language_model.layers = remove_layers(
            model.model.language_model.layers, dec_to_remove
        )
        model.config.text_config.num_hidden_layers = len(model.model.language_model.layers)
        layer_types = getattr(model.config.text_config, "layer_types", None)
        if layer_types is not None:
            model.config.text_config.layer_types = [
                lt for i, lt in enumerate(layer_types) if i not in dec_to_remove
            ]

    return enc_to_remove, dec_to_remove


def load_eval_data(dataset_name, n_total=None):
    """
    Loads and samples the evaluation set for dataset_name. Returns (ds, total).

    ASR uses its own loading path (a TSV read + plain sample, no HF dataset,
    no task_key stratification) since it isn't in DATASET_CONFIGS at all --
    every other dataset goes through load_eval_dataset()/sample_ds().
    """
    n_total = n_total if n_total is not None else DEFAULT_N_TOTAL[dataset_name]

    if dataset_name == "asr":
        if not ASR_DATASET_PATH:
            raise RuntimeError("ASR_DATASET_PATH must be set in the environment")
        ds = pd.read_csv(os.path.join(ASR_DATASET_PATH, "dev.tsv"), sep="\t") #sep="\t" for tab-separated values
        ds = ds.sample(n=n_total, random_state=42).reset_index(drop=True) # drop=True to avoid adding the old index as a column
        return ds, len(ds)

    ds = load_eval_dataset(dataset_name)
    return sample_ds(ds, n_total, task_key=DATASET_CONFIGS[dataset_name]["task_key"])


def run_candidate_eval(model, processor, ds, dataset_name, eval_output_file, model_name):
    """
    Runs the eval appropriate for dataset_name and returns the field(s) to
    merge into the candidate's result record: {"wer": ...} for asr,
    {"category_stats": ...} for everything else.
    """
    if dataset_name == "asr":
        evaluate_asr(model, processor, ASR_DATASET_PATH, eval_output_file, ds, model_name)
        return {"wer": compute_wer(eval_output_file)}
    return {"category_stats": run_eval(model, processor, ds, dataset_name,
                                       eval_output_file, model_name)}


def run_performance_candidate(n_enc, n_dec, encoder_importances, decoder_importances,
                              processor, ds, dataset_name, results_file, results_dir,
                              model_name):
    model, _ = load_model(model_name, load_processor=False)
    removed_enc, removed_dec = prune_candidate(model, n_enc, n_dec,
                                               encoder_importances, decoder_importances)

    eval_output_file = f"{results_dir}/{dataset_name}_results_enc{n_enc}_dec{n_dec}.jsonl"
    print(f"evaluating: {n_enc},{n_dec} on {dataset_name}")

    record = {
        "model_name": f"{model_name}_pruned_enc{n_enc}_dec{n_dec}",
        "arch": model_name,
        "dataset": dataset_name,
        "n_enc_removed": n_enc,
        "n_dec_removed": n_dec,
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

    print(f"Done: enc{n_enc}_dec{n_dec} -> results in {eval_output_file}")
    return record


def run_performance_sweep(dataset_name, encoder_importances, decoder_importances,
                          model_name, results_dir, n_total=None):
    processor = AutoProcessor.from_pretrained(MODEL_CONFIGS[model_name]["checkpoint"])

    ds, total = load_eval_data(dataset_name, n_total)
    print(f"Total samples in evaluation subset: {total}")

    os.makedirs(results_dir, exist_ok=True)
    performance_file = f"{results_dir}/performance_results_{dataset_name}.jsonl"

    for n_enc, n_dec in CANDIDATES:
        run_performance_candidate(n_enc, n_dec, encoder_importances, decoder_importances, processor, ds, dataset_name, performance_file, results_dir, model_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", choices=list(MODEL_CONFIGS.keys()), default="qwen2audio", help="Specify the model to use.")
    parser.add_argument("--dataset", type=str, default="mmsu", choices=["asr", "obqa", "mmsu", "all"],
                         help="Performance dataset, or 'all' to run asr, mmsu, and obqa.")
    parser.add_argument("--bi_file", type=str, required=True,
                         help="Path to the combined encoder+decoder BI JSON file (as written by shortGPT.compute_bi), used for both components.")
    parser.add_argument("--n_total", type=int, default=None,
                         help="Override eval sample size (defaults: asr=100, mmsu=500, obqa=500).")
    args = parser.parse_args()

    encoder_importances = load_bi_scores(args.bi_file, "encoder_bi")
    decoder_importances = load_bi_scores(args.bi_file, "decoder_bi")

    results_dir = os.path.join(RESULTS_DIR, f"{args.model}_shortGPT")
    os.makedirs(results_dir, exist_ok=True)

    datasets = ("asr", "mmsu", "obqa") if args.dataset == "all" else (args.dataset,)
    for dataset_name in datasets:
        run_performance_sweep(dataset_name, encoder_importances, decoder_importances,
                              args.model, results_dir, n_total=args.n_total)