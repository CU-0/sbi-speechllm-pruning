"""
Sweep over a set of (n_enc, n_dec) pruning candidates using SBI's scores:
an encoder file (SBI-Enc score, from sbi.encoder_bi) and a separate decoder
file (modality-split BI from sbi.decoder_bi, or a reverse-priority vector
from sbi.reverse_bi). Reuses the pruning/eval machinery from
shortGPT.sweep -- only the two-file score loading and output-directory
naming differ from the single-file baseline.

Run with:  python -m sbi.sweep -m qwen2audio --dataset mmsu --enc_score <encoder_sbi_file> --dec_score <decoder_bi_file> --key text
           python -m sbi.sweep -m qwen2audio --dataset mmsu --enc_score <encoder_sbi_file> --dec_score <reverse_bi_file> --key reverse
"""

import os
import argparse

from shortGPT.sweep import (
    DEFAULT_N_TOTAL,
    MODEL_CONFIGS,
    RESULTS_DIR,
    load_bi_scores,
    run_performance_sweep,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", choices=list(MODEL_CONFIGS.keys()), default="qwen2audio", help="Specify the model to use.")
    parser.add_argument("--dataset", type=str, default="mmsu", choices=["asr", "obqa", "mmsu", "all"],
                         help="Performance dataset, or 'all' to run asr, mmsu, and obqa.")
    parser.add_argument("--enc_score", type=str, required=True,
                         help="Path to the encoder SBI-Enc-score JSON file (as written by sbi.encoder_bi).")
    parser.add_argument("--dec_score", type=str, required=True,
                         help="Path to the decoder modality-split BI file (sbi.decoder_bi) or reverse-priority file (sbi.reverse_bi).")
    parser.add_argument("--key", type=str, default="text",
                         help="Which condition to select from --dec_score's 'bi' dict (e.g. 'text', 'audio', 'combined', 'reverse').")
    parser.add_argument("--n_total", type=int, default=None,
                         help="Override eval sample size (defaults: asr=100, mmsu=500, obqa=500).")
    args = parser.parse_args()

    encoder_importances = load_bi_scores(args.enc_score, "encoder_bi")
    decoder_importances = load_bi_scores(args.dec_score, "bi")[args.key]
    name = f"{args.key}_BI+modified_enc_BI"

    results_dir = os.path.join(RESULTS_DIR, f"{args.model}_{name}")
    os.makedirs(results_dir, exist_ok=True)

    datasets = ("asr", "mmsu", "obqa") if args.dataset == "all" else (args.dataset,)
    for dataset_name in datasets:
        run_performance_sweep(dataset_name, encoder_importances, decoder_importances,
                              args.model, results_dir, n_total=args.n_total)
