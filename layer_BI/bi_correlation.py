"""Spearman correlation between SBI-Dec (text positions of speech-text data)
and BI^text (original BI on the transcribed text-only calibration set),
plus top-k overlap and audio/text token statistics.

Usage: python bi_correlation.py file1 file2 ...
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, kendalltau

KS = (3, 6, 9, 12)
N_PERM = 100_000
rng = np.random.default_rng(0)


def perm_pvalue(x, y, rho_obs):
    """One-sided permutation test: P(rho >= rho_obs) under random layer order."""
    ry = np.argsort(np.argsort(y))
    rx = np.argsort(np.argsort(x))
    hits = 0
    for _ in range(N_PERM):
        if np.corrcoef(rx, rng.permutation(ry))[0, 1] >= rho_obs:
            hits += 1
    return (hits + 1) / (N_PERM + 1)


def topk(scores, k):
    return set(np.argsort(scores)[:k])  # k lowest-scoring = most prunable


def analyse(path):
    d = json.loads(Path(path).read_text())
    bi, tok = d["bi"], d["tok_counts"]
    sbi_dec = np.asarray(bi["text"])       # SBI-Dec, speech-text data
    bi_txt = np.asarray(bi["text_only"])   # BI^text, text-only data
    L = len(sbi_dec)

    rho, p = spearmanr(sbi_dec, bi_txt)
    tau, _ = kendalltau(sbi_dec, bi_txt)
    # like-for-like: prompt positions only (same tokens in both passes)
    rho_prompt, _ = spearmanr(bi["text_prompt"], bi["text_only_prompt"])

    overlap = {k: len(topk(sbi_dec, k) & topk(bi_txt, k)) for k in KS}

    n_aud, n_txt = tok["audio"], tok["text"]
    return dict(
        L=L, rho=rho, p=p, tau=tau,
        rho_prompt=rho_prompt, overlap=overlap,
        audio=n_aud, text=n_txt, ratio=n_aud / n_txt,
        audio_frac=n_aud / (n_aud + n_txt),
        text_only=tok["text_only"],
        cost_ratio=(n_aud + n_txt) / tok["text_only"],
    )


def main(paths):
    rows = {Path(p).stem.split("_layer")[0].split("-", 1)[-1]: analyse(p) for p in paths}

    print("== SBI-Dec vs BI^text ==")
    print(f"{'model':<16}{'L':>3}{'spearman rho':>15}{'p-value':>10}{'kendall tau':>15}"
          f"{'rho(prompt)':>15}     top-k overlap")
    for m, r in rows.items():
        ov = "  ".join(f"k={k}:{v}/{k}" for k, v in r["overlap"].items())
        print(f"{m:<16}{r['L']:>3}{r['rho']:>12.3f}{r['p']:>13.1e}"
              f"{r['tau']:>12.3f}{r['rho_prompt']:>15.3f}       {ov}")

    print("\n== Token counts (100 calibration samples) ==")
    print(f"{'model':<16}{'audio':>8}{'text':>7}{'audio:text':>14}{'audio %':>9}"
          f"{'text-only':>13}{'cost ratio':>13}")
    for m, r in rows.items():
        print(f"{m:<16}{r['audio']:>8}{r['text']:>7}{r['ratio']:>10.1f}:1"
              f"{100*r['audio_frac']:>8.1f}%{r['text_only']:>12}{r['cost_ratio']:>12.1f}x")
    A = sum(r["audio"] for r in rows.values())
    T = sum(r["text"] for r in rows.values())
    print(f"{'pooled':<16}{A:>8}{T:>7}{A/T:>10.1f}:1{100*A/(A+T):>8.1f}%")


if __name__ == "__main__":
    main(sys.argv[1:])