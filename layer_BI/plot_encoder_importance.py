'''
 Plots encoder-layer BI against SBI-Enc,
 one panel per model side by side, sharing a single legend.

 Run:
   python plot_encoder_importance.py --dir layer_BI
'''

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "serif"
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['mathtext.it'] = 'serif:italic'
plt.rcParams['mathtext.rm'] = 'serif'

CONDITIONS = [
    ("bi", "BI", "#D55E00", ":", "^"),
    ("ai", "SBI-Enc", "#0072B2", "-", "s"),
]

# model display name -> file prefix
# (<prefix>_layer_bi_100.jsonl = BI, <prefix>_encoder_bi_100.jsonl = AI)
MODELS = {
    "Audio Flamingo 3": "audioflamingo3",
    "Qwen2-Audio": "qwen2audio",
    "Voxtral": "voxtral",
}

TOP_K_PRUNABLE = 10


def load_encoder_bi(path):
    with open(path) as f:
        raw = json.load(f)
    return raw["encoder_bi"]


def most_prunable(v, k=TOP_K_PRUNABLE):
    """1-based indices (layer 1 = first layer) of the k lowest-score layers, most prunable first."""
    return (np.argsort(v)[:k] + 1).tolist()


def plot_one(ax, vectors, title, ylabel=None):
    order = [(key, lab, col, ls, mk) for key, lab, col, ls, mk in CONDITIONS if key in vectors]
    n_layers = len(next(iter(vectors.values())))

    for key, lab, col, ls, mk in order:
        v = vectors[key]
        ax.plot(range(1, len(v) + 1), v, color=col, ls=ls, lw=1.6, marker=mk, ms=7, label=lab)

    ax.set_yscale("log")
    ax.set_xlim(0.5, n_layers + 0.5)  # layers are 1-indexed; keep layer 1 off the left edge
    ax.set_xticks([1] + list(range(5, n_layers + 1, 5)))
    ax.set_xlabel("layer index", fontsize=14)
    if ylabel != None:
        ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.tick_params(axis="both", labelsize=14)
    ax.grid(alpha=.3)

    pad_width = max(len(lab) for _, lab, _, _, _ in order)
    note_lines = [f"top-{TOP_K_PRUNABLE} prunable layers:"]
    for key, lab, col, ls, mk in order:
        idx = most_prunable(vectors[key])
        note_lines.append(f"{lab.ljust(pad_width)}: " + " ".join(f"{i:>2d}" for i in idx))
    ax.text(0.5, 0.98, "\n".join(note_lines), transform=ax.transAxes,
             ha="center", va="top", multialignment="left", fontsize=14, fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.6", alpha=0.85))


def plot_all_models(base_dir, save_path=None):
    fig, axes = plt.subplots(1, len(MODELS), figsize=(6.5 * len(MODELS), 4.5))

    for ax, (name, prefix) in zip(axes, MODELS.items()):
        vectors = {
            "bi": load_encoder_bi(os.path.join(base_dir, f"{prefix}_layer_bi_100.jsonl")),
            "ai": load_encoder_bi(os.path.join(base_dir, f"{prefix}_encoder_bi_100.jsonl")),
        }
        if ax == axes[0]:
            plot_one(ax, vectors, title=name, ylabel='score')
        else:
            plot_one(ax, vectors, title=name)

    # One shared legend, deduped, one line across the figure.
    handles, labels = [], []
    seen = set()
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in seen:
                handles.append(h)
                labels.append(l)
                seen.add(l)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.legend(handles, labels, loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, 0.0), fontsize=16)

    if save_path:
        fig.savefig(save_path, dpi=150, format="pdf", bbox_inches="tight")
        print(f"wrote {save_path}")

    return fig


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="layer_BI", help="directory holding <prefix>_layer_bi_100.jsonl / <prefix>_encoder_bi_100.jsonl")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    out = args.out or os.path.join(args.dir, "encoder_importance.pdf")
    plot_all_models(args.dir, save_path=out)
