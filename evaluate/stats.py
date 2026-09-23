"""
Builds a per-model, per-category accuracy / format-error-rate summary table
from a sweep output file:

  - performance_results.jsonl: one record per candidate, each with
    "model_name" and "category_stats" (as written by run_performance_candidate,
    which gets category_stats straight from run_eval -- counts already
    aggregated per category, no need to re-read per-example result files).

Output table: one row per model, one pair of columns per category
("<category>_accuracy", "<category>_format_error_rate"), saved as CSV.
"""

import os
import json
import pandas as pd
import re
import matplotlib.pyplot as plt
import argparse


def load_jsonl(path):
    """Reads a .jsonl file, returns a list of row dicts."""
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def compute_category_metrics_from_stats(category_stats):
    """
    Converts a category_stats dict (as returned by run_eval:
    {category: {"total": ..., "correct": ..., "incorrect": ...,
    "format_error": ...}}) into {category: {"accuracy": ...,
    "format_error_rate": ..., "n": ...}}, plus an "overall" entry
    aggregated across all categories (not an average of per-category
    rates -- categories with more examples contribute proportionally more,
    matching how accuracy is normally reported).
    """
    metrics = {}
    total_all = 0
    correct_all = 0
    format_error_all = 0

    for category, stats in category_stats.items():
        n = stats.get("total", 0)
        n_correct = stats.get("correct", 0)
        n_format_error = stats.get("format_error", 0)
        metrics[category] = {
            "accuracy": round(n_correct / n, 4) if n else None,
            "format_error_rate": round(n_format_error / n, 4) if n else None,
            "n": n,
        }
        total_all += n
        correct_all += n_correct
        format_error_all += n_format_error

    metrics["overall"] = {
        "accuracy": round(correct_all / total_all, 4) if total_all else None,
        "format_error_rate": round(format_error_all / total_all, 4) if total_all else None,
        "n": total_all,
    }
    return metrics


def performance_table(performance_jsonl, output_file):
    """
    Reads performance_results.jsonl (one record per candidate, with
    "model_name" and "category_stats"), computes per-category accuracy and
    format-error-rate for each, and writes a combined table (one row per
    model, one column pair per category) to output_file as CSV.

    Returns the pandas DataFrame as well, in case you want to inspect it
    further in the same session without re-reading from disk.
    """
    records = load_jsonl(performance_jsonl)
    if not records:
        raise FileNotFoundError(f"No records found in {performance_jsonl!r}")

    table_rows = {}
    for record in records:
        model_name = record.get("model_name")
        if model_name is None:
            print(f"Skipping record with no model_name: {record}")
            continue

        category_stats = record.get("category_stats")
        row = {}
        if not category_stats:
            row["wer"] = record.get("wer")
        else:
            category_metrics = compute_category_metrics_from_stats(category_stats)
            for category, m in category_metrics.items():
                row[f"{category}_accuracy"] = m["accuracy"]
                row[f"{category}_format_error_rate"] = m["format_error_rate"]

        table_rows[model_name] = row

    df = pd.DataFrame.from_dict(table_rows, orient="index")
    df.index.name = "model"
    df = df.sort_index(axis=1)  # keep category columns grouped/alphabetical, stable across runs

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    df.to_csv(output_file)

    print(f"Wrote table ({df.shape[0]} models x {df.shape[1]} columns) to {output_file}")
    return df


def _extract_enc_dec(model_name):
    """
    Parses n_enc_removed / n_dec_removed out of a model name like
    'Qwen2-Audio-7B-Instruct-pruned-enc0-dec8' or
    'Qwen2-Audio-7B-Instruct-pruned_enc4_dec0'. Returns (n_enc, n_dec) or
    (None, None) if the pattern isn't found.
    """
    match = re.search(r"enc(\d+)[-_]dec(\d+)", model_name)
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


def performance_plot(csv_path, filename, mode):
    """
    Reads the performance table CSV (as written by performance_table) and
    plots one point per CSV row.

    If the CSV has "overall_accuracy" / "overall_format_error_rate" columns,
    plots overall format-error-rate (x-axis) vs overall accuracy (y-axis).
    Otherwise, falls back to a "wer" column (as written for asr candidates,
    which have no category_stats) and plots WER (y-axis) against number of
    layers removed (x-axis, pruning mode) or model/config (other modes).

    mode controls which rows are plotted:
      - "all": every row in the CSV with non-null overall metrics
      - "pruning": rows whose model names parse as pruning configs

    Points are annotated so each plotted row can be identified directly.
    """
    if mode not in ("all", "pruning"):
        raise ValueError(f"mode must be 'all' or 'pruning', got {mode!r}")

    df = pd.read_csv(csv_path, index_col="model")

    has_overall_metrics = "overall_accuracy" in df.columns and "overall_format_error_rate" in df.columns
    if not has_overall_metrics and "wer" not in df.columns:
        raise KeyError(
            "Expected either 'overall_accuracy'/'overall_format_error_rate' columns "
            "or a 'wer' column in CSV, found neither"
        )

    y_col, y_label = ("overall_accuracy", "Overall accuracy") if has_overall_metrics else ("wer", "WER")
    plot_df = df.dropna(subset=[y_col]).copy()

    if mode == "pruning":
        plot_df["n_enc_removed"], plot_df["n_dec_removed"] = zip(*plot_df.index.map(_extract_enc_dec))
        plot_df = plot_df.dropna(subset=["n_enc_removed", "n_dec_removed"])
        plot_df["n_enc_removed"] = plot_df["n_enc_removed"].astype(int)
        plot_df["n_dec_removed"] = plot_df["n_dec_removed"].astype(int)
        plot_df["label"] = plot_df.apply(
            lambda r: f"enc{r['n_enc_removed']}_dec{r['n_dec_removed']}", axis=1
        )
        plot_df["series"] = plot_df.apply(
            lambda r: "Encoder-only" if r["n_dec_removed"] == 0
            else ("Decoder-only" if r["n_enc_removed"] == 0 else "Mixed pruning"),
            axis=1,
        )
        if has_overall_metrics:
            x_col, x_label = "overall_format_error_rate", "Overall format error rate"
            title = "Pruning: overall accuracy vs overall format error rate"
        else:
            plot_df["n_layers_removed"] = plot_df["n_enc_removed"] + plot_df["n_dec_removed"]
            x_col, x_label = "n_layers_removed", "Number of layers removed"
            title = "Pruning: WER vs number of layers removed"

    else:
        plot_df["label"] = plot_df.index.astype(str)
        plot_df["series"] = plot_df.index.map(
            lambda model_name: "Pruning" if _extract_enc_dec(model_name)[0] is not None else "Other"
        )
        if has_overall_metrics:
            x_col, x_label = "overall_format_error_rate", "Overall format error rate"
            title = "All models: overall accuracy vs overall format error rate"
        else:
            x_col, x_label = "label", "Model"
            title = "All models: WER"

    if plot_df.empty:
        raise ValueError(f"No rows available to plot for mode={mode!r}.")

    style_by_series = {
        "Encoder-only": ("tab:blue", "o"),
        "Decoder-only": ("tab:red", "s"),
        "Mixed pruning": ("tab:green", "^"),
        "Pruning": ("tab:blue", "o"),
        "Other": ("tab:gray", "x"),
    }

    # One subplot per series (e.g. Encoder-only vs Decoder-only) so unrelated
    # x-axis scales (e.g. layers removed from different components) aren't
    # overlaid on the same panel.
    series_order = [s for s in style_by_series if s in plot_df["series"].unique()]
    fig, axes = plt.subplots(1, len(series_order), figsize=(6 * len(series_order), 6), sharey=True, squeeze=False)
    axes = axes[0]

    for ax, series_label in zip(axes, series_order):
        series_df = plot_df[plot_df["series"] == series_label]
        color, marker = style_by_series[series_label]
        x = series_df[x_col]
        y = series_df[y_col]
        labels = series_df["label"]

        ax.scatter(x, y, marker=marker, color=color, label=series_label)
        for xi, yi, label in zip(x, y, labels):
            ax.annotate(label, (xi, yi), textcoords="offset points", xytext=(5, 5),
                        fontsize=8, color=color)

        ax.set_title(series_label)
        ax.set_xlabel(x_label)

    axes[0].set_ylabel(y_label)
    fig.suptitle(title)

    fig.tight_layout()
    plt.savefig(filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MMSU evaluation statistics")
    subparsers = parser.add_subparsers(dest='command', required=True, help='Available commands')
    RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")  # default to "results" if not set

    # performance_table
    perf_table_parser = subparsers.add_parser('performance_table', help='Generate performance table')
    perf_table_parser.add_argument('filepath', help='Path to performance_results.jsonl')
    perf_table_parser.add_argument('-o', '--output', default='results_table.csv', help='Output file path (default: results_table.csv)')

    # performance_plot
    perf_plot_parser = subparsers.add_parser('performance_plot', help='Generate performance plot')
    perf_plot_parser.add_argument('filepath', help='Path to performance table CSV')
    perf_plot_parser.add_argument('-m', '--mode', default='all', choices=['all', 'pruning'], help='Row filter mode for performance plot (default: all)')
    perf_plot_parser.add_argument('-o', '--output', default='Performance.png', help='Output file path (default: Performance.png)')

    args = parser.parse_args()

    args.output = os.path.join(RESULTS_DIR, args.output)  # ensure output is in RESULTS_DIR
    if args.command == 'performance_table':  # python -m evaluate.stats performance_table performance_results.jsonl -o performance.csv
        performance_table(args.filepath, args.output)
    elif args.command == 'performance_plot':  # python -m evaluate.stats performance_plot merged_table.csv -m pruning -o performance.png
        performance_plot(args.filepath, args.output, mode=args.mode)