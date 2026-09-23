"""Generate a reverse decoder-priority BI vector to evaluate deepest-k pruning.

For L decoder layers, the scores are [(L - 1) / 100, ..., 0.01, 1.0].
The second-last layer therefore has the lowest score, and the final layer is
never selected before the earlier layers.

Run:
    python -m sbi.reverse_bi 32 --out layer_BI/qwen2audio_reverse_bi.json
"""

import argparse
import json


def reverse_scores(n_layers, step=0.01):
    if n_layers < 2:
        raise ValueError("at least two layers are required to protect the last layer")
    return [step * (n_layers - index - 1) for index in range(n_layers - 1)] + [1.0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("n_layers", type=int, help="number of decoder layers")
    parser.add_argument("--key", default="reverse", help="name for the generated BI vector")
    parser.add_argument("--step", type=float, default=0.01,
                        help="score increment between non-final layers")
    parser.add_argument("--out", default=None,
                        help="output JSON path (default: reverse_bi_<n_layers>.json)")
    args = parser.parse_args()

    scores = reverse_scores(args.n_layers, args.step)
    data = {"bi": {args.key: scores}}

    out = args.out or f"reverse_bi_{args.n_layers}.json"
    with open(out, "w") as file:
        json.dump(data, file, indent=2)
    print(f"wrote {out}: {args.key} = {scores}")