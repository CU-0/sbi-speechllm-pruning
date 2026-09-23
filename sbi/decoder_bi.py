# Does the BI ranking of the decoder (backbone) layers depend on WHICH token
# positions you average over, and on whether the content arrives as audio or as
# text?
#
# Standard BI averages cosine distance over every position in the sequence. In
# a speech LLM the sequence is heterogeneous -- audio embeddings from the
# adaptor sit alongside text tokens -- so that average silently mixes two
# distributions, weighted by whatever the prompt template happens to produce.
#
# --mode multimodal   ONE forward pass per item, split over positions:
#   audio            BI over audio-embedding positions
#   text             BI over all non-audio positions
#   text_prompt      BI over the question tokens only  <- matched-span condition
#   text_template    BI over chat scaffolding + system prompt (text - prompt)
#   combined         token-weighted (audio + text) == standard BI
#
# --mode text_only    a second, audio-free pass over the same items:
#   text_only        BI over all positions, transcript substituted for audio
#   text_only_prompt BI over the question tokens only  <- matched-span condition
#   text_drop        (--text-drop) prompt with NO transcript and no audio:
#                    the "content missing" ablation
#
# The two modes write into the SAME output file and merge, so they can be run
# in either order or on separate days. Per-item prompt-span lengths are stored
# by each mode; when both are present the merge checks them against each other.
#
# Run:
#   python -m sbi.decoder_bi --model qwen2audio

import os
import torch
import argparse
import json
from calibration.transcribe_calibration import ensure_transcripts, resolve_audio_path
from evaluate.evaluate_model import load_model
from shortGPT.compute_bi import block_influence, build_input_af3, build_input_qwen, build_input_voxtral

REPO_ROOT = os.getenv("REPO_ROOT") or os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
BI_SCORES_DIR = os.getenv("BI_SCORES_DIR", os.path.join(REPO_ROOT, "layer_bi"))
calibration_size = int(os.getenv("CALIBRATION_SIZE", "100"))
calibration_dataset_path = os.getenv("CALIBRATION_DATASET_PATH", None)
CALIBRATION_DATA_DIR = os.getenv("CALIBRATION_DATA_DIR")

os.makedirs(BI_SCORES_DIR, exist_ok=True)

if CALIBRATION_DATA_DIR is None:
    raise ValueError("CALIBRATION_DATA_DIR environment variable is not set.")


# ------------------------------------------------------------------- masking


def build_masks(input_ids, attention_mask, audio_token_id) -> dict:
    """Returns position information for audio and text tokens.
        parameters:
            input_ids: torch.Tensor of shape (B, T), the ID of each token in the input sequence.
            attention_mask: torch.Tensor of shape (B, T), the attention mask indicating valid tokens. invalid tokens exists because of padding.
            audio_token_id: int, the token ID representing audio tokens
        Returns:
            dict with position information for audio and text tokens.
        """
    valid = attention_mask == 1
    audio = (input_ids == audio_token_id) & valid
    text  = (input_ids != audio_token_id) & valid
    return {"all": valid, "audio": audio, "text": text}


def find_subsequence(haystack, needle):
    """Index of the first occurrence of `needle` in `haystack`, or None."""
    n, m = len(haystack), len(needle)
    if m == 0 or m > n:
        return None
    for s in range(n - m + 1):
        if haystack[s:s + m] == needle:
            return s
    return None


def prompt_span_mask(tokenizer, input_ids, attention_mask, prompt):
    """Boolean mask (B, T) selecting the tokens of `prompt` inside input_ids.

    Tokenizing `prompt` alone can give different tokens than it has in
    context, since BPE merges characters across the boundary with whatever
    precedes it (usually a newline). Try a few likely spellings, dropping
    the first token if needed, until one matches as a literal subsequence.

    Returns (mask, span_length). span_length == 0 means no match was found;
    the caller treats that as a miss rather than averaging an empty mask.
    """
    assert input_ids.shape[0] == 1, "prompt_span_mask assumes batch size 1"
    ids = input_ids[0].tolist()
    mask = torch.zeros_like(attention_mask, dtype=torch.bool)
    for variant in (prompt, "\n" + prompt, " " + prompt):
        needle = tokenizer(variant, add_special_tokens=False)["input_ids"]
        for drop in (0, 1):
            sub = needle[drop:]
            start = find_subsequence(ids, sub)
            if start is not None:
                mask[0, start:start + len(sub)] = True
                return mask, len(sub)
    return mask, 0


def build_text_only_input(processor, data, include_transcript=True):
    """Audio-free pass. Model-agnostic: render the chat template to a string and
    tokenise it directly, so neither processor's audio path is entered at all.

    include_transcript=False is the `text_drop` ablation -- the question with its
    referent removed, which is NOT the text-LLM regime and is only there to
    separate "content in text form" from "content absent".
    """
    text = f"{data['transcript']}\n{data['prompt']}" if include_transcript else data["prompt"]
    conversation = [{"role": "user", "content": [{"type": "text", "text": text}]}]
    return processor.apply_chat_template(
        conversation, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt")


# ------------------------------------------------------------- accumulation


def accumulate_bi_masked(hidden_states, mask, sums):
    """Accumulate block influence for the specified mask across all layers.

        Args:
            hidden_states (list of torch.Tensor): List of hidden states for each layer.
            mask (torch.Tensor): Boolean mask indicating the positions to consider.
            sums (list of float): List to accumulate the block influence sums for each layer.
    """
    num_layers = len(hidden_states) - 1
    for i in range(num_layers):
        bi = block_influence(hidden_states[i], hidden_states[i + 1])
        masked_bi = bi[mask]
        sums[i] += masked_bi.sum().item() #.item() will convert the tensor to a Python float


class BIAccumulator:
    """Per-condition running sums of BI and counts of contributing positions."""

    def __init__(self, n_layers, conditions):
        self.sums = {c: [0.0 for _ in range(n_layers)] for c in conditions}
        self.counts = {c: 0 for c in conditions}

    def add(self, condition, hidden_states, mask):
        n = int(mask.sum())
        if n == 0:
            return
        accumulate_bi_masked(hidden_states, mask, self.sums[condition])
        self.counts[condition] += n

    def means(self):
        return {c: [s / self.counts[c] for s in self.sums[c]] if self.counts[c] else None
                for c in self.sums}


def iter_calibration(calibration_file, dataset_path, max_samples):
    """Yield (index, data, resolved_audio_path) for the first max_samples items."""
    with open(calibration_file, "r") as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            data = json.loads(line)  # if the file is in JSONL format
            path = resolve_audio_path(data["audio_path"], dataset_path, data.get("corpus"))
            yield i, data, path


# ------------------------------------------------------------------ mode: mm


MULTIMODAL_CONDITIONS = ("audio", "text", "text_prompt", "text_template")


def compute_multimodal_bi(model_name, model, processor, n_layers, audio_token_id,
                          calibration_file, dataset_path, max_samples):
    """One forward pass per item; BI split over position types.

    Returns {"bi", "tok_counts", "span_lens"}. `span_lens` is the per-item
    prompt-span length, kept so a later text-only run can check that it is
    masking the same token string.
    """
    acc = BIAccumulator(n_layers, MULTIMODAL_CONDITIONS)
    diag = {"items": 0, "span_found": 0}
    span_lens = []

    for i, data, path in iter_calibration(calibration_file, dataset_path, max_samples):
        diag["items"] += 1
        inputs = model_config[model_name]["build_input_fn"](processor, data, path)
        inputs = inputs.to(model.device, dtype=model.dtype)

        mask = build_masks(inputs["input_ids"], inputs["attention_mask"], audio_token_id)

        assert mask["audio"].any(), (
            f"no audio positions found for {model_name} -- audio_token_id="
            f"{audio_token_id} does not appear in input_ids")

        mask["text_prompt"], span = prompt_span_mask(
            processor.tokenizer, inputs["input_ids"], inputs["attention_mask"],
            data["prompt"])
        mask["text_template"] = mask["text"] & ~mask["text_prompt"]
        span_lens.append(span)
        diag["span_found"] += span > 0

        with torch.inference_mode():
            hidden_states = model(**inputs, output_hidden_states=True).hidden_states

        if i == 0: # print token counts and number of layers for the first sample
            print(f"[{model_name}] {int(mask['audio'].sum())} audio / "
                  f"{int(mask['text'].sum())} text tokens "
                  f"({span} of them prompt), {len(hidden_states) - 1} layers")

        for c in MULTIMODAL_CONDITIONS:
            acc.add(c, hidden_states, mask[c])
        del hidden_states, inputs

    bi = acc.means()
    # standard BI: token-weighted mean over every multimodal position
    total = acc.counts["audio"] + acc.counts["text"]
    bi["combined"] = [(a + t) / total
                      for a, t in zip(acc.sums["audio"], acc.sums["text"])]

    print("multimodal diagnostics:", diag)
    return {"bi": bi, "tok_counts": acc.counts, "span_lens": span_lens}


# ------------------------------------------------------------- mode: text_only


def compute_text_only_bi(model, processor, n_layers, calibration_file,
                         max_samples, include_text_drop=False):
    """Audio-free pass(es) over the same items. No audio is loaded at all, so no
    dataset path or audio token id is needed."""
    conditions = ["text_only", "text_only_prompt"]
    if include_text_drop:
        conditions += ["text_drop", "text_drop_prompt"]
    acc = BIAccumulator(n_layers, conditions)
    diag = {"items": 0, "span_found": 0, "missing_transcript": 0}
    span_lens = []

    passes = [("text_only", "text_only_prompt", True)]
    if include_text_drop:
        passes.append(("text_drop", "text_drop_prompt", False))

    for i, data, _ in iter_calibration(calibration_file, None, max_samples):
        diag["items"] += 1
        if not data.get("transcript"):
            diag["missing_transcript"] += 1
            span_lens.append(0)
            continue

        for all_cond, span_cond, with_transcript in passes:
            inputs = build_text_only_input(processor, data, with_transcript)
            inputs = inputs.to(model.device)
            valid = inputs["attention_mask"] == 1
            span_mask, span = prompt_span_mask(
                processor.tokenizer, inputs["input_ids"], inputs["attention_mask"],
                data["prompt"])

            with torch.inference_mode():
                hidden_states = model(**inputs, output_hidden_states=True).hidden_states

            if i == 0 and all_cond == "text_only":
                print(f"[text_only] {int(valid.sum())} tokens "
                      f"({span} of them prompt), {len(hidden_states) - 1} layers")

            acc.add(all_cond, hidden_states, valid)
            acc.add(span_cond, hidden_states, span_mask)

            if all_cond == "text_only":
                span_lens.append(span)
                diag["span_found"] += span > 0

            del hidden_states, inputs

    print("text_only diagnostics:", diag)
    return {"bi": acc.means(), "tok_counts": acc.counts, "span_lens": span_lens}


# ----------------------------------------------------------------- merge / io


def load_results(out_file):
    """Only the current schema's two keys ("bi", "tok_counts") are kept -- any
    legacy top-level keys from older script versions (e.g. audio_bi,
    audio_tok_count) are dropped so the file gets overwritten clean instead
    of accumulating them."""
    if os.path.exists(out_file):
        with open(out_file) as f:
            existing = json.load(f)
        print(f"merging into existing {out_file}")
        return {key: existing.get(key, {}) for key in
                ("bi", "tok_counts")}
    return {"bi": {}, "tok_counts": {}}


def merge(results, mode, out):
    results["bi"].update(out["bi"])
    results["tok_counts"].update(out["tok_counts"])
    return results


def check_spans(mm_span_lens, to_span_lens):
    """The multimodal and text-only passes mask the same question tokens. If the
    per-item span lengths disagree, the matched comparison is not matched.
    Only meaningful when both modes were just computed in this run -- span
    lengths aren't persisted to the output file."""
    if not mm_span_lens or not to_span_lens:
        return
    n = min(len(mm_span_lens), len(to_span_lens))
    bad = [i for i in range(n) if mm_span_lens[i] != to_span_lens[i]]
    if bad:
        print(f"WARNING: prompt span length differs on {len(bad)}/{n} items "
              f"(first few: {bad[:5]}) -- text_prompt vs text_only_prompt is "
              f"NOT a matched comparison on those items")
    else:
        print(f"prompt spans agree on all {n} items")


model_config = {
    "qwen2audio": {
        "calibration_file": os.path.join(CALIBRATION_DATA_DIR, "calibration_data_qwen2audio.jsonl"),
        "build_input_fn": build_input_qwen,
        "audio_token_attr": "audio_token_index",   # Qwen2AudioConfig
    },
    "audioflamingo3": {
        "calibration_file": os.path.join(CALIBRATION_DATA_DIR, "calibration_data_audioflamingo3.jsonl"),
        "build_input_fn": build_input_af3,
        "audio_token_attr": "audio_token_id",      # AudioFlamingo3Config -- different name
    },
    "voxtral": {
        "calibration_file": os.path.join(CALIBRATION_DATA_DIR, "calibration_data_voxtral.jsonl"),
        "build_input_fn": build_input_voxtral,
        "audio_token_attr": "audio_token_id",
    },
}

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["qwen2audio", "audioflamingo3", "voxtral"], default="qwen2audio")
    p.add_argument("--mode", choices=["multimodal", "text_only", "both"], default="both",
                   help="which pass to run; results merge into the same file")
    p.add_argument("--text-drop", action="store_true",
                   help="text_only mode: also run the no-transcript ablation")
    p.add_argument("--skip-transcribe", action="store_true",
                   help="do not backfill missing transcripts with Whisper")
    args = p.parse_args()

    cfg = model_config[args.model]
    calibration_file = cfg["calibration_file"]
    out_file = os.path.join(BI_SCORES_DIR, f"{args.model}_layer_modality_bi_{calibration_size}.json")
    run_mm = args.mode in ("multimodal", "both")
    run_to = args.mode in ("text_only", "both")

    if run_to and not args.skip_transcribe:
        ensure_transcripts(calibration_file, calibration_dataset_path,
                           calibration_size)

    model, processor = load_model(args.model)

    num_dec_layers = model.config.text_config.num_hidden_layers
    results = load_results(out_file)
    mm_span_lens = to_span_lens = None

    if run_mm:
        out = compute_multimodal_bi(
            model_name=args.model,
            model=model,
            processor=processor,
            n_layers=num_dec_layers,
            audio_token_id=getattr(model.config, cfg["audio_token_attr"]),
            calibration_file=calibration_file,
            dataset_path=calibration_dataset_path,
            max_samples=calibration_size,
        )
        mm_span_lens = out["span_lens"]
        results = merge(results, "multimodal", out)

    if run_to:
        out = compute_text_only_bi(
            model=model,
            processor=processor,
            n_layers=num_dec_layers,
            calibration_file=calibration_file,
            max_samples=calibration_size,
            include_text_drop=args.text_drop,
        )
        to_span_lens = out["span_lens"]
        results = merge(results, "text_only", out)

    check_spans(mm_span_lens, to_span_lens)
    print("token counts:", results["tok_counts"])

    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {out_file}")