# Backfill a `transcript` field into a calibration jsonl.
#
# The text-only BI condition substitutes the transcript for the audio, so every
# calibration item needs one. This transcribes any item that is missing one
# with Whisper and rewrites the file in place (atomically: temp file + rename).
#
# Only the first `max_samples` items are considered -- those are the only ones
# the BI scripts read -- and every other line is written back verbatim, so
# re-running with a larger --samples only adds what is new.
#
# Whisper is loaded lazily and freed before returning, so calling this from
# decoder_bi.py before the speech LLM is loaded costs no resident memory.
#
# Run standalone:
#   python -m calibration.transcribe_calibration \
#       --calibration-file $CALIBRATION_DATA_DIR/sampled_audio.jsonl \
#       --samples 100

import argparse
import json
import os
import tempfile

import torch

WHISPER_CHECKPOINT = os.getenv("WHISPER_CHECKPOINT", "openai/whisper-large-v3-turbo")


def resolve_audio_path(audio_path, dataset_path, corpus=None):
    """If dataset_path is unset, return audio_path unchanged.

    Otherwise, if `corpus` appears as a path segment in audio_path, rebuild
    the path as dataset_path/corpus/<everything after corpus>. If not,
    fall back to dataset_path/<basename of audio_path>.
    """
    if not dataset_path:
        return audio_path
    parts = os.path.normpath(audio_path).split(os.sep)
    if corpus and corpus in parts:
        rel = parts[parts.index(corpus) + 1:]
        return os.path.join(dataset_path, corpus, *rel)
    return os.path.join(dataset_path, os.path.basename(audio_path))


def ensure_transcripts(calibration_file, dataset_path=None, max_samples=None,
                       checkpoint=WHISPER_CHECKPOINT, language="en", force=False):
    """Add `transcript` to any of the first max_samples items that lack one.

    Returns the number of items transcribed. Loads no model when there is
    nothing to do.
    """
    with open(calibration_file) as f:
        lines = f.read().splitlines()

    todo = []
    for i, line in enumerate(lines):
        if max_samples is not None and i >= max_samples:
            break
        if not line.strip():
            continue
        data = json.loads(line)
        if force or not data.get("transcript"):
            todo.append((i, data))

    n_considered = len(lines) if max_samples is None else min(len(lines), max_samples)
    if not todo:
        print(f"[transcripts] all {n_considered} items already have a transcript")
        return 0

    print(f"[transcripts] {len(todo)}/{n_considered} missing -- loading {checkpoint}")
    from transformers import pipeline

    asr = pipeline(
        "automatic-speech-recognition",
        model=checkpoint,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device=0 if torch.cuda.is_available() else -1,
        chunk_length_s=30,          # enables long-form for clips over 30 s
    )
    gen_kwargs = {"task": "transcribe"}
    if language:
        gen_kwargs["language"] = language

    try:
        for n, (i, data) in enumerate(todo, 1):
            path = resolve_audio_path(data["audio_path"], dataset_path, data.get("corpus"))
            data["transcript"] = asr(path, generate_kwargs=gen_kwargs)["text"].strip()
            lines[i] = json.dumps(data)
            if n % 10 == 0 or n == len(todo):
                print(f"[transcripts] {n}/{len(todo)}")
    finally:
        del asr
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out_dir = os.path.dirname(os.path.abspath(calibration_file))
    with tempfile.NamedTemporaryFile("w", dir=out_dir, delete=False) as tmp:
        tmp.write("\n".join(lines) + "\n")
        tmp_path = tmp.name
    os.replace(tmp_path, calibration_file)      # atomic; never a half-written file

    print(f"[transcripts] wrote {len(todo)} transcripts to {calibration_file}")
    return len(todo)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--calibration-file", required=True)
    p.add_argument("--dataset-path", default=os.getenv("CALIBRATION_DATASET_PATH"))
    p.add_argument("--samples", type=int, default=None)
    p.add_argument("--checkpoint", default=WHISPER_CHECKPOINT)
    p.add_argument("--language", default="en",
                   help="pass an empty string to let Whisper auto-detect")
    p.add_argument("--force", action="store_true",
                   help="re-transcribe items that already have a transcript")
    args = p.parse_args()

    ensure_transcripts(args.calibration_file, args.dataset_path, args.samples,
                       args.checkpoint, args.language or None, args.force)