import jiwer
from whisper_normalizer.english import EnglishTextNormalizer
import argparse
import os
import re
import pandas as pd
import librosa
import json
from evaluate.evaluate_model import MODEL_CONFIGS, load_model

asr_path = os.getenv("ASR_DATASET_PATH")
BASE_CHECKPOINT = os.getenv("BASE_CHECKPOINT")      # optional override, e.g. a pruned checkpoint
results_dir = os.getenv("RESULTS_DIR")

if not results_dir:
    raise RuntimeError("RESULTS_DIR must be set in the environment")
os.makedirs(results_dir, exist_ok=True)


def evaluate_asr(model, processor, asr_path, results_file, ds, model_name):
    inference_fn = MODEL_CONFIGS[model_name]["inference_fn"]
    prompt = "Transcribe the input speech."

    with open(results_file, "w") as f:
        for index, row in ds.iterrows():
            audio_path = os.path.join(asr_path, "clips", row["path"])
            audio, _ = librosa.load(audio_path, sr=16000)

            response, truncated = inference_fn(
                model, processor, prompt, audio,
                asr=True, max_new_tokens=75, return_truncated=True)

            result = {
                "path": row["path"],
                "gt": row["sentence"],
                "response": response,
                "truncated": truncated,
            }
            f.write(json.dumps(result) + "\n")
            f.flush()


QUOTES = {'’': "'", '‘': "'", '“': '"', '”': '"', '′': "'", '´': "'", '`': "'"}
BRACKETS = str.maketrans({c: ' ' for c in '()[]<>{}'})

def pre(s):
    # Preprocess the input string by normalizing quotes, removing brackets, and handling "cannot"
    for k, v in QUOTES.items():
        s = s.replace(k, v)
    s = s.translate(BRACKETS)
    return re.sub(r'\bcannot\b', 'can not', s, flags=re.I)
        
def compute_wer(results_file):
    """
        Compute the Word Error Rate (WER) from the results file.
    """
    normalizer = EnglishTextNormalizer()
    per_utt = []   # cache (S, D, I, N) per utterance

    with open(results_file, "r") as f:
        for line in f:
            result = json.loads(line)
            reference = pre(result["gt"])
            response = pre(result["response"])
            o = jiwer.process_words(normalizer(reference), normalizer(response))
            per_utt.append((o.substitutions, o.deletions, o.insertions,
                            o.substitutions + o.deletions + o.hits))

    S, D, I, N = (sum(x) for x in zip(*per_utt))
    corpus_wer = (S + D + I) / N
    print(f"Corpus WER: {corpus_wer}")
    return corpus_wer

if __name__ == "__main__":
    # to run:
    # python -m evaluate.asr_eval --model qwen2audio --mode eval
    # python -m evaluate.asr_eval --model qwen2audio --mode wer
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen2audio",
                        choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--mode", type=str, default="eval", choices=["eval", "wer"])
    parser.add_argument("--results_file", type=str, required=True)
    args = parser.parse_args()

    results_file = args.results_file

    if args.mode == "eval":
        model, processor = load_model(args.model)
        ds = pd.read_csv(os.path.join(asr_path, "dev.tsv"), sep="\t")
        print(len(ds), "samples loaded from", asr_path)
        ds = ds.sample(n=100, random_state=42).reset_index(drop=True)
        evaluate_asr(model, processor, asr_path, results_file, ds, args.model)
    elif args.mode == "wer":
        compute_wer(results_file)