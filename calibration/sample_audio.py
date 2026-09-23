# Sample audio files from various datasets to construct a calibration set to measure layer importance.
# run with: python -m calibration.sample_audio
from pathlib import Path
import random
import os
import json

REPO_ROOT = os.getenv("REPO_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Point these at your local copies of each dataset (or set the env vars below).
asr_path = os.getenv("ASR_DATASET_PATH", "/path/to/cv25_en")
emo_path = os.getenv("EMO_DATASET_PATH", "/path/to/IEMOCAP")
voxceleb = os.getenv("VOXCELEB_DATASET_PATH", "/path/to/voxceleb")
gaokao = os.getenv("GAOKAO_DATASET_PATH", "/path/to/Gaokao")
datasets = [
    (asr_path, ".mp3"),
    (emo_path, ".wav"),
    (voxceleb, ".wav"),
    (gaokao, ".wav"),
]

sample_file = os.path.join(REPO_ROOT, "calibration", "sampled_audio.jsonl")

# Simple recursive glob, if there's no manifest
def list_wavs(root_dir, type = ".wav"):
    # List all .wav files in the directory and its subdirectories
    return list(Path(root_dir).rglob(f"*{type}"))

random.seed(42)  # fixed seed = reproducible calibration set, important for a paper

def sample_files(file_list, n):
    n = min(n, len(file_list))
    return random.sample(file_list, n), n

if __name__ == "__main__":
    total_sampled = 0
    with open(sample_file, "w") as f:
        for dataset, exts in datasets:
            wav_files = list_wavs(dataset, exts)
            sampled_files, num_sampled = sample_files(wav_files, 30)  # number of samples per dataset
            total_sampled += num_sampled
            print(dataset, num_sampled)
            for file in sampled_files:
                data = {"audio_path": str(file), "corpus": Path(dataset).name, "prompt": "", "perplexity": 0.0} 
                f.write(json.dumps(data) + "\n")
    print(f"Total sampled files: {total_sampled}")