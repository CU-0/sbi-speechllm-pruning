"""
Copies the audio files listed in sampled_audio.jsonl into one destination
folder, organized by corpus. Optional -- nothing else in the pipeline
depends on this folder existing.

Run with:  python -m calibration.create_calibration_set
"""
import shutil
import os
import json

REPO_ROOT = os.getenv("REPO_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
calibration_data_file = os.path.join(REPO_ROOT, "calibration", "sampled_audio.jsonl")
destination_folder = os.getenv("CALIBRATION_DATASET_PATH", "/path/to/calibration_dataset")

def copy_calibration_data(calibration_data_file, destination_folder, size_limit=None):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    considered = 0
    failed = 0
    with open(calibration_data_file, "r") as f:
        for i,line in enumerate(f):
            if size_limit is not None and i >= size_limit:
                break
            considered += 1
            data = json.loads(line)
            audio_path = data["audio_path"]
            corpus = data.get("corpus")
            parts = os.path.normpath(audio_path).split(os.sep)
            rel = parts[parts.index(corpus) + 1:] if corpus and corpus in parts else [parts[-1]]
            destination_path = os.path.join(destination_folder, corpus or "", *rel)

            if os.path.exists(audio_path):
                os.makedirs(os.path.dirname(destination_path), exist_ok=True)
                shutil.copy(audio_path, destination_path)
            else:
                failed += 1
    print(f"Copied {considered-failed} files to {destination_folder}. Failed to copy {failed} files.")

if __name__ == "__main__":
    size_limit = None  # Set to None to copy all files
    copy_calibration_data(calibration_data_file, destination_folder, size_limit)