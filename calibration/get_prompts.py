# Generates a calibration question and perplexity score for each sampled
# audio clip, using the target model's own inference.
# Supports three model backends: Qwen2-Audio-7B-Instruct, Voxtral-Mini-3B-2507,
# and Audio Flamingo 3.
# run with:  python -m calibration.get_prompts --model qwen2audio
import os
import types
import json
import librosa
import torch
import argparse
from evaluate.evaluate_model import load_model
from calibration.transcribe_calibration import resolve_audio_path

calibration_dataset_path = os.getenv("CALIBRATION_DATASET_PATH", None)


def qwen_inference_with_score(model, processor, audio_path, prompt):
    conversation = [
        {"role": "user", "content": [
            {"type": "audio", "audio_url": "placeholder"},
            {"type": "text", "text": prompt},
        ]},
    ]

    audios = [
        librosa.load(audio_path, sr=processor.feature_extractor.sampling_rate)[0]
    ]
    with torch.inference_mode():
        text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=text_prompt, audio=audios, sampling_rate=16000, return_tensors="pt", padding=True).to(model.device, dtype=model.dtype)

        output = model.generate(**inputs, max_new_tokens=100, output_scores=True, return_dict_in_generate=True)
        generate_ids = output.sequences[:, inputs["input_ids"].shape[1]:]  # drop the echoed prompt, keep only new tokens
        response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    scores = torch.stack(output.scores, dim=0)
    log_probs = torch.log_softmax(scores, dim=-1)
    token_log_probs = log_probs.gather(
        dim=-1,
        index=generate_ids.transpose(0, 1).unsqueeze(-1)
    ).squeeze(-1)

    nll = -token_log_probs.mean(dim=0)
    perplexity = torch.exp(nll)

    return response, perplexity.item()


def voxtral_inference_with_score(model, processor, audio_path, prompt, max_new_tokens=100):
    conversation = [
            {"role": "user", "content": [
                {"type": "audio", "path": audio_path},
                {"type": "text", "text": prompt},
            ]},
        ]

    with torch.inference_mode():
        inputs = processor.apply_chat_template(conversation, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt",
            ).to(model.device, dtype=model.dtype)

        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, output_scores=True, return_dict_in_generate=True)
        generate_ids = outputs.sequences[:, inputs.input_ids.shape[1]:]  # drop the echoed prompt, keep only new tokens
        response = processor.batch_decode(generate_ids, skip_special_tokens=True)[0]

    scores = torch.stack(outputs.scores, dim=0)
    log_probs = torch.log_softmax(scores, dim=-1)
    token_log_probs = log_probs.gather(
        dim=-1,
        index=generate_ids.transpose(0, 1).unsqueeze(-1)
    ).squeeze(-1)

    nll = -token_log_probs.mean(dim=0)
    perplexity = torch.exp(nll)

    return response, perplexity.item()


def af3_inference_with_score(model, processor, audio_path, prompt):
    """
    AF3 is a standard transformers.GenerationMixin model -- no monkeypatch
    needed, output_scores/return_dict_in_generate work directly like Qwen2-Audio.
    Uses apply_chat_template(tokenize=True, return_dict=True) to get
    input_ids/attention_mask/input_features/input_features_mask in one call,
    per the model card's documented usage.
    """
    conversation = [
        {"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "audio", "path": audio_path},
        ]},
    ]

    with torch.inference_mode():
        inputs = processor.apply_chat_template(
            conversation, tokenize=True, add_generation_prompt=True, return_dict=True,
        ).to(model.device, dtype=model.dtype)

        output = model.generate(**inputs, max_new_tokens=100, output_scores=True, return_dict_in_generate=True)
        generate_ids = output.sequences[:, inputs["input_ids"].shape[1]:]  # drop the echoed prompt, keep only new tokens
        response = processor.batch_decode(generate_ids, skip_special_tokens=True)[0]

    scores = torch.stack(output.scores, dim=0)
    log_probs = torch.log_softmax(scores, dim=-1)
    token_log_probs = log_probs.gather(
        dim=-1,
        index=generate_ids.transpose(0, 1).unsqueeze(-1)
    ).squeeze(-1)

    nll = -token_log_probs.mean(dim=0)
    perplexity = torch.exp(nll)

    return response, perplexity.item()


def sort_by_perplexity(file):
    with open(file, "r") as f:
        rows = [json.loads(line) for line in f]

    rows_sorted = sorted(rows, key=lambda r: r["perplexity"])

    with open(file, "w") as f:
        for row in rows_sorted:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", choices=["qwen2audio", "voxtral", "audioflamingo3"], default="qwen2audio",
        help="Which audio LLM to use for generating calibration questions.",
    )
    args = parser.parse_args()

    REPO_ROOT = os.getenv("REPO_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    audio_samples_file = os.path.join(REPO_ROOT, "calibration", "sampled_audio.jsonl")

    prompt = (
        "Listen to the audio and generate one question about the speech. Use third-person perspective."
    )
    calibration_data_file = os.path.join(REPO_ROOT, "calibration", f"calibration_data_{args.model}.jsonl")

    if args.model == "qwen2audio":
        model, processor = load_model("qwen2audio", load_processor=True)
        infer_fn = lambda audio_path, prompt: qwen_inference_with_score(model, processor, audio_path, prompt)

    elif args.model == "voxtral":
        model, processor = load_model("voxtral", load_processor=True)
        infer_fn = lambda audio_path, prompt: voxtral_inference_with_score(model, processor, audio_path, prompt)

    elif args.model == "audioflamingo3":
        model, processor = load_model("audioflamingo3", load_processor=True)
        infer_fn = lambda audio_path, prompt: af3_inference_with_score(model, processor, audio_path, prompt)

    with open(audio_samples_file, "r") as f_in, open(calibration_data_file, "w") as f_out:
        for line in f_in:
            data = json.loads(line)
            audio_path = resolve_audio_path(data["audio_path"], calibration_dataset_path, data.get("corpus"))
            response, perplexity = infer_fn(audio_path, prompt)
            data["prompt"] = response
            data["perplexity"] = perplexity
            f_out.write(json.dumps(data) + "\n")
            f_out.flush()

    sort_by_perplexity(calibration_data_file)