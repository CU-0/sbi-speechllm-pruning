# run with python -m evaluate.evaluate_model --model <model_name> --dataset <dataset_name>
from datasets import load_dataset, Audio
from evaluate.verify import add_answer_char, get_answer_letter
from evaluate.prompt import get_question_text, get_prediction, extract_answer_obqa
import argparse
import json
import os
import torch
import random
import gc
import tempfile
import numpy as np
import soundfile as sf

QWEN_CHECKPOINT = os.getenv("QWEN_CHECKPOINT", "Qwen/Qwen2-Audio-7B-Instruct")
AF3_CHECKPOINT = os.getenv("AF3_CHECKPOINT", "nvidia/audio-flamingo-3-hf")  # transformers-native release, not nvidia/audio-flamingo-3
VOXTRAL_CHECKPOINT = os.getenv("VOXTRAL_CHECKPOINT", "mistralai/Voxtral-Mini-3B-2507")

DATASET_CONFIGS = {
    "mmsu": {
        "hf_repo": "ddwang2000/MMSU",
        "hf_config": None,
        "split": "train",
        "needs_answer_char": True,
        "needs_audio_cast": False,
        "task_key": "sub-sub-category",
        "include_text_prompt": True, 
    },
    "obqa": {
        "hf_repo": "hlt-lab/voicebench",
        "hf_config": "openbookqa",
        "split": "test",
        "needs_answer_char": False,
        "needs_audio_cast": True,
        "task_key": None,  # flat dataset, no subcategory to stratify over
        "include_text_prompt": False,  # audio *is* the spoken instruction (question+choices);
                                        # VoiceBench's own audio-modality eval passes audio only,
                                        # matching generate_audio() in their qwen2.py adapter
    },
}

PREDICTION_PARSERS = {
    "mmsu": get_prediction,
    "obqa": extract_answer_obqa,
}

def load_eval_dataset(dataset_name):
    cfg = DATASET_CONFIGS[dataset_name]

    if cfg["hf_config"] is not None:
        ds = load_dataset(cfg["hf_repo"], cfg["hf_config"], split=cfg["split"])
    else:
        ds = load_dataset(cfg["hf_repo"], split=cfg["split"])

    if cfg["needs_answer_char"]:
        ds = ds.map(add_answer_char, input_columns=["choice_a", "choice_b", "choice_c", "choice_d", "answer_gt"])

    if cfg["needs_audio_cast"]:
        ds = ds.cast_column("audio", Audio(sampling_rate=16_000))

    return ds


def normalize_row(row, dataset_name):
    """Dataset-agnostic view of a row: prompt text, gt answer letter, category, row id."""
    if dataset_name == "mmsu":
        return {
            "id": row["id"],
            "prompt": get_question_text(row),
            "answer_letter": row["answer_char"],
            "category": row["sub-sub-category"],
        }
    elif dataset_name == "obqa":
        return {
            "id": None,
            "prompt": row["prompt"],       # VoiceBench pre-formats the full MC prompt
            "answer_letter": row["reference"],  # VoiceBench's own MCQEvaluator reads this key
            "category": "openbookqa",       # VoiceBench openbookqa has no subcategories
        }
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def sample_ds(ds, n_total, task_key=None, seed=42):
    """
    Selects up to n_total rows from ds.

    If task_key names a column present in ds, stratifies as evenly as
    possible across that column's distinct values (remainder rows go to
    the first few tasks in sorted order). If task_key is None or not a
    column in ds, falls back to a plain random sample of n_total rows.

    Returns (subset, n_selected). n_selected can be < n_total if some
    tasks don't have enough rows to fill their even share.
    """
    random.seed(seed)

    if task_key is not None and task_key in ds.column_names:
        indices_by_task = {}
        for i, task in enumerate(ds[task_key]):
            indices_by_task.setdefault(task, []).append(i)

        tasks = sorted(indices_by_task.keys())
        base_n, remainder = divmod(n_total, len(tasks))

        selected_indices = []
        for i, task in enumerate(tasks):
            indices = indices_by_task[task]
            k = base_n + (1 if i < remainder else 0)
            k = min(k, len(indices))
            selected_indices.extend(random.sample(indices, k))

        selected_indices.sort()
    else:
        n = min(n_total, len(ds))
        selected_indices = sorted(random.sample(range(len(ds)), n))

    return ds.select(selected_indices), len(selected_indices)


def qwen_inference(model, processor, prompt, audio, asr=False, max_new_tokens=200, return_truncated=False):
    with torch.inference_mode():
        content = [{"type": "audio", "audio_url": "placeholder"}]
        if prompt is not None:
            content.append({"type": "text", "text": prompt})

        conversation = [{"role": "user", "content": content}]

        text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        if asr:
            text_prompt = text_prompt + "The transcription is \""

        inputs = processor(text=text_prompt, audio=[audio], sampling_rate=16000, return_tensors="pt", padding=True).to(model.device)

        generate_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        generate_ids = generate_ids[:, inputs["input_ids"].shape[1]:]
        response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        if asr: 
            response = response.split('"')[0] if '"' in response else response.strip()  # Extract the transcribed text from the response
        truncated = generate_ids.shape[1] >= max_new_tokens

    return (response, truncated) if return_truncated else response

def af3_inference(model, processor, prompt, audio, asr=False, max_new_tokens=200, return_truncated=False):
    with torch.inference_mode():
        content = []
        content.append({"type": "audio", "audio": audio})
        if prompt is not None:
            content.append({"type": "text", "text": prompt})

        conversation = [{"role": "user", "content": content}]
        inputs = processor.apply_chat_template(
            conversation, tokenize=True, add_generation_prompt=True, return_dict=True,
        ).to(model.device, model.dtype)

        generate_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        generate_ids = generate_ids[:, inputs["input_ids"].shape[1]:]
        if not asr:
            response = processor.batch_decode(
                generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        else: 
            response = processor.batch_decode(
                generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False, strip_prefix=True)[0]
        truncated = generate_ids.shape[1] >= max_new_tokens

    return (response, truncated) if return_truncated else response

def voxtral_inference(model, processor, prompt, audio, asr=False, max_new_tokens=200, return_truncated=False):
    """Voxtral takes audio by url/path/base64 only -- VoxtralProcessor.__call__
    rejects audio outright and the chat template has no raw-array form -- so the
    decoded dataset array goes through a temp wav. Transcription is the
    exception: apply_transcription_request accepts arrays directly.
    """
    with torch.inference_mode():
        if asr:
            inputs = processor.apply_transcription_request(
                audio=np.asarray(audio, dtype=np.float32), sampling_rate=16000,
                format=["wav"], language=["en"],
                model_id=MODEL_CONFIGS["voxtral"]["checkpoint"],
            ).to(model.device, model.dtype)
            generate_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        else:
            with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
                sf.write(tmp.name, np.asarray(audio, dtype=np.float32), 16000)
                content = [{"type": "audio", "path": tmp.name}]
                if prompt is not None:
                    content.append({"type": "text", "text": prompt})
                inputs = processor.apply_chat_template(
                    [{"role": "user", "content": content}],
                    tokenize=True, add_generation_prompt=True,
                    return_dict=True, return_tensors="pt",
                ).to(model.device, model.dtype)
                generate_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

        generate_ids = generate_ids[:, inputs["input_ids"].shape[1]:]
        response = processor.batch_decode(generate_ids, skip_special_tokens=True)[0]
        truncated = generate_ids.shape[1] >= max_new_tokens

    return (response, truncated) if return_truncated else response

MODEL_CONFIGS = {
    "qwen2audio": {
        "inference_fn": qwen_inference,
        "checkpoint": QWEN_CHECKPOINT,
    },
    "audioflamingo3": {
        "inference_fn": af3_inference,
        "checkpoint": AF3_CHECKPOINT,
    },
    "voxtral": {
        "inference_fn": voxtral_inference,
        "checkpoint": VOXTRAL_CHECKPOINT,
    },
}


def load_model(model_name, checkpoint=None, load_processor=True, dtype=torch.float16):
    checkpoint = checkpoint or MODEL_CONFIGS[model_name]["checkpoint"]
    if model_name == "qwen2audio":
        from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration as ModelCls
    elif model_name == "audioflamingo3":
        from transformers import AutoProcessor, AudioFlamingo3ForConditionalGeneration as ModelCls
    elif model_name == "voxtral":
        from transformers import AutoProcessor, VoxtralForConditionalGeneration as ModelCls
    processor = AutoProcessor.from_pretrained(checkpoint) if load_processor else None
    model = ModelCls.from_pretrained(
        checkpoint, dtype=dtype, device_map="auto").eval()
    
    for p in model.parameters():
        if p.is_floating_point() and p.dtype is not dtype:
            p.data = p.data.to(dtype)
    
    dts = {p.dtype for p in model.parameters()}
    assert len(dts) == 1, f"mixed dtypes in model: {dts}"

    # Checkpoints disagree on decoding defaults (Qwen2-Audio ships do_sample=True,
    # AF3 ships nothing) -- pin greedy so runs are comparable and reproducible.
    gc_ = model.generation_config
    gc_.do_sample = False
    gc_.temperature = gc_.top_p = gc_.top_k = None   # None, not 0 -- avoids the validate() warnings
    gc_.num_beams = 1
    gc_.repetition_penalty = 1.0

    return model, processor


def run_eval(model, processor, ds, dataset_name, output_file, model_name="qwen2audio"):
    category_stats = {}
    include_text_prompt = DATASET_CONFIGS[dataset_name]["include_text_prompt"]
    inference_fn = MODEL_CONFIGS[model_name]["inference_fn"]

    with open(output_file, "w") as f:
        for i, row in enumerate(ds):
            if i % (len(ds) // 20 or 1) == 0:
                print(f"Processing row {i}/{len(ds)}")
                torch.cuda.empty_cache()
                gc.collect()

            norm = normalize_row(row, dataset_name)
            model_prompt = norm["prompt"] if include_text_prompt else None
            response = inference_fn(model, processor, model_prompt, row["audio"]["array"],
                                    max_new_tokens=50) # restrict response length for MCQ tasks

            predicted = PREDICTION_PARSERS[dataset_name](response)

            if predicted is None:
                outcome = "format_error"
            elif predicted == norm["answer_letter"]:
                outcome = "correct"
            else:
                outcome = "incorrect"

            category = norm["category"]

            if category not in category_stats:
                category_stats[category] = {
                    "total": 0,
                    "correct": 0,
                    "incorrect": 0,
                    "format_error": 0,
                }

            category_stats[category]["total"] += 1
            category_stats[category][outcome] += 1

            result = {
                "row_id": norm["id"],
                "category": category,
                "outcome": outcome,
                "gt": norm["answer_letter"],
                "parsed": predicted,
                "response": response,
            }
            f.write(json.dumps(result) + "\n")
            f.flush()

    return category_stats


if __name__ == "__main__": # run with python -m evaluate.evaluate_model --model voxtral
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen2audio",
                        choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--dataset", type=str, default="mmsu", choices=list(DATASET_CONFIGS.keys()))
    args = parser.parse_args()

    ds = load_eval_dataset(args.dataset)

    model, processor = load_model(args.model)

    output_file = f"{args.model}_short_results_{args.dataset}.jsonl"

    run_eval(model, processor, ds, args.dataset, output_file, model_name=args.model)