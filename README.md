# SpeechLLM Compression

Layer-pruning speech LLMs via layer-importance scores tailored to the
encoder/adapter/decoder structure of speech LLMs, evaluated against the
original ShortGPT Block Influence (BI) score and a reverse-order baseline.

Given an audio-language model (Qwen2-Audio-7B-Instruct, Audio Flamingo 3, or
Voxtral-Mini-3B-2507), this repo:

1. computes a per-layer importance score for the encoder and/or decoder,
2. prunes the lowest-scoring layers (or a targeted subset), and
3. evaluates the pruned model's accuracy (MMSU, OpenBookQA, ASR) against the
   uncompressed baseline.

## Methods

Layer-importance scoring and pruning is split into two packages, both built
on the Block Influence (BI) score from Men et al. 2025 (ShortGPT): the
cosine distance between a layer's input and output hidden state, where
low-BI layers contribute little and are pruned first.

**`shortGPT/`** — the original ShortGPT baseline: plain BI, scored at each
component's own output, one score file per model covering both the encoder
and decoder.
- `compute_bi.py` — computes BI for a model's encoder and decoder (writing
  both under `"encoder_bi"`/`"decoder_bi"` in one file).
- `prune.py` — removes the selected layers from a loaded model.
- `sweep.py` — sweeps a fixed set of (n_enc, n_dec) pruning candidates from
  a single BI file: prune both components by it, evaluate on `--dataset`,
  write one result row per candidate.
- `custom_pruning.py` — targeted ablation: prune an explicit list of
  encoder/decoder layer indices (rather than picking the lowest-BI n
  layers), for isolating whether a performance drop comes from a specific
  layer or just from removing that many layers.

**`sbi/`** — Speech Block Influence (SBI, scored separately per component):
separate encoder and decoder scoring tailored to a speech LLM's
encoder/adapter/decoder structure.
- `encoder_bi.py` — computes SBI-Enc: scores encoder layers at
  the **adapter's output** (removing each layer and measuring the change
  in the adapter's output hidden state) rather than at the encoder's own
  output, since that's what the decoder actually receives.
- `decoder_bi.py` — computes SBI-Dec: splits decoder-layer BI
  by token modality (audio vs. text position), since the two are
  numerically dominated by whichever is more numerous in the calibration
  sequence. Also runs the text-only calibration BI measurement (BI computed on
  transcripts instead of audio).
- `reverse_bi.py` — generates a reverse-priority score vector (ranking the
  deepest layers most redundant, excluding the last layer) as a baseline
  to compare against BI-guided pruning.
- `sweep.py` — the same candidate sweep as `shortGPT/sweep.py`, but reading
  an encoder SBI-Enc-score file and a *separate* decoder file
  (modality-split BI or reverse-priority).

## Pipeline

```
calibration/  ->  layer_BI/  ->  shortGPT/sweep.py or sbi/sweep.py  ->  results/  ->  plot_*.py / run_stats.sh
(build calib.     (BI/SBI         (prune + evaluate)                     (metrics)    (figures)
 set)              scores)
```

1. **Build a calibration set** (`calibration/`): sample audio clips from your
   datasets (`sample_audio.py`), fetch/transcribe prompts
   (`get_prompts.py`, `transcribe_calibration.py`), and optionally copy the
   sampled files into a standalone folder (`create_calibration_set.py`).
   Pre-built calibration manifests for each model are already included
   (`calibration/calibration_data_*.jsonl`).
2. **Compute BI scores**: `shortGPT/compute_bi.py` for the baseline,
   `sbi/encoder_bi.py` + `sbi/decoder_bi.py` for this paper's
   method, written to `layer_BI/`.
3. **Prune + evaluate** (`shortGPT/sweep.py`, `sbi/sweep.py`,
   `shortGPT/custom_pruning.py`), writing results to
   `results/<run>/`.
4. **Summarize + plot**: `run_stats.sh` builds a CSV + PNG per task from a
   results folder; `layer_BI/plot_encoder_importance.py`,
   `layer_BI/plot_decoder_importance.py`, `results/plot_results.py`,
   `layer_BI/bi_correlation.py` produce the remaining figures/statistics.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt   # CUDA 12.6 wheels; edit the --extra-index-url line for a different CUDA version

cp env.sh.example env.sh          # fill in your HF token + dataset paths
source env.sh
```

## Citation

The paper describing this work is still in preparation (expected early 2027).
If you use this code before then, please cite the repository directly.

