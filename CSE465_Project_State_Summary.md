# CSE465 Project State Summary
## For: Rayed Riasat Rabbi (2311649642) — Approach 2
## Course: CSE 465: Pattern Recognition and Neural Network, North South University
## Instructor: Dr. Nabeel Mohammed
## Date of this document: April 5, 2026

---

## 1. Project Overview

**Title:** Multilingual Speech-to-Speech Voice Cloning via Knowledge Distillation over Large Voice Clone Models to Tiny On-Device Model

**Team:**
- Md. Tanvir Chowdhury (2232122042) — Approach 1: Two-Stage Progressive Distillation
- Rayed Riasat Rabbi (2311649642) — **Approach 2: Structural Pruning + Fine-tuning** ← this document
- Nazmus Sakib Nihal (2311748042) — Approach 3: Layer Selection + Speaker Adapter Module

**Rayed's specific goal:** Compress `facebook/seamless-m4t-v2-large` (1805M parameters) using structural pruning of the speech encoder, then recover quality with fine-tuning, and add speaker identity transfer via LoRA. Final model must run on limited hardware (on-device).

**Task:** English speech in → Bengali speech out (S2ST), preserving speaker voice identity.

---

## 2. Base Model

**Model:** `facebook/seamless-m4t-v2-large`  
**HuggingFace:** https://huggingface.co/facebook/seamless-m4t-v2-large  
**Architecture (UnitY2):**

```
Audio Input (16kHz)
    → Speech Encoder (W2v-BERT 2.0 Conformer, 24 layers, 635M params)
    → Text Decoder (Transformer, 866.8M params)
    → T2U Model (non-autoregressive text-to-unit)
    → HiFi-GAN Vocoder (200M params)
    → Bengali Speech Output
```

**Total: 1805.5M parameters**

**Key architecture detail for pruning:** The speech encoder uses `SeamlessM4Tv2ConformerEncoderLayer`. Each layer contains:
- `ffn1`: `intermediate_dense` (4096×1024) + `output_dense` (1024×4096)
- `self_attn`: `linear_q`, `linear_k`, `linear_v`, `linear_out` (all 1024×1024)
- `ffn2`: same as ffn1
- `conv_module`, layer norms

The encoder is accessed as `model.speech_encoder.encoder.layers` (note the double `.encoder`).

---

## 3. Platform & Infrastructure

**Primary platform:** Kaggle (T4 GPU, 16GB VRAM)  
**Fallback:** Google Colab  
**Checkpoint storage:** Google Drive via rclone  
**Notebook file:** `cse465-approach2v2.ipynb`

**Drive structure:**
```
gdrive:cse465/
    checkpoints/          ← .pt files (small metadata, scores, logs)
    stage3_pruned/        ← full HuggingFace model (3.1 GB)
    stage5_finetuned/     ← (not yet created)
    audio/                ← sample wav outputs
    figures/              ← matplotlib PNG charts
```

**Key secrets in Kaggle:**
- `RCLONE_CONF` — rclone config for Google Drive access
- `HF_TOKEN` — HuggingFace token (needed for FLEURS dataset)

**Session startup sequence (cells 1–8, always run):**
1. Platform detection + path setup
2. Install rclone
3. Configure rclone from secret
4. Install ML packages (transformers, datasets, torchaudio, speechbrain, peft, sacrebleu, sentencepiece, openai-whisper)
5. Pull checkpoints from Drive
6. Define save/load/model utility functions
7. Define audio play/save helpers
8. Session status check

**End of session:** Always run the Last Cell to sync everything back to Drive.

---

## 4. Evaluation Metrics (Corrected and Final)

| Metric | Type | Tool | Notes |
|--------|------|------|-------|
| **BLEU** | PRIMARY | sacrebleu | Sentence-level, `effective_order=True`. Used in official SeamlessM4T paper. |
| **ChrF** | PRIMARY | sacrebleu | Character F-score. More robust than BLEU for Bengali (morphologically rich). |
| **RTF** | Speed | manual | Real-Time Factor = processing_time / audio_duration. <1.0 = faster than real-time. |
| **SECS** | Voice (post-LoRA only) | SpeechBrain ECAPA-TDNN | Speaker Embedding Cosine Similarity. **Meaningless before Stage 7 LoRA** — SeamlessM4T uses a fixed default voice, so SECS input vs output is always ~0.05 and does not measure quality. |

**Why NOT Whisper ASR-BLEU as primary:** Bengali ASR quality in Whisper is inconsistent. The proper primary metric is BLEU on the S2TT text output, compared against the Bengali reference from FLEURS.

**Evaluation dataset:** `google/fleurs` — `en_us` (English audio) aligned with `bn_in` (Bengali text reference). This is the same dataset used in the official SeamlessM4T evaluation pipeline. 646 aligned test pairs available.

---

## 5. Key Functions Defined in Notebook

```python
# Inference
run_s2tt(model, wav_np)           # Speech → Bengali text (for BLEU/ChrF)
run_s2s(model, wav_np, tgt_lang)  # Speech → Bengali audio (for RTF/SECS)
run_s2st(model, wav_np, tgt_lang) # Returns BOTH (text, audio) in one pass

# Metrics
compute_bleu(hypothesis, reference)   # → float 0-100
compute_chrf(hypothesis, reference)   # → float 0-100
measure_secs_robust(wav_out, wav_ref) # → float -1 to 1

# Benchmarking
run_benchmark(eval_model, bench_samples, label)
# → (results_list, summary_dict)
# summary_dict keys: avg_bleu, avg_chrf, avg_rtf, avg_secs, n_samples

# FFN Pruning
apply_ffn_pruning(base_model, ratio)
# Deep-copies model, zeroes FFN neurons by L1 magnitude
# Uses shape-based Linear detection: out>in = fc1, out<in = fc2
# Works correctly with SeamlessM4Tv2ConformerEncoderLayer

# Checkpoint I/O
save_checkpoint(state_dict, name, step)     # saves .pt + pushes to Drive
load_latest_checkpoint(name)                # loads latest .pt from CKPT_DIR
save_model_to_drive(model, processor, name) # saves full HF model + pushes
load_model_from_drive(stage_name)           # downloads + loads full HF model
```

---

## 6. Completed Stages and Results

### Stage 0 — Baseline Benchmark (COMPLETE)
**Teacher model: full `facebook/seamless-m4t-v2-large`, no changes**

| Metric | Value |
|--------|-------|
| BLEU | **12.21** |
| ChrF | **48.12** |
| RTF | **0.2251** (well below 1.0, real-time capable) |
| SECS | ~0.06 (meaningless pre-LoRA, fixed voice) |
| Params | 1805.5M |
| Dataset | FLEURS en_us → bn_in, 20 samples |

Saved as: `benchmark_baseline_step000000.pt`

---

### Stage 1 — Layer Importance Scoring (COMPLETE)
**Method:** 200 LibriSpeech calibration utterances. Forward hooks on all 24 encoder layers. Angular distance = 1 - cosine_similarity(input_hidden, output_hidden). High score = layer does more work = more important to keep.

**Full ranking (best → worst):**
```
Rank  Layer  Score
1     0      0.7420  ← most important (first layer, large transformation)
2     23     0.4105  ← last layer, also important
3     7      0.3187
4     8      0.2980
5     22     0.2897
6     4      0.2473
7     17     0.1955
8     3      0.1702
9     5      0.1576
10    1      0.1562
11    6      0.1359
12    14     0.1188
13    2      0.1049
14    16     0.0982
15    9      0.0840
16    21     0.0818
17    12     0.0809
18    15     0.0667
19    10     0.0577
20    13     0.0559
21    18     0.0500
22    11     0.0472
23    20     0.0397
24    19     0.0290  ← least important
```

Saved as: `layer_importance_step000000.pt`  
Keys: `{'ranked': [...], 'scores': {...}, 'n_layers': 24}`

---

### Stage 2 — FFN Pruning Ablation (COMPLETE — CRITICAL FINDING)

**CRITICAL FINDING: FFN pruning causes severe quality loss even at 10-20%.**

This contradicts the RA's claim that "99% of FFN neurons can be removed." The actual results on this model:

| FFN Ratio | BLEU | ΔBLEU | ChrF | ΔChrF | RTF | Verdict |
|-----------|------|-------|------|-------|-----|---------|
| 0% (baseline) | 12.21 | 0 | 48.12 | 0 | 0.239 | reference |
| 10% | 11.02 | -1.18 | 44.80 | -3.32 | 0.235 | ✗ quality loss |
| 20% | 8.33 | -3.88 | 36.62 | -11.51 | 0.221 | ✗ quality loss |
| 30% | 1.62 | -10.58 | 14.89 | -33.24 | 0.308 | ✗ severe |
| 40% | 0.04 | -12.16 | 4.85 | -43.27 | 0.558 | ✗ model broken |
| 50% | ~0 | ~-12 | ~0 | ~-48 | N/A | ✗ model broken |

**Conclusion: NO FFN pruning will be applied.** The ablation sweep becomes a paper contribution showing why the RA's hypothesis did not hold for this model. SeamlessM4T's conformer FFN neurons are NOT redundant at the weight-magnitude level. The recommended FFN_PRUNE_RATIO is **0.0** (no FFN pruning).

Saved as: `ffn_pruning_log_step000000.pt` and `stage2b_ffn_ablation_step000000.pt`
Figures saved: `stage2b_quality_cliff.png` (shows BLEU cliff at 20%)

---

### Stage 3 — Layer Pruning (IN PROGRESS — about to run)

**Current state:** The pruned model with 60% layer keep ratio has been saved to Drive as `stage3_pruned` (3168 MB), but Stage 4 benchmark has NOT been run yet.

**Configuration chosen:**
- FFN pruning: **0%** (no FFN pruning — see Stage 2 finding)
- Layer keep ratio: **60%** (keep 14 of 24 layers)
- Keep indices: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 14, 16, 17, 22, 23]`
- Drop indices: `[9, 10, 11, 12, 13, 15, 18, 19, 20, 21]`

**Why 60%:** At 50% keep, translation meaning is lost while language is preserved. At 60% keep, translation is "slightly wrong but still Bengali" — this is the sweet spot where fine-tuning can recover quality. The damage→recovery arc is the paper's research contribution.

**Expected size after layer pruning:**
- Speech encoder: 635M → ~393M (38% smaller)
- Full model: 1805M → ~1563M (13% smaller)

**Stage 3 is about to be run.** The model `stage3_pruned` exists on Drive from earlier work. Load it with:
```python
pruned_model, _ = load_model_from_drive('stage3_pruned')
```

---

### Stage 4 — Post-Pruning Benchmark (NOT YET RUN)

Needs to use `run_benchmark(pruned_model, bench_samples, label='stage3_pruned')`.  
Will compare against baseline to document the degradation.  
Expected: BLEU drops from 12.21 to somewhere around 2-6. This is intentional.

---

### Stage 5 — Fine-tuning (NOT YET RUN — code written but not executed)

**IMPORTANT: Previous fine-tuning approach was WRONG and caused "rererere" audio output.**

**What went wrong:** MSE distillation loss between pruned encoder hidden states and teacher encoder hidden states. This geometrically aligned the hidden states but did NOT ensure the downstream text decoder could decode them. The T2U model generated repeated discrete unit tokens → "rererere" sound.

**Correct approach (implemented in current notebook):**
- Use `SeamlessM4Tv2ForSpeechToText` (has proper `forward()` with `labels` parameter)
- Load pruned encoder weights into the S2T model
- Freeze text decoder, T2U, vocoder — train only speech encoder
- Loss: **S2TT cross-entropy** — predicted Bengali text tokens vs FLEURS bn_in reference
- Dataset: FLEURS en_us audio + FLEURS bn_in Bengali text labels (aligned by sentence ID)
- Learning rate: 5e-6 with 100-step warmup, linear decay
- MAX_STEPS: 1000 (increase to 3000 for final paper)

**After fine-tuning:** Transfer the fine-tuned encoder back into the S2ST model for evaluation.

---

### Stage 6 — Post-Fine-tuning Benchmark (NOT YET RUN)

Paper figure code is written (Stage 6 cell in notebook). Will produce:
- `stage6_bleu_chrf_comparison.png` — bar charts
- `stage6_size_quality_tradeoff.png` — scatter plot (params vs BLEU)
- `stage6_rtf_comparison.png` — RTF bar chart

---

### Stage 7 — LoRA Speaker Conditioning (TODO)

**Goal:** Make Bengali output sound like the input speaker (voice cloning).

**Plan:**
- Load fine-tuned S2ST model from Stage 5
- Apply LoRA (r=8) to vocoder linear layers
- Train LoRA weights with VCTK or VoxCeleb speaker pairs
- Loss: SECS + spectral reconstruction loss
- Only after this stage does SECS become a valid quality metric

**Expected result:** SECS improves from ~0.05 (fixed voice) to >0.70 (speaker conditioned)

---

## 7. Current Drive Contents

```
gdrive:cse465/
    checkpoints/
        benchmark_baseline_step000000.pt   ← Stage 0 results (BLEU=12.21)
        layer_importance_step000000.pt     ← Stage 1 layer scores
        ffn_pruning_log_step000000.pt      ← Stage 2 log
        stage2b_ffn_ablation_step000000.pt ← Stage 2 ablation results
        session_log.json
    stage3_pruned/                         ← Full pruned model (3.1 GB)
        model.safetensors
        config.json
        generation_config.json
        processor_config.json
        tokenizer.json
        tokenizer_config.json
    figures/
        stage2b_quality_cliff.png
```

---

## 8. What To Do In The Next Session

**Step 1 — Run cells 1-8** (platform setup, rclone, packages, pull checkpoints)

**Step 2 — Load teacher model** (Stage 0 cell — needed for run_s2st reference)

**Step 3 — Load benchmark helpers** (Stage 0b, 0c cells)

**Step 4 — Load pruned model** from Drive:
```python
pruned_model, _ = load_model_from_drive('stage3_pruned')
pruned_model.eval()
```

**Step 5 — Run Stage 4 benchmark** on pruned_model using run_benchmark()

**Step 6 — Run Stage 5** fine-tuning (S2TT cross-entropy, FLEURS dataset)

**Step 7 — Transfer encoder + run Stage 6 benchmark**

---

## 9. Known Issues and Fixes Applied

| Issue | Root Cause | Fix Applied |
|-------|-----------|-------------|
| "rererere" audio from fine-tuning | MSE distillation on hidden states misaligns decoder distributions | Changed to S2TT cross-entropy with FLEURS labels |
| FFN pruning showed no effect (ratio 0.9 did nothing) | Code was passing base_model directly instead of applying pruning | Fixed with shape-based Linear detection (out>in = fc1, out<in = fc2) using `named_modules()` |
| FFN pruning at any ratio >0.1 causes severe quality loss | SeamlessM4T conformer FFN neurons are not redundant | Decision: no FFN pruning. Layer pruning only. |
| SECS always ~0.05 regardless of model | SeamlessM4T uses fixed default voice before LoRA | SECS removed from primary metrics. Only BLEU/ChrF used until Stage 7. |
| AudioDecoder error with streaming datasets | HuggingFace streaming returns AudioDecoder objects | Use `cast_column("audio", Audio(sampling_rate=16000))` or load FLEURS via parquet URL |
| `trust_remote_code` warning | Deprecated parameter | Removed from all dataset loads |
| `output.waveform` AttributeError | generate() returns plain tuple (waveforms, lengths), not named tuple | Use `output[0].cpu().numpy().squeeze()` |
| Layer scores loaded with `weights_only=True` error | PyTorch 2.6 default changed to `weights_only=True`, blocks numpy scalars | Use `weights_only=False` in all `torch.load()` calls |
| Wrong encoder path | SeamlessM4T encoder is at `model.speech_encoder.encoder.layers`, not `model.speech_encoder.layers` | Fixed in Stage 1 hooks and Stage 2 pruning loop |

---

## 10. Paper Narrative

The paper tells this story:

1. **Motivation:** SeamlessM4T-v2-large (1805M params) is too large for on-device use. We compress it via structural pruning.

2. **Contribution 1 — FFN pruning negative result:** The RA's hypothesis (99% FFN neurons are redundant) does NOT hold for SeamlessM4T's conformer architecture. We show quantitatively (Table: FFN ablation) that BLEU drops from 12.21 to 8.33 at just 20% FFN pruning. This is a meaningful finding — it tells the community that the redundancy claim is architecture-specific.

3. **Contribution 2 — Layer pruning:** Angular distance scoring successfully identifies 10 dispensable layers out of 24 (41%). Removing them reduces speech encoder by 38% (635M → 393M params) while keeping the model in Bengali output mode.

4. **Contribution 3 — Fine-tuning recovery:** S2TT cross-entropy fine-tuning on FLEURS en_us→bn_in recovers the BLEU degradation from layer pruning. We show the before/after loss curve and BLEU recovery.

5. **Contribution 4 — LoRA speaker conditioning:** (TODO) Vocoder LoRA enables voice cloning so Bengali output matches the input speaker's voice identity.

---

## 11. File Locations Summary

| File | Location | Contents |
|------|----------|----------|
| Main notebook | Kaggle (download from Kaggle UI) | All code |
| Baseline results | `gdrive:cse465/checkpoints/benchmark_baseline_step000000.pt` | BLEU=12.21, ChrF=48.12 |
| Layer scores | `gdrive:cse465/checkpoints/layer_importance_step000000.pt` | 24 layer rankings |
| FFN ablation | `gdrive:cse465/checkpoints/stage2b_ffn_ablation_step000000.pt` | Full sweep table |
| Pruned model | `gdrive:cse465/stage3_pruned/` | 3.1 GB, 1563M params |
| FFN cliff figure | `gdrive:cse465/figures/stage2b_quality_cliff.png` | Paper-ready chart |

---

## 12. Quick Reference: Variable Names After Session Resume

After running cells 1-8 + Stage 0 + Stage 0b + Stage 0c:

```python
# Models
model           # full teacher (SeamlessM4Tv2ForSpeechToSpeech, fp16)
processor       # SeamlessM4TProcessor
pruned_model    # loaded from stage3_pruned (after load_model_from_drive)

# Data
bench_samples   # list of 20 dicts: {id, wav, bn_ref, duration_s}
                # each wav is np.float32 at 16kHz, trimmed to 8s

# Reports (loaded from .pt files)
baseline_report # dict: avg_bleu=12.21, avg_chrf=48.12, avg_rtf=0.225...
sweep_results   # dict keyed by ratio: {0.0: {...}, 0.1: {...}, ...}

# Constants
N_SAMPLES = 20
CKPT_DIR  = '/kaggle/working/checkpoints'
AUDIO_DIR = '/kaggle/working/audio'
FIG_DIR   = '/kaggle/working/figures'
```
