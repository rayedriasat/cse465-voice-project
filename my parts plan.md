Let me read your current notebook first.Good — your setup cells (1–8 + last cell) are all working correctly. Drive is connected, rclone works, checkpoints sync. Now I'll give you the complete project cells to add after Cell 8.

Here's the full plan before the code:

```
STAGE 0  — Baseline benchmark    (teacher model, no changes)
STAGE 1  — Load & inspect model
STAGE 2  — FFN pruning
STAGE 3  — Layer pruning
STAGE 4  — Benchmark after pruning  (compare vs baseline)
STAGE 5  — Fine-tuning
STAGE 6  — Benchmark after fine-tuning
STAGE 7  — LoRA speaker conditioning
STAGE 8  — Final benchmark + report table
```

Every stage saves results to Drive. The benchmark cells produce the tables you need for your paper.

---

```python
# ╔══════════════════════════════════════════════════════════╗
# ║  STAGE 0 — Load teacher model + run baseline benchmark  ║
# ╚══════════════════════════════════════════════════════════╝
# Run this ONCE. Saves baseline numbers to compare everything against.
# Takes ~10 min to load + ~20 min to benchmark.

import torch
import torchaudio
import torch.nn.functional as F
import numpy as np
from transformers import SeamlessM4Tv2ForSpeechToSpeech, SeamlessM4TProcessor
from datasets import load_dataset

print("Loading processor...")
processor = SeamlessM4TProcessor.from_pretrained("facebook/seamless-m4t-v2-large")

print("Loading model (4.5 GB — takes 5-10 min)...")
model = SeamlessM4Tv2ForSpeechToSpeech.from_pretrained(
    "facebook/seamless-m4t-v2-large",
    torch_dtype=torch.float16,
    device_map="auto",
)
model.eval()

# ── Parameter count per component ────────────────────────────────────────────
def count_params(module):
    return sum(p.numel() for p in module.parameters()) / 1e6

print("\n── Model size breakdown ──────────────────────")
components = {
    'speech_encoder' : model.speech_encoder,
    'text_decoder'   : model.text_decoder,
    'vocoder'        : model.vocoder,
}
total = 0
for name, mod in components.items():
    p = count_params(mod)
    total += p
    print(f"  {name:<25} {p:>8.1f} M params")
print(f"  {'TOTAL':<25} {total:>8.1f} M params")
print(f"  {'(other)':<25} {count_params(model) - total:>8.1f} M params")
print(f"  {'FULL MODEL':<25} {count_params(model):>8.1f} M params")
```

---

```python
# ╔══════════════════════════════════════════════════════════╗
# ║  STAGE 0b — Benchmark helper functions                  ║
# ╚══════════════════════════════════════════════════════════╝
# Define these once. Reuse at every stage.

import time, jiwer, whisper as whisper_lib
from speechbrain.pretrained import EncoderClassifier

print("Loading Whisper for WER measurement...")
whisper_model = whisper_lib.load_model("base")   # 'base' is fast; use 'large' for final paper numbers

print("Loading speaker encoder for SECS measurement...")
spk_encoder = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    run_opts={"device": "cpu"},   # keep on CPU to save VRAM
)

def measure_wer(hypothesis: str, reference: str) -> float:
    """Word Error Rate between two strings. Lower is better."""
    return jiwer.wer(reference.lower(), hypothesis.lower())

def measure_secs(wav_out: np.ndarray, wav_ref: np.ndarray, sr: int = 16000) -> float:
    """
    Speaker Embedding Cosine Similarity.
    Compares voice identity of output vs reference. Higher is better (max 1.0).
    """
    def get_emb(wav):
        t = torch.tensor(wav, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            emb = spk_encoder.encode_batch(t)
        return F.normalize(emb.squeeze(), dim=-1)

    e1 = get_emb(wav_out)
    e2 = get_emb(wav_ref)
    return F.cosine_similarity(e1.unsqueeze(0), e2.unsqueeze(0)).item()

def measure_rtf(model_fn, inputs: dict) -> float:
    """
    Real-Time Factor = processing_time / audio_duration.
    RTF < 1.0 means faster than real-time. Lower is better.
    """
    # Warmup
    with torch.no_grad():
        model_fn(inputs)

    # Timed run
    start = time.time()
    with torch.no_grad():
        output = model_fn(inputs)
    elapsed = time.time() - start

    # Get output duration
    wav = output[0].cpu().numpy().squeeze()
    duration = len(wav) / model.config.sampling_rate
    rtf = elapsed / max(duration, 0.001)
    return rtf, elapsed, duration

def run_s2s(current_model, input_wav: np.ndarray, tgt_lang: str = "ben") -> np.ndarray:
    """Run one S2S inference. Returns output waveform as numpy array."""
    inputs = processor(audios=input_wav, sampling_rate=16000, return_tensors="pt")
    inputs = {k: v.to(current_model.device) for k, v in inputs.items()}
    with torch.no_grad():
        output = current_model.generate(**inputs, tgt_lang=tgt_lang)
    return output[0].cpu().numpy().squeeze()

print("Benchmark functions ready:")
print("  measure_wer(hypothesis, reference)")
print("  measure_secs(wav_out, wav_ref)")
print("  measure_rtf(model_fn, inputs)")
print("  run_s2s(model, input_wav, tgt_lang)")
```

---

```python
# ╔══════════════════════════════════════════════════════════╗
# ║  STAGE 0c — Load benchmark dataset (10 samples)         ║
# ╚══════════════════════════════════════════════════════════╝
# We use 10 samples for quick benchmarking during development.
# For the final paper numbers, change N_SAMPLES to 100.

N_SAMPLES = 10   # ← change to 100 for final paper run

print(f"Loading {N_SAMPLES} samples from LibriSpeech test-clean...")
ds = load_dataset("librispeech_asr", "clean", split="test", streaming=True)

bench_samples = []
for i, ex in enumerate(ds):
    if i >= N_SAMPLES:
        break
    wav = np.array(ex['audio']['array'], dtype=np.float32)
    sr  = ex['audio']['sampling_rate']
    if sr != 16000:
        wav = torchaudio.functional.resample(
            torch.tensor(wav), sr, 16000).numpy()
    # Trim to 8 seconds max
    wav = wav[:16000 * 8]
    bench_samples.append({
        'id'   : ex['id'],
        'wav'  : wav,
        'text' : ex['text'].lower(),
    })
    print(f"  [{i+1}/{N_SAMPLES}] {ex['id']}  {len(wav)/16000:.1f}s  \"{ex['text'][:50]}\"")

print(f"\nLoaded {len(bench_samples)} benchmark samples.")
```

---

```python
# ╔══════════════════════════════════════════════════════════╗
# ║  STAGE 0d — Run baseline benchmark (teacher model)      ║
# ╚══════════════════════════════════════════════════════════╝
# This is your "100% quality" reference point.
# Everything you do in pruning is measured against this.

print("Running baseline benchmark on teacher model...")
print("(This takes ~2 min for 10 samples)\n")

baseline_results = []

for i, sample in enumerate(bench_samples):
    print(f"  Sample {i+1}/{len(bench_samples)}: {sample['id']}")

    # Run S2S (English → Bengali)
    t0 = time.time()
    out_wav = run_s2s(model, sample['wav'], tgt_lang="ben")
    elapsed = time.time() - t0

    # RTF
    audio_dur = len(sample['wav']) / 16000
    rtf = elapsed / audio_dur

    # WER — transcribe the OUTPUT back to text using Whisper
    # (measures if the translated speech is intelligible)
    transcription = whisper_model.transcribe(
        out_wav.astype(np.float32),
        language="bn"   # Bengali
    )['text']

    # SECS — compare speaker identity of output vs input
    secs = measure_secs(out_wav, sample['wav'])

    result = {
        'id'            : sample['id'],
        'rtf'           : round(rtf, 4),
        'secs'          : round(secs, 4),
        'transcription' : transcription,
        'ref_text'      : sample['text'],
    }
    baseline_results.append(result)
    print(f"    RTF={rtf:.3f}  SECS={secs:.3f}  transcription: {transcription[:60]}")

# Summary statistics
avg_rtf  = np.mean([r['rtf']  for r in baseline_results])
avg_secs = np.mean([r['secs'] for r in baseline_results])

print(f"\n── Baseline Summary ──────────────────────────")
print(f"  Samples : {len(baseline_results)}")
print(f"  Avg RTF : {avg_rtf:.4f}  (lower = faster)")
print(f"  Avg SECS: {avg_secs:.4f}  (higher = more speaker similarity)")
print(f"  Model size: {count_params(model):.1f}M params")

# Save baseline results
baseline_report = {
    'stage'         : 'baseline_teacher',
    'n_samples'     : len(baseline_results),
    'avg_rtf'       : avg_rtf,
    'avg_secs'      : avg_secs,
    'model_params_M': count_params(model),
    'samples'       : baseline_results,
}
save_checkpoint(baseline_report, name='benchmark_baseline', step=0)
print("\nBaseline saved.")
```

---

```python
# ╔══════════════════════════════════════════════════════════╗
# ║  STAGE 1 — Layer importance scoring                     ║
# ╚══════════════════════════════════════════════════════════╝
# Score every encoder layer. Expensive (~20 min). Save results.
# You only need to run this once ever.

print("Loading calibration data (200 samples)...")
calib_ds = load_dataset("librispeech_asr", "clean", split="test", streaming=True)
calib_samples = []
for i, ex in enumerate(calib_ds):
    if i >= 200: break
    wav = np.array(ex['audio']['array'], dtype=np.float32)[:16000*6]
    calib_samples.append(wav)
print(f"Loaded {len(calib_samples)} calibration samples.")

# ── Register hooks ────────────────────────────────────────────────────────────
layer_inputs  = {}
layer_outputs = {}
hooks = []

for i, layer in enumerate(model.speech_encoder.layers):
    def make_hook(idx):
        def hook(module, inp, out):
            layer_inputs[idx]  = inp[0].detach().float().cpu()
            layer_outputs[idx] = (out[0] if isinstance(out, tuple) else out).detach().float().cpu()
        return hook
    hooks.append(layer.register_forward_hook(make_hook(i)))

# ── Run calibration ───────────────────────────────────────────────────────────
scores_acc = {i: [] for i in range(len(model.speech_encoder.layers))}

for idx, wav in enumerate(calib_samples):
    if idx % 25 == 0:
        print(f"  Calibrating {idx}/200...")
    try:
        inputs = processor(audios=wav, sampling_rate=16000, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            model.speech_encoder(
                input_features=inputs.get('input_features'),
                attention_mask=inputs.get('attention_mask'),
            )
        for i in layer_inputs:
            if i in layer_outputs:
                x = F.normalize(layer_inputs[i].float().mean(1), dim=-1)
                y = F.normalize(layer_outputs[i].float().mean(1), dim=-1)
                cos = (x * y).sum(-1).clamp(-1, 1).mean().item()
                scores_acc[i].append(1 - cos)   # angular distance: 0=useless, 2=very important
    except Exception as e:
        print(f"  Skipped {idx}: {e}")

# Remove hooks
for h in hooks:
    h.remove()

# Average and rank
final_scores = {i: np.mean(v) for i, v in scores_acc.items() if v}
ranked = sorted(final_scores.items(), key=lambda x: -x[1])

print("\n── Layer importance ranking ──────────────────")
print(f"  {'Rank':<6} {'Layer':<8} {'Score':<10} {'Bar'}")
for rank, (layer_idx, score) in enumerate(ranked):
    bar = '█' * int(score * 30)
    print(f"  {rank+1:<6} {layer_idx:<8} {score:<10.4f} {bar}")

# Save
save_checkpoint(
    {'layer_scores': final_scores, 'ranked': ranked, 'n_layers': len(ranked)},
    name='layer_importance',
    step=0
)
print("\nLayer scores saved.")
```

---

```python
# ╔══════════════════════════════════════════════════════════╗
# ║  STAGE 2 — FFN neuron pruning                           ║
# ╚══════════════════════════════════════════════════════════╝
# Zeros out low-magnitude FFN neurons in the speech encoder.
# Start at 70%, push to 90% if quality holds after fine-tuning.

import copy

FFN_PRUNE_RATIO = 0.70   # ← try 0.70, 0.80, 0.90 in separate runs

print(f"FFN pruning at {FFN_PRUNE_RATIO*100:.0f}% ratio...")

pruned_model = copy.deepcopy(model)
pruned_model.eval()

ffn_stats = []

for layer_idx, layer in enumerate(pruned_model.speech_encoder.layers):
    # SeamlessM4T uses different FFN attribute names — handle both
    ffn = None
    if hasattr(layer, 'feed_forward'):
        ffn = layer.feed_forward
    elif hasattr(layer, 'ffn'):
        ffn = layer.ffn

    if ffn is None:
        continue

    # Find the first linear projection in the FFN
    fc1 = None
    for attr in ['fc1', 'intermediate_dense', 'linear1', 'dense']:
        if hasattr(ffn, attr):
            fc1 = getattr(ffn, attr)
            break
    fc2 = None
    for attr in ['fc2', 'output_dense', 'linear2', 'out_proj']:
        if hasattr(ffn, attr):
            fc2 = getattr(ffn, attr)
            break

    if fc1 is None or fc2 is None:
        continue

    weight = fc1.weight.data.float()   # [ffn_dim, hidden_dim]
    scores = weight.abs().mean(dim=1)  # [ffn_dim]
    threshold = torch.quantile(scores, FFN_PRUNE_RATIO)
    mask = scores > threshold

    fc1.weight.data[~mask] = 0
    if fc1.bias is not None:
        fc1.bias.data[~mask] = 0
    fc2.weight.data[:, ~mask] = 0

    kept = mask.sum().item()
    total = len(mask)
    ffn_stats.append({'layer': layer_idx, 'kept': kept, 'total': total,
                      'pruned_pct': (1 - kept/total)*100})

print(f"\n── FFN pruning results ───────────────────────")
for s in ffn_stats:
    print(f"  Layer {s['layer']:>2}: kept {s['kept']:>4}/{s['total']} neurons "
          f"({s['pruned_pct']:.1f}% pruned)")

# Count effective parameters (zeros still take memory but not compute)
orig_p   = count_params(model.speech_encoder)
pruned_p = count_params(pruned_model.speech_encoder)
print(f"\n  Speech encoder params: {orig_p:.1f}M (unchanged — zeros still stored)")
print(f"  Note: sparse compute benefit requires sparse inference engine")
print(f"  To actually reduce size: apply layer pruning next (Stage 3)")

# Quick sanity check — run one sample
test_wav = bench_samples[0]['wav']
try:
    out = run_s2s(pruned_model, test_wav, 'ben')
    print(f"\n  Sanity check passed — output shape: {out.shape}")
    play(out, pruned_model.config.sampling_rate, "FFN-pruned output (Bengali)")
except Exception as e:
    print(f"\n  Sanity check failed: {e}")

save_checkpoint(
    {'ffn_prune_ratio': FFN_PRUNE_RATIO, 'ffn_stats': ffn_stats},
    name='ffn_pruning_log',
    step=0
)
```

---

```python
# ╔══════════════════════════════════════════════════════════╗
# ║  STAGE 3 — Layer pruning (remove weakest encoder layers) ║
# ╚══════════════════════════════════════════════════════════╝
# Load layer scores from Stage 1, remove bottom 40% of layers.

# Load scores saved in Stage 1
scores_state = load_latest_checkpoint('layer_importance')
ranked = scores_state['ranked']

LAYER_KEEP_RATIO = 0.60   # keep top 60% of layers (remove 40%)

n_layers = len(ranked)
n_keep   = int(n_layers * LAYER_KEEP_RATIO)
keep_indices = sorted([idx for idx, _ in ranked[:n_keep]])
drop_indices = [i for i in range(n_layers) if i not in keep_indices]

print(f"Original layers : {n_layers}")
print(f"Keeping {n_keep} layers : {keep_indices}")
print(f"Dropping {len(drop_indices)} layers: {drop_indices}")

# Apply on top of FFN-pruned model
pruned_model.speech_encoder.layers = torch.nn.ModuleList(
    [pruned_model.speech_encoder.layers[i] for i in keep_indices]
)

orig_enc_p   = count_params(model.speech_encoder)
pruned_enc_p = count_params(pruned_model.speech_encoder)
full_orig    = count_params(model)
full_pruned  = count_params(pruned_model)

print(f"\n── Size after layer pruning ──────────────────")
print(f"  Speech encoder: {orig_enc_p:.1f}M → {pruned_enc_p:.1f}M  "
      f"({(1-pruned_enc_p/orig_enc_p)*100:.1f}% reduction)")
print(f"  Full model    : {full_orig:.1f}M → {full_pruned:.1f}M  "
      f"({(1-full_pruned/full_orig)*100:.1f}% reduction)")

# Sanity check
test_wav = bench_samples[0]['wav']
try:
    out = run_s2s(pruned_model, test_wav, 'ben')
    print(f"\n  Sanity check passed — output shape: {out.shape}")
    play(out, pruned_model.config.sampling_rate, "After layer pruning (Bengali)")
except Exception as e:
    print(f"\n  Sanity check FAILED: {e}")
    print("  Try reducing LAYER_KEEP_RATIO to 0.70 if this keeps failing")

# Save the pruned model as a full HuggingFace model
save_model_to_drive(pruned_model, processor, 'stage3_pruned')
```

---

```python
# ╔══════════════════════════════════════════════════════════╗
# ║  STAGE 4 — Benchmark after pruning                      ║
# ╚══════════════════════════════════════════════════════════╝
# Measures quality BEFORE fine-tuning.
# This shows how much pruning degraded the model.

print("Benchmarking pruned model (before fine-tuning)...")

pruned_results = []

for i, sample in enumerate(bench_samples):
    print(f"  Sample {i+1}/{len(bench_samples)}: {sample['id']}")
    t0      = time.time()
    out_wav = run_s2s(pruned_model, sample['wav'], tgt_lang="ben")
    elapsed = time.time() - t0

    rtf  = elapsed / (len(sample['wav']) / 16000)
    secs = measure_secs(out_wav, sample['wav'])
    transcription = whisper_model.transcribe(
        out_wav.astype(np.float32), language="bn")['text']

    pruned_results.append({
        'id': sample['id'], 'rtf': round(rtf, 4),
        'secs': round(secs, 4), 'transcription': transcription,
    })
    print(f"    RTF={rtf:.3f}  SECS={secs:.3f}")

avg_rtf_p  = np.mean([r['rtf']  for r in pruned_results])
avg_secs_p = np.mean([r['secs'] for r in pruned_results])

# ── Comparison table ──────────────────────────────────────────────────────────
print("\n")
print("╔══════════════════════════════════════════════════════════════╗")
print("║  BENCHMARK TABLE — Pruning effect                           ║")
print("╠══════════════╦══════════════╦══════════════╦════════════════╣")
print("║  Stage       ║  Params (M)  ║  Avg RTF     ║  Avg SECS      ║")
print("╠══════════════╬══════════════╬══════════════╬════════════════╣")
print(f"║  Baseline    ║  {count_params(model):>10.1f}  ║  "
      f"{baseline_report['avg_rtf']:>10.4f}  ║  {baseline_report['avg_secs']:>12.4f}  ║")
print(f"║  Pruned      ║  {count_params(pruned_model):>10.1f}  ║  "
      f"{avg_rtf_p:>10.4f}  ║  {avg_secs_p:>12.4f}  ║")
print("╚══════════════╩══════════════╩══════════════╩════════════════╝")
print(f"\n  RTF  improvement : {baseline_report['avg_rtf']/avg_rtf_p:.2f}x faster")
print(f"  SECS degradation : {(baseline_report['avg_secs']-avg_secs_p)*100:.1f} points")
print(f"  Size reduction   : {(1-count_params(pruned_model)/count_params(model))*100:.1f}%")

pruned_report = {
    'stage': 'after_pruning',
    'ffn_prune_ratio'  : FFN_PRUNE_RATIO,
    'layer_keep_ratio' : LAYER_KEEP_RATIO,
    'layers_kept'      : keep_indices,
    'n_samples'        : len(pruned_results),
    'avg_rtf'          : avg_rtf_p,
    'avg_secs'         : avg_secs_p,
    'model_params_M'   : count_params(pruned_model),
    'samples'          : pruned_results,
}
save_checkpoint(pruned_report, name='benchmark_after_pruning', step=0)
print("\nPruning benchmark saved.")
```

---

```python
# ╔══════════════════════════════════════════════════════════╗
# ║  STAGE 5 — Fine-tuning the pruned model                 ║
# ╚══════════════════════════════════════════════════════════╝
# Fine-tune for a few steps to recover quality lost in pruning.
# Full fine-tuning takes hours — we use a small subset to start.
# Increase MAX_STEPS for a longer run in a dedicated session.

from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler

MAX_STEPS   = 500    # ← increase to 2000+ for real fine-tuning
SAVE_EVERY  = 100
LOG_EVERY   = 20
LR          = 1e-5
BATCH_SIZE  = 1      # T4 has 16GB — keep at 1 for safety

print("Loading fine-tuning data (MLS English)...")
ft_ds = load_dataset(
    "facebook/multilingual_librispeech", "english",
    split="train", streaming=True
)

# Resume from checkpoint if it exists
ft_state = load_latest_checkpoint('finetune')
start_step = ft_state['step'] if ft_state else 0

# Set up optimizer — only update speech encoder (frozen rest)
for param in pruned_model.parameters():
    param.requires_grad = False
for param in pruned_model.speech_encoder.parameters():
    param.requires_grad = True

optimizer = AdamW(
    [p for p in pruned_model.parameters() if p.requires_grad],
    lr=LR
)
if ft_state and 'optimizer_state' in ft_state:
    optimizer.load_state_dict(ft_state['optimizer_state'])

scaler = GradScaler()
pruned_model.train()

print(f"Starting fine-tuning from step {start_step}...")
print(f"Trainable params: {sum(p.numel() for p in pruned_model.parameters() if p.requires_grad)/1e6:.1f}M")

step       = start_step
loss_log   = []
data_iter  = iter(ft_ds)

while step < MAX_STEPS:
    try:
        batch = next(data_iter)
    except StopIteration:
        data_iter = iter(ft_ds)
        batch = next(data_iter)

    wav = np.array(batch['audio']['array'], dtype=np.float32)
    if len(wav) < 1600:   # skip very short clips
        continue
    wav = wav[:16000 * 10]   # cap at 10s

    inputs = processor(audios=wav, sampling_rate=16000, return_tensors="pt")
    inputs = {k: v.to(pruned_model.device) for k, v in inputs.items()}
    labels = processor(
        text=batch['transcript'],
        return_tensors="pt"
    ).input_ids.to(pruned_model.device)

    try:
        optimizer.zero_grad()
        with autocast():
            out = pruned_model(**inputs, labels=labels)
            loss = out.loss

        if torch.isnan(loss):
            print(f"  Step {step}: NaN loss — skipping")
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(pruned_model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        loss_log.append(loss.item())
        step += 1

        if step % LOG_EVERY == 0:
            avg_loss = np.mean(loss_log[-LOG_EVERY:])
            print(f"  Step {step:>5}/{MAX_STEPS}  loss={avg_loss:.4f}")

        if step % SAVE_EVERY == 0:
            save_checkpoint({
                'step'            : step,
                'loss'            : loss.item(),
                'avg_loss'        : np.mean(loss_log[-SAVE_EVERY:]),
                'optimizer_state' : optimizer.state_dict(),
                # Don't save full model state here — too large
                # Use save_model_to_drive at the end instead
            }, name='finetune', step=step)

    except RuntimeError as e:
        if 'out of memory' in str(e):
            print(f"  OOM at step {step} — clearing cache")
            torch.cuda.empty_cache()
            continue
        raise e

pruned_model.eval()
print(f"\nFine-tuning complete. Final avg loss: {np.mean(loss_log[-50:]):.4f}")
save_model_to_drive(pruned_model, processor, 'stage5_finetuned')
```

---

```python
# ╔══════════════════════════════════════════════════════════╗
# ║  STAGE 6 — Benchmark after fine-tuning                  ║
# ╚══════════════════════════════════════════════════════════╝

print("Benchmarking fine-tuned model...")
ft_results = []

for i, sample in enumerate(bench_samples):
    print(f"  Sample {i+1}/{len(bench_samples)}")
    t0      = time.time()
    out_wav = run_s2s(pruned_model, sample['wav'], tgt_lang="ben")
    elapsed = time.time() - t0

    rtf  = elapsed / (len(sample['wav']) / 16000)
    secs = measure_secs(out_wav, sample['wav'])

    ft_results.append({'id': sample['id'], 'rtf': round(rtf,4), 'secs': round(secs,4)})
    print(f"    RTF={rtf:.3f}  SECS={secs:.3f}")

avg_rtf_ft  = np.mean([r['rtf']  for r in ft_results])
avg_secs_ft = np.mean([r['secs'] for r in ft_results])

# ── Full comparison table so far ─────────────────────────────────────────────
print("\n")
print("╔══════════════════════════════════════════════════════════════════╗")
print("║  BENCHMARK TABLE — Running comparison                           ║")
print("╠═══════════════════╦════════════╦════════════╦═══════════════════╣")
print("║  Stage            ║  Params(M) ║  Avg RTF   ║  Avg SECS         ║")
print("╠═══════════════════╬════════════╬════════════╬═══════════════════╣")
print(f"║  Baseline         ║  {baseline_report['model_params_M']:>8.1f}  ║  "
      f"{baseline_report['avg_rtf']:>8.4f}  ║  {baseline_report['avg_secs']:>15.4f}  ║")
print(f"║  After pruning    ║  {pruned_report['model_params_M']:>8.1f}  ║  "
      f"{pruned_report['avg_rtf']:>8.4f}  ║  {pruned_report['avg_secs']:>15.4f}  ║")
print(f"║  After fine-tune  ║  {count_params(pruned_model):>8.1f}  ║  "
      f"{avg_rtf_ft:>8.4f}  ║  {avg_secs_ft:>15.4f}  ║")
print("╚═══════════════════╩════════════╩════════════╩═══════════════════╝")

ft_report = {
    'stage': 'after_finetuning', 'n_samples': len(ft_results),
    'avg_rtf': avg_rtf_ft, 'avg_secs': avg_secs_ft,
    'model_params_M': count_params(pruned_model), 'samples': ft_results,
}
save_checkpoint(ft_report, name='benchmark_after_finetune', step=0)
```

---

```python
# ╔══════════════════════════════════════════════════════════╗
# ║  STAGE 7 — LoRA speaker conditioning                    ║
# ╚══════════════════════════════════════════════════════════╝
# Add LoRA adapters to the vocoder so it can accept speaker embeddings.
# Train only LoRA weights (~15MB) — everything else stays frozen.

from peft import get_peft_model, LoraConfig

print("Adding LoRA adapters to vocoder...")

# Find which linear layer names exist in the vocoder
vocoder_linear_names = []
for name, module in pruned_model.vocoder.named_modules():
    if isinstance(module, torch.nn.Linear):
        vocoder_linear_names.append(name)
print(f"  Found {len(vocoder_linear_names)} linear layers in vocoder")
print(f"  First few: {vocoder_linear_names[:5]}")

# Target the attention projection layers
target_modules = [n.split('.')[-1] for n in vocoder_linear_names
                  if any(k in n for k in ['proj', 'query', 'key', 'value', 'dense'])]
target_modules = list(set(target_modules))   # unique names only
if not target_modules:
    target_modules = [vocoder_linear_names[0].split('.')[-1]]  # fallback: first layer
print(f"  LoRA target modules: {target_modules}")

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=target_modules,
    lora_dropout=0.05,
    bias="none",
)

pruned_model.vocoder = get_peft_model(pruned_model.vocoder, lora_config)
lora_params = sum(p.numel() for p in pruned_model.vocoder.parameters()
                  if p.requires_grad) / 1e6
print(f"  LoRA trainable params: {lora_params:.2f}M")

# Load VCTK for speaker conditioning training
print("\nLoading VCTK dataset for speaker training...")
vctk = load_dataset("speech-recognition-community-v2/vctk", split="train", streaming=True)

# Freeze everything except LoRA
for param in pruned_model.parameters():
    param.requires_grad = False
for param in pruned_model.vocoder.parameters():
    if hasattr(param, 'lora'):
        param.requires_grad = True
# Re-enable LoRA params specifically
for name, param in pruned_model.vocoder.named_parameters():
    if 'lora_' in name:
        param.requires_grad = True

LORA_STEPS  = 300
LORA_LR     = 5e-4
lora_optimizer = AdamW(
    [p for p in pruned_model.parameters() if p.requires_grad], lr=LORA_LR
)

print(f"Training LoRA for {LORA_STEPS} steps...")
pruned_model.train()
lora_loss_log = []
data_iter = iter(vctk)

for step in range(LORA_STEPS):
    try:
        batch = next(data_iter)
    except StopIteration:
        data_iter = iter(vctk)
        batch = next(data_iter)

    try:
        wav = np.array(batch['audio']['array'], dtype=np.float32)
        if len(wav) < 1600: continue
        wav = wav[:16000 * 6]

        inputs = processor(audios=wav, sampling_rate=16000, return_tensors="pt")
        inputs = {k: v.to(pruned_model.device) for k, v in inputs.items()}
        labels = processor(text=batch.get('text', batch.get('sentence', '')),
                          return_tensors="pt").input_ids.to(pruned_model.device)

        lora_optimizer.zero_grad()
        with autocast():
            out = pruned_model(**inputs, labels=labels)
            loss = out.loss

        if torch.isnan(loss): continue

        scaler.scale(loss).backward()
        scaler.step(lora_optimizer)
        scaler.update()

        lora_loss_log.append(loss.item())

        if step % 50 == 0:
            print(f"  LoRA step {step:>4}/{LORA_STEPS}  loss={np.mean(lora_loss_log[-50:]):.4f}")

        if step % 100 == 0 and step > 0:
            # Save only the LoRA adapter weights (tiny)
            lora_weights = {k: v.cpu() for k, v in pruned_model.named_parameters()
                           if 'lora_' in k}
            save_checkpoint({'step': step, 'lora_weights': lora_weights,
                           'loss': loss.item()}, name='lora_adapter', step=step)
    except RuntimeError as e:
        if 'out of memory' in str(e):
            torch.cuda.empty_cache()
            continue
        raise e

pruned_model.eval()
print("LoRA training complete.")
save_model_to_drive(pruned_model, processor, 'stage7_lora')
```

---

```python
# ╔══════════════════════════════════════════════════════════╗
# ║  STAGE 8 — Final benchmark + full paper report table    ║
# ╚══════════════════════════════════════════════════════════╝

print("Running final benchmark...")
final_results = []

for i, sample in enumerate(bench_samples):
    print(f"  Sample {i+1}/{len(bench_samples)}")
    t0      = time.time()
    out_wav = run_s2s(pruned_model, sample['wav'], tgt_lang="ben")
    elapsed = time.time() - t0

    rtf    = elapsed / (len(sample['wav']) / 16000)
    secs   = measure_secs(out_wav, sample['wav'])
    transc = whisper_model.transcribe(out_wav.astype(np.float32), language="bn")['text']

    final_results.append({
        'id': sample['id'], 'rtf': round(rtf,4),
        'secs': round(secs,4), 'transcription': transc,
    })

    # Save a sample audio for each
    save_audio(out_wav, pruned_model.config.sampling_rate,
               f"final_bengali_{i+1}.wav", f"Final output {i+1}")

avg_rtf_f  = np.mean([r['rtf']  for r in final_results])
avg_secs_f = np.mean([r['secs'] for r in final_results])

# ══════════════════════════════════════════════════════════════════
# FULL PAPER REPORT TABLE
# ══════════════════════════════════════════════════════════════════
print("\n\n")
print("╔══════════════════════════════════════════════════════════════════════════════════╗")
print("║              PAPER TABLE — Approach 2: Structural Pruning + LoRA               ║")
print("╠═══════════════════════╦════════════╦════════════╦════════════╦══════════════════╣")
print("║  Stage                ║  Params(M) ║  Size Δ    ║  Avg RTF   ║  Avg SECS        ║")
print("╠═══════════════════════╬════════════╬════════════╬════════════╬══════════════════╣")

stages = [
    ("1. Baseline (teacher)",  baseline_report['model_params_M'], baseline_report['avg_rtf'],  baseline_report['avg_secs']),
    ("2. FFN pruned",          pruned_report['model_params_M'],   pruned_report['avg_rtf'],     pruned_report['avg_secs']),
    ("3. + Layer pruned",      pruned_report['model_params_M'],   pruned_report['avg_rtf'],     pruned_report['avg_secs']),
    ("4. + Fine-tuned",        ft_report['model_params_M'],       ft_report['avg_rtf'],         ft_report['avg_secs']),
    ("5. + LoRA (final)",      count_params(pruned_model),        avg_rtf_f,                    avg_secs_f),
]

baseline_params = stages[0][1]
for name, params, rtf, secs in stages:
    reduction = (1 - params/baseline_params) * 100
    delta_str = f"-{reduction:.1f}%" if reduction > 0 else "baseline"
    print(f"║  {name:<21}  ║  {params:>8.1f}  ║  {delta_str:>8}  ║  {rtf:>8.4f}  ║  {secs:>14.4f}  ║")

print("╚═══════════════════════╩════════════╩════════════╩════════════╩══════════════════╝")

print(f"""
Key results:
  Parameter reduction : {(1 - count_params(pruned_model)/baseline_report['model_params_M'])*100:.1f}%
  Speed improvement   : {baseline_report['avg_rtf']/avg_rtf_f:.2f}x faster
  SECS retention      : {avg_secs_f/baseline_report['avg_secs']*100:.1f}% of baseline
  FFN prune ratio used: {FFN_PRUNE_RATIO*100:.0f}%
  Layers kept         : {len(keep_indices)}/{n_layers} ({LAYER_KEEP_RATIO*100:.0f}%)
""")

# Save complete report as JSON
full_report = {
    'approach'         : 'Structural Pruning + LoRA (Approach 2)',
    'ffn_prune_ratio'  : FFN_PRUNE_RATIO,
    'layer_keep_ratio' : LAYER_KEEP_RATIO,
    'layers_kept'      : keep_indices,
    'baseline'         : baseline_report,
    'after_pruning'    : pruned_report,
    'after_finetuning' : ft_report,
    'final'            : {
        'avg_rtf'        : avg_rtf_f,
        'avg_secs'       : avg_secs_f,
        'model_params_M' : count_params(pruned_model),
        'samples'        : final_results,
    }
}
report_path = f"{CKPT_DIR}/full_report.json"
with open(report_path, 'w') as f:
    json.dump(full_report, f, indent=2)
subprocess.run(f'rclone copy {report_path} gdrive:cse465/', shell=True)
print("Full report saved to Drive as full_report.json")
```

---

## How to run this across sessions

| Session | Cells to run | What you're doing |
|---|---|---|
| 1 | Setup 1–8, Stage 0, 0b, 0c, 0d, 1 | Baseline + layer scoring (save and stop) |
| 2 | Setup 1–8, Stage 2, 3, 4 | Pruning + first benchmark |
| 3 | Setup 1–8, Stage 5 | Fine-tuning (run as long as quota allows) |
| 4 | Setup 1–8, Stage 6, 7 | Fine-tune benchmark + LoRA |
| 5 | Setup 1–8, Stage 8 | Final benchmark + paper table |

At the start of sessions 2–5, `load_model_from_drive('stage3_pruned')` or `load_model_from_drive('stage5_finetuned')` brings back where you left off. Sessions 2 onwards skip Stage 0 and 1 entirely since those results are already saved as checkpoints.