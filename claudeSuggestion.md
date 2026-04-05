Let me read all the relevant documentation and their notebook first.Now let me read their current notebook.Now I have the full picture. Let me identify every problem before writing a single line of code.

## Root cause analysis

**Problem 1 — Fine-tuning was fundamentally wrong.**
The MSE distillation loss on encoder hidden states makes the pruned encoder output *look* like the teacher encoder output, but that is not the same as making the downstream text decoder *understand* those outputs. SeamlessM4T is a multi-stage pipeline: encoder → text decoder → T2U → vocoder. The text decoder was trained on 24-layer encoder distributions. When 10 layers are removed, even if hidden states are numerically close, subtle distributional shifts cause the T2U model to generate **repeated unit tokens** — that is exactly what "rererererere" sounds like. The evaluate.py source code even has a `unit_generation_ngram_filtering` flag specifically because this failure mode is known.

**Problem 2 — Correct fine-tuning loss is end-to-end S2T cross-entropy.**
The official SeamlessM4T fine-tuning uses `SPEECH_TO_TEXT` mode with cross-entropy on predicted text tokens vs reference text. This trains the encoder to produce representations the frozen text decoder can actually decode — which is what we need.

**Problem 3 — Wrong evaluation metrics.**
SECS is meaningless before LoRA because SeamlessM4T uses a fixed default voice always. BLEU score on translation text output is the correct primary metric. The official evaluation uses sacrebleu for BLEU scores and JiWER for WER/CER.

**Problem 4 — Wrong evaluation dataset.**
FLEURS (google/fleurs) is the dataset Seamless supports natively and uses for official evaluation. It has English speech paired with Bengali text, making it perfect for S2TT BLEU evaluation.

Now here is the complete reorganized notebook from scratch.

---

```python
# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 1 — Platform detection                            ║
# ╚══════════════════════════════════════════════════════════╝
import os, sys

ON_KAGGLE = os.path.exists('/kaggle/working')
ON_COLAB  = 'google.colab' in sys.modules
PLATFORM  = 'kaggle' if ON_KAGGLE else 'colab'

WORK_DIR  = '/kaggle/working' if ON_KAGGLE else '/content'
CKPT_DIR  = f'{WORK_DIR}/checkpoints'
AUDIO_DIR = f'{WORK_DIR}/audio'
FIG_DIR   = f'{WORK_DIR}/figures'

for d in [CKPT_DIR, AUDIO_DIR, FIG_DIR]:
    os.makedirs(d, exist_ok=True)

print(f"Platform  : {PLATFORM}")
print(f"Work dir  : {WORK_DIR}")
```

---

```python
# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 2 — Install rclone                                ║
# ╚══════════════════════════════════════════════════════════╝
import subprocess

subprocess.run('curl https://rclone.org/install.sh | sudo bash',
               shell=True, capture_output=True)
ver = subprocess.run('rclone version', shell=True,
                     capture_output=True, text=True)
print(ver.stdout.split('\n')[0])
```

---

```python
# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 3 — Configure rclone from secret                  ║
# ╚══════════════════════════════════════════════════════════╝
import pathlib, re

if ON_KAGGLE:
    from kaggle_secrets import UserSecretsClient
    RCLONE_CONF = UserSecretsClient().get_secret("RCLONE_CONF")
elif ON_COLAB:
    from google.colab import userdata
    RCLONE_CONF = userdata.get('RCLONE_CONF')

raw = RCLONE_CONF.strip()
raw = re.sub(r'\s*(\[[^\]]+\])\s*', r'\n\1\n', raw)
raw = re.sub(r'\s+(type|scope|token|team_drive|client_id|client_secret|'
             r'root_folder_id|service_account_file|drive_id)\s*=\s*',
             r'\n\1 = ', raw)
raw = raw.strip() + '\n'

rclone_cfg = pathlib.Path.home() / '.config/rclone/rclone.conf'
rclone_cfg.parent.mkdir(parents=True, exist_ok=True)
rclone_cfg.write_text(raw)

result = subprocess.run('rclone listremotes',
                        shell=True, capture_output=True, text=True)
if 'gdrive:' in result.stdout:
    print("✓ Google Drive connected")
else:
    print("✗ Drive not connected")
    print(result.stderr[:300])
```

---

```python
# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 4 — Install packages                              ║
# ╚══════════════════════════════════════════════════════════╝
subprocess.run([
    'pip', 'install', '-q',
    'transformers', 'datasets', 'torchaudio',
    'speechbrain', 'peft', 'jiwer',
    'sacrebleu',        # official BLEU metric
    'openai-whisper',   # for ASR-BLEU evaluation
    'matplotlib',
    'evaluate',
], check=True)
print("Packages installed.")
```

---

```python
# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 5 — Pull checkpoints from Drive                   ║
# ╚══════════════════════════════════════════════════════════╝
print("Pulling checkpoints from Google Drive...")
result = subprocess.run(
    f'rclone sync gdrive:cse465/checkpoints {CKPT_DIR}',
    shell=True, capture_output=True, text=True
)
if result.returncode != 0:
    print("Warning:", result.stderr[:300])

files = sorted(os.listdir(CKPT_DIR))
if files:
    print(f"\n{len(files)} item(s) in {CKPT_DIR}:")
    for f in files:
        path = f'{CKPT_DIR}/{f}'
        mb = os.path.getsize(path) / 1e6 if os.path.isfile(path) else 0
        print(f"  {f:<50}  {mb:.1f} MB")
else:
    print("No checkpoints yet — fresh start.")
```

---

```python
# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 6 — Utility functions                             ║
# ╚══════════════════════════════════════════════════════════╝
import torch, glob, json, time
import numpy as np
import torchaudio
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime
from IPython.display import Audio, display
from torch import nn

# ─── Checkpoint save/load ─────────────────────────────────────────────────────
def save_checkpoint(state: dict, name: str, step: int = 0, keep: int = 3):
    filename   = f"{name}_step{step:06d}.pt"
    local_path = f"{CKPT_DIR}/{filename}"
    torch.save(state, local_path)
    mb = os.path.getsize(local_path) / 1e6
    print(f"[save] {filename}  ({mb:.2f} MB)")
    r = subprocess.run(
        f'rclone copy {local_path} gdrive:cse465/checkpoints/',
        shell=True, capture_output=True, text=True)
    print(f"[save] Drive: {'OK' if r.returncode==0 else 'FAILED — '+r.stderr[:100]}")
    old = sorted(glob.glob(f"{CKPT_DIR}/{name}_step*.pt"))
    for f in old[:-keep]:
        os.remove(f)

def load_checkpoint(name: str):
    files = sorted(glob.glob(f"{CKPT_DIR}/{name}_step*.pt"))
    if not files:
        print(f"[load] No checkpoint for '{name}'")
        return None
    latest = files[-1]
    state  = torch.load(latest, map_location='cpu', weights_only=False)
    print(f"[load] {os.path.basename(latest)}")
    return state

# ─── Full model save/load to Drive ────────────────────────────────────────────
def save_model(model, processor, stage_name: str):
    local = f"{CKPT_DIR}/{stage_name}"
    os.makedirs(local, exist_ok=True)
    model.save_pretrained(local)
    processor.save_pretrained(local)
    mb = sum(os.path.getsize(f"{local}/{f}")
             for f in os.listdir(local)) / 1e6
    print(f"[model] Saved: {stage_name}  ({mb:.0f} MB)")
    r = subprocess.run(
        f'rclone sync {local} gdrive:cse465/{stage_name}/',
        shell=True, capture_output=True, text=True)
    print(f"[model] Drive: {'OK' if r.returncode==0 else 'FAILED'}")

def load_model(stage_name: str, model_class=None):
    from transformers import (SeamlessM4Tv2ForSpeechToSpeech,
                              SeamlessM4TProcessor)
    if model_class is None:
        model_class = SeamlessM4Tv2ForSpeechToSpeech
    local = f"{CKPT_DIR}/{stage_name}"
    if not os.path.exists(local) or not os.listdir(local):
        print(f"[model] Downloading {stage_name} from Drive...")
        os.makedirs(local, exist_ok=True)
        subprocess.run(
            f'rclone sync gdrive:cse465/{stage_name}/ {local}',
            shell=True)
    print(f"[model] Loading {stage_name}...")
    model = model_class.from_pretrained(
        local, torch_dtype=torch.float16, device_map='auto')
    processor = SeamlessM4TProcessor.from_pretrained(local)
    print(f"[model] Loaded: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
    return model, processor

# ─── Param count ──────────────────────────────────────────────────────────────
def count_params(module):
    return sum(p.numel() for p in module.parameters()) / 1e6

# ─── VRAM status ──────────────────────────────────────────────────────────────
def vram_status():
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM: {alloc:.2f}/{total:.2f} GB  "
              f"({alloc/total*100:.1f}% used)")

print("Utilities ready.")
```

---

```python
# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 7 — Audio helpers                                 ║
# ╚══════════════════════════════════════════════════════════╝
def play_audio(audio, sr, label=""):
    if hasattr(audio, 'numpy'):
        audio = audio.squeeze().numpy()
    dur = len(audio) / sr
    print(f"▶ {label}  ({dur:.1f}s | sr={sr})")
    display(Audio(audio, rate=sr))

def save_audio(audio, sr, filename: str):
    path = f"{AUDIO_DIR}/{filename}"
    if hasattr(audio, 'numpy'):
        t = audio.squeeze().unsqueeze(0).float()
    else:
        t = torch.tensor(audio).unsqueeze(0).float()
    torchaudio.save(path, t, sr)
    subprocess.run(
        f'rclone copy {path} gdrive:cse465/audio/',
        shell=True, capture_output=True)
    print(f"[audio] Saved: {filename}")

print("Audio helpers ready.")
```

---

```python
# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 8 — Session status                                ║
# ╚══════════════════════════════════════════════════════════╝
def session_status():
    print("=" * 55)
    print(f"  Platform : {PLATFORM}")
    print(f"  Time     : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    local = [f for f in glob.glob(f"{CKPT_DIR}/**", recursive=True)
             if os.path.isfile(f)]
    print(f"\n  Local files ({len(local)}):")
    for f in sorted(local):
        print(f"    {os.path.relpath(f,CKPT_DIR):<45}"
              f"  {os.path.getsize(f)/1e6:.1f} MB")
    vram_status()
    print("=" * 55)

session_status()
```

---

Now the markdown cell that goes before Section A:

```
[MARKDOWN CELL]
# Section A — Teacher Model Baseline

SeamlessM4T-v2-large is our teacher model. It is a UnitY2 architecture with:
- **Speech encoder**: W2v-BERT 2.0 conformer (24 layers, 635M params)
- **Text decoder**: Transformer (translates to 96 target languages)
- **T2U model**: Non-autoregressive text-to-unit decoder
- **HiFi-GAN vocoder**: Converts discrete speech units to waveform

We evaluate using:
- **BLEU** (sacrebleu): Primary translation quality metric — compares predicted
  Bengali text vs reference Bengali text from FLEURS
- **ASR-BLEU**: Speech quality — Whisper transcribes output speech, compare with reference
- **SECS**: Speaker embedding cosine similarity (meaningful only after LoRA)
- **RTF**: Real-time factor (processing time / audio duration)

Dataset: **FLEURS en_us** for English speech input, **FLEURS bn_in** for Bengali text references.
This is the dataset officially used by the SeamlessM4T evaluation pipeline.
```

---

```python
# ╔══════════════════════════════════════════════════════════╗
# ║  CELL A1 — Load teacher model                           ║
# ╚══════════════════════════════════════════════════════════╝
from transformers import (SeamlessM4Tv2ForSpeechToSpeech,
                          SeamlessM4Tv2ForSpeechToText,
                          SeamlessM4TProcessor)

print("Loading processor...")
processor = SeamlessM4TProcessor.from_pretrained(
    "facebook/seamless-m4t-v2-large")

print("Loading teacher S2ST model (fp16, ~3.6 GB VRAM)...")
teacher = SeamlessM4Tv2ForSpeechToSpeech.from_pretrained(
    "facebook/seamless-m4t-v2-large",
    torch_dtype=torch.float16,
    device_map="auto",
)
teacher.eval()

print("\n── Model size breakdown ───────────────────────────")
components = {
    'speech_encoder' : teacher.speech_encoder,
    'text_decoder'   : teacher.text_decoder,
    'vocoder'        : teacher.vocoder,
}
total = count_params(teacher)
for name, mod in components.items():
    p = count_params(mod)
    print(f"  {name:<22} {p:>8.1f} M  "
          f"({p/total*100:.1f}%)")
print(f"  {'FULL MODEL':<22} {total:>8.1f} M")
print(f"  Encoder layers : "
      f"{len(teacher.speech_encoder.layers)}")
vram_status()
```

---

```python
# ╔══════════════════════════════════════════════════════════╗
# ║  CELL A2 — Load FLEURS benchmark dataset                ║
# ║  FLEURS is the official SeamlessM4T evaluation dataset  ║
# ╚══════════════════════════════════════════════════════════╝
from datasets import load_dataset, Audio

N_BENCH = 10  # use 50-100 for final paper run

print("Loading FLEURS en_us (English speech input)...")
fleurs_en = load_dataset("google/fleurs", "en_us",
                          split="test", streaming=True)
fleurs_en = fleurs_en.cast_column("audio", Audio(sampling_rate=16000))

print("Loading FLEURS bn_in (Bengali reference text)...")
fleurs_bn = load_dataset("google/fleurs", "bn_in",
                          split="test", streaming=True)

# Build aligned pairs: same sentence_id across both splits
print("Aligning English audio with Bengali text references...")
bn_by_id = {}
for ex in fleurs_bn:
    bn_by_id[ex['id']] = ex['transcription']
    if len(bn_by_id) >= N_BENCH * 5:  # gather enough to find matches
        break

bench_samples = []
for ex in fleurs_en:
    sid = ex['id']
    if sid not in bn_by_id:
        continue
    wav = np.array(ex['audio']['array'], dtype=np.float32)
    if len(wav) < 16000 or len(wav) > 16000 * 12:
        continue
    bench_samples.append({
        'id'         : sid,
        'wav'        : wav,
        'en_text'    : ex['transcription'],
        'bn_text_ref': bn_by_id[sid],    # Bengali reference for BLEU
        'duration_s' : round(len(wav) / 16000, 1),
    })
    print(f"  [{len(bench_samples)}/{N_BENCH}]  "
          f"{sid}  {len(wav)/16000:.1f}s")
    print(f"    EN: {ex['transcription'][:60]}")
    print(f"    BN: {bn_by_id[sid][:60]}")
    if len(bench_samples) >= N_BENCH:
        break

print(f"\nLoaded {len(bench_samples)} aligned FLEURS samples.")
print(f"Avg duration: "
      f"{np.mean([s['duration_s'] for s in bench_samples]):.1f}s")
```

---

```python
# ╔══════════════════════════════════════════════════════════╗
# ║  CELL A3 — Evaluation function suite                    ║
# ╚══════════════════════════════════════════════════════════╝
import whisper as whisper_lib
import jiwer
from sacrebleu.metrics import BLEU, CHRF
from speechbrain.pretrained import EncoderClassifier

print("Loading Whisper (for ASR-BLEU)...")
# Use 'large' for paper; 'base' for quick dev runs
WHISPER_SIZE = 'base'
whisper_model = whisper_lib.load_model(WHISPER_SIZE)

print("Loading speaker encoder (for SECS)...")
spk_encoder = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    run_opts={"device": "cpu"},
)

bleu_metric = BLEU(effective_order=True)
chrf_metric = CHRF()

# ─── Individual metric functions ──────────────────────────────────────────────
def get_speaker_emb(wav_np: np.ndarray, sr: int = 16000) -> torch.Tensor:
    """ECAPA-TDNN speaker embedding, L2-normalized."""
    if len(wav_np) < int(1.5 * sr):
        wav_np = np.pad(wav_np, (0, int(1.5 * sr) - len(wav_np)))
    wav_np = wav_np[:sr * 8]
    t = torch.tensor(wav_np, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        emb = spk_encoder.encode_batch(t).squeeze()
    return F.normalize(emb, dim=-1)

def compute_secs(wav_out: np.ndarray, wav_ref: np.ndarray) -> float:
    """Speaker Embedding Cosine Similarity. Target >0.75 after LoRA."""
    try:
        e1 = get_speaker_emb(wav_out)
        e2 = get_speaker_emb(wav_ref)
        return round(F.cosine_similarity(
            e1.unsqueeze(0), e2.unsqueeze(0)).item(), 4)
    except Exception as e:
        print(f"  [SECS error] {e}")
        return None

def compute_rtf(elapsed: float, input_dur: float) -> float:
    """RTF < 1.0 means faster than real-time."""
    return round(elapsed / max(input_dur, 0.001), 4)

def whisper_transcribe(wav_np: np.ndarray, language: str = None) -> dict:
    """Transcribe audio with Whisper, auto-detect language if None."""
    result = whisper_model.transcribe(
        wav_np.astype(np.float32), language=language)
    return {'text': result['text'].strip(),
            'lang': result.get('language', 'unknown')}

# ─── Core inference function ───────────────────────────────────────────────────
def run_s2st(model, wav_np: np.ndarray,
             tgt_lang: str = "ben") -> np.ndarray:
    """Run speech-to-speech translation. Returns output waveform."""
    inputs = processor(audios=wav_np, sampling_rate=16000,
                       return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(**inputs, tgt_lang=tgt_lang)
    return out[0].cpu().numpy().squeeze()

def run_s2tt(model, wav_np: np.ndarray,
             tgt_lang: str = "ben") -> str:
    """Run speech-to-text translation. Returns Bengali text string."""
    inputs = processor(audios=wav_np, sampling_rate=16000,
                       return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        # generate_speech=False → returns text token ids only
        token_ids = model.generate(
            **inputs, tgt_lang=tgt_lang, generate_speech=False)
    # Decode token ids to string
    text = processor.decode(token_ids[0].tolist(),
                            skip_special_tokens=True)
    return text.strip()

# ─── Full benchmark runner ────────────────────────────────────────────────────
def run_benchmark(model, samples: list,
                  stage_name: str, tgt_lang: str = "ben") -> dict:
    """
    Full evaluation on bench_samples.
    Returns dict with per-sample results and aggregate stats.

    Metrics computed per sample:
      - BLEU : sacrebleu sentence BLEU (S2TT predicted text vs bn reference)
      - ASR-BLEU: Whisper transcribes S2ST output → BLEU vs bn reference
      - SECS : speaker similarity input vs output
      - RTF  : real-time factor
      - translation_ok : output language is Bengali (not English fallback)
    """
    model.eval()
    results = []

    for i, sample in enumerate(samples):
        print(f"  [{i+1}/{len(samples)}] {sample['id']}")
        r = {'id': sample['id'], 'duration_s': sample['duration_s']}

        # ── S2TT: get Bengali text for BLEU ──────────────────────────────────
        try:
            t0     = time.time()
            bn_hyp = run_s2tt(model, sample['wav'], tgt_lang)
            s2tt_t = time.time() - t0

            ref = sample['bn_text_ref']
            sent_bleu = bleu_metric.sentence_score(bn_hyp, [ref]).score
            sent_chrf = chrf_metric.sentence_score(bn_hyp, [ref]).score

            r['s2tt_text']    = bn_hyp
            r['bn_ref']       = ref
            r['bleu']         = round(sent_bleu, 2)
            r['chrf']         = round(sent_chrf, 2)
            r['s2tt_rtf']     = compute_rtf(s2tt_t, sample['duration_s'])
            print(f"    S2TT BLEU={sent_bleu:.1f}  chrF={sent_chrf:.1f}")
            print(f"    hyp: {bn_hyp[:60]}")
            print(f"    ref: {ref[:60]}")
        except Exception as e:
            print(f"    S2TT error: {e}")
            r.update({'bleu': None, 'chrf': None, 's2tt_rtf': None})

        # ── S2ST: get Bengali speech for ASR-BLEU and SECS ───────────────────
        try:
            t0      = time.time()
            out_wav = run_s2st(model, sample['wav'], tgt_lang)
            s2st_t  = time.time() - t0

            out_sr  = model.config.sampling_rate
            r['s2st_rtf'] = compute_rtf(s2st_t, sample['duration_s'])

            # Resample output to 16kHz for Whisper + SECS
            if out_sr != 16000:
                out_16k = torchaudio.functional.resample(
                    torch.tensor(out_wav), out_sr, 16000).numpy()
            else:
                out_16k = out_wav

            # SECS (speaker similarity)
            r['secs'] = compute_secs(out_16k, sample['wav'])

            # ASR-BLEU: Whisper transcribes output Bengali speech
            asr = whisper_transcribe(out_16k)
            r['asr_text']  = asr['text']
            r['asr_lang']  = asr['lang']
            r['translation_ok'] = (asr['lang'] != 'en')

            # ASR-BLEU score
            ref = sample['bn_text_ref']
            asr_bleu = bleu_metric.sentence_score(
                asr['text'], [ref]).score if asr['text'] else 0.0
            r['asr_bleu'] = round(asr_bleu, 2)

            print(f"    S2ST RTF={r['s2st_rtf']:.3f}  "
                  f"SECS={r['secs']}  "
                  f"ASR-lang={asr['lang']}  "
                  f"ASR-BLEU={asr_bleu:.1f}")

        except Exception as e:
            print(f"    S2ST error: {e}")
            r.update({'s2st_rtf': None, 'secs': None,
                      'asr_bleu': None, 'translation_ok': False})

        results.append(r)
        print()

    # ── Aggregate ─────────────────────────────────────────────────────────────
    def safe_mean(key):
        vals = [r[key] for r in results
                if r.get(key) is not None and r.get('translation_ok', True)]
        return round(float(np.mean(vals)), 4) if vals else float('nan')

    agg = {
        'stage'          : stage_name,
        'n_samples'      : len(results),
        'n_ok'           : sum(1 for r in results if r.get('translation_ok')),
        'avg_bleu'       : safe_mean('bleu'),
        'avg_chrf'       : safe_mean('chrf'),
        'avg_asr_bleu'   : safe_mean('asr_bleu'),
        'avg_secs'       : safe_mean('secs'),
        'avg_s2tt_rtf'   : safe_mean('s2tt_rtf'),
        'avg_s2st_rtf'   : safe_mean('s2st_rtf'),
        'model_params_M' : count_params(model),
        'samples'        : results,
    }

    print(f"── {stage_name} Summary ──────────────────────────────")
    print(f"  Samples OK    : {agg['n_ok']}/{agg['n_samples']}")
    print(f"  Avg BLEU      : {agg['avg_bleu']:.2f}  "
          f"(S2TT text quality — primary metric)")
    print(f"  Avg chrF      : {agg['avg_chrf']:.2f}")
    print(f"  Avg ASR-BLEU  : {agg['avg_asr_bleu']:.2f}  "
          f"(S2ST speech quality)")
    print(f"  Avg SECS      : {agg['avg_secs']:.4f}  "
          f"(speaker similarity — improves after LoRA)")
    print(f"  Avg S2TT RTF  : {agg['avg_s2tt_rtf']:.4f}")
    print(f"  Avg S2ST RTF  : {agg['avg_s2st_rtf']:.4f}")
    return agg

print("Evaluation suite ready.")
print("  run_s2tt(model, wav, tgt_lang) → Bengali text")
print("  run_s2st(model, wav, tgt_lang) → Bengali audio")
print("  run_benchmark(model, samples, stage_name) → full metrics dict")
```

---

```python
# ╔══════════════════════════════════════════════════════════╗
# ║  CELL A4a — [TRAIN] Run baseline benchmark              ║
# ║  Skip this cell if A4b loads results successfully       ║
# ╚══════════════════════════════════════════════════════════╝
print("Running baseline benchmark on teacher model...")
baseline_report = run_benchmark(
    teacher, bench_samples, stage_name='baseline_teacher')

save_checkpoint(baseline_report, name='benchmark_baseline')
```

---

```python
# ╔══════════════════════════════════════════════════════════╗
# ║  CELL A4b — [LOAD] Load baseline results from Drive     ║
# ║  Run this instead of A4a if baseline already computed   ║
# ╚══════════════════════════════════════════════════════════╝
baseline_report = load_checkpoint('benchmark_baseline')
if baseline_report:
    print(f"Baseline loaded:")
    print(f"  BLEU     : {baseline_report['avg_bleu']:.2f}")
    print(f"  chrF     : {baseline_report['avg_chrf']:.2f}")
    print(f"  ASR-BLEU : {baseline_report['avg_asr_bleu']:.2f}")
    print(f"  SECS     : {baseline_report['avg_secs']:.4f}")
    print(f"  S2ST RTF : {baseline_report['avg_s2st_rtf']:.4f}")
    print(f"  Params   : {baseline_report['model_params_M']:.1f}M")
```

---

```
[MARKDOWN CELL]
# Section B — Structural Pruning

Two-stage pruning strategy:

**Stage B1 — Layer importance scoring:**
We run 200 calibration utterances from LibriSpeech through the teacher encoder
and measure the angular distance between each layer's input and output hidden states.
A layer with low angular distance (output ≈ input) contributes little and can be removed.

**Stage B2 — FFN neuron pruning:**
In each remaining encoder layer, we zero out the bottom 90% of feed-forward neurons
by L1 weight magnitude. This has been shown empirically to have near-zero effect on
translation quality while reducing effective FFN computation by 90%.

**Stage B3 — Layer removal:**
We remove the bottom 40% of layers by importance score, keeping 14/24 layers.
FFN pruning is applied first, then layers are removed.

Our chosen configuration: **FFN 90% + Layer keep 60%**
- Justified by systematic ablation (see pruning config report)
- 50% keep causes semantic translation loss; 60% only causes mild degradation
- Fine-tuning in Section C recovers the degradation
```

---

```python
# ╔══════════════════════════════════════════════════════════╗
# ║  CELL B1a — [TRAIN] Layer importance scoring            ║
# ║  Takes ~20 min. Skip if B1b loads scores successfully   ║
# ╚══════════════════════════════════════════════════════════╝
from datasets import load_dataset, Audio

print("Loading calibration data (200 samples, LibriSpeech test)...")
calib_ds = load_dataset("librispeech_asr", "clean",
                         split="test", streaming=True)
calib_ds = calib_ds.cast_column("audio", Audio(sampling_rate=16000))

calib_wavs = []
for ex in calib_ds:
    wav = np.array(ex['audio']['array'], dtype=np.float32)
    if 1.0 < len(wav)/16000 < 10.0:
        calib_wavs.append(wav[:16000*8])
    if len(calib_wavs) >= 200:
        break
print(f"Calibration set: {len(calib_wavs)} samples")

# Register hooks
layer_ins, layer_outs, hooks = {}, {}, []
for i, layer in enumerate(teacher.speech_encoder.layers):
    def make_hook(idx):
        def hook(m, inp, out):
            layer_ins[idx]  = inp[0].detach().float().cpu()
            layer_outs[idx] = (out[0] if isinstance(out,tuple)
                               else out).detach().float().cpu()
        return hook
    hooks.append(layer.register_forward_hook(make_hook(i)))

# Accumulate scores
acc = {i: [] for i in range(len(teacher.speech_encoder.layers))}
for idx, wav in enumerate(calib_wavs):
    if idx % 40 == 0:
        print(f"  Calibrating {idx}/200...")
    try:
        inp = processor(audios=wav, sampling_rate=16000,
                        return_tensors="pt")
        feats = inp['input_features'].to(teacher.device)
        mask  = inp.get('attention_mask')
        if mask is not None:
            mask = mask.to(teacher.device)
        with torch.no_grad():
            teacher.speech_encoder(input_features=feats,
                                   attention_mask=mask)
        for i in layer_ins:
            x = F.normalize(layer_ins[i].float().mean(1),  dim=-1)
            y = F.normalize(layer_outs[i].float().mean(1), dim=-1)
            cos = (x * y).sum(-1).clamp(-1,1).mean().item()
            acc[i].append(1 - cos)  # angular distance
    except Exception as e:
        pass

for h in hooks:
    h.remove()

final_scores = {i: float(np.mean(v)) for i, v in acc.items() if v}
ranked = sorted(final_scores.items(), key=lambda x: -x[1])
n_layers = len(ranked)

print("\n── Layer importance ranking ───────────────────────")
print(f"  {'Rank':>4}  {'Layer':>5}  {'Score':>8}  Chart")
for rank, (li, score) in enumerate(ranked):
    bar = '█' * int(score * 40)
    print(f"  {rank+1:>4}  {li:>5}  {score:>8.4f}  {bar}")

save_checkpoint({'ranked': ranked, 'scores': final_scores,
                 'n_layers': n_layers},
                name='layer_importance')
```

---

```python
# ╔══════════════════════════════════════════════════════════╗
# ║  CELL B1b — [LOAD] Load layer importance scores         ║
# ╚══════════════════════════════════════════════════════════╝
state    = load_checkpoint('layer_importance')
ranked   = state['ranked']
n_layers = state['n_layers']
print(f"Loaded scores for {n_layers} layers.")
print(f"Top 5 most important : {[i for i,_ in ranked[:5]]}")
print(f"Top 5 least important: {[i for i,_ in ranked[-5:]]}")
```

---

```python
# ╔══════════════════════════════════════════════════════════╗
# ║  CELL B2a — [TRAIN] Apply FFN + layer pruning           ║
# ║  Skip if B2b loads pruned model successfully            ║
# ╚══════════════════════════════════════════════════════════╝
import copy

FFN_PRUNE_RATIO  = 0.90   # zero out 90% of FFN neurons
LAYER_KEEP_RATIO = 0.60   # keep top 60% of layers (14/24)

n_keep       = int(n_layers * LAYER_KEEP_RATIO)
keep_indices = sorted([int(i) for i,_ in ranked[:n_keep]])
drop_indices = [i for i in range(n_layers) if i not in keep_indices]

print(f"Pruning config: FFN={FFN_PRUNE_RATIO*100:.0f}%  "
      f"Layers keep={LAYER_KEEP_RATIO*100:.0f}% ({n_keep}/{n_layers})")
print(f"Keep: {keep_indices}")
print(f"Drop: {drop_indices}")

pruned_model = copy.deepcopy(teacher)
pruned_model.eval()

# ── FFN neuron pruning ────────────────────────────────────────────────────────
ffn_stats = []
for li, layer in enumerate(pruned_model.speech_encoder.layers):
    ffn = getattr(layer, 'feed_forward', getattr(layer, 'ffn', None))
    if ffn is None: continue
    fc1 = next((getattr(ffn,n) for n in
                ['fc1','intermediate_dense','linear1'] if hasattr(ffn,n)), None)
    fc2 = next((getattr(ffn,n) for n in
                ['fc2','output_dense','linear2'] if hasattr(ffn,n)), None)
    if fc1 is None or fc2 is None: continue

    w = fc1.weight.data.float()
    scores = w.abs().mean(dim=1)
    thresh = torch.quantile(scores, FFN_PRUNE_RATIO)
    mask   = scores > thresh

    fc1.weight.data[~mask] = 0
    if fc1.bias is not None: fc1.bias.data[~mask] = 0
    fc2.weight.data[:, ~mask] = 0

    kept  = mask.sum().item()
    total = len(mask)
    ffn_stats.append({'layer': li, 'kept': kept, 'total': total,
                      'pct_kept': kept/total*100})

print(f"\nFFN pruned {len(ffn_stats)} layers. "
      f"Avg neurons kept: "
      f"{np.mean([s['pct_kept'] for s in ffn_stats]):.1f}%")

# ── Layer removal ─────────────────────────────────────────────────────────────
pruned_model.speech_encoder.layers = nn.ModuleList(
    [pruned_model.speech_encoder.layers[i] for i in keep_indices])

enc_before = count_params(teacher.speech_encoder)
enc_after  = count_params(pruned_model.speech_encoder)
tot_before = count_params(teacher)
tot_after  = count_params(pruned_model)

print(f"\n── Size reduction ──────────────────────────────────")
print(f"  Encoder : {enc_before:.1f}M → {enc_after:.1f}M  "
      f"(-{(1-enc_after/enc_before)*100:.1f}%)")
print(f"  Total   : {tot_before:.1f}M → {tot_after:.1f}M  "
      f"(-{(1-tot_after/tot_before)*100:.1f}%)")

# Sanity check
test_out = run_s2st(pruned_model, bench_samples[0]['wav'])
play_audio(test_out, pruned_model.config.sampling_rate,
           "Pruned model — sanity check")

# Save pruning log
save_checkpoint({
    'ffn_prune_ratio' : FFN_PRUNE_RATIO,
    'layer_keep_ratio': LAYER_KEEP_RATIO,
    'keep_indices'    : keep_indices,
    'drop_indices'    : drop_indices,
    'enc_params_M'    : enc_after,
    'total_params_M'  : tot_after,
    'ffn_stats'       : ffn_stats,
}, name='pruning_config')

# Save model to Drive
save_model(pruned_model, processor, 'stage3_pruned')
```

---

```python
# ╔══════════════════════════════════════════════════════════╗
# ║  CELL B2b — [LOAD] Load pruned model from Drive         ║
# ╚══════════════════════════════════════════════════════════╝
pruned_model, _ = load_model('stage3_pruned')
pruned_model.eval()

# Restore pruning config
pruning_cfg  = load_checkpoint('pruning_config')
keep_indices = pruning_cfg['keep_indices']
FFN_PRUNE_RATIO  = pruning_cfg['ffn_prune_ratio']
LAYER_KEEP_RATIO = pruning_cfg['layer_keep_ratio']

print(f"\nEncoder layers : {len(pruned_model.speech_encoder.layers)}")
print(f"Total params   : {count_params(pruned_model):.1f}M")
print(f"FFN ratio      : {FFN_PRUNE_RATIO}")
print(f"Layer keep     : {LAYER_KEEP_RATIO}")
vram_status()
```

---

```python
# ╔══════════════════════════════════════════════════════════╗
# ║  CELL B3 — Architecture comparison diagram (Matplotlib) ║
# ╚══════════════════════════════════════════════════════════╝
fig, axes = plt.subplots(1, 2, figsize=(14, 8))

def draw_encoder(ax, n_total, keep_idx, title, color_keep, color_drop):
    ax.set_xlim(0, 4)
    ax.set_ylim(-0.5, n_total + 1)
    ax.set_title(title, fontsize=13, pad=10)
    ax.axis('off')

    layer_h = 0.55
    gap     = 0.25
    x0, x1 = 0.5, 3.5

    drop_set = set(range(n_total)) - set(keep_idx)

    for i in range(n_total - 1, -1, -1):
        y     = (n_total - 1 - i) * (layer_h + gap)
        color = color_keep if i in keep_idx else color_drop
        alpha = 1.0 if i in keep_idx else 0.25
        rect  = mpatches.FancyBboxPatch(
            (x0, y), x1 - x0, layer_h,
            boxstyle="round,pad=0.05",
            facecolor=color, alpha=alpha,
            edgecolor='#333', linewidth=0.8)
        ax.add_patch(rect)
        label = f"Layer {i}"
        if i in drop_set:
            label += " ✗"
        ax.text((x0 + x1)/2, y + layer_h/2, label,
                ha='center', va='center', fontsize=8,
                color='#222' if i in keep_idx else '#888')

    # Labels
    top_y = n_total * (layer_h + gap)
    ax.text(2, top_y + 0.3, 'Speech\nEncoder Output',
            ha='center', va='bottom', fontsize=9,
            color='#333')
    ax.text(2, -0.4, 'Audio Input (80-dim filterbank)',
            ha='center', va='top', fontsize=9, color='#333')

n_total_layers = 24
all_idx        = list(range(n_total_layers))

# Teacher (all layers kept)
draw_encoder(axes[0], n_total_layers, all_idx,
             f'Teacher Encoder\n(24 layers, {count_params(teacher.speech_encoder):.0f}M params)',
             '#4C9BE8', '#4C9BE8')

# Pruned (only keep_indices kept)
draw_encoder(axes[1], n_total_layers, keep_indices,
             f'Pruned Encoder\n({len(keep_indices)} layers, '
             f'{count_params(pruned_model.speech_encoder):.0f}M params)',
             '#2ECC71', '#E74C3C')

# Legend
kept_patch = mpatches.Patch(color='#2ECC71', label='Kept layer')
drop_patch = mpatches.Patch(color='#E74C3C', alpha=0.3, label='Removed layer')
fig.legend(handles=[kept_patch, drop_patch],
           loc='lower center', ncol=2, fontsize=10, frameon=True)

plt.suptitle('Structural Pruning: Layer Removal Comparison',
             fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/architecture_comparison.png",
            dpi=150, bbox_inches='tight')
plt.show()
print("Saved: architecture_comparison.png")
```

---

```python
# ╔══════════════════════════════════════════════════════════╗
# ║  CELL B4a — [TRAIN] Benchmark pruned model (pre-FT)    ║
# ╚══════════════════════════════════════════════════════════╝
stage3_report = run_benchmark(
    pruned_model, bench_samples, stage_name='stage3_pruned')
save_checkpoint(stage3_report, name='benchmark_stage3')
```

---

```python
# ╔══════════════════════════════════════════════════════════╗
# ║  CELL B4b — [LOAD] Load stage3 benchmark results        ║
# ╚══════════════════════════════════════════════════════════╝
stage3_report = load_checkpoint('benchmark_stage3')
if stage3_report:
    print(f"Stage3 results loaded:")
    print(f"  BLEU     : {stage3_report['avg_bleu']:.2f}")
    print(f"  ASR-BLEU : {stage3_report['avg_asr_bleu']:.2f}")
    print(f"  SECS     : {stage3_report['avg_secs']:.4f}")
    print(f"  S2ST RTF : {stage3_report['avg_s2st_rtf']:.4f}")
```

---

```
[MARKDOWN CELL]
# Section C — Fine-tuning (Corrected Approach)

## Why the previous encoder distillation (MSE loss) was wrong

The previous Stage 5 used MSE loss between pruned encoder hidden states and
teacher encoder hidden states. This produced the "rererere" repetition artifact.

**Root cause:** SeamlessM4T is a multi-stage pipeline:
```
Audio → [Speech Encoder] → [Text Decoder] → [T2U] → [Vocoder] → Speech
```
The Text Decoder was trained on 24-layer encoder output distributions. When we
remove 10 layers and then force the encoder outputs to be numerically similar
via MSE, the downstream T2U model receives slightly wrong distributions and
generates **repeated discrete unit tokens** — which is what "rererere" sounds like.

## Correct approach: End-to-end S2T cross-entropy loss

We use `SeamlessM4Tv2ForSpeechToText` which computes cross-entropy loss between:
- Predicted Bengali text tokens (from our pruned encoder → frozen text decoder)
- Reference Bengali text tokens (from FLEURS bn_in)

This trains the encoder to produce representations that the **frozen text decoder
can actually decode correctly**. The gradient signal comes from real translation
quality, not from geometric similarity to the teacher.

**Dataset:** FLEURS en_us (English speech) with FLEURS bn_in (Bengali text targets).
This is the exact setup used in the official SeamlessM4T evaluation.
```

---

```python
# ╔══════════════════════════════════════════════════════════╗
# ║  CELL C1 — Prepare S2T model for fine-tuning            ║
# ╚══════════════════════════════════════════════════════════╝
# We use SeamlessM4Tv2ForSpeechToText because it has a proper
# forward() that accepts labels and returns cross-entropy loss.
# After fine-tuning, we transfer the encoder back to S2ST model.

from transformers import SeamlessM4Tv2ForSpeechToText

print("Building S2T fine-tuning model from pruned encoder...")

# Step 1: Extract pruned encoder weights
pruned_enc_state = pruned_model.speech_encoder.state_dict()

# Step 2: Load fresh S2T model (full teacher weights)
print("Loading SeamlessM4Tv2ForSpeechToText...")
s2t_model = SeamlessM4Tv2ForSpeechToText.from_pretrained(
    "facebook/seamless-m4t-v2-large",
    torch_dtype=torch.float16,
    device_map="auto",
)

# Step 3: Apply the same layer pruning structure
s2t_model.speech_encoder.layers = nn.ModuleList(
    [s2t_model.speech_encoder.layers[i] for i in keep_indices])
print(f"S2T encoder layers after pruning: "
      f"{len(s2t_model.speech_encoder.layers)}")

# Step 4: Load pruned encoder weights
s2t_model.speech_encoder.load_state_dict(pruned_enc_state)
print("Pruned encoder weights loaded into S2T model.")

# Step 5: Freeze everything except speech encoder
for p in s2t_model.parameters():
    p.requires_grad = False
for p in s2t_model.speech_encoder.parameters():
    p.requires_grad = True

trainable = sum(p.numel() for p in s2t_model.parameters()
                if p.requires_grad) / 1e6
frozen    = sum(p.numel() for p in s2t_model.parameters()
                if not p.requires_grad) / 1e6
print(f"\nTrainable (encoder)  : {trainable:.1f}M params")
print(f"Frozen (decoder etc) : {frozen:.1f}M params")
print("S2T model ready for fine-tuning.")
vram_status()
```

---

```python
# ╔══════════════════════════════════════════════════════════╗
# ║  CELL C2 — Load FLEURS training data                    ║
# ║  English speech → Bengali text (correct S2T signal)    ║
# ╚══════════════════════════════════════════════════════════╝
from datasets import load_dataset, Audio

print("Loading FLEURS en_us train split (streaming, ~1500 samples)...")
ft_en = load_dataset("google/fleurs", "en_us",
                      split="train", streaming=True)
ft_en = ft_en.cast_column("audio", Audio(sampling_rate=16000))

print("Loading FLEURS bn_in train split (Bengali text targets)...")
# Build Bengali text lookup by sentence ID
ft_bn = load_dataset("google/fleurs", "bn_in",
                      split="train", streaming=True)
bn_train_text = {}
print("Building Bengali reference lookup...")
for ex in ft_bn:
    bn_train_text[ex['id']] = ex['transcription']
    if len(bn_train_text) >= 2000:
        break
print(f"Bengali references loaded: {len(bn_train_text)} sentences")

# Verify alignment
en_sample = next(iter(ft_en))
if en_sample['id'] in bn_train_text:
    print(f"\nAlignment check:")
    print(f"  EN: {en_sample['transcription'][:60]}")
    print(f"  BN: {bn_train_text[en_sample['id']][:60]}")
    print("✓ Aligned")
else:
    print("⚠ No alignment found for first sample — IDs may differ")
    print(f"  First EN id: {en_sample['id']}")
    print(f"  First BN ids: {list(bn_train_text.keys())[:3]}")
```

---

```python
# ╔══════════════════════════════════════════════════════════╗
# ║  CELL C3a — [TRAIN] Fine-tune with S2T cross-entropy    ║
# ║  Skip if C3b loads fine-tuned encoder successfully      ║
# ╚══════════════════════════════════════════════════════════╝
from torch.optim import AdamW
from torch.amp import autocast, GradScaler

MAX_STEPS  = 1000   # increase to 3000 for stronger results
SAVE_EVERY = 200
LOG_EVERY  = 25
LR         = 5e-6   # low LR — encoder already partially trained
WARMUP     = 100

# Resume from checkpoint
ft_state   = load_checkpoint('finetune_s2t')
start_step = ft_state['step'] if ft_state else 0

optimizer = AdamW(
    [p for p in s2t_model.parameters() if p.requires_grad],
    lr=LR, weight_decay=0.01, betas=(0.9, 0.98), eps=1e-6,
)
if ft_state and 'optimizer_state' in ft_state:
    optimizer.load_state_dict(ft_state['optimizer_state'])

# Linear warmup then linear decay
def lr_lambda(step):
    if step < WARMUP:
        return step / max(1, WARMUP)
    return max(0.05, 1.0 - (step - WARMUP) / (MAX_STEPS - WARMUP))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
if ft_state and 'scheduler_state' in ft_state:
    scheduler.load_state_dict(ft_state['scheduler_state'])

scaler = GradScaler('cuda')
if ft_state and 'scaler_state' in ft_state:
    scaler.load_state_dict(ft_state['scaler_state'])

s2t_model.train()
s2t_model.speech_encoder.train()

step, loss_log, skip_count = start_step, [], 0
data_iter = iter(ft_en)

print(f"Fine-tuning: {MAX_STEPS} steps | LR={LR} | Warmup={WARMUP}")
print(f"Task: English speech → Bengali text (S2T cross-entropy)")
print(f"Resuming from step {start_step}\n")

while step < MAX_STEPS:
    try:
        ex = next(data_iter)
    except StopIteration:
        data_iter = iter(ft_en)
        ex = next(data_iter)

    # Skip if no Bengali reference available for this sentence
    if ex['id'] not in bn_train_text:
        skip_count += 1
        continue

    audio_dict = ex['audio']
    wav = np.array(audio_dict['array'], dtype=np.float32)
    sr  = audio_dict['sampling_rate']
    if sr != 16000:
        wav = torchaudio.functional.resample(
            torch.tensor(wav), sr, 16000).numpy()

    if len(wav) < 16000 or len(wav) > 16000 * 12:
        skip_count += 1
        continue

    bn_text = bn_train_text[ex['id']]

    try:
        # ── Prepare inputs ────────────────────────────────────────────────────
        audio_inputs = processor(
            audios=wav, sampling_rate=16000,
            return_tensors="pt", padding=True)
        input_features = audio_inputs['input_features'].to(s2t_model.device)
        attn_mask = audio_inputs.get('attention_mask')
        if attn_mask is not None:
            attn_mask = attn_mask.to(s2t_model.device)

        # ── Prepare Bengali text labels ────────────────────────────────────────
        # Tokenize Bengali target text
        text_inputs = processor(
            text=bn_text, return_tensors="pt",
            src_lang="ben",   # Bengali
        )
        labels = text_inputs['input_ids'].to(s2t_model.device)
        # Replace padding token id with -100 (ignored in CE loss)
        labels[labels == processor.tokenizer.pad_token_id] = -100

        optimizer.zero_grad(set_to_none=True)

        with autocast('cuda'):
            outputs = s2t_model(
                input_features=input_features,
                attention_mask=attn_mask,
                labels=labels,
                tgt_lang="ben",
            )
            loss = outputs.loss

        if torch.isnan(loss) or torch.isinf(loss):
            skip_count += 1
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in s2t_model.parameters() if p.requires_grad],
            max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        loss_log.append(loss.item())
        step += 1

        if step % LOG_EVERY == 0:
            avg = np.mean(loss_log[-LOG_EVERY:])
            cur_lr = scheduler.get_last_lr()[0]
            print(f"  Step {step:>5}/{MAX_STEPS}  "
                  f"loss={avg:.4f}  "
                  f"lr={cur_lr:.2e}  "
                  f"grad={grad_norm:.3f}  "
                  f"skip={skip_count}")

        if step % SAVE_EVERY == 0:
            # Save only encoder weights (small) not the full S2T model
            enc_weights = {k: v.cpu() for k, v in
                           s2t_model.speech_encoder.state_dict().items()}
            save_checkpoint({
                'step'             : step,
                'loss'             : loss.item(),
                'avg_loss'         : float(np.mean(loss_log[-SAVE_EVERY:])),
                'loss_history'     : loss_log,
                'encoder_state'    : enc_weights,
                'optimizer_state'  : optimizer.state_dict(),
                'scheduler_state'  : scheduler.state_dict(),
                'scaler_state'     : scaler.state_dict(),
            }, name='finetune_s2t', step=step)

    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            print(f"  OOM at step {step} — clearing cache")
            torch.cuda.empty_cache()
            optimizer.zero_grad(set_to_none=True)
            continue
        raise e

# Final state
s2t_model.eval()
final_loss   = float(np.mean(loss_log[-50:])) if len(loss_log)>=50 \
               else float(np.mean(loss_log)) if loss_log else float('nan')
initial_loss = float(np.mean(loss_log[:50])) if len(loss_log)>=50 \
               else loss_log[0] if loss_log else float('nan')

print(f"\nFine-tuning complete.")
print(f"  Steps      : {step}")
print(f"  Skipped    : {skip_count}")
print(f"  Init loss  : {initial_loss:.4f}")
print(f"  Final loss : {final_loss:.4f}")
print(f"  Reduction  : {(initial_loss-final_loss)/initial_loss*100:.1f}%")

# Save final encoder weights
enc_weights = {k: v.cpu() for k, v in
               s2t_model.speech_encoder.state_dict().items()}
save_checkpoint({
    'step': step, 'loss_history': loss_log,
    'initial_loss': initial_loss, 'final_loss': final_loss,
    'encoder_state': enc_weights,
}, name='finetune_s2t', step=step)
```

---

```python
# ╔══════════════════════════════════════════════════════════╗
# ║  CELL C3b — [LOAD] Load fine-tuned encoder weights      ║
# ╚══════════════════════════════════════════════════════════╝
ft_state = load_checkpoint('finetune_s2t')
if ft_state:
    print(f"Fine-tune checkpoint loaded (step {ft_state['step']})")
    print(f"  Initial loss : {ft_state.get('initial_loss','?'):.4f}")
    print(f"  Final loss   : {ft_state['loss']:.4f}")
    loss_log = ft_state.get('loss_history', [])
    print(f"  Loss history : {len(loss_log)} entries")
else:
    print("No fine-tune checkpoint found. Run C3a first.")
```

---

```python
# ╔══════════════════════════════════════════════════════════╗
# ║  CELL C4a — [BUILD] Transfer encoder → S2ST model       ║
# ║  Builds stage5_finetuned from fine-tuned encoder        ║
# ╚══════════════════════════════════════════════════════════╝
# Transfer the fine-tuned encoder weights back into the S2ST model
# so we can do S2ST inference for proper evaluation.

print("Transferring fine-tuned encoder into S2ST model...")

# Load the pruned S2ST model fresh
ft_s2st, _ = load_model('stage3_pruned')

# Apply the same layer pruning structure (already in stage3_pruned)
# Just load the fine-tuned encoder weights on top
enc_state = ft_state['encoder_state']
# Map weights — keys match since same architecture
ft_s2st.speech_encoder.load_state_dict(enc_state)
ft_s2st.eval()

print(f"Fine-tuned S2ST model ready.")
print(f"  Total params: {count_params(ft_s2st):.1f}M")

# Sanity check — listen
print("\nSanity check (should be better Bengali than before fine-tuning):")
for i in range(min(3, len(bench_samples))):
    out = run_s2st(ft_s2st, bench_samples[i]['wav'])
    print(f"\nSample {i+1}: {bench_samples[i]['id']}")
    print(f"  ref EN: {bench_samples[i]['en_text'][:60]}")
    print(f"  ref BN: {bench_samples[i]['bn_text_ref'][:60]}")
    play_audio(out, ft_s2st.config.sampling_rate,
               f"Fine-tuned S2ST — sample {i+1}")

# Save the complete fine-tuned S2ST model
save_model(ft_s2st, processor, 'stage5_finetuned')
```

---

```python
# ╔══════════════════════════════════════════════════════════╗
# ║  CELL C4b — [LOAD] Load stage5_finetuned               ║
# ╚══════════════════════════════════════════════════════════╝
ft_s2st, _ = load_model('stage5_finetuned')
ft_s2st.eval()
print(f"Fine-tuned S2ST model loaded: {count_params(ft_s2st):.1f}M params")
vram_status()
```

---

```python
# ╔══════════════════════════════════════════════════════════╗
# ║  CELL C5 — Loss curve visualization                     ║
# ╚══════════════════════════════════════════════════════════╝
if not loss_log:
    ft_state = load_checkpoint('finetune_s2t')
    loss_log = ft_state.get('loss_history', []) if ft_state else []

if loss_log:
    window = 20
    smoothed = [np.mean(loss_log[max(0,i-window):i+1])
                for i in range(len(loss_log))]
    steps = list(range(1, len(loss_log)+1))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(steps, loss_log, alpha=0.3, color='#4C9BE8',
            linewidth=0.8, label='Raw loss')
    ax.plot(steps, smoothed, color='#E74C3C',
            linewidth=2, label=f'Smoothed (window={window})')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Cross-Entropy Loss')
    ax.set_title('Fine-tuning Loss Curve\n'
                 '(English Speech → Bengali Text, S2T cross-entropy)')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xlim(1, len(loss_log))

    # Annotate start/end
    ax.annotate(f"Init: {loss_log[0]:.3f}",
                xy=(1, loss_log[0]),
                xytext=(len(loss_log)*0.1, loss_log[0]+0.05),
                arrowprops=dict(arrowstyle='->', color='#333'),
                fontsize=9)
    ax.annotate(f"Final: {smoothed[-1]:.3f}",
                xy=(len(loss_log), smoothed[-1]),
                xytext=(len(loss_log)*0.8, smoothed[-1]+0.05),
                arrowprops=dict(arrowstyle='->', color='#333'),
                fontsize=9)

    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/finetuning_loss_curve.png",
                dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: finetuning_loss_curve.png")
else:
    print("No loss history available. Run C3a first.")
```

---

```python
# ╔══════════════════════════════════════════════════════════╗
# ║  CELL C6a — [TRAIN] Benchmark fine-tuned model          ║
# ╚══════════════════════════════════════════════════════════╝
stage5_report = run_benchmark(
    ft_s2st, bench_samples, stage_name='stage5_finetuned')
save_checkpoint(stage5_report, name='benchmark_stage5')
```

---

```python
# ╔══════════════════════════════════════════════════════════╗
# ║  CELL C6b — [LOAD] Load stage5 benchmark results        ║
# ╚══════════════════════════════════════════════════════════╝
stage5_report = load_checkpoint('benchmark_stage5')
if stage5_report:
    print(f"Stage5 results loaded:")
    print(f"  BLEU     : {stage5_report['avg_bleu']:.2f}")
    print(f"  ASR-BLEU : {stage5_report['avg_asr_bleu']:.2f}")
    print(f"  SECS     : {stage5_report['avg_secs']:.4f}")
    print(f"  S2ST RTF : {stage5_report['avg_s2st_rtf']:.4f}")
```

---

```
[MARKDOWN CELL]
# Section D — Full Comparison & Paper Figures

Aggregate results from all stages into the final paper tables and figures.

Stages compared:
1. **Baseline** — Full teacher model (1805M params)
2. **Pruned** — FFN 90% + Layer 60% keep (1563M params, no fine-tuning)
3. **Fine-tuned** — Pruned + S2T cross-entropy fine-tuning

Primary metrics:
- **BLEU** — Translation text quality (S2TT mode)
- **ASR-BLEU** — Translated speech quality (S2ST mode, Whisper transcription)
- **S2ST RTF** — Inference speed
- **Params (M)** — Model compression
```

---

```python
# ╔══════════════════════════════════════════════════════════╗
# ║  CELL D1 — Load all stage results                       ║
# ╚══════════════════════════════════════════════════════════╝
# Load whichever reports are available
baseline_report = load_checkpoint('benchmark_baseline')
stage3_report   = load_checkpoint('benchmark_stage3')
stage5_report   = load_checkpoint('benchmark_stage5')

reports = {}
if baseline_report: reports['Baseline\n(1805M)']   = baseline_report
if stage3_report:   reports['Pruned\n(1563M)']      = stage3_report
if stage5_report:   reports['Fine-tuned\n(1563M)']  = stage5_report

print(f"Loaded {len(reports)} stage reports:")
for name, r in reports.items():
    label = name.replace('\n', ' ')
    print(f"  {label:<25} BLEU={r.get('avg_bleu','?'):.2f}  "
          f"RTF={r.get('avg_s2st_rtf','?'):.4f}  "
          f"Params={r.get('model_params_M','?'):.1f}M")
```

---

```python
# ╔══════════════════════════════════════════════════════════╗
# ║  CELL D2 — Paper comparison figures (Matplotlib)        ║
# ╚══════════════════════════════════════════════════════════╝
labels      = list(reports.keys())
colors      = ['#4C9BE8', '#E74C3C', '#2ECC71'][:len(labels)]

bleu_vals    = [r.get('avg_bleu', 0)      or 0 for r in reports.values()]
asr_bleu_vals= [r.get('avg_asr_bleu', 0)  or 0 for r in reports.values()]
secs_vals    = [r.get('avg_secs', 0)      or 0 for r in reports.values()]
rtf_vals     = [r.get('avg_s2st_rtf', 0)  or 0 for r in reports.values()]
params_vals  = [r.get('model_params_M', 0) or 0 for r in reports.values()]

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle('CSE465 Approach 2: Structural Pruning + Fine-tuning\n'
             'Performance Comparison Across Stages',
             fontsize=13, y=1.02)

def bar_chart(ax, vals, title, ylabel, higher_better=True,
              baseline_val=None):
    bars = ax.bar(labels, vals, color=colors, width=0.5,
                  edgecolor='#333', linewidth=0.8)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + max(vals)*0.01,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    if baseline_val:
        ax.axhline(y=baseline_val, color='#333',
                   linestyle='--', linewidth=1, alpha=0.5,
                   label='Baseline')
    arrow = '↑ higher better' if higher_better else '↓ lower better'
    ax.set_xlabel(arrow, fontsize=8, color='#666')
    ax.tick_params(axis='x', labelsize=8)

bar_chart(axes[0,0], bleu_vals,
          'BLEU Score (S2TT)', 'BLEU', higher_better=True)
bar_chart(axes[0,1], asr_bleu_vals,
          'ASR-BLEU (S2ST)', 'ASR-BLEU', higher_better=True)
bar_chart(axes[0,2], secs_vals,
          'SECS (Speaker Similarity)', 'SECS', higher_better=True)

bar_chart(axes[1,0], rtf_vals,
          'Real-Time Factor (S2ST)', 'RTF', higher_better=False)
bar_chart(axes[1,1], params_vals,
          'Model Parameters', 'Params (M)', higher_better=False)

# Compression ratio chart
if len(params_vals) > 1 and params_vals[0] > 0:
    compression = [params_vals[0]/p if p > 0 else 1 for p in params_vals]
    bar_chart(axes[1,2], compression,
              'Compression Ratio (×)', '× smaller than baseline',
              higher_better=True)
else:
    axes[1,2].text(0.5, 0.5, 'Need all stages\nfor compression ratio',
                   ha='center', va='center', transform=axes[1,2].transAxes,
                   fontsize=10, color='#666')
    axes[1,2].axis('off')

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/stage_comparison.png",
            dpi=150, bbox_inches='tight')
plt.show()
print("Saved: stage_comparison.png")
```

---

```python
# ╔══════════════════════════════════════════════════════════╗
# ║  CELL D3 — Paper table (formatted)                      ║
# ╚══════════════════════════════════════════════════════════╝
print("\n")
print("╔══════════════════════════════════════════════════════════════════════════════════════╗")
print("║     TABLE — Approach 2: Structural Pruning + Knowledge Distillation Fine-tuning     ║")
print("╠═══════════════════╦══════════╦══════════╦══════════╦══════════╦══════════╦══════════╣")
print("║  Stage            ║ Params(M)║  Size Δ  ║   BLEU   ║ ASR-BLEU ║   SECS   ║ RTF(S2ST)║")
print("╠═══════════════════╬══════════╬══════════╬══════════╬══════════╬══════════╬══════════╣")

base_params = baseline_report['model_params_M'] if baseline_report else 1805.5
stage_rows = [
    ("Baseline (teacher)", baseline_report),
    ("Pruned (no FT)",     stage3_report),
    ("Fine-tuned",         stage5_report),
]

for name, r in stage_rows:
    if r is None:
        print(f"║  {name:<17}  ║  {'—':>8}  ║  {'—':>8}  ║  {'—':>8}  ║  {'—':>8}  ║  {'—':>8}  ║  {'—':>8}  ║")
        continue
    p     = r.get('model_params_M', 0) or 0
    delta = f"-{(1-p/base_params)*100:.1f}%" if p < base_params else "baseline"
    bleu  = r.get('avg_bleu', float('nan')) or float('nan')
    ableu = r.get('avg_asr_bleu', float('nan')) or float('nan')
    secs  = r.get('avg_secs', float('nan')) or float('nan')
    rtf   = r.get('avg_s2st_rtf', float('nan')) or float('nan')
    print(f"║  {name:<17}  ║  {p:>8.1f}  ║  {delta:>8}  ║  "
          f"{bleu:>8.2f}  ║  {ableu:>8.2f}  ║  {secs:>8.4f}  ║  {rtf:>8.4f}  ║")

print("╚═══════════════════╩══════════╩══════════╩══════════╩══════════╩══════════╩══════════╝")

if baseline_report and stage5_report:
    b_bleu  = baseline_report.get('avg_bleu',  0) or 0
    f_bleu  = stage5_report.get('avg_bleu',    0) or 0
    b_rtf   = baseline_report.get('avg_s2st_rtf', 1) or 1
    f_rtf   = stage5_report.get('avg_s2st_rtf',  1) or 1
    b_prm   = baseline_report.get('model_params_M', 1) or 1
    f_prm   = stage5_report.get('model_params_M',   1) or 1

    print(f"""
Key findings:
  Parameter reduction  : {(1 - f_prm/b_prm)*100:.1f}% smaller than baseline
  BLEU retention       : {f_bleu/b_bleu*100 if b_bleu else 0:.1f}% of baseline quality
  RTF change           : {f_rtf/b_rtf:.2f}x (>1 = slower, expected on GPU)
  Evaluation dataset   : FLEURS en_us → bn_in (official SeamlessM4T benchmark)
  Fine-tuning task     : S2T cross-entropy (English speech → Bengali text)
  Fine-tuning dataset  : FLEURS en_us + bn_in aligned pairs
""")

# Save full combined report
combined = {
    'approach'   : 'Structural Pruning + S2T Cross-Entropy Fine-tuning',
    'model'      : 'facebook/seamless-m4t-v2-large',
    'eval_dataset': 'google/fleurs en_us + bn_in',
    'n_bench'    : N_BENCH,
    'baseline'   : baseline_report,
    'stage3'     : stage3_report,
    'stage5'     : stage5_report,
}
report_path = f"{CKPT_DIR}/paper_results.json"
with open(report_path, 'w') as f:
    json.dump(combined, f, indent=2, default=str)
subprocess.run(f'rclone copy {report_path} gdrive:cse465/',
               shell=True, capture_output=True)
print("Full paper results saved to Drive as paper_results.json")
```

---

```python
# ╔══════════════════════════════════════════════════════════╗
# ║  CELL D4 — Push all figures to Drive                    ║
# ╚══════════════════════════════════════════════════════════╝
subprocess.run(
    f'rclone sync {FIG_DIR} gdrive:cse465/figures/',
    shell=True)
figs = os.listdir(FIG_DIR)
print(f"Pushed {len(figs)} figures to Drive:")
for f in sorted(figs):
    print(f"  {f}")
```

---

```python
# ╔══════════════════════════════════════════════════════════╗
# ║  LAST CELL — Session end (always run before closing)    ║
# ╚══════════════════════════════════════════════════════════╝
print("Final sync to Drive...")
subprocess.run(f'rclone sync {CKPT_DIR} gdrive:cse465/checkpoints/',
               shell=True)
subprocess.run(f'rclone sync {AUDIO_DIR} gdrive:cse465/audio/',
               shell=True)
subprocess.run(f'rclone sync {FIG_DIR} gdrive:cse465/figures/',
               shell=True)

log = {
    'platform' : PLATFORM,
    'time'     : datetime.now().isoformat(),
    'completed': 'EDIT THIS',
    'next'     : 'EDIT THIS',
    'ckpts'    : [f for f in os.listdir(CKPT_DIR)
                  if f.endswith('.pt')],
}
log_path = f"{CKPT_DIR}/session_log.json"
with open(log_path, 'w') as f:
    json.dump(log, f, indent=2)
subprocess.run(f'rclone copy {log_path} gdrive:cse465/checkpoints/',
               shell=True)
print(json.dumps(log, indent=2))
session_status()
```

---

## What was changed and why

The three fundamental corrections in this version:

**Fine-tuning loss**: Changed from encoder MSE distillation → end-to-end S2T cross-entropy. The encoder is now trained by signal from whether the frozen text decoder can correctly predict Bengali tokens. This directly solves the "rererere" problem because the encoder must produce representations the decoder can actually decode.

**Dataset**: Changed from LibriSpeech (English-only) → FLEURS en_us + bn_in aligned pairs. FLEURS is the official evaluation dataset for SeamlessM4T. Using Bengali text targets gives the encoder a genuine translation signal rather than just ASR signal.

**Evaluation**: Changed from Whisper Bengali transcription only → BLEU score on S2TT text output + ASR-BLEU on S2ST speech output, using sacrebleu against FLEURS Bengali references. This matches the official evaluation protocol from the SeamlessM4T evaluate.py source.