This is a well-scoped and exciting project. Here's the full breakdown — first a pipeline diagram of the unified system, then step-by-step per person.---

## The unified goal

User speaks Language A (in their voice) → model outputs Language B **in the same voice**. SeamlessM4T-v2-large already handles the translation; it just uses a fixed default voice. Your job: add voice identity transfer and compress the whole thing to run locally.

The key insight is that all three approaches share the **same speaker encoder** idea — you extract a voice fingerprint from the input audio and condition the speech synthesis on it. What differs is *how you compress the translation backbone*.

---

## Person 1 — Tanvir: Knowledge Distillation

**Your job**: Build a small student network (~50M params) trained to imitate SeamlessM4T's internal representations, then plug in a speaker conditioning module.

**Step 1 — Environment**
```bash
pip install transformers torchaudio speechbrain einops
# Clone the SeamlessM4T model for teacher
from transformers import SeamlessM4Tv2Model, SeamlessM4TProcessor
teacher = SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large")
teacher.eval()  # frozen — never update its weights
```

**Step 2 — Understand the teacher's architecture**

Run this to see layer names and sizes:
```python
for name, param in teacher.named_parameters():
    print(name, param.shape)
```
You're interested in: `speech_encoder`, `text_decoder`, `vocoder`. The teacher speech encoder is ~600M alone. You will distill it down to ~8M with a 6-layer causal conformer.

**Step 3 — Build the speaker encoder first (this is standalone)**

Use SpeechBrain's pretrained ECAPA-TDNN. This extracts a 192-dim speaker embedding from 3–5 seconds of audio:
```python
from speechbrain.pretrained import EncoderClassifier
spk_enc = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb")
# Input: raw waveform tensor [batch, time]
# Output: embedding [batch, 192]
embedding = spk_enc.encode_batch(waveform)
```
Test it immediately: verify that two recordings of the same person have cosine similarity > 0.85, and two different people < 0.6.

**Step 4 — Design the student architecture**

You're building 5 modules (see proposal, Methodology 1). Start with just the content encoder + mel decoder. Don't build everything at once.

```
Input audio
    → Causal Content Encoder (6-layer conv-transformer, ~8M)   ← distill from teacher speech encoder
    → Speaker Encoder ECAPA-TDNN (~10M, pretrained, frozen)
    → Language ID (3-layer CNN, ~1M)
    → Mel Decoder with FiLM conditioning (~12M)               ← FiLM = Feature-wise Linear Modulation
    → HiFi-GAN Lite vocoder (~12M)
```

FiLM conditioning is how speaker identity enters the decoder. For each decoder layer, you compute `gamma, beta = Linear(speaker_embedding)` and then apply `output = gamma * layer_output + beta`. This is ~5 lines of code per layer.

**Step 5 — Stage 1 training: content encoder distillation**

Before training the full model, just train the content encoder to mimic the teacher's speech encoder features on your dataset:

```python
# Loss: MSE between student and teacher encoder hidden states
loss = F.mse_loss(student_encoder_output, teacher_encoder_output.detach())
```

Dataset for this stage: **LibriTTS** (English, 585h, already in HuggingFace). This is fast to train and tells you if your architecture works before you invest in multilingual data.

**Step 6 — Stage 2 training: full pipeline with speaker loss**

Now train end-to-end. Your loss has 3 parts:
- Mel reconstruction loss (compare student mel output to teacher mel output)
- Speaker similarity loss: cosine similarity between speaker embeddings of output and reference should be > 0.82
- Cycle consistency: if you convert A→B, then B→A, you should get back to A

**Step 7 — Evaluation checkpoints**
- After stage 1: check WER on LibriTTS test set using Whisper (should be < 8%)
- After stage 2: run SECS (Speaker Embedding Cosine Similarity) on VCTK test set
- Target: SECS > 0.82, WER < 5%, RTF < 0.3

---

## Person 2 — Rayed: Structural Pruning + Speaker LoRA

**Your job**: Start from the actual SeamlessM4T-v2-large weights, aggressively prune them (as the RA described — 99% of FFN neurons can be removed with minimal quality loss), fine-tune the pruned model, then add speaker conditioning via LoRA adapters.

**Step 1 — Load and inspect the model**

```python
from transformers import SeamlessM4Tv2ForSpeechToSpeech, SeamlessM4TProcessor
import torch

model = SeamlessM4Tv2ForSpeechToSpeech.from_pretrained(
    "facebook/seamless-m4t-v2-large",
    torch_dtype=torch.float16
)
# Check parameter count per component
def count_params(module):
    return sum(p.numel() for p in module.parameters()) / 1e6
print(f"Speech encoder: {count_params(model.speech_encoder):.1f}M")
print(f"Text decoder: {count_params(model.text_decoder):.1f}M")
print(f"Vocoder: {count_params(model.vocoder):.1f}M")
```

**Step 2 — FFN neuron pruning (the RA's key insight)**

In transformer FFN layers, most neurons are near-zero on any given input. You score each neuron by its average activation magnitude across a calibration set, then zero out the lowest 70-80% (the RA said 99% is possible — start at 70% to be safe, then push further):

```python
def prune_ffn(layer, prune_ratio=0.7):
    # Score neurons by L1 norm of their weight vectors
    weight = layer.fc1.weight.data  # [ffn_dim, hidden_dim]
    scores = weight.abs().mean(dim=1)  # [ffn_dim]
    threshold = torch.quantile(scores, prune_ratio)
    mask = scores > threshold
    # Apply mask — zero out pruned neurons
    layer.fc1.weight.data[~mask] = 0
    layer.fc2.weight.data[:, ~mask] = 0
    return mask.sum().item()  # return surviving neuron count
```

Run this on a calibration set of ~500 utterances from LibriTTS to compute real activation statistics before pruning.

**Step 3 — Layer-level pruning**

After FFN pruning, also remove full attention layers that contribute little. The RA mentioned ~50% of layers can be dropped. Measure each layer's contribution by computing the cosine similarity between its input and output — a layer whose output is near-identical to its input is doing almost nothing:

```python
def layer_importance(model, calibration_data):
    importances = []
    for i, layer in enumerate(model.speech_encoder.layers):
        sims = []
        for batch in calibration_data:
            with torch.no_grad():
                inp = get_layer_input(model, i, batch)   # hook-based
                out = layer(inp)[0]
                sim = F.cosine_similarity(inp.mean(1), out.mean(1)).mean()
                sims.append(sim.item())
        importances.append((i, 1 - sum(sims)/len(sims)))  # higher = more important
    return sorted(importances, key=lambda x: -x[1])
```

Keep the top 50% by importance, rebuild the model with `nn.ModuleList` of only those layers.

**Step 4 — Post-pruning fine-tuning**

The pruned model will degrade. Fine-tune for 2–3 epochs on **Multilingual LibriSpeech** (8 languages, available at `facebook/multilingual_librispeech` on HuggingFace). Use a small learning rate (1e-5) and the full SeamlessM4T loss.

**Step 5 — Add speaker conditioning via LoRA**

You don't need to redesign the vocoder. Instead, add LoRA adapters to the vocoder's linear layers. LoRA injects low-rank matrices (rank 8–16) that learn to modulate output based on speaker embedding:

```python
from peft import get_peft_model, LoraConfig, TaskType

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["k_proj", "v_proj", "out_proj"],  # in vocoder attention
    lora_dropout=0.05,
)
model.vocoder = get_peft_model(model.vocoder, lora_config)
```

Then prepend the speaker embedding to the vocoder's conditioning vector before each forward pass. Train only the LoRA weights (frozen everything else) on a voice conversion dataset like **VCTK**.

**Step 6 — Quantize to INT8**

After fine-tuning:
```python
from transformers import BitsAndBytesConfig
quantization_config = BitsAndBytesConfig(load_in_8bit=True)
# Reload the pruned checkpoint with quantization
model_quantized = SeamlessM4Tv2ForSpeechToSpeech.from_pretrained(
    "./pruned_checkpoint",
    quantization_config=quantization_config
)
```

This halves memory with ~1% quality loss. Measure RTF before and after on your RTX 3060 Ti.

**Step 7 — Evaluation**

Track model size at each stage: original → after FFN prune → after layer prune → after quantization. Your notebook should have a table: size (MB), WER, SECS, RTF. This comparison *is* your result.

---

## Person 3 — Nihal: Layer Selection + Speaker Adapter Module

**Your job**: Implement the RA's layer selection method (find the 50% of layers that actually matter), extract them into a compact model, then build a plug-in speaker adapter that works independently of the backbone.

**Step 1 — Understand what "layer selection" means here**

The RA described a recent paper where they scored transformer layers by how much each one changes the representation. Your version: for each layer `i`, compute the angular distance between the input hidden state and the output hidden state across a calibration corpus. Large angular distance = layer is doing real work. Small = mostly pass-through, safe to remove.

**Step 2 — Set up calibration pipeline**

```python
from transformers import SeamlessM4Tv2Model, SeamlessM4TProcessor
import torch, torch.nn.functional as F

model = SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large")
processor = SeamlessM4TProcessor.from_pretrained("facebook/seamless-m4t-v2-large")

# Register hooks to capture layer inputs/outputs
layer_inputs, layer_outputs = {}, {}
def make_hook(i):
    def hook(module, inp, out):
        layer_inputs[i] = inp[0].detach()
        layer_outputs[i] = out[0].detach()
    return hook

for i, layer in enumerate(model.speech_encoder.layers):
    layer.register_forward_hook(make_hook(i))
```

Run 200–300 utterances from LibriTTS through the model, recording `layer_inputs[i]` and `layer_outputs[i]` for each layer `i`. Then compute the angular distance for each:

```python
def angular_distance(x, y):
    # x, y: [batch, seq, hidden]
    x_norm = F.normalize(x.mean(1), dim=-1)
    y_norm = F.normalize(y.mean(1), dim=-1)
    cos_sim = (x_norm * y_norm).sum(-1).clamp(-1, 1)
    return (1 - cos_sim).mean().item()  # 0=pass-through, 2=max change

scores = {i: angular_distance(layer_inputs[i], layer_outputs[i]) for i in layer_inputs}
ranked = sorted(scores.items(), key=lambda x: -x[1])
print("Most important layers:", [i for i,_ in ranked[:len(ranked)//2]])
```

**Step 3 — Build the compact model using selected layers**

You now have a list of the top-N layer indices. Build a new model that only uses those layers:

```python
selected_indices = [i for i, _ in ranked[:len(ranked)//2]]
selected_layers = nn.ModuleList([
    model.speech_encoder.layers[i] for i in sorted(selected_indices)
])
# Replace the full layer stack
model.speech_encoder.layers = selected_layers
```

Save this as your base checkpoint. Test WER immediately — it will degrade somewhat, which tells you the cost of removal. If WER degrades more than 5 percentage points, try keeping 60% of layers instead.

**Step 4 — Add lightweight adapter layers between the selected layers**

To help the model recover from missing layers, insert small "bridge adapters" between consecutive selected layers. These are simple 2-layer MLPs that smooth the representation gap:

```python
class BridgeAdapter(nn.Module):
    def __init__(self, hidden_dim=1024, bottleneck=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, hidden_dim)
        )
        self.norm = nn.LayerNorm(hidden_dim)
    def forward(self, x):
        return self.norm(x + self.net(x))  # residual
```

Each adapter is only ~130K params. If you have 12 selected layers with 11 adapters between them, that's ~1.4M params total — negligible.

**Step 5 — Speaker adapter module (your unique contribution)**

This is the module that handles voice identity. Design it as a fully standalone unit that sits between the translator's output and the vocoder, so it can be swapped or fine-tuned independently:

```python
class SpeakerAdapter(nn.Module):
    def __init__(self, spk_dim=192, mel_dim=80, hidden=256):
        super().__init__()
        # Projects speaker embedding to conditioning vector
        self.spk_proj = nn.Linear(spk_dim, hidden)
        # Cross-attention: mel queries, speaker as key/value
        self.cross_attn = nn.MultiheadAttention(mel_dim, num_heads=4, batch_first=True)
        self.spk_to_kv = nn.Linear(hidden, mel_dim)
        self.out_norm = nn.LayerNorm(mel_dim)

    def forward(self, mel, spk_embedding):
        # mel: [B, T, 80], spk_embedding: [B, 192]
        spk = self.spk_proj(spk_embedding).unsqueeze(1)  # [B, 1, 256]
        kv = self.spk_to_kv(spk)                         # [B, 1, 80]
        out, _ = self.cross_attn(mel, kv, kv)
        return self.out_norm(mel + out)
```

The cross-attention mechanism lets each mel frame attend to the speaker embedding, pulling in the voice characteristics at the spectrogram level before the vocoder converts to audio.

**Step 6 — Training the speaker adapter**

Train only the speaker adapter (freeze everything else) on paired data: take VCTK corpus (110 speakers, 44h English), run audio through the pipeline without speaker conditioning to get a baseline mel, then train the adapter to transform this baseline mel so that speaker embeddings of the output match speaker embeddings of the reference speaker.

```python
# Loss: speaker embedding cosine similarity
spk_emb_out = speaker_encoder(vocoder(adapter_output_mel))
spk_emb_ref = speaker_encoder(reference_waveform)
loss = 1 - F.cosine_similarity(spk_emb_out, spk_emb_ref).mean()
```

**Step 7 — Evaluation focus**

Your unique metric is **cross-lingual SECS** — does the voice identity survive when the language changes? Test with FLEURS: take 20 speakers, each saying the same content in 2 different languages through your model. Compute SECS between the two outputs. Target > 0.75. This is the benchmark your proposal calls "new" — Nihal's section should own this number.

---

## Shared infrastructure (all three of you)

**Datasets to download first** (these are free and on HuggingFace):
```python
# All three of you need these
from datasets import load_dataset
libri = load_dataset("librispeech_asr", "clean", split="train.100")
vctk = load_dataset("vctk", split="train")
fleurs = load_dataset("google/fleurs", "all", split="test")
mls = load_dataset("facebook/multilingual_librispeech", "english", split="train")
```

**Evaluation script (share this)**:
```python
import whisper, torch
from speechbrain.pretrained import EncoderClassifier

whisper_model = whisper.load_model("large")
spk_model = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb")

def evaluate(output_wav, reference_wav, reference_text, lang="en"):
    # WER
    result = whisper_model.transcribe(output_wav, language=lang)
    wer = compute_wer(reference_text, result["text"])
    # SECS
    emb_out = spk_model.encode_batch(output_wav)
    emb_ref = spk_model.encode_batch(reference_wav)
    secs = F.cosine_similarity(emb_out, emb_ref).item()
    return {"WER": wer, "SECS": secs}
```

**Recommended task split**:

- Tanvir (Approach 1) — owns the baseline pipeline and the `SpeakerEncoder + FiLM` implementation. Shares ECAPA-TDNN code with Nihal.
- Rayed (Approach 2) — owns the pruning + quantization experiments. Documents the size/quality tradeoff table.
- Nihal (Approach 3) — owns the layer scoring analysis and cross-lingual SECS benchmark. Shares the evaluation script with everyone.

Start all three approaches with a working SeamlessM4T baseline on LibriTTS English first, *then* add speaker conditioning, *then* tackle multilingual. Don't try to solve everything at once.