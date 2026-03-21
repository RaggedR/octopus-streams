"""
Octopus Streams — Experiment 5: Ablation on Natural Language

Same ablation design as 03, but with WikiText-103 inputs.
Added: L3 hub ablation (the newly discovered natural-language coordination center)
and the L3H6-centered cluster from experiment 04.
"""

import torch
import numpy as np
from transformer_lens import HookedTransformer
from datasets import load_dataset
from pathlib import Path
from functools import partial
import json

MODEL_NAME = "pythia-70m"
SEQ_LEN = 128
SEED = 42
OUT_DIR = Path(__file__).parent / "results"

# --- Load model ---
print(f"Loading {MODEL_NAME}...")
model = HookedTransformer.from_pretrained(MODEL_NAME)
n_layers = model.cfg.n_layers
n_heads = model.cfg.n_heads
print(f"  {n_layers} layers x {n_heads} heads\n")

# --- Load natural language data ---
print("Loading WikiText-103...")
dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
np.random.seed(SEED)
indices = np.random.choice(len(dataset), size=2000, replace=False)

eval_tokens = []
eval_texts = []
for idx in indices:
    text = dataset[int(idx)]["text"]
    tokens = model.to_tokens(text, prepend_bos=True)
    if tokens.shape[1] >= SEQ_LEN:
        eval_tokens.append(tokens[0, :SEQ_LEN].unsqueeze(0))
        eval_texts.append(text[:200])
    if len(eval_tokens) >= 100:
        break

token_batch = torch.cat(eval_tokens, dim=0)
print(f"  {len(eval_tokens)} sequences of length {SEQ_LEN}\n")

# --- Ablation groups ---
ABLATION_GROUPS = {
    "none (baseline)": [],
    "early arm (L0+L1)": [(l, h) for l in [0, 1] for h in range(8)],
    "late arm (L3+L4+L5)": [(l, h) for l in [3, 4, 5] for h in range(8)],
    "layer 0 only": [(0, h) for h in range(8)],
    "layer 3 only (nat-lang hub)": [(3, h) for h in range(8)],
    "layer 5 only": [(5, h) for h in range(8)],
    "L3H6 cluster (L2H6+L3H1+L3H5+L3H6+L4H7)": [(2, 6), (3, 1), (3, 5), (3, 6), (4, 7)],
    "random-token chain (L3H0+L4H2+L5H3)": [(3, 0), (4, 2), (5, 3)],
    "random 5 heads (control)": [(0, 2), (1, 6), (2, 4), (4, 0), (5, 1)],
}


def ablate_heads(z, hook, heads_to_ablate, layer):
    for (l, h) in heads_to_ablate:
        if l == layer:
            z[:, :, h, :] = 0.0
    return z


def evaluate(model, token_batch, heads_to_ablate):
    total_loss = 0.0
    total_tokens = 0
    correct = 0
    BATCH = 10

    for i in range(0, len(token_batch), BATCH):
        batch = token_batch[i:i+BATCH]

        fwd_hooks = []
        if heads_to_ablate:
            layers_involved = set(l for l, h in heads_to_ablate)
            for layer in layers_involved:
                hook_fn = partial(ablate_heads, heads_to_ablate=heads_to_ablate, layer=layer)
                fwd_hooks.append((f"blocks.{layer}.attn.hook_z", hook_fn))

        with torch.no_grad():
            if fwd_hooks:
                logits = model.run_with_hooks(batch, fwd_hooks=fwd_hooks)
            else:
                logits = model(batch)

        shift_logits = logits[:, :-1, :]
        shift_targets = batch[:, 1:]

        loss = torch.nn.functional.cross_entropy(
            shift_logits.reshape(-1, shift_logits.shape[-1]),
            shift_targets.reshape(-1),
            reduction="sum",
        )
        total_loss += loss.item()
        total_tokens += shift_targets.numel()

        preds = shift_logits.argmax(dim=-1)
        correct += (preds == shift_targets).sum().item()

    avg_loss = total_loss / total_tokens
    return {
        "avg_loss": round(avg_loss, 4),
        "perplexity": round(np.exp(avg_loss), 2),
        "accuracy": round(correct / total_tokens, 4),
    }


# --- Run ---
results = {}
for name, heads in ABLATION_GROUPS.items():
    print(f"Ablating: {name} ({len(heads)} heads)...")
    m = evaluate(model, token_batch, heads)
    results[name] = {"heads_ablated": len(heads), **m}
    print(f"  ppl: {m['perplexity']:.2f}  acc: {m['accuracy']:.4f}")

# --- Summary ---
baseline_ppl = results["none (baseline)"]["perplexity"]
print(f"\n{'='*80}")
print(f"{'Ablation Group':<45} {'Heads':>5} {'Perplexity':>12} {'Accuracy':>10}")
print(f"{'-'*80}")
for name, r in results.items():
    ratio = r["perplexity"] / baseline_ppl
    marker = "" if name == "none (baseline)" else f" ({ratio:.1f}x)"
    print(f"{name:<45} {r['heads_ablated']:>5} {r['perplexity']:>12.2f}{marker:<8} {r['accuracy']:>10.4f}")
print(f"{'='*80}")

with open(OUT_DIR / "05_natural_ablation.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {OUT_DIR / '05_natural_ablation.json'}")
