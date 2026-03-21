"""
Octopus Streams — Experiment 3: Ablation

Zero out groups of heads and measure what breaks.

Groups (from PCA findings):
  - "Early arm": L0 + L1 (16 heads) — left side of PC1
  - "Late arm": L3 + L4 + L5 (24 heads) — right side of PC1
  - "Bridge": L2 (8 heads) — split between early and late
  - Individual chain: L3H0, L4H2, L5H3 — the strongest correlated triplet

We measure:
  - Perplexity on natural text (overall capability)
  - Next-token accuracy (can it still predict?)
  - Per-token loss distribution (does damage concentrate on specific token types?)
"""

import torch
import numpy as np
from transformer_lens import HookedTransformer
from pathlib import Path
from functools import partial
import json

# --- Config ---
MODEL_NAME = "pythia-70m"
OUT_DIR = Path(__file__).parent / "results"
OUT_DIR.mkdir(exist_ok=True)

# --- Test sentences ---
# Diverse inputs to test different capabilities
TEST_TEXTS = [
    # Factual / knowledge
    "The capital of France is",
    "Water boils at a temperature of",
    "The largest planet in our solar system is",
    # Syntactic completion
    "The cat sat on the mat and the dog sat on the",
    "She gave him the book that he had been looking for since",
    "Neither the students nor the teacher was able to",
    # Pattern / repetition
    "A B C D E F G H I J K",
    "1 2 3 4 5 6 7 8 9 10 11",
    "red blue red blue red blue red",
    # Reasoning
    "If all cats are animals and all animals breathe, then cats",
    "John is taller than Mary. Mary is taller than Sue. The tallest person is",
    # Code-like
    "def fibonacci(n):\n    if n <= 1:\n        return",
    "for i in range(10):\n    print(",
    # Longer context
    "Once upon a time, in a kingdom far away, there lived a young princess who dreamed of exploring the world beyond the castle walls. One morning, she decided to",
    "The experiment required careful preparation. First, the researchers calibrated the instruments. Then, they measured the initial temperature. Finally, they",
]

# --- Ablation groups ---
ABLATION_GROUPS = {
    "none (baseline)": [],
    "early arm (L0+L1)": [(l, h) for l in [0, 1] for h in range(8)],
    "late arm (L3+L4+L5)": [(l, h) for l in [3, 4, 5] for h in range(8)],
    "bridge (L2)": [(2, h) for h in range(8)],
    "correlated chain (L3H0+L4H2+L5H3)": [(3, 0), (4, 2), (5, 3)],
    "random 3 heads": [(0, 3), (2, 5), (4, 7)],  # control: 3 random heads
    "all layer 0": [(0, h) for h in range(8)],
    "all layer 5": [(5, h) for h in range(8)],
}


def ablate_heads(z, hook, heads_to_ablate, layer):
    """Hook function: zero out specified heads' outputs."""
    for (l, h) in heads_to_ablate:
        if l == layer:
            z[:, :, h, :] = 0.0
    return z


def evaluate_with_ablation(model, texts, heads_to_ablate):
    """Run model with specified heads zeroed out, return metrics."""
    tokens_list = [model.to_tokens(t) for t in texts]

    total_loss = 0.0
    total_tokens = 0
    correct_predictions = 0
    total_predictions = 0
    completions = []

    for tokens in tokens_list:
        # Set up hooks for ablation
        fwd_hooks = []
        if heads_to_ablate:
            layers_involved = set(l for l, h in heads_to_ablate)
            for layer in layers_involved:
                hook_fn = partial(ablate_heads, heads_to_ablate=heads_to_ablate, layer=layer)
                fwd_hooks.append((f"blocks.{layer}.attn.hook_z", hook_fn))

        with torch.no_grad():
            if fwd_hooks:
                logits = model.run_with_hooks(tokens, fwd_hooks=fwd_hooks)
            else:
                logits = model(tokens)

        # Loss: cross-entropy on next-token prediction
        # logits shape: (1, seq_len, vocab_size)
        # Shift: predict token[i+1] from logits[i]
        shift_logits = logits[0, :-1, :]  # (seq_len-1, vocab)
        shift_targets = tokens[0, 1:]      # (seq_len-1,)

        loss = torch.nn.functional.cross_entropy(shift_logits, shift_targets, reduction="sum")
        total_loss += loss.item()
        total_tokens += shift_targets.shape[0]

        # Accuracy: does argmax match?
        preds = shift_logits.argmax(dim=-1)
        correct_predictions += (preds == shift_targets).sum().item()
        total_predictions += shift_targets.shape[0]

        # Generate 10-token completion for qualitative comparison
        if fwd_hooks:
            # Manual generation with hooks
            gen_tokens = tokens.clone()
            for _ in range(10):
                out = model.run_with_hooks(gen_tokens, fwd_hooks=fwd_hooks)
                next_token = out[0, -1, :].argmax(dim=-1, keepdim=True).unsqueeze(0)
                gen_tokens = torch.cat([gen_tokens, next_token], dim=1)
        else:
            gen_tokens = tokens.clone()
            for _ in range(10):
                out = model(gen_tokens)
                next_token = out[0, -1, :].argmax(dim=-1, keepdim=True).unsqueeze(0)
                gen_tokens = torch.cat([gen_tokens, next_token], dim=1)

        completion = model.to_string(gen_tokens[0, tokens.shape[1]:])
        completions.append(completion)

    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    accuracy = correct_predictions / total_predictions

    return {
        "avg_loss": avg_loss,
        "perplexity": perplexity,
        "accuracy": accuracy,
        "completions": completions,
    }


# --- Run ablations ---
print(f"Loading {MODEL_NAME}...")
model = HookedTransformer.from_pretrained(MODEL_NAME)
print(f"  {model.cfg.n_layers} layers x {model.cfg.n_heads} heads = {model.cfg.n_layers * model.cfg.n_heads} total\n")

results = {}

for group_name, heads in ABLATION_GROUPS.items():
    print(f"Ablating: {group_name} ({len(heads)} heads)...")
    metrics = evaluate_with_ablation(model, TEST_TEXTS, heads)
    results[group_name] = {
        "heads_ablated": len(heads),
        "perplexity": round(metrics["perplexity"], 2),
        "accuracy": round(metrics["accuracy"], 4),
        "avg_loss": round(metrics["avg_loss"], 4),
    }
    print(f"  perplexity: {metrics['perplexity']:.2f}  accuracy: {metrics['accuracy']:.4f}")

    # Show a few completions
    for i in [0, 3, 6, 13]:
        prompt = TEST_TEXTS[i][:50]
        print(f"    \"{prompt}...\" → \"{metrics['completions'][i]}\"")
    print()

# --- Summary table ---
print("=" * 75)
print(f"{'Ablation Group':<35} {'Heads':>5} {'Perplexity':>12} {'Accuracy':>10}")
print("-" * 75)
baseline_ppl = results["none (baseline)"]["perplexity"]
for name, r in results.items():
    ppl_ratio = r["perplexity"] / baseline_ppl
    marker = "" if name == "none (baseline)" else f" ({ppl_ratio:.1f}x)"
    print(f"{name:<35} {r['heads_ablated']:>5} {r['perplexity']:>12.2f}{marker:<8} {r['accuracy']:>10.4f}")
print("=" * 75)

# --- Save ---
with open(OUT_DIR / "03_ablation.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to {OUT_DIR / '03_ablation.json'}")
