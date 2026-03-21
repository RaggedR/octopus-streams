"""
Octopus Streams — Experiment 8: Head Interpretation (Layer 3)

Directly examines what each of the 8 attention heads in Layer 3 computes.
This complements the SAE decomposition (Experiment 7) by looking at heads
individually rather than decomposing the residual stream into features.

For each head we ask:
  1. What does it attend to? (attention pattern analysis)
  2. What does its output mean? (direct logit attribution via W_U)
  3. How does it vary across domains? (per-domain output norms)

Outputs:
  results/08_attention_patterns.png   — mean attention pattern per head
  results/08_logit_attribution.png    — top promoted/suppressed tokens per head
  results/08_head_profiles.png        — per-domain output norms + head roles
  results/08_head_analysis.json       — machine-readable results
  results/08_head_analysis.txt        — human-readable summary
"""

import torch
import numpy as np
from transformer_lens import HookedTransformer
from datasets import load_dataset
from pathlib import Path
import json
import matplotlib.pyplot as plt

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME = "pythia-70m"
LAYER = 0
N_HEADS = 8
N_SAMPLES = 200
SEQ_LEN = 128
SEED = 42
BATCH_SIZE = 10
OUT_DIR = Path(__file__).parent / "results"
OUT_DIR.mkdir(exist_ok=True)
PREFIX = f"08_L{LAYER}_"  # layer-aware output prefix

DOMAIN_ORDER = ["random", "wiki", "poetry", "prose", "code"]
DOMAIN_LABELS = {
    "random": "Random Tokens",
    "wiki": "WikiText",
    "poetry": "Poetry",
    "prose": "Prose (PG19)",
    "code": "Python Code",
}
DOMAIN_COLOURS = {
    "random": "#7f7f7f",
    "wiki": "#1f77b4",
    "poetry": "#e377c2",
    "prose": "#2ca02c",
    "code": "#ff7f0e",
}


# ── Load model ────────────────────────────────────────────────────────────────
print(f"Loading {MODEL_NAME}...")
model = HookedTransformer.from_pretrained(MODEL_NAME)
print(f"  {model.cfg.n_layers} layers, {model.cfg.n_heads} heads/layer, "
      f"d_model={model.cfg.d_model}, d_head={model.cfg.d_head}")


# ── Domain loaders (same as experiments 06/07) ───────────────────────────────

def load_random_tokens(model):
    np.random.seed(SEED)
    tokens = np.random.randint(0, model.cfg.d_vocab, size=(N_SAMPLES, SEQ_LEN))
    return torch.tensor(tokens, dtype=torch.long), N_SAMPLES


def load_wiki_tokens(model):
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    np.random.seed(SEED + 10)
    indices = np.random.permutation(len(ds))
    tokens_list = []
    for idx in indices:
        text = ds[int(idx)]["text"]
        if not text or len(text.strip()) < 50:
            continue
        toks = model.to_tokens(text, prepend_bos=True)
        if toks.shape[1] >= SEQ_LEN:
            tokens_list.append(toks[0, :SEQ_LEN].unsqueeze(0))
        if len(tokens_list) >= N_SAMPLES:
            break
    return torch.cat(tokens_list, dim=0), len(tokens_list)


def load_poetry(model):
    ds = load_dataset("suayptalha/Poetry-Foundation-Poems", split="train")
    np.random.seed(SEED + 1)
    n_candidates = min(N_SAMPLES * 10, len(ds))
    indices = np.random.choice(len(ds), size=n_candidates, replace=False)
    tokens_list = []
    for idx in indices:
        text = ds[int(idx)]["Poem"]
        if not text or len(text.strip()) < 50:
            continue
        toks = model.to_tokens(text, prepend_bos=True)
        if toks.shape[1] >= SEQ_LEN:
            tokens_list.append(toks[0, :SEQ_LEN].unsqueeze(0))
        if len(tokens_list) >= N_SAMPLES:
            break
    return torch.cat(tokens_list, dim=0), len(tokens_list)


def load_prose_gutenberg(model):
    ds = load_dataset("sedthh/gutenberg_english", split="train", streaming=True)
    ds = ds.shuffle(seed=SEED, buffer_size=1000)
    tokens_list = []
    for example in ds:
        text = example["TEXT"]
        start = len(text) // 4
        chunk = text[start : start + 2000]
        if len(chunk) < 500:
            continue
        toks = model.to_tokens(chunk, prepend_bos=True)
        if toks.shape[1] >= SEQ_LEN:
            tokens_list.append(toks[0, :SEQ_LEN].unsqueeze(0))
        if len(tokens_list) >= N_SAMPLES:
            break
    return torch.cat(tokens_list, dim=0), len(tokens_list)


def load_code_python(model):
    ds = load_dataset(
        "Nan-Do/code-search-net-python", split="train", streaming=True
    )
    ds = ds.shuffle(seed=SEED + 2, buffer_size=1000)
    tokens_list = []
    for example in ds:
        code = example["code"]
        if not code or len(code.strip()) < 50:
            continue
        toks = model.to_tokens(code, prepend_bos=True)
        if toks.shape[1] >= SEQ_LEN:
            tokens_list.append(toks[0, :SEQ_LEN].unsqueeze(0))
        if len(tokens_list) >= N_SAMPLES:
            break
    return torch.cat(tokens_list, dim=0), len(tokens_list)


LOADERS = {
    "random": load_random_tokens,
    "wiki": load_wiki_tokens,
    "poetry": load_poetry,
    "prose": load_prose_gutenberg,
    "code": load_code_python,
}


# ── Collect activations and attention patterns ────────────────────────────────


def collect_head_data(model, token_batch, layer):
    """
    Collect per-head data from a batch of token sequences.

    Uses hook_z (per-head output before W_O, shape: batch, pos, n_heads, d_head)
    and manually applies W_O to get per-head contributions to the residual stream.

    Returns:
        attention_patterns: (n_samples, n_heads, seq_len, seq_len)
        head_outputs: (n_samples, seq_len, n_heads, d_model) — per-head after W_O
        head_norms: (n_samples, n_heads) — mean output norm per head per sample
    """
    pattern_hook = f"blocks.{layer}.attn.hook_pattern"
    z_hook = f"blocks.{layer}.attn.hook_z"

    # W_O: (n_heads, d_head, d_model) — per-head output projection
    W_O = model.blocks[layer].attn.W_O.detach().cpu().float()
    b_O = model.blocks[layer].attn.b_O.detach().cpu().float()

    all_patterns = []
    all_results = []
    all_norms = []

    for start in range(0, len(token_batch), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(token_batch))
        batch = token_batch[start:end]

        _, cache = model.run_with_cache(
            batch,
            names_filter=lambda name: name in (pattern_hook, z_hook),
        )

        # Attention patterns: (batch, n_heads, seq, seq)
        patterns = cache[pattern_hook].detach().cpu()
        all_patterns.append(patterns)

        # hook_z: (batch, pos, n_heads, d_head)
        z = cache[z_hook].detach().cpu().float()

        # Apply W_O per head: z[..., h, :] @ W_O[h] -> (batch, pos, d_model)
        # Result shape: (batch, pos, n_heads, d_model)
        b, s, nh, dh = z.shape
        results = torch.einsum("bshe,hed->bshd", z, W_O)
        # Add bias split equally across heads
        results = results + b_O / nh
        all_results.append(results)

        # Head norms: (batch, n_heads) — mean over positions
        norms = results.norm(dim=-1).mean(dim=1)  # (batch, n_heads)
        all_norms.append(norms)

        del cache

    return (
        torch.cat(all_patterns, dim=0),
        torch.cat(all_results, dim=0),
        torch.cat(all_norms, dim=0),
    )


# ── Load domain data ──────────────────────────────────────────────────────────
print("\nLoading domain data...")
domain_tokens = {}
for domain in DOMAIN_ORDER:
    print(f"  {DOMAIN_LABELS[domain]}...", end=" ", flush=True)
    token_batch, n = LOADERS[domain](model)
    domain_tokens[domain] = token_batch
    print(f"{n} samples")


# ── Main analysis ─────────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print(f"LAYER {LAYER}: Analysing {N_HEADS} heads")
print(f"{'=' * 70}")

domain_patterns = {}   # domain -> mean attention pattern (n_heads, seq, seq)
domain_norms = {}      # domain -> mean head norms (n_heads,)
domain_head_results = {}  # domain -> head_outputs for DLA (first batch only)
domain_head_norms_all = {}  # domain -> (n_samples, n_heads) norms

for domain in DOMAIN_ORDER:
    print(f"\n  Processing {DOMAIN_LABELS[domain]}...", flush=True)
    patterns, results, norms = collect_head_data(
        model, domain_tokens[domain], LAYER
    )

    # Mean attention pattern across all samples
    domain_patterns[domain] = patterns.mean(dim=0).numpy()  # (n_heads, seq, seq)

    # Mean head norms
    domain_norms[domain] = norms.mean(dim=0).numpy()  # (n_heads,)
    domain_head_norms_all[domain] = norms.numpy()  # (n_samples, n_heads)

    # Keep head results for first 20 samples (for DLA)
    domain_head_results[domain] = results[:20]  # (20, seq, n_heads, d_model)

    print(f"    Head norms: {', '.join(f'H{h}={domain_norms[domain][h]:.2f}' for h in range(N_HEADS))}")


# ── Analysis 1: Direct Logit Attribution ──────────────────────────────────────
print(f"\n{'─' * 70}")
print("Direct Logit Attribution: what tokens does each head promote?")
print(f"{'─' * 70}")

# Get the unembedding matrix W_U: (d_model, d_vocab)
W_U = model.W_U.detach().cpu().float()  # (d_model, d_vocab)

head_dla = {}  # head_idx -> {"top_promoted": [...], "top_suppressed": [...]}

# Use WikiText results for DLA (most natural)
wiki_results = domain_head_results["wiki"]  # (20, seq, n_heads, d_model)

for h in range(N_HEADS):
    # Get this head's mean output direction across all positions and samples
    head_out = wiki_results[:, :, h, :]  # (20, seq, d_model)
    mean_direction = head_out.mean(dim=(0, 1))  # (d_model,)

    # Project through unembedding to get logit contribution
    logit_contrib = mean_direction @ W_U  # (d_vocab,)

    # Top promoted tokens
    top_k = 15
    top_vals, top_ids = logit_contrib.topk(top_k)
    promoted = []
    for val, tid in zip(top_vals, top_ids):
        tok_str = model.tokenizer.decode([tid.item()]).replace("\n", "\\n")
        promoted.append({"token": tok_str, "logit": round(val.item(), 4)})

    # Top suppressed tokens
    bot_vals, bot_ids = logit_contrib.topk(top_k, largest=False)
    suppressed = []
    for val, tid in zip(bot_vals, bot_ids):
        tok_str = model.tokenizer.decode([tid.item()]).replace("\n", "\\n")
        suppressed.append({"token": tok_str, "logit": round(val.item(), 4)})

    head_dla[h] = {"top_promoted": promoted, "top_suppressed": suppressed}

    print(f"\n  L{LAYER}H{h}:")
    promoted_str = ", ".join(f"'{p['token']}'" for p in promoted[:5])
    suppressed_str = ", ".join(f"'{s['token']}'" for s in suppressed[:5])
    print(f"    Promotes:   {promoted_str}")
    print(f"    Suppresses: {suppressed_str}")


# ── Analysis 2: Attention pattern characterisation ────────────────────────────
print(f"\n{'─' * 70}")
print("Attention pattern analysis")
print(f"{'─' * 70}")

head_attn_profile = {}

for h in range(N_HEADS):
    # Use WikiText patterns
    pattern = domain_patterns["wiki"][h]  # (seq, seq) — mean over samples

    # Compute attention statistics
    # 1. Mean attention to self (diagonal)
    self_attn = np.diag(pattern).mean()

    # 2. Mean attention to position 0 (BOS)
    bos_attn = pattern[:, 0].mean()

    # 3. Mean attention to previous token
    prev_attn = np.mean([pattern[i, i - 1] for i in range(1, SEQ_LEN)])

    # 4. Attention entropy (how spread out is attention?)
    # Average entropy across destination positions
    eps = 1e-10
    entropies = []
    for pos in range(SEQ_LEN):
        # Only consider positions up to pos (causal mask)
        p = pattern[pos, :pos + 1]
        p = p / (p.sum() + eps)
        ent = -np.sum(p * np.log(p + eps))
        max_ent = np.log(pos + 1)
        entropies.append(ent / (max_ent + eps))  # normalized 0-1
    mean_entropy = np.mean(entropies)

    # 5. Positional bias: what's the average relative position attended to?
    avg_lookback = 0.0
    for dest in range(1, SEQ_LEN):
        for src in range(dest + 1):
            avg_lookback += pattern[dest, src] * (dest - src)
    avg_lookback /= (SEQ_LEN - 1)

    head_attn_profile[h] = {
        "self_attention": round(float(self_attn), 4),
        "bos_attention": round(float(bos_attn), 4),
        "prev_token_attention": round(float(prev_attn), 4),
        "entropy_normalized": round(float(mean_entropy), 4),
        "avg_lookback": round(float(avg_lookback), 4),
    }

    # Classify head type
    head_type = "other"
    if bos_attn > 0.3:
        head_type = "BOS-attending"
    elif prev_attn > 0.3:
        head_type = "previous-token"
    elif self_attn > 0.3:
        head_type = "self-attending"
    elif mean_entropy > 0.7:
        head_type = "broad/distributed"
    elif mean_entropy < 0.3:
        head_type = "narrow/focused"

    head_attn_profile[h]["type"] = head_type

    print(f"  L{LAYER}H{h}: type={head_type:>20s}  "
          f"self={self_attn:.3f}  bos={bos_attn:.3f}  "
          f"prev={prev_attn:.3f}  entropy={mean_entropy:.3f}  "
          f"lookback={avg_lookback:.1f}")


# ── Analysis 3: Per-domain head output comparison ─────────────────────────────
print(f"\n{'─' * 70}")
print("Per-domain head output norms")
print(f"{'─' * 70}")

# For each head, compute domain preference (ratio of max to min domain norm)
head_domain_ratios = {}
for h in range(N_HEADS):
    norms = [domain_norms[d][h] for d in DOMAIN_ORDER]
    max_norm = max(norms)
    min_norm = max(min(norms), 1e-8)
    ratio = max_norm / min_norm
    max_domain = DOMAIN_ORDER[np.argmax(norms)]
    min_domain = DOMAIN_ORDER[np.argmin(norms)]
    head_domain_ratios[h] = {
        "max_domain": max_domain,
        "min_domain": min_domain,
        "ratio": round(float(ratio), 2),
        "per_domain": {d: round(float(domain_norms[d][h]), 4) for d in DOMAIN_ORDER},
    }
    norms_str = "  ".join(
        f"{d[:4]}={domain_norms[d][h]:.3f}" for d in DOMAIN_ORDER
    )
    print(f"  L{LAYER}H{h}: ratio={ratio:.2f}x  "
          f"max={max_domain:>7s}  min={min_domain:>7s}  {norms_str}")


# ── Figure 1: Attention patterns ──────────────────────────────────────────────
print("\nGenerating figures...", flush=True)

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle(f"Layer {LAYER}: Mean Attention Patterns (WikiText)", fontsize=16, y=0.98)

# Show first 32 positions for readability
SHOW_LEN = 32

for h in range(N_HEADS):
    ax = axes[h // 4, h % 4]
    pattern = domain_patterns["wiki"][h][:SHOW_LEN, :SHOW_LEN]
    im = ax.imshow(pattern, cmap="Blues", aspect="auto", vmin=0, vmax=0.5)
    ax.set_title(f"L{LAYER}H{h} ({head_attn_profile[h]['type']})", fontsize=11)
    ax.set_xlabel("Source position")
    ax.set_ylabel("Dest position")

fig.colorbar(im, ax=axes, shrink=0.6, label="Attention weight")
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT_DIR / f"{PREFIX}attention_patterns.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved {PREFIX}attention_patterns.png")


# ── Figure 2: Direct Logit Attribution ────────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(20, 12))
fig.suptitle(f"Layer {LAYER}: Direct Logit Attribution (top promoted tokens)",
             fontsize=16, y=0.98)

for h in range(N_HEADS):
    ax = axes[h // 4, h % 4]
    promoted = head_dla[h]["top_promoted"][:10]
    tokens = [p["token"][:12] for p in promoted]
    logits = [p["logit"] for p in promoted]

    bars = ax.barh(range(len(tokens)), logits, color="#2196F3")
    ax.set_yticks(range(len(tokens)))
    ax.set_yticklabels([f"'{t}'" for t in tokens], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Logit contribution")
    ax.set_title(f"L{LAYER}H{h}", fontsize=12, fontweight="bold")
    ax.axvline(x=0, color="black", linewidth=0.5)

fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT_DIR / f"{PREFIX}logit_attribution.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved {PREFIX}logit_attribution.png")


# ── Figure 3: Per-domain head profiles ────────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle(f"Layer {LAYER}: Per-Head Output Norms by Domain", fontsize=16, y=0.98)

for h in range(N_HEADS):
    ax = axes[h // 4, h % 4]
    norms = [domain_norms[d][h] for d in DOMAIN_ORDER]
    colours = [DOMAIN_COLOURS[d] for d in DOMAIN_ORDER]
    labels = [DOMAIN_LABELS[d] for d in DOMAIN_ORDER]

    bars = ax.bar(range(len(DOMAIN_ORDER)), norms, color=colours)
    ax.set_xticks(range(len(DOMAIN_ORDER)))
    ax.set_xticklabels([l.replace(" ", "\n") for l in labels], fontsize=8)
    ax.set_ylabel("Mean output norm")
    ratio = head_domain_ratios[h]["ratio"]
    ax.set_title(f"L{LAYER}H{h} (ratio={ratio:.1f}×)", fontsize=12, fontweight="bold")

fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT_DIR / f"{PREFIX}head_profiles.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved {PREFIX}head_profiles.png")


# ── Figure 4: Attention pattern comparison across domains ─────────────────────
# For each head, show how its attention differs between domains
fig, axes = plt.subplots(N_HEADS, len(DOMAIN_ORDER), figsize=(25, 20))
fig.suptitle(f"Layer {LAYER}: Attention Patterns Across All Domains "
             f"(first {SHOW_LEN} positions)", fontsize=16, y=0.99)

for h in range(N_HEADS):
    for di, domain in enumerate(DOMAIN_ORDER):
        ax = axes[h, di]
        pattern = domain_patterns[domain][h][:SHOW_LEN, :SHOW_LEN]
        ax.imshow(pattern, cmap="Blues", aspect="auto", vmin=0, vmax=0.5)
        if h == 0:
            ax.set_title(DOMAIN_LABELS[domain], fontsize=11, fontweight="bold")
        if di == 0:
            ax.set_ylabel(f"H{h}", fontsize=12, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(OUT_DIR / f"{PREFIX}attention_domain_grid.png", dpi=100, bbox_inches="tight")
plt.close(fig)
print(f"  Saved {PREFIX}attention_domain_grid.png")


# ── Save JSON ─────────────────────────────────────────────────────────────────
analysis = {
    "layer": LAYER,
    "n_heads": N_HEADS,
    "heads": {},
}

for h in range(N_HEADS):
    analysis["heads"][str(h)] = {
        "attention_profile": head_attn_profile[h],
        "domain_norms": head_domain_ratios[h],
        "logit_attribution": head_dla[h],
    }

with open(OUT_DIR / f"{PREFIX}head_analysis.json", "w") as f:
    json.dump(analysis, f, indent=2)
print(f"\n  Saved {PREFIX}head_analysis.json")


# ── Human-readable summary ───────────────────────────────────────────────────
lines = []
lines.append(f"{'=' * 80}")
lines.append(f"LAYER {LAYER}: HEAD-BY-HEAD INTERPRETATION")
lines.append(f"{'=' * 80}")
lines.append("")

for h in range(N_HEADS):
    profile = head_attn_profile[h]
    domains = head_domain_ratios[h]
    dla = head_dla[h]

    lines.append(f"── L{LAYER}H{h} {'─' * 60}")
    lines.append(f"  Type: {profile['type']}")
    lines.append(f"  Attention: self={profile['self_attention']:.3f}  "
                 f"bos={profile['bos_attention']:.3f}  "
                 f"prev={profile['prev_token_attention']:.3f}  "
                 f"entropy={profile['entropy_normalized']:.3f}  "
                 f"lookback={profile['avg_lookback']:.1f}")
    lines.append(f"  Domain preference: {domains['ratio']:.1f}× "
                 f"(max={domains['max_domain']}, min={domains['min_domain']})")

    norms_str = "  ".join(
        f"{d}={domains['per_domain'][d]:.3f}" for d in DOMAIN_ORDER
    )
    lines.append(f"  Per-domain norms: {norms_str}")

    promoted = ", ".join(f"'{p['token']}' ({p['logit']:.3f})" for p in dla["top_promoted"][:5])
    suppressed = ", ".join(f"'{s['token']}' ({s['logit']:.3f})" for s in dla["top_suppressed"][:5])
    lines.append(f"  Promotes:   {promoted}")
    lines.append(f"  Suppresses: {suppressed}")
    lines.append("")

txt_path = OUT_DIR / f"{PREFIX}head_analysis.txt"
with open(txt_path, "w") as f:
    f.write("\n".join(lines))
print(f"  Saved {txt_path}")


# ── Console summary ──────────────────────────────────────────────────────────
print(f"\n{'=' * 80}")
print(f"LAYER {LAYER} HEAD INTERPRETATION SUMMARY")
print(f"{'=' * 80}")

for h in range(N_HEADS):
    profile = head_attn_profile[h]
    domains = head_domain_ratios[h]
    promoted = head_dla[h]["top_promoted"][:3]
    promoted_str = ", ".join(f"'{p['token']}'" for p in promoted)

    print(f"\n  L{LAYER}H{h}: {profile['type']}")
    print(f"    Domain ratio: {domains['ratio']:.1f}× "
          f"(max={domains['max_domain']}, min={domains['min_domain']})")
    print(f"    Promotes: {promoted_str}")

print(f"\n{'=' * 80}")
print("Done!")
