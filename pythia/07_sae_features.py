"""
Octopus Streams — Experiment 7: SAE Feature Analysis

Train sparse autoencoders on Pythia-70m's residual stream at layers 0, 3, 5
to decompose what features each layer computes.

Prior experiments established:
  - Layer 0 is the irreplaceable "spinal cord" (17.8× perplexity when ablated)
  - Layer 3 is a natural-language coordination hub (4.5×)
  - Different domains (wiki, poetry, prose, code) reconfigure head coalitions

This experiment asks: what monosemantic features do those critical layers compute?

Outputs:
  results/07_sae_layer{0,3,5}_weights.pt
  results/07_sae_training_history.json
  results/07_feature_census.json
  results/07_top_features.json
  results/07_sae_training_curves.png
  results/07_domain_feature_heatmap.png
  results/07_feature_specificity.png
  results/07_layer0_critical_features.png
  results/07_layer3_wiki_vs_code.png
"""

import torch
import torch.nn as nn
import numpy as np
from transformer_lens import HookedTransformer
from datasets import load_dataset
from pathlib import Path
import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME = "pythia-70m"
D_MODEL = 512
EXPANSION = 8
N_FEATURES = D_MODEL * EXPANSION  # 4096

SAE_LAYERS = [0, 3, 5]

# SAE training
N_TRAIN_SEQ = 2000  # WikiText sequences for SAE training
SEQ_LEN = 128
SAE_STEPS = 50_000
SAE_BATCH = 128  # activation vectors per SAE training step
SAE_LR = 3e-4
TOP_K = 50  # number of active features per input (guarantees L0=50)
LOG_EVERY = 1000

# Domain analysis
N_SAMPLES = 200  # samples per domain (matching exps 01–06)
BATCH_SIZE = 10  # model forward-pass batch size
SEED = 42

OUT_DIR = Path(__file__).parent / "results"
OUT_DIR.mkdir(exist_ok=True)

# Domain definitions (reused from exp 06)
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

# ── Device ────────────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")


# ── Sparse Autoencoder ───────────────────────────────────────────────────────


class SparseAutoencoder(nn.Module):
    """TopK sparse autoencoder.

    encode: pre = W_enc @ (x - b_dec) + b_enc
            h   = TopK(ReLU(pre), k)   — keep only the top-k activations
    decode: x_hat = W_dec @ h + b_dec

    TopK guarantees exact sparsity (L0 = k) by construction, avoiding the
    L1 tuning pitfalls of standard ReLU SAEs.  Decoder columns are
    constrained to unit norm after each optimiser step.
    """

    def __init__(self, d_model, n_features, k=TOP_K):
        super().__init__()
        self.d_model = d_model
        self.n_features = n_features
        self.k = k

        self.encoder = nn.Linear(d_model, n_features)
        self.decoder = nn.Linear(n_features, d_model)

        # Tied initialisation: encoder ≈ transpose of decoder
        nn.init.kaiming_uniform_(self.decoder.weight)
        self.encoder.weight.data = self.decoder.weight.data.T.clone()
        nn.init.zeros_(self.encoder.bias)
        nn.init.zeros_(self.decoder.bias)

        self._constrain_decoder()

    def _constrain_decoder(self):
        """Project decoder columns to unit norm."""
        with torch.no_grad():
            norms = self.decoder.weight.norm(dim=0, keepdim=True).clamp(min=1e-8)
            self.decoder.weight.div_(norms)

    def forward(self, x):
        x_centered = x - self.decoder.bias
        pre_acts = self.encoder(x_centered)

        # TopK selection with straight-through gradient
        topk_values, topk_indices = pre_acts.topk(self.k, dim=-1)
        topk_values = torch.relu(topk_values)  # ensure non-negative

        h = torch.zeros_like(pre_acts)
        h.scatter_(-1, topk_indices, topk_values)

        x_hat = self.decoder(h)
        return x_hat, h


# ── Load model ────────────────────────────────────────────────────────────────
print(f"\nLoading {MODEL_NAME}...")
model = HookedTransformer.from_pretrained(MODEL_NAME)
n_layers = model.cfg.n_layers
n_heads = model.cfg.n_heads
print(f"  {n_layers} layers, d_model={model.cfg.d_model}\n")


# ── Data loaders ──────────────────────────────────────────────────────────────


def load_wikitext_tokens(model, n_sequences, seq_len, seed):
    """Load WikiText-103 sequences as a token batch."""
    print(f"  Loading WikiText-103 ({n_sequences} sequences, {seq_len} tokens each)...")
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")

    np.random.seed(seed)
    indices = np.random.permutation(len(ds))

    tokens_list = []
    for idx in indices:
        text = ds[int(idx)]["text"]
        if not text or len(text.strip()) < 50:
            continue
        toks = model.to_tokens(text, prepend_bos=True)
        if toks.shape[1] >= seq_len:
            tokens_list.append(toks[0, :seq_len].unsqueeze(0))
        if len(tokens_list) >= n_sequences:
            break

    actual = len(tokens_list)
    print(f"    Got {actual}/{n_sequences} sequences")
    return torch.cat(tokens_list, dim=0)


def load_random_tokens(model):
    """Generate random token sequences."""
    print("  Generating random token sequences...")
    np.random.seed(SEED)
    vocab_size = model.cfg.d_vocab
    tokens = np.random.randint(0, vocab_size, size=(N_SAMPLES, SEQ_LEN))
    return torch.tensor(tokens, dtype=torch.long), N_SAMPLES


def load_wiki_tokens(model):
    """Load WikiText samples for domain analysis (separate seed from training)."""
    tokens = load_wikitext_tokens(model, N_SAMPLES, SEQ_LEN, SEED + 10)
    return tokens, len(tokens)


def load_poetry(model):
    """Load poetry samples from Poetry Foundation dataset."""
    print("  Loading Poetry Foundation dataset...")
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

    actual = len(tokens_list)
    print(f"    Yield: {actual}/{N_SAMPLES} poems")
    if actual < 50:
        raise ValueError(f"Too few poetry samples ({actual}). Need at least 50.")
    return torch.cat(tokens_list, dim=0), actual


def load_prose_gutenberg(model):
    """Load narrative prose from Project Gutenberg (streaming)."""
    print("  Loading Gutenberg English dataset (streaming)...")
    ds = load_dataset("sedthh/gutenberg_english", split="train", streaming=True)
    ds = ds.shuffle(seed=SEED, buffer_size=1000)

    tokens_list = []
    texts_tried = 0

    for example in ds:
        text = example["TEXT"]
        texts_tried += 1
        start = len(text) // 4
        chunk = text[start : start + 2000]
        if len(chunk) < 500:
            continue
        toks = model.to_tokens(chunk, prepend_bos=True)
        if toks.shape[1] >= SEQ_LEN:
            tokens_list.append(toks[0, :SEQ_LEN].unsqueeze(0))
        if len(tokens_list) >= N_SAMPLES:
            break
        if texts_tried % 100 == 0:
            print(f"    ... tried {texts_tried} texts, got {len(tokens_list)} samples")

    actual = len(tokens_list)
    print(f"    Yield: {actual}/{N_SAMPLES} from {texts_tried} texts")
    return torch.cat(tokens_list, dim=0), actual


def load_code_python(model):
    """Load Python code from CodeSearchNet (streaming)."""
    print("  Loading CodeSearchNet Python (streaming)...")
    ds = load_dataset(
        "Nan-Do/code-search-net-python", split="train", streaming=True
    )
    ds = ds.shuffle(seed=SEED + 2, buffer_size=1000)

    tokens_list = []
    funcs_tried = 0

    for example in ds:
        code = example["code"]
        funcs_tried += 1
        if not code or len(code.strip()) < 50:
            continue
        toks = model.to_tokens(code, prepend_bos=True)
        if toks.shape[1] >= SEQ_LEN:
            tokens_list.append(toks[0, :SEQ_LEN].unsqueeze(0))
        if len(tokens_list) >= N_SAMPLES:
            break
        if funcs_tried % 500 == 0:
            print(f"    ... tried {funcs_tried} functions, got {len(tokens_list)} samples")

    actual = len(tokens_list)
    print(f"    Yield: {actual}/{N_SAMPLES} from {funcs_tried} functions")
    return torch.cat(tokens_list, dim=0), actual


DOMAIN_LOADERS = {
    "random": load_random_tokens,
    "wiki": load_wiki_tokens,
    "poetry": load_poetry,
    "prose": load_prose_gutenberg,
    "code": load_code_python,
}


# ── Activation collection ────────────────────────────────────────────────────


def collect_activations(model, token_batch, layer, batch_size=20):
    """Run tokens through model, collect residual stream at a layer.

    Returns: (n_sequences × seq_len, d_model) tensor on CPU.
    """
    hook_name = f"blocks.{layer}.hook_resid_post"
    n_seq = len(token_batch)
    all_acts = []

    for start in range(0, n_seq, batch_size):
        end = min(start + batch_size, n_seq)
        batch = token_batch[start:end]

        _, cache = model.run_with_cache(
            batch,
            names_filter=lambda name, hn=hook_name: name == hn,
        )

        acts = cache[hook_name].detach().cpu()  # (batch, seq_len, d_model)
        acts = acts.reshape(-1, acts.shape[-1])  # (batch * seq_len, d_model)
        all_acts.append(acts)

        del cache

        if (start // batch_size) % 10 == 0:
            print(f"    Layer {layer}: {end}/{n_seq} sequences")

    return torch.cat(all_acts, dim=0).float()


# ── Feature analysis helpers ─────────────────────────────────────────────────


def get_feature_activations(sae, activation_vectors, act_scale=1.0):
    """Run activation vectors through SAE, return feature activations.

    Normalises by act_scale (from training) so the SAE sees the same
    distribution it was trained on.
    """
    sae.eval()
    all_h = []
    chunk_size = 4096

    with torch.no_grad():
        for start in range(0, len(activation_vectors), chunk_size):
            end = min(start + chunk_size, len(activation_vectors))
            x = activation_vectors[start:end] / act_scale
            _, h = sae(x)
            all_h.append(h.cpu())

    return torch.cat(all_h, dim=0)


def find_max_activating_tokens(sae, model, token_batch, layer, top_k=20,
                               act_scale=1.0):
    """Find tokens that maximally activate the top-k features.

    Returns dict: feat_idx -> {mean_activation, top_tokens: [(str, value)]}
    """
    hook_name = f"blocks.{layer}.hook_resid_post"
    all_acts = []
    all_tokens = []

    for start in range(0, len(token_batch), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(token_batch))
        batch = token_batch[start:end]

        _, cache = model.run_with_cache(
            batch,
            names_filter=lambda name, hn=hook_name: name == hn,
        )

        acts = cache[hook_name].detach().cpu()
        toks = batch.cpu()
        b, s, d = acts.shape
        all_acts.append(acts.reshape(b * s, d))
        all_tokens.append(toks.reshape(b * s))

        del cache

    acts_flat = torch.cat(all_acts, dim=0).float()
    tokens_flat = torch.cat(all_tokens, dim=0)

    feature_acts = get_feature_activations(sae, acts_flat, act_scale=act_scale)
    mean_acts = feature_acts.mean(dim=0)
    top_features = mean_acts.argsort(descending=True)[:top_k]

    result = {}
    for feat_idx in top_features:
        fi = feat_idx.item()
        feat_vals = feature_acts[:, fi]
        top_positions = feat_vals.argsort(descending=True)[:5]

        examples = []
        for pos in top_positions:
            token_id = tokens_flat[pos.item()].item()
            token_str = model.tokenizer.decode([token_id])
            examples.append((token_str, round(feat_vals[pos.item()].item(), 4)))

        result[fi] = {
            "mean_activation": round(mean_acts[fi].item(), 6),
            "top_tokens": examples,
        }

    return result, mean_acts


# ── SAE training ──────────────────────────────────────────────────────────────


def train_sae(activation_pool, layer):
    """Train a TopK sparse autoencoder on a pool of activation vectors.

    Normalises activations to unit variance for stable training.
    Loss is pure MSE (no L1 needed — TopK guarantees L0 = k).

    Returns (sae, history, act_scale).
    """
    # Normalise to unit variance for stable training
    act_scale = activation_pool.std().item()
    activation_pool = activation_pool / act_scale
    print(f"    Activation scale: {act_scale:.4f} (normalised to unit variance)")

    n_vectors = len(activation_pool)
    print(f"\n  Training SAE for layer {layer} ({n_vectors} vectors, "
          f"{SAE_STEPS} steps, k={TOP_K})...")

    sae = SparseAutoencoder(D_MODEL, N_FEATURES, k=TOP_K).to(DEVICE)
    optimizer = torch.optim.Adam(sae.parameters(), lr=SAE_LR)

    history = {"loss": [], "mse": [], "l0": [], "step": []}
    t0 = time.time()

    for step in range(SAE_STEPS):
        idx = np.random.randint(0, n_vectors, size=SAE_BATCH)
        x = activation_pool[idx].to(DEVICE)

        x_hat, h = sae(x)
        mse = (x - x_hat).pow(2).mean()

        optimizer.zero_grad()
        mse.backward()
        optimizer.step()

        sae._constrain_decoder()

        if (step + 1) % LOG_EVERY == 0:
            l0 = (h > 0).float().sum(dim=1).mean().item()
            elapsed = time.time() - t0

            history["loss"].append(mse.item())
            history["mse"].append(mse.item())
            history["l0"].append(l0)
            history["step"].append(step + 1)

            print(
                f"    Step {step+1:>6}/{SAE_STEPS}  MSE={mse.item():.6f}  "
                f"L0={l0:.0f}  ({elapsed:.0f}s)"
            )

    # Final evaluation on larger sample
    sae.eval()
    with torch.no_grad():
        eval_idx = np.random.randint(0, n_vectors, size=min(4096, n_vectors))
        x = activation_pool[eval_idx].to(DEVICE)
        x_hat, h = sae(x)
        final_l0 = (h > 0).float().sum(dim=1).mean().item()
        n_dead = ((h > 0).float().sum(dim=0) == 0).sum().item()
        frac_dead = n_dead / N_FEATURES
        final_mse = (x - x_hat).pow(2).mean().item()

    print(f"    Final: L0={final_l0:.0f}  MSE={final_mse:.4f}  "
          f"dead={n_dead}/{N_FEATURES} ({frac_dead:.1%})")

    if frac_dead > 0.5:
        print("    ⚠ WARNING: >50% dead features! Consider increasing k.")

    return sae.cpu(), history, act_scale


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1: Train SAEs
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("PHASE 1: Train Sparse Autoencoders")
print("=" * 70)

wiki_train_tokens = load_wikitext_tokens(model, N_TRAIN_SEQ, SEQ_LEN, SEED + 100)
print(f"  Training pool: {len(wiki_train_tokens)} sequences × {SEQ_LEN} tokens\n")

trained_saes = {}
all_histories = {}
act_scales = {}  # per-layer normalisation factor

for layer in SAE_LAYERS:
    print(f"\n{'─' * 60}")
    print(f"Layer {layer}")
    print(f"{'─' * 60}")

    print("  Collecting residual stream activations...")
    act_pool = collect_activations(model, wiki_train_tokens, layer, batch_size=20)
    print(f"    Activation pool: {act_pool.shape}")

    sae, history, act_scale = train_sae(act_pool, layer)

    weight_path = OUT_DIR / f"07_sae_layer{layer}_weights.pt"
    torch.save(
        {
            "state_dict": sae.state_dict(),
            "d_model": D_MODEL,
            "n_features": N_FEATURES,
            "layer": layer,
            "top_k": TOP_K,
            "steps": SAE_STEPS,
            "act_scale": act_scale,
        },
        weight_path,
    )
    print(f"    Saved weights to {weight_path}")

    trained_saes[layer] = sae
    all_histories[layer] = history
    act_scales[layer] = act_scale

    del act_pool

with open(OUT_DIR / "07_sae_training_history.json", "w") as f:
    json.dump({str(k): v for k, v in all_histories.items()}, f, indent=2)
print(f"\nSaved training history to {OUT_DIR / '07_sae_training_history.json'}")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2: Domain Feature Analysis
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PHASE 2: Domain Feature Analysis")
print("=" * 70)

# Load domain data
domain_tokens = {}
for domain in DOMAIN_ORDER:
    print(f"\n═══ {DOMAIN_LABELS[domain]} ═══")
    loader_fn = DOMAIN_LOADERS[domain]
    token_batch, n_actual = loader_fn(model)
    domain_tokens[domain] = token_batch
    print(f"  → {n_actual} samples loaded")

# Collect feature activations: domains × layers
# domain_features[domain][layer] = mean feature activation vector (n_features,)
domain_features = {d: {} for d in DOMAIN_ORDER}
top_features_info = {}

for layer in SAE_LAYERS:
    sae = trained_saes[layer]
    sae.eval()
    scale = act_scales[layer]
    print(f"\n  Analysing layer {layer} features across domains (scale={scale:.4f})...")

    for domain in DOMAIN_ORDER:
        token_batch = domain_tokens[domain]
        acts = collect_activations(model, token_batch, layer, batch_size=BATCH_SIZE)
        feat_acts = get_feature_activations(sae, acts, act_scale=scale)
        mean_act = feat_acts.mean(dim=0).numpy()
        domain_features[domain][layer] = mean_act

        frac_alive = (feat_acts > 0).float().mean(dim=0)
        n_alive = (frac_alive > 0.01).sum().item()
        print(
            f"    {DOMAIN_LABELS[domain]:>20s}: "
            f"{n_alive} features alive (>1% activation rate)"
        )

    # Max-activating tokens (using wiki data as reference)
    print(f"  Finding max-activating tokens for layer {layer}...")
    top_info, _ = find_max_activating_tokens(
        sae, model, domain_tokens["wiki"], layer, top_k=20,
        act_scale=scale,
    )
    top_features_info[layer] = top_info


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3: Feature Census
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PHASE 3: Feature Census")
print("=" * 70)

ALIVE_THRESHOLD = 1e-3

census = {}
for layer in SAE_LAYERS:
    acts_matrix = np.stack([domain_features[d][layer] for d in DOMAIN_ORDER])

    max_per_feat = acts_matrix.max(axis=0)

    dead = max_per_feat < ALIVE_THRESHOLD
    alive_mask = ~dead

    activated_in = (acts_matrix > ALIVE_THRESHOLD).sum(axis=0)

    universal = alive_mask & (activated_in == len(DOMAIN_ORDER))
    domain_specific = alive_mask & (activated_in == 1)
    shared = alive_mask & (activated_in >= 2) & (activated_in < len(DOMAIN_ORDER))

    n_alive = int(alive_mask.sum())
    census[layer] = {
        "total": N_FEATURES,
        "dead": int(dead.sum()),
        "alive": n_alive,
        "universal": int(universal.sum()),
        "shared": int(shared.sum()),
        "domain_specific": int(domain_specific.sum()),
        "pct_dead": round(float(dead.mean()) * 100, 1),
        "pct_universal": round(
            float(universal.sum()) / max(n_alive, 1) * 100, 1
        ),
        "pct_domain_specific": round(
            float(domain_specific.sum()) / max(n_alive, 1) * 100, 1
        ),
    }

    c = census[layer]
    print(f"\n  Layer {layer}:")
    print(f"    Alive:           {c['alive']:>6} / {N_FEATURES}")
    print(f"    Dead:            {c['dead']:>6} ({c['pct_dead']}%)")
    print(f"    Universal:       {c['universal']:>6} ({c['pct_universal']}% of alive)")
    print(f"    Shared (2–4):    {c['shared']:>6}")
    print(f"    Domain-specific: {c['domain_specific']:>6} ({c['pct_domain_specific']}% of alive)")

with open(OUT_DIR / "07_feature_census.json", "w") as f:
    json.dump({str(k): v for k, v in census.items()}, f, indent=2)

serialisable_top = {}
for layer, info in top_features_info.items():
    serialisable_top[str(layer)] = {str(k): v for k, v in info.items()}
with open(OUT_DIR / "07_top_features.json", "w") as f:
    json.dump(serialisable_top, f, indent=2)

print(f"\nSaved census to {OUT_DIR / '07_feature_census.json'}")
print(f"Saved top features to {OUT_DIR / '07_top_features.json'}")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 4: Figures
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PHASE 4: Generate Figures")
print("=" * 70)

# ── Figure 1: Training Curves ────────────────────────────────────────────────

print("\nGenerating Figure 1: Training curves...")
fig, axes = plt.subplots(len(SAE_LAYERS), 2, figsize=(14, 4 * len(SAE_LAYERS)))

for row, layer in enumerate(SAE_LAYERS):
    h = all_histories[layer]
    steps = h["step"]

    # Left: MSE loss
    ax = axes[row, 0]
    ax.plot(steps, h["mse"], color="#1f77b4", linewidth=1.5)
    ax.set_ylabel("MSE", fontsize=10)
    ax.set_title(f"Layer {layer} — Reconstruction Loss (TopK, k={TOP_K})", fontsize=11)
    ax.set_xlabel("Step", fontsize=10)
    ax.grid(alpha=0.3)

    # Right: L0 sparsity (should be constant at TOP_K)
    ax = axes[row, 1]
    ax.plot(steps, h["l0"], color="#2ca02c", linewidth=1.5)
    ax.axhline(TOP_K, color="gray", linestyle="--", alpha=0.5, linewidth=0.8,
               label=f"k={TOP_K}")
    ax.set_ylabel("L0 (active features)", fontsize=10)
    ax.set_title(f"Layer {layer} — Sparsity (L0)", fontsize=11)
    ax.set_xlabel("Step", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

plt.suptitle("SAE Training Curves (Exp 07)", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(OUT_DIR / "07_sae_training_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved {OUT_DIR / '07_sae_training_curves.png'}")

# ── Figure 2: Domain Feature Heatmap ─────────────────────────────────────────

print("Generating Figure 2: Domain feature heatmap...")
N_TOP = 30
fig, axes = plt.subplots(1, len(SAE_LAYERS), figsize=(6 * len(SAE_LAYERS), 8))

for col, layer in enumerate(SAE_LAYERS):
    ax = axes[col]

    acts_matrix = np.stack([domain_features[d][layer] for d in DOMAIN_ORDER])
    max_per_feat = acts_matrix.max(axis=0)
    top_idx = max_per_feat.argsort()[::-1][:N_TOP]

    submatrix = acts_matrix[:, top_idx].T  # (N_TOP, n_domains)
    row_max = submatrix.max(axis=1, keepdims=True)
    row_max = np.where(row_max > 0, row_max, 1.0)
    submatrix_norm = submatrix / row_max

    im = ax.imshow(submatrix_norm, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(DOMAIN_ORDER)))
    ax.set_xticklabels(
        [DOMAIN_LABELS[d] for d in DOMAIN_ORDER],
        fontsize=8, rotation=45, ha="right",
    )
    ax.set_yticks(range(N_TOP))
    ax.set_yticklabels([f"F{i}" for i in top_idx], fontsize=6)
    ax.set_title(f"Layer {layer} — Top {N_TOP} Features", fontsize=11)
    ax.set_ylabel("Feature index", fontsize=9)
    plt.colorbar(im, ax=ax, shrink=0.6, label="Relative activation")

plt.suptitle("Domain × Feature Activation Heatmap (Exp 07)", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(OUT_DIR / "07_domain_feature_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved {OUT_DIR / '07_domain_feature_heatmap.png'}")

# ── Figure 3: Feature Specificity (stacked bars) ─────────────────────────────

print("Generating Figure 3: Feature specificity...")
fig, ax = plt.subplots(figsize=(8, 5))

categories = ["dead", "domain_specific", "shared", "universal"]
cat_labels = ["Dead", "Domain-specific", "Shared (2–4 domains)", "Universal"]
cat_colors = ["#d62728", "#ff7f0e", "#1f77b4", "#2ca02c"]

x = np.arange(len(SAE_LAYERS))
width = 0.5
bottoms = np.zeros(len(SAE_LAYERS))

for cat, label, color in zip(categories, cat_labels, cat_colors):
    values = [census[layer][cat] for layer in SAE_LAYERS]
    ax.bar(x, values, width, bottom=bottoms, label=label, color=color, alpha=0.85)

    for i, v in enumerate(values):
        if v > N_FEATURES * 0.03:
            ax.text(
                x[i], bottoms[i] + v / 2, str(v),
                ha="center", va="center", fontsize=8, fontweight="bold",
                color="white",
            )

    bottoms += values

ax.set_xticks(x)
ax.set_xticklabels([f"Layer {l}" for l in SAE_LAYERS], fontsize=11)
ax.set_ylabel("Number of features", fontsize=11)
ax.set_title("Feature Specificity by Layer (Exp 07)", fontsize=13)
ax.legend(loc="upper right", fontsize=9)
ax.set_ylim(0, N_FEATURES * 1.05)
plt.tight_layout()
plt.savefig(OUT_DIR / "07_feature_specificity.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved {OUT_DIR / '07_feature_specificity.png'}")

# ── Figure 4: Layer 0 Critical Features ──────────────────────────────────────

print("Generating Figure 4: Layer 0 critical features...")
fig, ax = plt.subplots(figsize=(12, 6))

acts_matrix_l0 = np.stack([domain_features[d][0] for d in DOMAIN_ORDER])
mean_across_domains = acts_matrix_l0.mean(axis=0)
top10_l0 = mean_across_domains.argsort()[::-1][:10]

x_pos = np.arange(10)
bar_width = 0.15

for i, domain in enumerate(DOMAIN_ORDER):
    vals = acts_matrix_l0[i, top10_l0]
    ax.bar(
        x_pos + i * bar_width - 2 * bar_width, vals, bar_width,
        label=DOMAIN_LABELS[domain], color=DOMAIN_COLOURS[domain], alpha=0.85,
    )

ax.set_xticks(x_pos)
ax.set_xticklabels([f"F{idx}" for idx in top10_l0], fontsize=9)
ax.set_xlabel("Feature index", fontsize=11)
ax.set_ylabel("Mean activation", fontsize=11)
ax.set_title("Layer 0 — Top 10 Features by Domain (Exp 07)", fontsize=13)
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "07_layer0_critical_features.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved {OUT_DIR / '07_layer0_critical_features.png'}")

# ── Figure 5: Layer 3 Wiki vs Code Scatter ────────────────────────────────────

print("Generating Figure 5: Layer 3 wiki vs code scatter...")
fig, ax = plt.subplots(figsize=(8, 8))

wiki_acts = domain_features["wiki"][3]
code_acts = domain_features["code"][3]

alive = (wiki_acts > ALIVE_THRESHOLD) | (code_acts > ALIVE_THRESHOLD)
wiki_alive = wiki_acts[alive]
code_alive = code_acts[alive]

total_act = wiki_alive + code_alive + 1e-8
ratio = (code_alive - wiki_alive) / total_act  # -1=wiki, +1=code
colors = plt.cm.RdBu_r((ratio + 1) / 2)

ax.scatter(wiki_alive, code_alive, c=colors, s=12, alpha=0.6, edgecolors="none")

max_val = max(wiki_alive.max(), code_alive.max())
ax.plot([0, max_val], [0, max_val], "k--", linewidth=0.8, alpha=0.4, label="y = x")

# Label extreme outliers
feature_indices = np.where(alive)[0]
p95_wiki = np.percentile(wiki_alive, 95)
p95_code = np.percentile(code_alive, 95)

for i in range(len(wiki_alive)):
    if abs(ratio[i]) > 0.7 and (wiki_alive[i] > p95_wiki or code_alive[i] > p95_code):
        ax.annotate(
            f"F{feature_indices[i]}", (wiki_alive[i], code_alive[i]),
            fontsize=6, alpha=0.7,
        )

ax.set_xlabel("Mean feature activation (WikiText)", fontsize=11)
ax.set_ylabel("Mean feature activation (Python Code)", fontsize=11)
ax.set_title("Layer 3 — Wiki vs Code Feature Activations (Exp 07)", fontsize=13)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
ax.set_aspect("equal")
plt.tight_layout()
plt.savefig(OUT_DIR / "07_layer3_wiki_vs_code.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved {OUT_DIR / '07_layer3_wiki_vs_code.png'}")


# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 80}")
print("SAE FEATURE ANALYSIS SUMMARY")
print(f"{'=' * 80}")

print(f"\n  {'Layer':>8} {'Alive':>8} {'Dead':>8} {'Univ':>8} {'Shared':>8} {'Specific':>8}")
print(f"  {'-' * 56}")
for layer in SAE_LAYERS:
    c = census[layer]
    print(
        f"  {layer:>8} {c['alive']:>8} {c['dead']:>8} {c['universal']:>8} "
        f"{c['shared']:>8} {c['domain_specific']:>8}"
    )

print(f"\n  TOP FEATURES (max-activating tokens from WikiText):")
for layer in SAE_LAYERS:
    print(f"\n  Layer {layer}:")
    info = top_features_info[layer]
    for feat_idx, feat_data in list(info.items())[:5]:
        tokens_str = ", ".join(
            [f"'{t}' ({v:.2f})" for t, v in feat_data["top_tokens"][:3]]
        )
        print(f"    F{feat_idx:>4}: mean={feat_data['mean_activation']:.4f}  top: {tokens_str}")

print(f"\n{'=' * 80}")
print("Done! Check results/07_*.png for visualisations.")
