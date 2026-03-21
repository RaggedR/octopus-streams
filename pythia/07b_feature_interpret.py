"""
Octopus Streams — Experiment 7b: Feature Interpretation

Loads trained SAEs from experiment 07 and interprets features by finding
their max-activating tokens.  Focuses on Layer 3 (the coordination hub)
but also covers Layers 0 and 5 for completeness.

For each layer, classifies features into:
  - Universal (active in all 5 domains)
  - Shared   (active in 2–4 domains)
  - Domain-specific (active in exactly 1 domain)
  - Dead     (never active)

Then finds max-activating tokens for all non-dead features, producing a
human-readable catalogue.

Outputs:
  results/07b_feature_catalogue.json   — full machine-readable catalogue
  results/07b_feature_catalogue.txt    — human-readable summary
"""

import torch
import torch.nn as nn
import numpy as np
from transformer_lens import HookedTransformer
from datasets import load_dataset
from pathlib import Path
import json
import time

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME = "pythia-70m"
D_MODEL = 512
N_FEATURES = 4096
TOP_K = 50
SEQ_LEN = 128
N_SAMPLES = 200
BATCH_SIZE = 10
SEED = 42

LAYERS = [0, 3, 5]
ALIVE_THRESHOLD = 1e-3

OUT_DIR = Path(__file__).parent / "results"

DOMAIN_ORDER = ["random", "wiki", "poetry", "prose", "code"]
DOMAIN_LABELS = {
    "random": "Random Tokens",
    "wiki": "WikiText",
    "poetry": "Poetry",
    "prose": "Prose (PG19)",
    "code": "Python Code",
}


# ── SAE class (must match training) ──────────────────────────────────────────


class SparseAutoencoder(nn.Module):
    def __init__(self, d_model, n_features, k=TOP_K):
        super().__init__()
        self.d_model = d_model
        self.n_features = n_features
        self.k = k
        self.encoder = nn.Linear(d_model, n_features)
        self.decoder = nn.Linear(n_features, d_model)

    def _constrain_decoder(self):
        with torch.no_grad():
            norms = self.decoder.weight.norm(dim=0, keepdim=True).clamp(min=1e-8)
            self.decoder.weight.div_(norms)

    def forward(self, x):
        x_centered = x - self.decoder.bias
        pre_acts = self.encoder(x_centered)
        topk_values, topk_indices = pre_acts.topk(self.k, dim=-1)
        topk_values = torch.relu(topk_values)
        h = torch.zeros_like(pre_acts)
        h.scatter_(-1, topk_indices, topk_values)
        x_hat = self.decoder(h)
        return x_hat, h


# ── Load model ────────────────────────────────────────────────────────────────
print(f"Loading {MODEL_NAME}...")
model = HookedTransformer.from_pretrained(MODEL_NAME)
print(f"  {model.cfg.n_layers} layers, d_model={model.cfg.d_model}\n")


# ── Load trained SAEs ─────────────────────────────────────────────────────────
saes = {}
act_scales = {}

for layer in LAYERS:
    path = OUT_DIR / f"07_sae_layer{layer}_weights.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    sae = SparseAutoencoder(D_MODEL, N_FEATURES, k=TOP_K)
    sae.load_state_dict(checkpoint["state_dict"])
    sae.eval()
    saes[layer] = sae
    act_scales[layer] = checkpoint["act_scale"]
    print(f"  Loaded layer {layer} SAE (act_scale={act_scales[layer]:.4f})")


# ── Domain loaders ────────────────────────────────────────────────────────────


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


# ── Helpers ───────────────────────────────────────────────────────────────────


def collect_activations_with_tokens(model, token_batch, layer):
    """Collect residual stream activations AND corresponding token IDs."""
    hook_name = f"blocks.{layer}.hook_resid_post"
    all_acts = []
    all_toks = []

    for start in range(0, len(token_batch), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(token_batch))
        batch = token_batch[start:end]

        _, cache = model.run_with_cache(
            batch,
            names_filter=lambda name, hn=hook_name: name == hn,
        )

        acts = cache[hook_name].detach().cpu()
        b, s, d = acts.shape
        all_acts.append(acts.reshape(b * s, d))
        all_toks.append(batch.cpu().reshape(b * s))

        del cache

    return torch.cat(all_acts, dim=0).float(), torch.cat(all_toks, dim=0)


def get_feature_acts(sae, acts, act_scale):
    """Run activations through SAE, return feature activations."""
    all_h = []
    with torch.no_grad():
        for start in range(0, len(acts), 4096):
            end = min(start + 4096, len(acts))
            x = acts[start:end] / act_scale
            _, h = sae(x)
            all_h.append(h)
    return torch.cat(all_h, dim=0)


# ── Load domain data ─────────────────────────────────────────────────────────
print("Loading domain data...")
domain_tokens = {}
for domain in DOMAIN_ORDER:
    print(f"  {DOMAIN_LABELS[domain]}...", end=" ", flush=True)
    token_batch, n = LOADERS[domain](model)
    domain_tokens[domain] = token_batch
    print(f"{n} samples")


# ── Main interpretation loop ──────────────────────────────────────────────────
catalogue = {}  # layer -> list of feature dicts

for layer in LAYERS:
    sae = saes[layer]
    scale = act_scales[layer]
    print(f"\n{'=' * 70}")
    print(f"Layer {layer}")
    print(f"{'=' * 70}")

    # Step 1: Compute mean activation per feature per domain
    domain_mean_acts = {}  # domain -> (n_features,) array
    domain_all_feat_acts = {}  # domain -> (n_positions, n_features) tensor
    domain_all_tokens = {}  # domain -> (n_positions,) tensor

    for domain in DOMAIN_ORDER:
        acts, toks = collect_activations_with_tokens(
            model, domain_tokens[domain], layer
        )
        feat_acts = get_feature_acts(sae, acts, scale)
        domain_mean_acts[domain] = feat_acts.mean(dim=0).numpy()
        domain_all_feat_acts[domain] = feat_acts
        domain_all_tokens[domain] = toks
        print(f"  {DOMAIN_LABELS[domain]:>20s}: collected {len(acts)} positions")

    # Step 2: Classify features
    acts_matrix = np.stack([domain_mean_acts[d] for d in DOMAIN_ORDER])  # (5, 4096)
    max_per_feat = acts_matrix.max(axis=0)

    dead_mask = max_per_feat < ALIVE_THRESHOLD
    alive_mask = ~dead_mask
    activated_in = (acts_matrix > ALIVE_THRESHOLD).sum(axis=0)

    # Which domain is the dominant one for each feature?
    dominant_domain_idx = acts_matrix.argmax(axis=0)

    layer_features = []

    for fi in range(N_FEATURES):
        if dead_mask[fi]:
            continue

        n_domains = int(activated_in[fi])
        if n_domains == 5:
            category = "universal"
        elif n_domains == 1:
            category = "domain-specific"
        else:
            category = f"shared-{n_domains}"

        dom_idx = dominant_domain_idx[fi]
        dominant = DOMAIN_ORDER[dom_idx]

        # Find top-5 activating tokens across ALL domains
        top_tokens = []
        for domain in DOMAIN_ORDER:
            feat_col = domain_all_feat_acts[domain][:, fi]
            if feat_col.max() < ALIVE_THRESHOLD:
                continue
            top5_pos = feat_col.argsort(descending=True)[:5]
            for pos in top5_pos:
                tok_id = domain_all_tokens[domain][pos.item()].item()
                tok_str = model.tokenizer.decode([tok_id]).replace("\n", "\\n")
                top_tokens.append({
                    "token": tok_str,
                    "activation": round(feat_col[pos.item()].item(), 3),
                    "domain": domain,
                })

        # Sort by activation, keep top 8
        top_tokens.sort(key=lambda x: x["activation"], reverse=True)
        top_tokens = top_tokens[:8]

        # Per-domain mean activations
        per_domain = {
            d: round(float(domain_mean_acts[d][fi]), 4) for d in DOMAIN_ORDER
        }

        layer_features.append({
            "feature": fi,
            "category": category,
            "dominant_domain": dominant,
            "mean_activation": round(float(max_per_feat[fi]), 4),
            "per_domain": per_domain,
            "top_tokens": top_tokens,
        })

    # Sort: domain-specific first, then shared, then universal; within each by activation
    cat_order = {"domain-specific": 0, "shared-2": 1, "shared-3": 2, "shared-4": 3, "universal": 4}
    layer_features.sort(key=lambda f: (cat_order.get(f["category"], 5), -f["mean_activation"]))

    catalogue[layer] = layer_features
    print(f"\n  Catalogued {len(layer_features)} alive features")

    # Free memory
    del domain_all_feat_acts, domain_all_tokens


# ── Save JSON catalogue ──────────────────────────────────────────────────────
serialisable = {str(k): v for k, v in catalogue.items()}
with open(OUT_DIR / "07b_feature_catalogue.json", "w") as f:
    json.dump(serialisable, f, indent=2)
print(f"\nSaved {OUT_DIR / '07b_feature_catalogue.json'}")


# ── Write human-readable text ─────────────────────────────────────────────────
lines = []
for layer in LAYERS:
    features = catalogue[layer]
    lines.append(f"{'=' * 80}")
    lines.append(f"LAYER {layer}  —  {len(features)} alive features")
    lines.append(f"{'=' * 80}")
    lines.append("")

    # Group by category
    from itertools import groupby
    for category, group in groupby(features, key=lambda f: f["category"]):
        group = list(group)
        lines.append(f"── {category.upper()} ({len(group)} features) {'─' * 40}")
        lines.append("")

        for feat in group:
            fi = feat["feature"]
            dom = feat["dominant_domain"]
            mean = feat["mean_activation"]
            tokens_str = ", ".join(
                f"'{t['token']}' ({t['activation']:.1f}, {t['domain']})"
                for t in feat["top_tokens"][:5]
            )
            per_dom_str = "  ".join(
                f"{d[:4]}={v:.3f}" for d, v in feat["per_domain"].items()
            )

            lines.append(f"  F{fi:<5d}  [{dom:>7s}]  mean={mean:.3f}  {per_dom_str}")
            lines.append(f"          top: {tokens_str}")
            lines.append("")

    lines.append("")

txt_path = OUT_DIR / "07b_feature_catalogue.txt"
with open(txt_path, "w") as f:
    f.write("\n".join(lines))
print(f"Saved {txt_path}")


# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'=' * 80}")
print("FEATURE INTERPRETATION SUMMARY")
print(f"{'=' * 80}")

for layer in LAYERS:
    features = catalogue[layer]
    cats = {}
    for f in features:
        cats[f["category"]] = cats.get(f["category"], 0) + 1

    print(f"\n  Layer {layer}: {len(features)} alive features")
    for cat in sorted(cats, key=lambda c: cat_order.get(c, 5)):
        print(f"    {cat:>20s}: {cats[cat]}")

    # Show top 5 domain-specific features
    specific = [f for f in features if f["category"] == "domain-specific"]
    if specific:
        print(f"\n    Top domain-specific features:")
        for feat in specific[:10]:
            tokens = ", ".join(f"'{t['token']}'" for t in feat["top_tokens"][:3])
            print(f"      F{feat['feature']:<5d} [{feat['dominant_domain']:>7s}]  "
                  f"mean={feat['mean_activation']:.3f}  → {tokens}")

print(f"\n{'=' * 80}")
print(f"Done! Full catalogue: {txt_path}")
