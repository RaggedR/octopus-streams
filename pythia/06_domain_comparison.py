"""
Octopus Streams — Experiment 6: Domain Comparison (Poetry vs Prose vs Code)

Does the attention-head coalition structure change across linguistically
distinct domains? We compare five input types:
  - Random tokens   (from experiment 01)
  - WikiText         (from experiment 04)
  - Poetry           (Poetry Foundation via HuggingFace)
  - Narrative prose   (Project Gutenberg English via sedthh/gutenberg_english)
  - Python code      (CodeSearchNet — Python functions)

For each domain we extract per-head output norms, build a 48×48 correlation
matrix, run PCA, then compare all five via a pairwise similarity matrix.

Outputs:
  results/06_{domain}_{correlations.json, corr_matrix.npy, head_norms.npy, pca_components.npy}
  results/06_similarity_matrix.npy
  results/06_domain_grid.png
  results/06_similarity_matrix.png
  results/06_pca_overlay.png
"""

import torch
import numpy as np
from transformer_lens import HookedTransformer
from datasets import load_dataset
from sklearn.decomposition import PCA
from pathlib import Path
import json
import matplotlib.pyplot as plt

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME = "pythia-70m"
N_SAMPLES = 200
SEQ_LEN = 128
SEED = 42
BATCH_SIZE = 10
OUT_DIR = Path(__file__).parent / "results"
OUT_DIR.mkdir(exist_ok=True)

# Domain definitions: (key, label, colour for plots)
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

# ── Fail-fast: check that 01 and 04 results exist ────────────────────────────
for prefix in ["01", "04"]:
    for suffix in ["corr_matrix.npy", "correlations.json", "pca_components.npy"]:
        fname = f"{prefix}_{suffix}" if prefix == "01" else (
            f"{prefix}_{suffix}" if suffix != "correlations.json"
            else f"{prefix}_natural_correlations.json"
        )
        path = OUT_DIR / fname
        if not path.exists():
            raise FileNotFoundError(
                f"Required result from experiment {prefix} not found: {path}\n"
                f"Run experiment {prefix} first."
            )

print("✓ Found existing results from experiments 01 and 04\n")

# ── Load model ────────────────────────────────────────────────────────────────
print(f"Loading {MODEL_NAME}...")
model = HookedTransformer.from_pretrained(MODEL_NAME)
n_layers = model.cfg.n_layers
n_heads = model.cfg.n_heads
total_heads = n_layers * n_heads
print(f"  {n_layers} layers × {n_heads} heads = {total_heads} total\n")


# ── Shared analysis functions ─────────────────────────────────────────────────

def extract_head_norms(model, token_batch):
    """Extract mean per-head output norms for each sample in the batch."""
    n_samples = len(token_batch)
    head_norms = np.zeros((n_samples, total_heads))

    for batch_start in range(0, n_samples, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, n_samples)
        batch = token_batch[batch_start:batch_end]

        _, cache = model.run_with_cache(
            batch,
            names_filter=lambda name: name.endswith(".attn.hook_z"),
        )

        for layer in range(n_layers):
            hook_name = f"blocks.{layer}.attn.hook_z"
            z = cache[hook_name].detach().cpu().numpy()
            for head in range(n_heads):
                head_idx = layer * n_heads + head
                norms = np.linalg.norm(z[:, :, head, :], axis=-1)
                head_norms[batch_start:batch_end, head_idx] = norms.mean(axis=1)

        del cache
        if (batch_start // BATCH_SIZE) % 5 == 0:
            print(f"    processed {batch_end}/{n_samples}")

    return head_norms


def run_domain_analysis(head_norms):
    """Compute correlation matrix and PCA from head norms."""
    corr_matrix = np.corrcoef(head_norms.T)

    head_norms_std = (head_norms - head_norms.mean(axis=0)) / (
        head_norms.std(axis=0) + 1e-8
    )
    pca = PCA()
    pca.fit(head_norms_std)

    return corr_matrix, pca


def save_domain_results(domain_key, head_norms, corr_matrix, pca, data_source, n_actual):
    """Save all results for a domain."""
    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)
    n_80 = int(np.searchsorted(cumulative, 0.80) + 1)

    results = {
        "model": MODEL_NAME,
        "domain": domain_key,
        "data_source": data_source,
        "n_samples": int(n_actual),
        "seq_len": SEQ_LEN,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "total_heads": total_heads,
        "pca_explained_variance": explained.tolist(),
        "pca_cumulative_variance": cumulative.tolist(),
        "n_components_80pct": n_80,
    }

    prefix = f"06_{domain_key}"
    with open(OUT_DIR / f"{prefix}_correlations.json", "w") as f:
        json.dump(results, f, indent=2)
    np.save(OUT_DIR / f"{prefix}_corr_matrix.npy", corr_matrix)
    np.save(OUT_DIR / f"{prefix}_head_norms.npy", head_norms)
    np.save(OUT_DIR / f"{prefix}_pca_components.npy", pca.components_)

    return results


# ── Domain loaders ────────────────────────────────────────────────────────────

def load_poetry(model):
    """Load poetry samples from Poetry Foundation dataset."""
    print("  Loading Poetry Foundation dataset...")
    ds = load_dataset("suayptalha/Poetry-Foundation-Poems", split="train")
    print(f"    Column names: {ds.column_names}")
    print(f"    Total poems: {len(ds)}")

    np.random.seed(SEED + 1)
    # Oversample 10× to handle short poems
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
    print(f"    Yield: {actual}/{N_SAMPLES} poems pass {SEQ_LEN}-token filter")
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

        # Skip first quarter to avoid dedications/headers/Gutenberg boilerplate
        start = len(text) // 4
        # Take a 2000-char window from the interior
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
    """Load Python code from CodeSearchNet (Python subset, streaming)."""
    print("  Loading CodeSearchNet Python (streaming)...")
    ds = load_dataset(
        "Nan-Do/code-search-net-python", split="train", streaming=True,
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


# ── Load existing results ─────────────────────────────────────────────────────

def load_existing_domain(prefix, json_name):
    """Load previously computed results for a domain."""
    with open(OUT_DIR / json_name) as f:
        meta = json.load(f)
    corr = np.load(OUT_DIR / f"{prefix}_corr_matrix.npy")
    components = np.load(OUT_DIR / f"{prefix}_pca_components.npy")
    explained = np.array(meta["pca_explained_variance"])
    return corr, components, explained, meta


print("Loading existing results...")
random_corr, random_components, random_explained, random_meta = load_existing_domain(
    "01", "01_correlations.json"
)
wiki_corr, wiki_components, wiki_explained, wiki_meta = load_existing_domain(
    "04", "04_natural_correlations.json"
)
print("  ✓ Random tokens (01)")
print("  ✓ WikiText (04)\n")

# ── Run new domains ───────────────────────────────────────────────────────────

LOADERS = {
    "poetry": (load_poetry, "suayptalha/Poetry-Foundation-Poems"),
    "prose": (load_prose_gutenberg, "sedthh/gutenberg_english"),
    "code": (load_code_python, "Nan-Do/code-search-net-python"),
}

# Store all results for cross-domain comparison
all_corr = {"random": random_corr, "wiki": wiki_corr}
all_components = {"random": random_components, "wiki": wiki_components}
all_explained = {"random": random_explained, "wiki": wiki_explained}

for domain_key, (loader_fn, data_source) in LOADERS.items():
    print(f"═══ {DOMAIN_LABELS[domain_key]} ═══")

    print("  Loading data...")
    token_batch, n_actual = loader_fn(model)

    print(f"  Extracting head norms ({n_actual} samples)...")
    head_norms = extract_head_norms(model, token_batch)

    print("  Computing correlations + PCA...")
    corr_matrix, pca = run_domain_analysis(head_norms)

    print("  Saving results...")
    meta = save_domain_results(domain_key, head_norms, corr_matrix, pca, data_source, n_actual)

    all_corr[domain_key] = corr_matrix
    all_components[domain_key] = pca.components_
    all_explained[domain_key] = np.array(meta["pca_explained_variance"])

    print(f"  ✓ Done — PC1={meta['pca_explained_variance'][0]:.3f}, "
          f"80% at {meta['n_components_80pct']} components\n")

# ── Build 5×5 pairwise similarity matrix ──────────────────────────────────────

print("Computing pairwise similarity matrix...")
mask = np.triu(np.ones((total_heads, total_heads), dtype=bool), k=1)
n_domains = len(DOMAIN_ORDER)
similarity = np.zeros((n_domains, n_domains))

for i, d1 in enumerate(DOMAIN_ORDER):
    for j, d2 in enumerate(DOMAIN_ORDER):
        if i == j:
            similarity[i, j] = 1.0
        elif j > i:
            r = np.corrcoef(all_corr[d1][mask], all_corr[d2][mask])[0, 1]
            similarity[i, j] = r
            similarity[j, i] = r

np.save(OUT_DIR / "06_similarity_matrix.npy", similarity)

# ── Figure 1: Domain Grid (5 rows × 3 cols) ──────────────────────────────────

print("Generating Figure 1: Domain grid...")
fig, axes = plt.subplots(n_domains, 3, figsize=(18, n_domains * 3.5))
colours_by_layer = plt.cm.viridis(np.linspace(0, 1, n_layers))

for row, domain in enumerate(DOMAIN_ORDER):
    corr = all_corr[domain]
    explained = all_explained[domain]
    components = all_components[domain]
    label = DOMAIN_LABELS[domain]

    # Col 0: Correlation matrix
    ax = axes[row, 0]
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    for b in range(1, n_layers):
        ax.axhline(b * n_heads - 0.5, color="black", linewidth=0.5, alpha=0.5)
        ax.axvline(b * n_heads - 0.5, color="black", linewidth=0.5, alpha=0.5)
    ax.set_xticks(range(0, total_heads, 8))
    ax.set_xticklabels([f"L{l}" for l in range(n_layers)], fontsize=7)
    ax.set_yticks(range(0, total_heads, 8))
    ax.set_yticklabels([f"L{l}" for l in range(n_layers)], fontsize=7)
    ax.set_title(f"{label} — Correlations", fontsize=10)
    plt.colorbar(im, ax=ax, shrink=0.7)

    # Col 1: Scree plot
    ax = axes[row, 1]
    n_show = min(20, len(explained))
    ax.bar(range(1, n_show + 1), explained[:n_show], alpha=0.7, color=DOMAIN_COLOURS[domain])
    ax.plot(range(1, n_show + 1), np.cumsum(explained[:n_show]),
            "o-", color="orange", markersize=3)
    ax.axhline(1 / total_heads, color="red", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.set_title(f"{label} — PCA Scree", fontsize=10)
    ax.set_xlabel("PC", fontsize=8)
    ax.set_ylabel("Variance", fontsize=8)
    ax.set_ylim(0, max(explained[:n_show]) * 1.15)

    # Col 2: PCA biplot (PC1 vs PC2 loadings)
    ax = axes[row, 2]
    loadings = components[:2].T  # (total_heads, 2)
    for l in range(n_layers):
        idx = slice(l * n_heads, (l + 1) * n_heads)
        ax.scatter(
            loadings[idx, 0], loadings[idx, 1],
            c=[colours_by_layer[l]], s=60, label=f"L{l}",
            edgecolors="white", linewidth=0.4,
        )
        for h in range(n_heads):
            hi = l * n_heads + h
            ax.annotate(
                f"H{h}", (loadings[hi, 0], loadings[hi, 1]),
                fontsize=5, ha="center", va="bottom", alpha=0.6,
            )
    ax.axhline(0, color="gray", linewidth=0.4, alpha=0.5)
    ax.axvline(0, color="gray", linewidth=0.4, alpha=0.5)
    ax.set_xlabel(f"PC1 ({explained[0]:.1%})", fontsize=8)
    ax.set_ylabel(f"PC2 ({explained[1]:.1%})", fontsize=8)
    ax.set_title(f"{label} — Head Loadings", fontsize=10)
    ax.legend(fontsize=6, ncol=2)

plt.suptitle("Octopus Streams — Domain Comparison (Exp 06)", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(OUT_DIR / "06_domain_grid.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved {OUT_DIR / '06_domain_grid.png'}")

# ── Figure 2: Similarity Matrix ──────────────────────────────────────────────

print("Generating Figure 2: Similarity matrix...")
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(similarity, cmap="YlOrRd", vmin=0, vmax=1, aspect="equal")

domain_labels_short = [DOMAIN_LABELS[d] for d in DOMAIN_ORDER]
ax.set_xticks(range(n_domains))
ax.set_xticklabels(domain_labels_short, fontsize=9, rotation=30, ha="right")
ax.set_yticks(range(n_domains))
ax.set_yticklabels(domain_labels_short, fontsize=9)

# Annotate each cell
for i in range(n_domains):
    for j in range(n_domains):
        colour = "white" if similarity[i, j] > 0.7 else "black"
        ax.text(j, i, f"{similarity[i, j]:.3f}", ha="center", va="center",
                fontsize=10, fontweight="bold", color=colour)

plt.colorbar(im, ax=ax, shrink=0.8, label="Pearson r (between correlation matrices)")
ax.set_title("Pairwise Domain Similarity\n(correlation between head-coalition structures)", fontsize=12)
plt.tight_layout()
plt.savefig(OUT_DIR / "06_similarity_matrix.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved {OUT_DIR / '06_similarity_matrix.png'}")

# ── Figure 3: PCA Overlay ────────────────────────────────────────────────────

print("Generating Figure 3: PCA overlay...")
fig, ax = plt.subplots(figsize=(10, 8))

for domain in DOMAIN_ORDER:
    components = all_components[domain]
    loadings = components[:2].T  # (total_heads, 2)
    colour = DOMAIN_COLOURS[domain]
    label = DOMAIN_LABELS[domain]

    # Plot individual heads with low alpha
    ax.scatter(
        loadings[:, 0], loadings[:, 1],
        c=colour, s=20, alpha=0.25, edgecolors="none",
    )

    # Layer centroids with connecting trajectory
    centroids = []
    for l in range(n_layers):
        idx = slice(l * n_heads, (l + 1) * n_heads)
        cx = loadings[idx, 0].mean()
        cy = loadings[idx, 1].mean()
        centroids.append((cx, cy))

    centroids = np.array(centroids)
    ax.plot(
        centroids[:, 0], centroids[:, 1],
        "o-", color=colour, markersize=8, linewidth=2, label=label,
        markeredgecolor="white", markeredgewidth=0.8,
    )

    # Label layer centroids
    for l, (cx, cy) in enumerate(centroids):
        ax.annotate(
            f"L{l}", (cx, cy), fontsize=6, ha="center", va="bottom",
            color=colour, fontweight="bold",
        )

ax.axhline(0, color="gray", linewidth=0.4, alpha=0.5)
ax.axvline(0, color="gray", linewidth=0.4, alpha=0.5)
ax.set_xlabel("PC1 loading", fontsize=11)
ax.set_ylabel("PC2 loading", fontsize=11)
ax.set_title(
    "Domain PCA Overlay — Layer Centroid Trajectories\n"
    "(faint dots = individual heads; connected markers = layer centroids)",
    fontsize=12,
)
ax.legend(fontsize=10, loc="best")
plt.tight_layout()
plt.savefig(OUT_DIR / "06_pca_overlay.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved {OUT_DIR / '06_pca_overlay.png'}")

# ── Summary table ─────────────────────────────────────────────────────────────

print(f"\n{'='*80}")
print("DOMAIN COMPARISON SUMMARY")
print(f"{'='*80}")
print(f"  {'Domain':<20} {'PC1':>8} {'PC1+PC2':>10} {'80% at':>8}")
print(f"  {'-'*50}")
for domain in DOMAIN_ORDER:
    exp = all_explained[domain]
    cum2 = exp[0] + exp[1]
    # Compute n_80 from explained variance
    cumsum = np.cumsum(exp)
    n80 = int(np.searchsorted(cumsum, 0.80) + 1)
    print(f"  {DOMAIN_LABELS[domain]:<20} {exp[0]:>8.3f} {cum2:>10.3f} {n80:>8}")

print(f"\n  PAIRWISE SIMILARITY (r between correlation matrices):")
print(f"  {'':>20}", end="")
for d in DOMAIN_ORDER:
    print(f"  {DOMAIN_LABELS[d]:>12}", end="")
print()
for i, d1 in enumerate(DOMAIN_ORDER):
    print(f"  {DOMAIN_LABELS[d1]:<20}", end="")
    for j, d2 in enumerate(DOMAIN_ORDER):
        print(f"  {similarity[i, j]:>12.3f}", end="")
    print()

print(f"{'='*80}")
print("\nDone! Check results/06_*.png for visualizations.")
