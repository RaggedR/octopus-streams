"""
Octopus Streams — Experiment 4: Head Correlations on Natural Language

Repeat the correlation/PCA analysis from 01, but with real text instead of
random tokens. This tests whether the early/late arm structure is an artifact
of random inputs or a genuine architectural property.

Data source: OpenWebText samples via HuggingFace datasets.
"""

import torch
import numpy as np
from transformer_lens import HookedTransformer
from datasets import load_dataset
from pathlib import Path
import json

# --- Config ---
MODEL_NAME = "pythia-70m"
N_SAMPLES = 200
SEQ_LEN = 128
SEED = 42
OUT_DIR = Path(__file__).parent / "results"
OUT_DIR.mkdir(exist_ok=True)

# --- Load model ---
print(f"Loading {MODEL_NAME}...")
model = HookedTransformer.from_pretrained(MODEL_NAME)
n_layers = model.cfg.n_layers
n_heads = model.cfg.n_heads
total_heads = n_layers * n_heads
print(f"  {n_layers} layers x {n_heads} heads = {total_heads} total")

# --- Load natural language data ---
print(f"\nLoading WikiText samples...")
dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
np.random.seed(SEED)
indices = np.random.choice(len(dataset), size=N_SAMPLES * 2, replace=False)

# Tokenize and filter to sequences that are long enough
all_tokens = []
for idx in indices:
    text = dataset[int(idx)]["text"]
    tokens = model.to_tokens(text, prepend_bos=True)
    if tokens.shape[1] >= SEQ_LEN:
        all_tokens.append(tokens[0, :SEQ_LEN].unsqueeze(0))
    if len(all_tokens) >= N_SAMPLES:
        break

print(f"  Got {len(all_tokens)} sequences of length {SEQ_LEN}")
token_batch = torch.cat(all_tokens, dim=0)  # (N_SAMPLES, SEQ_LEN)

# --- Extract head output norms (same method as 01) ---
print(f"\nExtracting head activations across {len(all_tokens)} inputs...")

head_norms = np.zeros((len(all_tokens), total_heads))
BATCH_SIZE = 10

for batch_start in range(0, len(all_tokens), BATCH_SIZE):
    batch_end = min(batch_start + BATCH_SIZE, len(all_tokens))
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
        print(f"  processed {batch_end}/{len(all_tokens)}")

# --- Correlation matrix ---
print("\nComputing head-to-head correlation matrix...")
corr_matrix = np.corrcoef(head_norms.T)

# --- PCA ---
print("Running PCA...")
from sklearn.decomposition import PCA

head_norms_std = (head_norms - head_norms.mean(axis=0)) / (head_norms.std(axis=0) + 1e-8)
pca = PCA()
pca.fit(head_norms_std)
explained = pca.explained_variance_ratio_
cumulative = np.cumsum(explained)

print(f"\n  Top 10 PCA components (explained variance):")
for i in range(min(10, len(explained))):
    print(f"    PC{i+1}: {explained[i]:.3f}  (cumulative: {cumulative[i]:.3f})")

n_80 = np.searchsorted(cumulative, 0.80) + 1
print(f"\n  Components for 80% variance: {n_80} out of {total_heads}")

# --- Load random-token results for comparison ---
with open(OUT_DIR / "01_correlations.json") as f:
    random_meta = json.load(f)
random_corr = np.load(OUT_DIR / "01_corr_matrix.npy")
random_explained = np.array(random_meta["pca_explained_variance"])

# --- Save ---
results = {
    "model": MODEL_NAME,
    "data_source": "wikitext-103-raw-v1",
    "n_samples": len(all_tokens),
    "seq_len": SEQ_LEN,
    "n_layers": n_layers,
    "n_heads": n_heads,
    "total_heads": total_heads,
    "pca_explained_variance": explained.tolist(),
    "pca_cumulative_variance": cumulative.tolist(),
    "n_components_80pct": int(n_80),
}

with open(OUT_DIR / "04_natural_correlations.json", "w") as f:
    json.dump(results, f, indent=2)
np.save(OUT_DIR / "04_corr_matrix.npy", corr_matrix)
np.save(OUT_DIR / "04_head_norms.npy", head_norms)
np.save(OUT_DIR / "04_pca_components.npy", pca.components_)

# --- Comparison visualization ---
import matplotlib.pyplot as plt

labels = [f"L{l}H{h}" for l in range(n_layers) for h in range(n_heads)]

fig, axes = plt.subplots(2, 3, figsize=(20, 12))

# Row 1: Random tokens (from experiment 01)
ax = axes[0, 0]
im = ax.imshow(random_corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
for i in range(1, n_layers):
    ax.axhline(i * n_heads - 0.5, color="black", linewidth=0.5, alpha=0.5)
    ax.axvline(i * n_heads - 0.5, color="black", linewidth=0.5, alpha=0.5)
ax.set_xticks(range(0, total_heads, 8))
ax.set_xticklabels([f"L{l}" for l in range(n_layers)], fontsize=9)
ax.set_yticks(range(0, total_heads, 8))
ax.set_yticklabels([f"L{l}" for l in range(n_layers)], fontsize=9)
ax.set_title("Random Tokens — Correlation Matrix", fontsize=11)
plt.colorbar(im, ax=ax, shrink=0.8)

ax = axes[0, 1]
ax.bar(range(1, 21), random_explained[:20], alpha=0.7, color="steelblue")
ax.plot(range(1, 21), np.cumsum(random_explained[:20]), "o-", color="orange", markersize=4)
ax.axhline(1 / total_heads, color="red", linestyle="--", alpha=0.5)
ax.set_title("Random Tokens — PCA Scree", fontsize=11)
ax.set_xlabel("PC")
ax.set_ylabel("Variance")

random_loadings = np.load(OUT_DIR / "01_pca_components.npy")[:2].T
ax = axes[0, 2]
colors_map = plt.cm.viridis(np.linspace(0, 1, n_layers))
for l in range(n_layers):
    idx = slice(l * n_heads, (l + 1) * n_heads)
    ax.scatter(random_loadings[idx, 0], random_loadings[idx, 1],
               c=[colors_map[l]], s=80, label=f"L{l}", edgecolors="white", linewidth=0.5)
    for h in range(n_heads):
        head_idx = l * n_heads + h
        ax.annotate(f"H{h}", (random_loadings[head_idx, 0], random_loadings[head_idx, 1]),
                     fontsize=6, ha="center", va="bottom", alpha=0.7)
ax.axhline(0, color="gray", linewidth=0.5, alpha=0.5)
ax.axvline(0, color="gray", linewidth=0.5, alpha=0.5)
ax.set_xlabel(f"PC1 ({random_explained[0]:.1%})")
ax.set_ylabel(f"PC2 ({random_explained[1]:.1%})")
ax.set_title("Random Tokens — Head Loadings", fontsize=11)
ax.legend(fontsize=7)

# Row 2: Natural language
ax = axes[1, 0]
im = ax.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
for i in range(1, n_layers):
    ax.axhline(i * n_heads - 0.5, color="black", linewidth=0.5, alpha=0.5)
    ax.axvline(i * n_heads - 0.5, color="black", linewidth=0.5, alpha=0.5)
ax.set_xticks(range(0, total_heads, 8))
ax.set_xticklabels([f"L{l}" for l in range(n_layers)], fontsize=9)
ax.set_yticks(range(0, total_heads, 8))
ax.set_yticklabels([f"L{l}" for l in range(n_layers)], fontsize=9)
ax.set_title("Natural Language — Correlation Matrix", fontsize=11)
plt.colorbar(im, ax=ax, shrink=0.8)

ax = axes[1, 1]
ax.bar(range(1, 21), explained[:20], alpha=0.7, color="steelblue")
ax.plot(range(1, 21), np.cumsum(explained[:20]), "o-", color="orange", markersize=4)
ax.axhline(1 / total_heads, color="red", linestyle="--", alpha=0.5)
ax.set_title("Natural Language — PCA Scree", fontsize=11)
ax.set_xlabel("PC")
ax.set_ylabel("Variance")

nat_loadings = pca.components_[:2].T
ax = axes[1, 2]
for l in range(n_layers):
    idx = slice(l * n_heads, (l + 1) * n_heads)
    ax.scatter(nat_loadings[idx, 0], nat_loadings[idx, 1],
               c=[colors_map[l]], s=80, label=f"L{l}", edgecolors="white", linewidth=0.5)
    for h in range(n_heads):
        head_idx = l * n_heads + h
        ax.annotate(f"H{h}", (nat_loadings[head_idx, 0], nat_loadings[head_idx, 1]),
                     fontsize=6, ha="center", va="bottom", alpha=0.7)
ax.axhline(0, color="gray", linewidth=0.5, alpha=0.5)
ax.axvline(0, color="gray", linewidth=0.5, alpha=0.5)
ax.set_xlabel(f"PC1 ({explained[0]:.1%})")
ax.set_ylabel(f"PC2 ({explained[1]:.1%})")
ax.set_title("Natural Language — Head Loadings", fontsize=11)
ax.legend(fontsize=7)

# --- Correlation between the two correlation matrices ---
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
r_between = np.corrcoef(random_corr[mask], corr_matrix[mask])[0, 1]

plt.suptitle(
    f"Octopus Streams — Random vs Natural Language\n"
    f"(correlation between correlation matrices: r = {r_between:.3f})",
    fontsize=14, y=1.02,
)
plt.tight_layout()
plt.savefig(OUT_DIR / "04_comparison.png", dpi=150, bbox_inches="tight")
print(f"\nSaved to {OUT_DIR / '04_comparison.png'}")

# --- Print summary ---
print(f"\n{'='*60}")
print(f"COMPARISON SUMMARY")
print(f"{'='*60}")
print(f"  {'Metric':<40} {'Random':>10} {'Natural':>10}")
print(f"  {'-'*60}")
print(f"  {'PC1 explained variance':<40} {random_explained[0]:>10.3f} {explained[0]:>10.3f}")
print(f"  {'PC1+PC2 cumulative':<40} {sum(random_explained[:2]):>10.3f} {sum(explained[:2]):>10.3f}")
print(f"  {'Components for 80%':<40} {random_meta['n_components_80pct']:>10} {n_80:>10}")
print(f"  {'Corr matrix similarity (r)':<40} {r_between:>10.3f} {'':>10}")
print(f"{'='*60}")

# Top correlated pairs in natural language
upper = corr_matrix.copy()
upper[~mask] = 0
flat_idx = np.argsort(upper.ravel())[::-1]
print(f"\nTop 10 most correlated head pairs (natural language):")
for rank, idx in enumerate(flat_idx[:10]):
    i, j = divmod(idx, total_heads)
    li, hi = divmod(i, n_heads)
    lj, hj = divmod(j, n_heads)
    print(f"  {rank+1}. L{li}H{hi} <-> L{lj}H{hj}  r={upper.ravel()[idx]:.3f}")
