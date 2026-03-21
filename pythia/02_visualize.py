"""
Octopus Streams — Visualization

Generates:
  1. Correlation heatmap (48x48 heads)
  2. PCA scree plot (explained variance)
  3. PCA biplot (heads projected onto PC1 vs PC2)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

RESULTS = Path(__file__).parent / "results"
OUT = RESULTS

# --- Load data ---
corr = np.load(OUT / "01_corr_matrix.npy")
components = np.load(OUT / "01_pca_components.npy")
head_norms = np.load(OUT / "01_head_norms.npy")

with open(OUT / "01_correlations.json") as f:
    meta = json.load(f)

n_layers = meta["n_layers"]
n_heads = meta["n_heads"]
total_heads = meta["total_heads"]
explained = np.array(meta["pca_explained_variance"])

# Head labels
labels = [f"L{l}H{h}" for l in range(n_layers) for h in range(n_heads)]

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# --- 1. Correlation heatmap ---
ax = axes[0]
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
ax.set_xticks(range(0, total_heads, 8))
ax.set_xticklabels([f"L{l}" for l in range(n_layers)], fontsize=9)
ax.set_yticks(range(0, total_heads, 8))
ax.set_yticklabels([f"L{l}" for l in range(n_layers)], fontsize=9)
ax.set_title("Head-to-Head Correlation Matrix", fontsize=12)
# Add layer boundary lines
for i in range(1, n_layers):
    ax.axhline(i * n_heads - 0.5, color="black", linewidth=0.5, alpha=0.5)
    ax.axvline(i * n_heads - 0.5, color="black", linewidth=0.5, alpha=0.5)
plt.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")

# --- 2. Scree plot ---
ax = axes[1]
cumulative = np.cumsum(explained)
ax.bar(range(1, len(explained) + 1), explained, alpha=0.7, color="steelblue", label="Individual")
ax.plot(range(1, len(explained) + 1), cumulative, "o-", color="orange", markersize=4, label="Cumulative")
ax.axhline(1 / total_heads, color="red", linestyle="--", alpha=0.5, label=f"Uniform ({1/total_heads:.3f})")
ax.set_xlabel("Principal Component")
ax.set_ylabel("Explained Variance Ratio")
ax.set_title("PCA Scree Plot", fontsize=12)
ax.legend(fontsize=8)
ax.set_xlim(0.5, 20.5)

# --- 3. PCA biplot (PC1 vs PC2) ---
ax = axes[2]

# Project heads onto PC1 and PC2
# Standardize head_norms
head_std = (head_norms - head_norms.mean(axis=0)) / (head_norms.std(axis=0) + 1e-8)
projected = head_std @ components[:2].T  # (n_samples, 2) — but we want head loadings

# Head loadings = correlation of each head with each PC
loadings = components[:2].T  # (total_heads, 2)

# Color by layer
colors = plt.cm.viridis(np.linspace(0, 1, n_layers))
for l in range(n_layers):
    idx = slice(l * n_heads, (l + 1) * n_heads)
    ax.scatter(loadings[idx, 0], loadings[idx, 1],
               c=[colors[l]], s=80, label=f"Layer {l}", zorder=3, edgecolors="white", linewidth=0.5)
    for h in range(n_heads):
        head_idx = l * n_heads + h
        ax.annotate(f"H{h}", (loadings[head_idx, 0], loadings[head_idx, 1]),
                     fontsize=6, ha="center", va="bottom", alpha=0.7)

ax.axhline(0, color="gray", linewidth=0.5, alpha=0.5)
ax.axvline(0, color="gray", linewidth=0.5, alpha=0.5)
ax.set_xlabel(f"PC1 ({explained[0]:.1%} variance)")
ax.set_ylabel(f"PC2 ({explained[1]:.1%} variance)")
ax.set_title("Head Loadings on PC1 vs PC2", fontsize=12)
ax.legend(fontsize=7, loc="best")

plt.suptitle("Octopus Streams — Pythia-70m Head Correlation Analysis", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(OUT / "02_head_correlations.png", dpi=150, bbox_inches="tight")
print(f"Saved to {OUT / '02_head_correlations.png'}")
plt.close()
