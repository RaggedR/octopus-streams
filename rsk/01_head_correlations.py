"""
Experiment 01: Head output correlation matrix + PCA.

Mirror of pythia/01_head_correlations.py for the RSK encoder model.

For each of 500 random permutations, extract per-head output norms (the L2
norm of each head's contribution to the residual stream, averaged over token
positions). Then compute the 48×48 Pearson correlation matrix and PCA.

The question: do heads co-activate in structured ways that suggest stream-like
organisation, or are they largely independent?
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

import torch

from load_model import load_rsk_model
from hooks import extract_all_head_outputs, head_output_norms
from domains import generate_uniform, domain_batch


N_SAMPLES = 500
BATCH_SIZE = 50
RESULTS_DIR = Path(__file__).parent / "results"


def run_experiment(n: int):
    print(f"\n{'='*60}")
    print(f"  Experiment 01 — Head Correlations (n={n})")
    print(f"{'='*60}")

    model, config = load_rsk_model(n)
    n_layers = config.num_layers
    n_heads = config.nhead
    total_heads = n_layers * n_heads

    # Generate uniform random permutations
    perms = generate_uniform(n, N_SAMPLES, seed=42)

    # Collect head output norms in batches
    all_norms = []
    for i in range(0, N_SAMPLES, BATCH_SIZE):
        batch_perms = perms[i : i + BATCH_SIZE]
        values, positions = domain_batch(batch_perms)

        result = extract_all_head_outputs(model, values, positions)
        norms = head_output_norms(result, n_layers, n_heads)
        all_norms.append(norms.numpy())

        del result
        if (i // BATCH_SIZE + 1) % 5 == 0:
            print(f"  Processed {i + len(batch_perms)}/{N_SAMPLES} samples")

    head_norms = np.concatenate(all_norms, axis=0)  # (N_SAMPLES, 48)
    print(f"  Head norms shape: {head_norms.shape}")

    # Correlation matrix
    corr_matrix = np.corrcoef(head_norms.T)  # (48, 48)

    # PCA
    norms_std = (head_norms - head_norms.mean(axis=0)) / (head_norms.std(axis=0) + 1e-8)
    pca = PCA()
    pca_coords = pca.fit_transform(norms_std)

    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)
    n_80 = int(np.searchsorted(cumulative, 0.80)) + 1

    print(f"\n  PCA results:")
    print(f"    Top 5 components explain: {cumulative[4]:.1%}")
    print(f"    Components for 80%: {n_80} / {total_heads}")
    verdict = "Strong" if n_80 < total_heads * 0.3 else "Weak"
    print(f"    Evidence for low-dim structure: {verdict}")

    # Top correlated and anti-correlated pairs
    head_labels = [f"L{l}H{h}" for l in range(n_layers) for h in range(n_heads)]
    pairs = []
    for i in range(total_heads):
        for j in range(i + 1, total_heads):
            pairs.append((corr_matrix[i, j], head_labels[i], head_labels[j]))
    pairs.sort(key=lambda x: x[0], reverse=True)

    print(f"\n  Top 10 correlated pairs:")
    for r, a, b in pairs[:10]:
        print(f"    {a} <-> {b}: r={r:.3f}")
    print(f"\n  Top 10 anti-correlated pairs:")
    for r, a, b in pairs[-10:]:
        print(f"    {a} <-> {b}: r={r:.3f}")

    # --- Plots ---
    prefix = f"01_n{n}"

    # 1. Correlation heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    plt.colorbar(im, ax=ax, shrink=0.8)
    # Layer boundary lines
    for boundary in range(n_heads, total_heads, n_heads):
        ax.axhline(boundary - 0.5, color="black", linewidth=0.5, alpha=0.5)
        ax.axvline(boundary - 0.5, color="black", linewidth=0.5, alpha=0.5)
    ax.set_xticks(range(0, total_heads, n_heads))
    ax.set_xticklabels([f"L{l}" for l in range(n_layers)])
    ax.set_yticks(range(0, total_heads, n_heads))
    ax.set_yticklabels([f"L{l}" for l in range(n_layers)])
    ax.set_title(f"RSK n={n}: Head Output Norm Correlation (48×48)")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"{prefix}_correlation_heatmap.png", dpi=150)
    plt.close(fig)

    # 2. PCA scatter (PC1 vs PC2, coloured by layer)
    fig, ax = plt.subplots(figsize=(8, 6))
    colours = plt.cm.tab10(np.linspace(0, 1, n_layers))
    for l in range(n_layers):
        idx = list(range(l * n_heads, (l + 1) * n_heads))
        ax.scatter(
            pca.components_[0, idx],
            pca.components_[1, idx],
            c=[colours[l]],
            label=f"Layer {l}",
            s=80,
            edgecolors="black",
            linewidth=0.5,
        )
        for h, i in enumerate(idx):
            ax.annotate(
                f"H{h}",
                (pca.components_[0, i], pca.components_[1, i]),
                fontsize=7,
                ha="center",
                va="bottom",
            )
    ax.set_xlabel(f"PC1 ({explained[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({explained[1]:.1%} variance)")
    ax.set_title(f"RSK n={n}: PCA of Head Output Norms")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"{prefix}_pca_scatter.png", dpi=150)
    plt.close(fig)

    # 3. Scree plot
    fig, ax1 = plt.subplots(figsize=(8, 4))
    x = np.arange(1, len(explained) + 1)
    ax1.bar(x, explained, alpha=0.6, label="Individual")
    ax2 = ax1.twinx()
    ax2.plot(x, cumulative, "r-o", markersize=3, label="Cumulative")
    ax2.axhline(0.8, color="gray", linestyle="--", alpha=0.5)
    ax2.axvline(n_80, color="gray", linestyle="--", alpha=0.5)
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Variance Explained")
    ax2.set_ylabel("Cumulative Variance")
    ax1.set_title(f"RSK n={n}: PCA Scree Plot ({n_80} PCs for 80%)")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"{prefix}_scree_plot.png", dpi=150)
    plt.close(fig)

    # Save data
    np.save(RESULTS_DIR / f"{prefix}_corr_matrix.npy", corr_matrix)
    np.save(RESULTS_DIR / f"{prefix}_head_norms.npy", head_norms)
    np.save(RESULTS_DIR / f"{prefix}_pca_components.npy", pca.components_)

    stats = {
        "n": n,
        "n_samples": N_SAMPLES,
        "total_heads": total_heads,
        "pca_variance_explained": explained.tolist(),
        "pca_cumulative": cumulative.tolist(),
        "n_components_80pct": n_80,
        "top_10_correlated": [
            {"pair": f"{a}-{b}", "r": float(r)} for r, a, b in pairs[:10]
        ],
        "top_10_anticorrelated": [
            {"pair": f"{a}-{b}", "r": float(r)} for r, a, b in pairs[-10:]
        ],
    }
    with open(RESULTS_DIR / f"{prefix}_correlations.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n  Saved results to {RESULTS_DIR}/{prefix}_*")
    return stats


if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for n in [8, 10]:
        run_experiment(n)
