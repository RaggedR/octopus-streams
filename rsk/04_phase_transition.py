"""
Experiment 04: Attention entropy phase transition across n.

Tests whether the sharp boundary between "structured attention" (layers 0–3)
and "uniform attention" (layers 4–5) shifts rightward as n increases.

Hypothesis: larger n requires more reverse-bumping steps, so the model needs
more layers for computation, pushing the entropy phase transition deeper.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import torch

from load_model import load_rsk_model
from hooks import extract_all_head_outputs
from domains import generate_uniform, domain_batch


N_SAMPLES = 200
BATCH_SIZE = 50
RESULTS_DIR = Path(__file__).parent / "results"


def compute_layer_entropies(model, config, values, positions):
    """
    Compute mean row entropy of attention patterns for each head.

    Returns:
        entropies: dict of "L{l}H{h}" → float (mean normalised row entropy)
    """
    result = extract_all_head_outputs(model, values, positions)
    n_layers = config.num_layers
    n_heads = config.nhead
    seq_len = values.shape[1]
    max_entropy = np.log(seq_len)

    entropies = {}
    for l in range(n_layers):
        for h in range(n_heads):
            attn = result[f"L{l}H{h}_attn"].numpy()  # (batch, seq, seq)
            # Per-row entropy, averaged over rows and batch
            row_ent = []
            for b in range(attn.shape[0]):
                for i in range(seq_len):
                    row = attn[b, i]
                    row = row / (row.sum() + 1e-10)
                    e = -np.sum(row * np.log(row + 1e-10))
                    row_ent.append(e / max_entropy)
            entropies[f"L{l}H{h}"] = float(np.mean(row_ent))

    return entropies


def run_experiment():
    print(f"{'='*60}")
    print(f"  Experiment 04 — Attention Entropy Phase Transition")
    print(f"{'='*60}")

    all_results = {}

    for n in [8, 10, 15]:
        print(f"\n  n={n}:")
        model, config = load_rsk_model(n)
        n_layers = config.num_layers
        n_heads = config.nhead

        perms = generate_uniform(n, N_SAMPLES, seed=42)

        # Collect entropies in batches
        batch_entropies = []
        for i in range(0, N_SAMPLES, BATCH_SIZE):
            batch_perms = perms[i : i + BATCH_SIZE]
            values, positions = domain_batch(batch_perms)
            ent = compute_layer_entropies(model, config, values, positions)
            batch_entropies.append(ent)

        # Average across batches
        head_entropies = {}
        for key in batch_entropies[0]:
            head_entropies[key] = np.mean([b[key] for b in batch_entropies])

        # Per-layer mean and std entropy
        layer_stats = {}
        for l in range(n_layers):
            head_ents = [head_entropies[f"L{l}H{h}"] for h in range(n_heads)]
            layer_stats[l] = {
                "mean": float(np.mean(head_ents)),
                "std": float(np.std(head_ents)),
                "min": float(np.min(head_ents)),
                "max": float(np.max(head_ents)),
                "per_head": {f"H{h}": head_ents[h] for h in range(n_heads)},
            }
            print(f"    Layer {l}: entropy = {layer_stats[l]['mean']:.3f} "
                  f"± {layer_stats[l]['std']:.3f} "
                  f"[{layer_stats[l]['min']:.3f} – {layer_stats[l]['max']:.3f}]")

        all_results[n] = layer_stats

    # --- Plot: entropy phase transition comparison ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colours = {8: "steelblue", 10: "coral", 15: "seagreen"}
    markers = {8: "o", 10: "s", 15: "D"}
    layers = list(range(6))

    # Left: mean entropy per layer with error bars
    for n in [8, 10, 15]:
        means = [all_results[n][l]["mean"] for l in layers]
        stds = [all_results[n][l]["std"] for l in layers]
        ax1.errorbar(
            layers, means, yerr=stds,
            color=colours[n], marker=markers[n], markersize=8,
            linewidth=2, capsize=4, label=f"n={n}",
        )

    ax1.set_xlabel("Layer", fontsize=12)
    ax1.set_ylabel("Mean Normalised Row Entropy", fontsize=12)
    ax1.set_title("Attention Entropy by Layer", fontsize=13)
    ax1.set_xticks(layers)
    ax1.set_xticklabels([f"L{l}" for l in layers])
    ax1.axhline(1.0, color="gray", linestyle=":", alpha=0.5, label="Maximum (uniform)")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.4, 1.05)

    # Right: per-head entropy heatmap for each n (stacked)
    for idx, n in enumerate([8, 10, 15]):
        row_data = []
        for l in layers:
            for h in range(8):
                row_data.append(all_results[n][l]["per_head"][f"H{h}"])
        grid = np.array(row_data).reshape(6, 8)

        ax_sub = fig.add_axes([0.56 + idx * 0.155, 0.15, 0.13, 0.7])
        im = ax_sub.imshow(grid, cmap="RdYlGn_r", aspect="auto", vmin=0.5, vmax=1.0)
        ax_sub.set_xticks(range(8))
        ax_sub.set_xticklabels([f"H{h}" for h in range(8)], fontsize=6)
        ax_sub.set_yticks(range(6))
        ax_sub.set_yticklabels([f"L{l}" for l in range(6)], fontsize=7)
        ax_sub.set_title(f"n={n}", fontsize=9, color=colours[n], fontweight="bold")

    # Remove the empty ax2 we created
    ax2.remove()

    fig.savefig(RESULTS_DIR / "04_phase_transition.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Simpler line-only plot for clarity
    fig, ax = plt.subplots(figsize=(8, 5))
    for n in [8, 10, 15]:
        means = [all_results[n][l]["mean"] for l in layers]
        ax.plot(layers, means, color=colours[n], marker=markers[n],
                markersize=10, linewidth=2.5, label=f"n={n}")
    ax.set_xlabel("Layer", fontsize=13)
    ax.set_ylabel("Mean Attention Entropy (normalised)", fontsize=13)
    ax.set_title("Does the Pipeline Extend for Larger n?", fontsize=14)
    ax.set_xticks(layers)
    ax.set_xticklabels([f"L{l}" for l in layers], fontsize=11)
    ax.axhline(1.0, color="gray", linestyle=":", alpha=0.4)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.5, 1.05)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "04_phase_transition_lines.png", dpi=150)
    plt.close(fig)

    # Save JSON
    serialisable = {}
    for n, stats in all_results.items():
        serialisable[str(n)] = {
            str(l): s for l, s in stats.items()
        }
    with open(RESULTS_DIR / "04_phase_transition.json", "w") as f:
        json.dump(serialisable, f, indent=2)

    print(f"\n  Saved to {RESULTS_DIR}/04_*")
    return all_results


if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_experiment()
