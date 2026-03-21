"""
Experiment 02: Layer and head ablation.

Mirror of pythia/03_ablation.py, adapted for the RSK classification task.

Ablation metric: greedy exact-match accuracy (the model's native metric) rather
than perplexity. Also reports per-position accuracy and mean cross-entropy loss.

We zero out head outputs in the manual forward pass and measure how accuracy
degrades, revealing which heads/layers are critical for the inverse RSK task.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn.functional as F

from load_model import load_rsk_model
from hooks import extract_all_head_outputs
from domains import generate_uniform, domain_batch

import sys
RSK_DIR = Path.home() / "git" / "paul" / "rsk"
if str(RSK_DIR) not in sys.path:
    sys.path.insert(0, str(RSK_DIR))
from train import masked_greedy_decode


N_SAMPLES = 500
BATCH_SIZE = 50
RESULTS_DIR = Path(__file__).parent / "results"


def build_ablation_groups(n_layers: int, n_heads: int) -> dict[str, list[tuple[int, int]]]:
    """Define head groups to ablate."""
    groups = {
        "none (baseline)": [],
    }

    # Individual layers
    for l in range(n_layers):
        groups[f"layer {l}"] = [(l, h) for h in range(n_heads)]

    # Layer pairs
    groups["early (L0+L1)"] = [(l, h) for l in [0, 1] for h in range(n_heads)]
    groups["middle (L2+L3)"] = [(l, h) for l in [2, 3] for h in range(n_heads)]
    groups["late (L4+L5)"] = [(l, h) for l in [4, 5] for h in range(n_heads)]

    # All-but-one layer
    for l in range(n_layers):
        others = [(ll, h) for ll in range(n_layers) if ll != l for h in range(n_heads)]
        groups[f"all except L{l}"] = others

    # Individual heads (all 48 — useful for finding the most critical)
    for l in range(n_layers):
        for h in range(n_heads):
            groups[f"L{l}H{h}"] = [(l, h)]

    # Random control
    groups["random 3 heads"] = [(0, 3), (2, 5), (4, 7)]

    return groups


def compute_ablation_metrics(
    model, values, positions, targets, zero_heads
) -> dict:
    """
    Run model with ablated heads and compute metrics.

    Args:
        model: RSKEncoder
        values: (batch, 2n)
        positions: (batch, 2n, 3)
        targets: (batch, n) — 0-indexed ground truth
        zero_heads: set of (layer, head) to ablate

    Returns:
        dict with loss, per_position accuracy, greedy_exact_match accuracy
    """
    result = extract_all_head_outputs(model, values, positions, zero_heads=zero_heads)
    logits = result["logits"]  # (batch, n, n)
    n = logits.shape[1]

    # Cross-entropy loss
    loss = F.cross_entropy(
        logits.reshape(-1, n), targets.reshape(-1), reduction="mean"
    ).item()

    # Per-position accuracy (unconstrained argmax)
    preds = logits.argmax(dim=-1)  # (batch, n)
    per_pos = (preds == targets).float().mean().item()

    # Greedy exact-match (with permutation constraint)
    greedy_preds = masked_greedy_decode(logits)
    exact = (greedy_preds == targets).all(dim=-1).float().mean().item()

    return {"loss": loss, "per_position": per_pos, "greedy_exact_match": exact}


def run_experiment(n: int):
    print(f"\n{'='*60}")
    print(f"  Experiment 02 — Ablation Study (n={n})")
    print(f"{'='*60}")

    model, config = load_rsk_model(n)
    n_layers = config.num_layers
    n_heads = config.nhead

    # Generate test data
    perms = generate_uniform(n, N_SAMPLES, seed=123)  # different seed from exp 01

    # Prepare targets
    targets_list = []
    for sigma in perms:
        targets_list.append(torch.tensor([v - 1 for v in sigma], dtype=torch.long))
    all_targets = torch.stack(targets_list)

    # Prepare input batches
    all_values, all_positions = domain_batch(perms)

    groups = build_ablation_groups(n_layers, n_heads)
    results = {}

    for name, heads in groups.items():
        zero_set = set(heads) if heads else set()
        n_ablated = len(heads)

        # Process in batches
        batch_losses, batch_per_pos, batch_exact = [], [], []

        for i in range(0, N_SAMPLES, BATCH_SIZE):
            j = min(i + BATCH_SIZE, N_SAMPLES)
            v_batch = all_values[i:j]
            p_batch = all_positions[i:j]
            t_batch = all_targets[i:j]

            metrics = compute_ablation_metrics(model, v_batch, p_batch, t_batch, zero_set)
            batch_losses.append(metrics["loss"])
            batch_per_pos.append(metrics["per_position"])
            batch_exact.append(metrics["greedy_exact_match"])

        avg_loss = np.mean(batch_losses)
        avg_per_pos = np.mean(batch_per_pos)
        avg_exact = np.mean(batch_exact)

        results[name] = {
            "n_ablated": n_ablated,
            "loss": float(avg_loss),
            "per_position": float(avg_per_pos),
            "greedy_exact_match": float(avg_exact),
        }

        # Only print non-individual-head results to keep output manageable
        is_individual = name.startswith("L") and name[1].isdigit() and "H" in name and len(name) <= 5
        if not is_individual:
            print(
                f"  {name:30s} | ablated={n_ablated:2d} | "
                f"loss={avg_loss:.4f} | pos_acc={avg_per_pos:.4f} | "
                f"exact={avg_exact:.4f}"
            )

    # Find most critical individual heads
    print(f"\n  Most critical individual heads (by exact-match drop):")
    baseline = results["none (baseline)"]["greedy_exact_match"]
    head_drops = []
    for l in range(n_layers):
        for h in range(n_heads):
            name = f"L{l}H{h}"
            drop = baseline - results[name]["greedy_exact_match"]
            head_drops.append((drop, name, results[name]))
    head_drops.sort(reverse=True)
    for drop, name, metrics in head_drops[:10]:
        print(
            f"    {name}: exact_match drop = {drop:+.4f} "
            f"(loss={metrics['loss']:.4f})"
        )

    # --- Plots ---
    prefix = f"02_n{n}"

    # 1. Layer ablation bar chart
    layer_names = [f"L{l}" for l in range(n_layers)]
    layer_exact = [results[f"layer {l}"]["greedy_exact_match"] for l in range(n_layers)]
    layer_loss = [results[f"layer {l}"]["loss"] for l in range(n_layers)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.bar(layer_names, layer_exact, color="steelblue")
    ax1.axhline(baseline, color="red", linestyle="--", alpha=0.7, label="Baseline")
    ax1.set_ylabel("Greedy Exact Match")
    ax1.set_title(f"n={n}: Accuracy When Layer Ablated")
    ax1.legend()

    ax2.bar(layer_names, layer_loss, color="coral")
    ax2.axhline(results["none (baseline)"]["loss"], color="red", linestyle="--", alpha=0.7)
    ax2.set_ylabel("Cross-Entropy Loss")
    ax2.set_title(f"n={n}: Loss When Layer Ablated")

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"{prefix}_layer_ablation.png", dpi=150)
    plt.close(fig)

    # 2. Per-head ablation heatmap
    head_exact_grid = np.zeros((n_layers, n_heads))
    for l in range(n_layers):
        for h in range(n_heads):
            head_exact_grid[l, h] = results[f"L{l}H{h}"]["greedy_exact_match"]

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(head_exact_grid, cmap="RdYlGn", aspect="auto")
    plt.colorbar(im, ax=ax, label="Exact Match After Ablation")
    ax.set_xticks(range(n_heads))
    ax.set_xticklabels([f"H{h}" for h in range(n_heads)])
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels([f"L{l}" for l in range(n_layers)])
    ax.set_title(f"n={n}: Exact Match When Individual Head Ablated")

    # Annotate cells
    for l in range(n_layers):
        for h in range(n_heads):
            ax.text(h, l, f"{head_exact_grid[l, h]:.2f}",
                    ha="center", va="center", fontsize=7,
                    color="white" if head_exact_grid[l, h] < 0.5 else "black")

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"{prefix}_head_ablation_heatmap.png", dpi=150)
    plt.close(fig)

    # 3. Group ablation summary
    group_names = [
        "none (baseline)", "early (L0+L1)", "middle (L2+L3)", "late (L4+L5)",
        "random 3 heads",
    ] + [f"all except L{l}" for l in range(n_layers)]

    fig, ax = plt.subplots(figsize=(12, 5))
    x = range(len(group_names))
    exact_vals = [results[g]["greedy_exact_match"] for g in group_names]
    bars = ax.bar(x, exact_vals, color="steelblue")
    ax.set_xticks(x)
    ax.set_xticklabels(group_names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Greedy Exact Match")
    ax.set_title(f"n={n}: Group Ablation Results")
    for bar, val in zip(bars, exact_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.2f}", ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"{prefix}_group_ablation.png", dpi=150)
    plt.close(fig)

    # Save JSON
    with open(RESULTS_DIR / f"{prefix}_ablation.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Saved results to {RESULTS_DIR}/{prefix}_*")
    return results


if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for n in [8, 10]:
        run_experiment(n)
