"""
Experiment 06: Growth Diagram vs Bumping — Attention Pattern Analysis.

For each attention head in the n=10 model, we measure whether Q-tokens
attend to P-tokens by VALUE (Fomin's growth diagram / local rule) or
by POSITION (Schensted's reverse bumping path).

Growth diagram prediction: Q-token with value i attends to P-token
with value σ(i) — the "hit" at cell (i, σ(i)) of the growth diagram.

Bumping prediction: Q-token with value i attends to P-tokens at
(row, col) positions on the reverse bumping path for step i.

Key context: the model is encoder-only (no causal mask). It processes
all 20 tokens in parallel. Growth diagrams are naturally parallel
(each local rule application is independent given boundaries).
Sequential bumping requires causal ordering that the architecture
doesn't enforce.
"""

import copy
import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch

from load_model import load_rsk_model
from hooks import extract_all_head_outputs
from domains import generate_uniform, domain_batch

import sys
sys.path.insert(0, str(Path.home() / "git" / "paul" / "rsk"))
from rsk import rsk_forward, rsk_inverse


# ── Config ────────────────────────────────────────────────────────────────────

N_SAMPLES = 500
BATCH_SIZE = 50
SEED = 42
N_LAYERS = 6
N_HEADS = 8

RESULTS_DIR = Path(__file__).parent / "results"


# ── Bumping path computation ─────────────────────────────────────────────────

def reverse_bump_traced(tableau, row_idx, col_idx):
    """Reverse bump with path tracing. Returns (tableau, value, path)."""
    tableau = copy.deepcopy(tableau)
    path = [(row_idx, col_idx)]
    value = tableau[row_idx].pop(col_idx)
    if not tableau[row_idx]:
        tableau.pop(row_idx)
    r = row_idx - 1
    while r >= 0:
        row = tableau[r]
        pos = len(row) - 1
        while pos >= 0 and row[pos] >= value:
            pos -= 1
        if pos < 0:
            raise ValueError(f"Reverse bump failed at row {r}")
        path.append((r, pos))
        value, row[pos] = row[pos], value
        r -= 1
    return tableau, value, path


def compute_bumping_paths(P, Q):
    """
    Run inverse RSK step by step, return bumping paths keyed by Q-entry.

    Returns:
        dict: q_entry (1..n) → list of (row, col) cells on the bumping path
    """
    P, Q = copy.deepcopy(P), copy.deepcopy(Q)
    n = sum(len(row) for row in Q)
    paths = {}

    for i in range(n, 0, -1):
        for ri, row in enumerate(Q):
            for ci, val in enumerate(row):
                if val == i:
                    Q[ri].pop(ci)
                    if not Q[ri]:
                        Q.pop(ri)
                    P, _, path = reverse_bump_traced(P, ri, ci)
                    paths[i] = path
                    break
            else:
                continue
            break

    return paths


# ── Attention analysis ────────────────────────────────────────────────────────

def analyze_batch(model, perms, n):
    """
    For a batch of permutations, compute per-head attention scores for
    growth diagram targets vs bumping path targets.

    Returns per-head arrays of (growth_diagram_attn, bumping_attn, baseline_attn).
    """
    values, positions = domain_batch(perms)
    device = next(model.parameters()).device
    values, positions = values.to(device), positions.to(device)

    result = extract_all_head_outputs(model, values, positions)

    batch_size = values.shape[0]
    seq_len = 2 * n

    # Per-head accumulators
    gd_scores = {(l, h): [] for l in range(N_LAYERS) for h in range(N_HEADS)}
    bp_scores = {(l, h): [] for l in range(N_LAYERS) for h in range(N_HEADS)}
    baseline_scores = {(l, h): [] for l in range(N_LAYERS) for h in range(N_HEADS)}

    # Cross-tableau attention breakdown accumulators
    cross = {(l, h): {"pp": [], "pq": [], "qp": [], "qq": []}
             for l in range(N_LAYERS) for h in range(N_HEADS)}

    for b in range(batch_size):
        sigma = perms[b]  # 1-indexed
        P, Q = rsk_forward(sigma)
        bumping_paths = compute_bumping_paths(P, Q)

        # Build token maps: value → sequence index
        p_val_to_idx = {}
        q_val_to_idx = {}
        p_indices = []
        q_indices = []
        # Also: (row, col) → sequence index for P-tokens
        p_pos_to_idx = {}

        for s in range(seq_len):
            v = values[b, s].item()
            r, c, t = positions[b, s].tolist()
            if t == 0:  # P token
                p_val_to_idx[v] = s
                p_indices.append(s)
                p_pos_to_idx[(r, c)] = s
            else:  # Q token
                q_val_to_idx[v] = s
                q_indices.append(s)

        for l in range(N_LAYERS):
            for h in range(N_HEADS):
                attn = result[f"L{l}H{h}_attn"][b]  # (seq, seq)

                # Cross-tableau breakdown
                p_set = set(p_indices)
                q_set = set(q_indices)
                pp = sum(attn[i, j].item() for i in p_set for j in p_set) / (n * seq_len)
                pq = sum(attn[i, j].item() for i in p_set for j in q_set) / (n * seq_len)
                qp = sum(attn[i, j].item() for i in q_set for j in p_set) / (n * seq_len)
                qq = sum(attn[i, j].item() for i in q_set for j in q_set) / (n * seq_len)
                cross[(l, h)]["pp"].append(pp)
                cross[(l, h)]["pq"].append(pq)
                cross[(l, h)]["qp"].append(qp)
                cross[(l, h)]["qq"].append(qq)

                # Growth diagram + bumping scores
                gd_sum = 0.0
                bp_sum = 0.0
                base_sum = 0.0
                n_steps = 0

                for step_i in range(1, n + 1):
                    if step_i not in q_val_to_idx:
                        continue
                    q_idx = q_val_to_idx[step_i]
                    sigma_i = sigma[step_i - 1]  # σ(i), 1-indexed

                    # Growth diagram: attention to P-token with value σ(i)
                    if sigma_i in p_val_to_idx:
                        gd_sum += attn[q_idx, p_val_to_idx[sigma_i]].item()

                    # Bumping: attention to P-tokens on bumping path for step i
                    if step_i in bumping_paths:
                        path = bumping_paths[step_i]
                        path_indices = [p_pos_to_idx[rc] for rc in path if rc in p_pos_to_idx]
                        if path_indices:
                            bp_sum += sum(attn[q_idx, pi].item() for pi in path_indices) / len(path_indices)

                    # Baseline: mean attention from Q[i] to all P-tokens
                    base_sum += sum(attn[q_idx, pi].item() for pi in p_indices) / len(p_indices)
                    n_steps += 1

                if n_steps > 0:
                    gd_scores[(l, h)].append(gd_sum / n_steps)
                    bp_scores[(l, h)].append(bp_sum / n_steps)
                    baseline_scores[(l, h)].append(base_sum / n_steps)

    return gd_scores, bp_scores, baseline_scores, cross


def run_experiment(n: int):
    print(f"\n{'='*60}")
    print(f"  Experiment 06 — Growth Diagram Attention (n={n})")
    print(f"{'='*60}")

    model, config = load_rsk_model(n)

    perms = generate_uniform(n, N_SAMPLES, seed=SEED)
    print(f"\n  Analyzing {N_SAMPLES} permutations...")

    # Accumulate across batches
    all_gd = {(l, h): [] for l in range(N_LAYERS) for h in range(N_HEADS)}
    all_bp = {(l, h): [] for l in range(N_LAYERS) for h in range(N_HEADS)}
    all_base = {(l, h): [] for l in range(N_LAYERS) for h in range(N_HEADS)}
    all_cross = {(l, h): {"pp": [], "pq": [], "qp": [], "qq": []}
                 for l in range(N_LAYERS) for h in range(N_HEADS)}

    for i in range(0, N_SAMPLES, BATCH_SIZE):
        batch = perms[i : i + BATCH_SIZE]
        gd, bp, base, cross = analyze_batch(model, batch, n)

        for key in all_gd:
            all_gd[key].extend(gd[key])
            all_bp[key].extend(bp[key])
            all_base[key].extend(base[key])
            for cat in ("pp", "pq", "qp", "qq"):
                all_cross[key][cat].extend(cross[key][cat])

        done = min(i + BATCH_SIZE, N_SAMPLES)
        print(f"    {done}/{N_SAMPLES} done")

    # ── Compute summary statistics ────────────────────────────────────────

    head_stats = {}
    for l in range(N_LAYERS):
        for h in range(N_HEADS):
            key = (l, h)
            gd_mean = float(np.mean(all_gd[key])) if all_gd[key] else 0
            bp_mean = float(np.mean(all_bp[key])) if all_bp[key] else 0
            base_mean = float(np.mean(all_base[key])) if all_base[key] else 0

            head_stats[f"L{l}H{h}"] = {
                "growth_diagram": round(gd_mean, 4),
                "bumping_path": round(bp_mean, 4),
                "baseline": round(base_mean, 4),
                "gd_ratio": round(gd_mean / max(base_mean, 1e-10), 2),
                "bp_ratio": round(bp_mean / max(base_mean, 1e-10), 2),
                "cross_tableau": {
                    cat: round(float(np.mean(all_cross[key][cat])), 4)
                    for cat in ("pp", "pq", "qp", "qq")
                },
            }

    # ── Print results ─────────────────────────────────────────────────────

    print(f"\n  {'Head':>6s}  {'GD attn':>8s}  {'BP attn':>8s}  {'Base':>8s}  "
          f"{'GD/Base':>8s}  {'BP/Base':>8s}  {'Q→P':>6s}")
    print(f"  {'─'*60}")

    for l in range(N_LAYERS):
        for h in range(N_HEADS):
            s = head_stats[f"L{l}H{h}"]
            qp = s["cross_tableau"]["qp"]
            print(f"  L{l}H{h}  {s['growth_diagram']:8.4f}  {s['bumping_path']:8.4f}  "
                  f"{s['baseline']:8.4f}  {s['gd_ratio']:8.2f}  {s['bp_ratio']:8.2f}  "
                  f"{qp:6.4f}")
        if l < N_LAYERS - 1:
            print()

    # Highlight heads with strongest growth diagram signal
    ranked = sorted(head_stats.items(), key=lambda x: -x[1]["gd_ratio"])
    print(f"\n  Top 10 heads by growth diagram ratio (GD/baseline):")
    for name, s in ranked[:10]:
        print(f"    {name}: GD={s['growth_diagram']:.4f}, "
              f"BP={s['bumping_path']:.4f}, ratio={s['gd_ratio']:.2f}×")

    # ── Figures ───────────────────────────────────────────────────────────

    prefix = f"06_n{n}"

    # 1. Growth diagram ratio heatmap (layers × heads)
    gd_matrix = np.zeros((N_LAYERS, N_HEADS))
    bp_matrix = np.zeros((N_LAYERS, N_HEADS))
    for l in range(N_LAYERS):
        for h in range(N_HEADS):
            gd_matrix[l, h] = head_stats[f"L{l}H{h}"]["gd_ratio"]
            bp_matrix[l, h] = head_stats[f"L{l}H{h}"]["bp_ratio"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    im0 = axes[0].imshow(gd_matrix, cmap="YlOrRd", aspect="auto", vmin=0.5)
    axes[0].set_title("Growth Diagram Ratio (GD attn / baseline)")
    axes[0].set_xlabel("Head")
    axes[0].set_ylabel("Layer")
    axes[0].set_xticks(range(N_HEADS))
    axes[0].set_yticks(range(N_LAYERS))
    for l in range(N_LAYERS):
        for h in range(N_HEADS):
            axes[0].text(h, l, f"{gd_matrix[l,h]:.1f}", ha="center", va="center", fontsize=7)
    plt.colorbar(im0, ax=axes[0], shrink=0.8)

    im1 = axes[1].imshow(bp_matrix, cmap="YlOrRd", aspect="auto", vmin=0.5)
    axes[1].set_title("Bumping Path Ratio (BP attn / baseline)")
    axes[1].set_xlabel("Head")
    axes[1].set_ylabel("Layer")
    axes[1].set_xticks(range(N_HEADS))
    axes[1].set_yticks(range(N_LAYERS))
    for l in range(N_LAYERS):
        for h in range(N_HEADS):
            axes[1].text(h, l, f"{bp_matrix[l,h]:.1f}", ha="center", va="center", fontsize=7)
    plt.colorbar(im1, ax=axes[1], shrink=0.8)

    fig.suptitle(f"RSK n={n}: Growth Diagram vs Bumping Path Attention", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"{prefix}_gd_vs_bp_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2. Cross-tableau attention by layer (stacked bars)
    fig, ax = plt.subplots(figsize=(10, 5))
    categories = ["pp", "pq", "qp", "qq"]
    labels = ["P→P", "P→Q", "Q→P", "Q→Q"]
    colors = ["#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78"]

    # Average across heads within each layer
    layer_cross = {cat: [] for cat in categories}
    for l in range(N_LAYERS):
        for cat in categories:
            vals = [head_stats[f"L{l}H{h}"]["cross_tableau"][cat] for h in range(N_HEADS)]
            layer_cross[cat].append(float(np.mean(vals)))

    x = np.arange(N_LAYERS)
    bottoms = np.zeros(N_LAYERS)
    for cat, label, color in zip(categories, labels, colors):
        vals = layer_cross[cat]
        ax.bar(x, vals, 0.6, bottom=bottoms, label=label, color=color, alpha=0.85)
        bottoms += vals

    ax.set_xticks(x)
    ax.set_xticklabels([f"Layer {l}" for l in range(N_LAYERS)])
    ax.set_ylabel("Attention fraction")
    ax.set_title(f"RSK n={n}: Cross-Tableau Attention by Layer")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"{prefix}_cross_tableau.png", dpi=150)
    plt.close(fig)

    # 3. Top heads: GD vs BP bar comparison
    top_heads = [name for name, _ in ranked[:12]]
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(top_heads))
    width = 0.3

    gd_vals = [head_stats[h]["growth_diagram"] for h in top_heads]
    bp_vals = [head_stats[h]["bumping_path"] for h in top_heads]
    base_vals = [head_stats[h]["baseline"] for h in top_heads]

    ax.bar(x - width, gd_vals, width, label="Growth diagram target", color="#2ca02c")
    ax.bar(x, bp_vals, width, label="Bumping path targets", color="#ff7f0e")
    ax.bar(x + width, base_vals, width, label="Baseline (uniform Q→P)", color="#d62728", alpha=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(top_heads, fontsize=8)
    ax.set_ylabel("Mean attention")
    ax.set_title(f"RSK n={n}: Top Heads by Growth Diagram Attention")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"{prefix}_top_heads_gd_bp.png", dpi=150)
    plt.close(fig)

    # ── Save JSON ─────────────────────────────────────────────────────────

    with open(RESULTS_DIR / f"{prefix}_growth_diagram.json", "w") as f:
        json.dump({"n": n, "n_samples": N_SAMPLES, "head_stats": head_stats}, f, indent=2)

    print(f"\n  Saved to {RESULTS_DIR}/{prefix}_growth_diagram.*")
    return head_stats


if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_experiment(10)
