"""
Experiment 03: Attention patterns, Direct Logit Attribution, Domain Sensitivity.

Mirror of pythia/08_head_interpretation.py, adapted for the RSK encoder.

Three analyses for all 48 heads:

1. Attention patterns — mean attention matrices per head across domain samples.
   No BOS token here (encoder, not decoder), so we look for archetypes like:
   uniform, diagonal, column-focused, cross-tableau (P→Q or Q→P).

2. Direct Logit Attribution (DLA) — project per-head outputs through the
   classification heads to measure each attention head's direct contribution
   to predicting σ(i).

3. Domain sensitivity — per-head output norms across all 5 combinatorial
   domains, revealing which heads specialise on different structures.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import torch

from load_model import load_rsk_model
from hooks import extract_all_head_outputs, head_output_norms
from domains import generate_all_domains, domain_batch, DOMAINS


N_SAMPLES = 200
BATCH_SIZE = 50
RESULTS_DIR = Path(__file__).parent / "results"

DOMAIN_NAMES = list(DOMAINS.keys())


def collect_head_data(model, config, values, positions):
    """
    Extract per-head attention patterns and residual-stream outputs.

    Returns:
        attn_patterns: dict of "L{l}H{h}" → (batch, seq, seq) numpy arrays
        head_outputs:  dict of "L{l}H{h}" → (batch, d_model) mean-pooled numpy arrays
        head_norms:    (batch, 48) numpy array of output norms
    """
    result = extract_all_head_outputs(model, values, positions)
    n_layers = config.num_layers
    n_heads = config.nhead

    attn_patterns = {}
    head_outputs = {}
    for l in range(n_layers):
        for h in range(n_heads):
            key = f"L{l}H{h}"
            attn_patterns[key] = result[f"{key}_attn"].numpy()
            # Mean-pool head output over token positions for DLA
            head_outputs[key] = result[f"{key}_output"].mean(dim=1).numpy()  # (batch, d_model)

    norms = head_output_norms(result, n_layers, n_heads).numpy()  # (batch, 48)
    return attn_patterns, head_outputs, norms


def characterise_attention(mean_attn: np.ndarray, n: int) -> dict:
    """
    Characterise attention pattern archetype for an RSK encoder head.

    Args:
        mean_attn: (seq, seq) mean attention matrix, seq = 2n
        n: permutation size

    Returns:
        dict with attention statistics and archetype label
    """
    seq = mean_attn.shape[0]

    # Diagonal attention (self-attending)
    diag = np.diag(mean_attn).mean()

    # Adjacent attention (prev/next token)
    prev_attn = np.mean([mean_attn[i, i - 1] for i in range(1, seq)])
    next_attn = np.mean([mean_attn[i, i + 1] for i in range(seq - 1)])

    # Cross-tableau attention: P tokens (0:n) attending to Q tokens (n:2n) and vice versa
    p_to_q = mean_attn[:n, n:].mean()  # P attending to Q
    q_to_p = mean_attn[n:, :n].mean()  # Q attending to P
    within_p = mean_attn[:n, :n].mean()
    within_q = mean_attn[n:, n:].mean()
    cross_ratio = (p_to_q + q_to_p) / (within_p + within_q + 1e-8)

    # Entropy (how spread out is attention)
    flat = mean_attn.mean(axis=0)  # average attention received per position
    flat = flat / (flat.sum() + 1e-8)
    entropy = -np.sum(flat * np.log(flat + 1e-10))
    max_entropy = np.log(seq)
    norm_entropy = entropy / max_entropy

    # Per-row entropy (how focused is each query)
    row_entropies = []
    for i in range(seq):
        row = mean_attn[i]
        row = row / (row.sum() + 1e-8)
        e = -np.sum(row * np.log(row + 1e-10))
        row_entropies.append(e / max_entropy)
    mean_row_entropy = np.mean(row_entropies)

    # Column focus (does attention concentrate on specific positions?)
    col_attn = mean_attn.mean(axis=0)
    col_max = col_attn.max()
    col_max_pos = int(col_attn.argmax())

    # Classify archetype
    if diag > 0.3:
        archetype = "self-attending"
    elif cross_ratio > 1.5:
        if p_to_q > q_to_p * 1.5:
            archetype = "P→Q cross-tableau"
        elif q_to_p > p_to_q * 1.5:
            archetype = "Q→P cross-tableau"
        else:
            archetype = "cross-tableau"
    elif col_max > 0.15:
        tableau = "P" if col_max_pos < n else "Q"
        archetype = f"column-focused ({tableau}[{col_max_pos % n}])"
    elif mean_row_entropy > 0.8:
        archetype = "broad/uniform"
    elif mean_row_entropy < 0.4:
        archetype = "narrow/focused"
    else:
        archetype = "mixed"

    return {
        "self_attn": float(diag),
        "prev_attn": float(prev_attn),
        "next_attn": float(next_attn),
        "p_to_q": float(p_to_q),
        "q_to_p": float(q_to_p),
        "within_p": float(within_p),
        "within_q": float(within_q),
        "cross_ratio": float(cross_ratio),
        "mean_row_entropy": float(mean_row_entropy),
        "col_max": float(col_max),
        "col_max_pos": col_max_pos,
        "archetype": archetype,
    }


def compute_dla(model, config, head_outputs: dict) -> np.ndarray:
    """
    Direct Logit Attribution: project per-head mean-pooled outputs through
    each classification head.

    Returns:
        dla: (n_layers * n_heads, n, n) — dla[h, i, j] = head h's contribution
             to predicting σ(i+1) = j+1
    """
    n_layers = config.num_layers
    n_heads = config.nhead
    n = config.n
    total_heads = n_layers * n_heads

    dla = np.zeros((total_heads, n, n))

    for l in range(n_layers):
        for h in range(n_heads):
            idx = l * n_heads + h
            # Mean over batch: (d_model,)
            mean_output = head_outputs[f"L{l}H{h}"].mean(axis=0)
            mean_output_t = torch.from_numpy(mean_output).float()

            # Project through each classification head
            for i, cls_head in enumerate(model.heads):
                with torch.no_grad():
                    logit_contrib = cls_head(mean_output_t)  # (n,)
                    dla[idx, i] = logit_contrib.numpy()

    return dla


def run_experiment(n: int):
    print(f"\n{'='*60}")
    print(f"  Experiment 03 — Head Interpretation (n={n})")
    print(f"{'='*60}")

    model, config = load_rsk_model(n)
    n_layers = config.num_layers
    n_heads = config.nhead
    total_heads = n_layers * n_heads
    seq_len = 2 * n
    head_labels = [f"L{l}H{h}" for l in range(n_layers) for h in range(n_heads)]

    # Generate all domains
    all_domains = generate_all_domains(n, N_SAMPLES, seed=42)

    # === Collect data per domain ===
    domain_attn = {}     # domain → head → (N, seq, seq)
    domain_outputs = {}  # domain → head → (N, d_model)
    domain_norms = {}    # domain → (N, 48)

    for domain_name in DOMAIN_NAMES:
        print(f"  Processing domain: {domain_name}")
        perms = all_domains[domain_name]

        all_attn = {f"L{l}H{h}": [] for l in range(n_layers) for h in range(n_heads)}
        all_out = {f"L{l}H{h}": [] for l in range(n_layers) for h in range(n_heads)}
        all_norms = []

        for i in range(0, N_SAMPLES, BATCH_SIZE):
            batch_perms = perms[i : i + BATCH_SIZE]
            values, positions = domain_batch(batch_perms)
            attn_p, head_out, norms = collect_head_data(model, config, values, positions)

            for key in all_attn:
                all_attn[key].append(attn_p[key])
                all_out[key].append(head_out[key])
            all_norms.append(norms)

        domain_attn[domain_name] = {
            k: np.concatenate(v, axis=0) for k, v in all_attn.items()
        }
        domain_outputs[domain_name] = {
            k: np.concatenate(v, axis=0) for k, v in all_out.items()
        }
        domain_norms[domain_name] = np.concatenate(all_norms, axis=0)

    prefix = f"03_n{n}"

    # === Analysis 1: Attention patterns ===
    print(f"\n  Attention pattern analysis (uniform domain):")
    uniform_attn = domain_attn["uniform"]
    head_analysis = {}

    for l in range(n_layers):
        for h in range(n_heads):
            key = f"L{l}H{h}"
            mean_attn = uniform_attn[key].mean(axis=0)  # (seq, seq)
            stats = characterise_attention(mean_attn, n)
            head_analysis[key] = stats
            print(f"    {key}: {stats['archetype']:30s} "
                  f"(self={stats['self_attn']:.2f}, cross={stats['cross_ratio']:.2f}, "
                  f"entropy={stats['mean_row_entropy']:.2f})")

    # Plot: attention patterns grid (8 heads per layer, one layer per row)
    fig, axes = plt.subplots(n_layers, n_heads, figsize=(2.5 * n_heads, 2.5 * n_layers))
    for l in range(n_layers):
        for h in range(n_heads):
            ax = axes[l, h]
            key = f"L{l}H{h}"
            mean_attn = uniform_attn[key].mean(axis=0)
            ax.imshow(mean_attn, cmap="Blues", aspect="equal", vmin=0)
            ax.set_title(f"{key}\n{head_analysis[key]['archetype']}", fontsize=6)
            # Mark P/Q boundary
            ax.axhline(n - 0.5, color="red", linewidth=0.5, alpha=0.5)
            ax.axvline(n - 0.5, color="red", linewidth=0.5, alpha=0.5)
            ax.set_xticks([])
            ax.set_yticks([])
            if h == 0:
                ax.set_ylabel(f"Layer {l}", fontsize=8)
    fig.suptitle(f"RSK n={n}: Mean Attention Patterns (uniform domain)", fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"{prefix}_attention_patterns.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # === Analysis 2: Direct Logit Attribution ===
    print(f"\n  Direct Logit Attribution (uniform domain):")
    dla = compute_dla(model, config, domain_outputs["uniform"])

    # For each attention head, find which σ-position it most influences
    dla_magnitude = np.abs(dla).mean(axis=2)  # (48, n) — mean over target values
    most_influenced = dla_magnitude.argmax(axis=1)  # (48,) — which σ(i) each head most affects
    max_influence = dla_magnitude.max(axis=1)  # (48,) — magnitude

    for idx in range(total_heads):
        l, h = divmod(idx, n_heads)
        pos = most_influenced[idx]
        mag = max_influence[idx]
        print(f"    L{l}H{h}: strongest influence on σ({pos+1}), magnitude={mag:.3f}")

    # Plot: DLA heatmap — heads × positions
    fig, ax = plt.subplots(figsize=(max(8, n), 10))
    im = ax.imshow(dla_magnitude, cmap="viridis", aspect="auto")
    plt.colorbar(im, ax=ax, label="Mean |logit contribution|")
    ax.set_xticks(range(n))
    ax.set_xticklabels([f"σ({i+1})" for i in range(n)], fontsize=8)
    ax.set_yticks(range(total_heads))
    ax.set_yticklabels(head_labels, fontsize=6)
    # Layer boundaries
    for boundary in range(n_heads, total_heads, n_heads):
        ax.axhline(boundary - 0.5, color="white", linewidth=0.5)
    ax.set_title(f"RSK n={n}: Direct Logit Attribution (heads × σ positions)")
    ax.set_xlabel("Permutation position")
    ax.set_ylabel("Attention head")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"{prefix}_dla_heatmap.png", dpi=150)
    plt.close(fig)

    # Plot: Top-contributing heads per position
    fig, axes = plt.subplots(1, min(n, 8), figsize=(2.5 * min(n, 8), 4))
    if n < 2:
        axes = [axes]
    for i in range(min(n, 8)):
        ax = axes[i]
        pos_dla = dla_magnitude[:, i]
        sorted_idx = np.argsort(pos_dla)[::-1][:10]
        ax.barh(range(len(sorted_idx)), pos_dla[sorted_idx], color="steelblue")
        ax.set_yticks(range(len(sorted_idx)))
        ax.set_yticklabels([head_labels[j] for j in sorted_idx], fontsize=7)
        ax.set_title(f"σ({i+1})", fontsize=9)
        ax.invert_yaxis()
    fig.suptitle(f"RSK n={n}: Top heads per σ-position (DLA)", fontsize=11)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"{prefix}_dla_top_heads.png", dpi=150)
    plt.close(fig)

    # === Analysis 3: Domain sensitivity ===
    print(f"\n  Domain sensitivity:")

    # Per-head mean norm across domains
    domain_mean_norms = {}  # head_label → {domain: mean_norm}
    for idx in range(total_heads):
        l, h = divmod(idx, n_heads)
        key = f"L{l}H{h}"
        domain_mean_norms[key] = {}
        for domain_name in DOMAIN_NAMES:
            mean_norm = domain_norms[domain_name][:, idx].mean()
            domain_mean_norms[key][domain_name] = float(mean_norm)

    # Find most domain-sensitive heads
    sensitivities = []
    for idx in range(total_heads):
        l, h = divmod(idx, n_heads)
        key = f"L{l}H{h}"
        norms_by_domain = [domain_mean_norms[key][d] for d in DOMAIN_NAMES]
        ratio = max(norms_by_domain) / (min(norms_by_domain) + 1e-8)
        max_domain = DOMAIN_NAMES[np.argmax(norms_by_domain)]
        min_domain = DOMAIN_NAMES[np.argmin(norms_by_domain)]
        sensitivities.append((ratio, key, max_domain, min_domain))

    sensitivities.sort(reverse=True)
    print(f"  Most domain-sensitive heads:")
    for ratio, key, max_d, min_d in sensitivities[:10]:
        print(f"    {key}: ratio={ratio:.2f}× (max={max_d}, min={min_d})")

    # Plot: head profiles — output norm by domain
    fig, axes = plt.subplots(n_layers, n_heads, figsize=(2 * n_heads, 2 * n_layers))
    for l in range(n_layers):
        for h in range(n_heads):
            ax = axes[l, h]
            key = f"L{l}H{h}"
            norms = [domain_mean_norms[key][d] for d in DOMAIN_NAMES]
            bars = ax.bar(range(len(DOMAIN_NAMES)), norms, color=[
                "steelblue", "coral", "seagreen", "goldenrod", "mediumpurple"
            ])
            ax.set_xticks([])
            ax.set_title(f"{key}", fontsize=7)
            if h == 0:
                ax.set_ylabel(f"L{l}", fontsize=8)
    # Add legend to last subplot
    axes[-1, -1].legend(
        [plt.Rectangle((0, 0), 1, 1, fc=c) for c in
         ["steelblue", "coral", "seagreen", "goldenrod", "mediumpurple"]],
        DOMAIN_NAMES, fontsize=5, loc="upper right"
    )
    fig.suptitle(f"RSK n={n}: Head Output Norms by Domain", fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"{prefix}_domain_profiles.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Plot: domain × head attention pattern grid (heads on x, domains on y)
    fig, axes = plt.subplots(
        len(DOMAIN_NAMES), n_heads,
        figsize=(2 * n_heads, 2 * len(DOMAIN_NAMES))
    )
    # Show only layer 0 for the domain grid (to keep it manageable)
    for di, domain_name in enumerate(DOMAIN_NAMES):
        for h in range(n_heads):
            ax = axes[di, h]
            key = f"L0H{h}"
            mean_attn = domain_attn[domain_name][key].mean(axis=0)
            ax.imshow(mean_attn, cmap="Blues", aspect="equal", vmin=0)
            ax.axhline(n - 0.5, color="red", linewidth=0.3, alpha=0.5)
            ax.axvline(n - 0.5, color="red", linewidth=0.3, alpha=0.5)
            ax.set_xticks([])
            ax.set_yticks([])
            if h == 0:
                ax.set_ylabel(domain_name, fontsize=8)
            if di == 0:
                ax.set_title(f"L0H{h}", fontsize=8)
    fig.suptitle(f"RSK n={n}: Layer 0 Attention × Domain", fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"{prefix}_attention_domain_grid.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # === Save results ===
    analysis = {
        "n": n,
        "n_samples": N_SAMPLES,
        "head_analysis": head_analysis,
        "dla_most_influenced": {
            head_labels[i]: {"position": int(most_influenced[i]), "magnitude": float(max_influence[i])}
            for i in range(total_heads)
        },
        "domain_mean_norms": domain_mean_norms,
        "domain_sensitivity_ranking": [
            {"head": key, "ratio": float(ratio), "max_domain": max_d, "min_domain": min_d}
            for ratio, key, max_d, min_d in sensitivities
        ],
    }

    with open(RESULTS_DIR / f"{prefix}_interpretation.json", "w") as f:
        json.dump(analysis, f, indent=2)

    # Human-readable summary
    lines = [
        f"RSK n={n} Head Interpretation Summary",
        f"{'='*50}",
        f"",
        f"Attention Pattern Archetypes:",
    ]
    archetype_counts = {}
    for key in head_labels:
        arch = head_analysis[key]["archetype"]
        archetype_counts[arch] = archetype_counts.get(arch, 0) + 1
        lines.append(f"  {key}: {arch}")

    lines.append(f"\nArchetype distribution:")
    for arch, count in sorted(archetype_counts.items(), key=lambda x: -x[1]):
        lines.append(f"  {arch}: {count} heads")

    lines.append(f"\nDirect Logit Attribution (top per position):")
    for i in range(n):
        sorted_idx = np.argsort(dla_magnitude[:, i])[::-1][:3]
        top3 = ", ".join(f"{head_labels[j]}({dla_magnitude[j,i]:.2f})" for j in sorted_idx)
        lines.append(f"  σ({i+1}): {top3}")

    lines.append(f"\nMost domain-sensitive heads:")
    for ratio, key, max_d, min_d in sensitivities[:10]:
        lines.append(f"  {key}: {ratio:.2f}× ({max_d} vs {min_d})")

    summary_text = "\n".join(lines)
    with open(RESULTS_DIR / f"{prefix}_interpretation.txt", "w") as f:
        f.write(summary_text)

    print(f"\n  Saved results to {RESULTS_DIR}/{prefix}_*")
    return analysis


if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for n in [8, 10]:
        run_experiment(n)
