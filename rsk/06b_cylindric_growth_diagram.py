"""
Experiment 06b: Cylindric Growth Diagram — Attention Pattern Analysis.

For the cylindric CPP model (profile 10101010, T=8, 10 ALCD labels, 99.98%),
we test whether attention heads implement the Burge local rule by checking
attention between partition triples.

Unlike permutation RSK where bumping is an alternative explanation, the
cylindric growth diagram bijection has NO alternative algorithm — it is
DEFINED by the local rule. Any structure the model learned must correspond
to the Burge local rule applied recursively.

The recursive 𝔏_i composition (thesis §4.2) processes non-wrapping
inversions in a fixed order: 0, 2, 1, 4, 3, 2, 6, 5, 4, 3.

Each step involves a partition triple (left, center, right). The dependency
structure forces a minimum depth of 4 layers:
  Layer 0: steps 0,1,3,6 (positions 0,2,4,6) — independent
  Layer 1: steps 2,4,7 (positions 1,3,5) — depend on layer 0
  Layer 2: steps 5,8 (positions 2,4) — depend on layer 1
  Layer 3: step 9 (position 3) — depends on layer 2

We measure: does attention at each layer concentrate on the predicted
partition triples for that depth level?
"""

import json
import random
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path.home() / "git" / "paul" / "rsk"))

import torch
from model import RSKEncoder
from config import ModelConfig
from data import CylindricSamplingDataset, encode_cpp
from hooks import extract_all_head_outputs
from rsk import (
    growth_diagram_forward, growth_diagram_inverse,
    sample_gamma, sample_alcd, _num_alcd_labels,
    _is_pi_min, _non_wrapping_inversions, _swap_profile,
)

# ── Config ────────────────────────────────────────────────────────────────────

PROFILE = (1, 0, 1, 0, 1, 0, 1, 0)
T = len(PROFILE)
MAX_LABEL = 3
MAX_GAMMA_PARTS = 3
MAX_GAMMA_SIZE = 4

N_SAMPLES = 500
BATCH_SIZE = 25  # smaller batches — 104 tokens per sample
SEED = 42
N_LAYERS = 6
N_HEADS = 8

RESULTS_DIR = Path(__file__).parent / "results"
CHECKPOINT_DIR = Path.home() / "git" / "paul" / "rsk" / "checkpoints"


# ── Growth diagram dependency analysis ────────────────────────────────────────

def compute_inversion_sequence(profile):
    """Trace the sequence of non-wrapping inversions processed by growth_diagram_inverse."""
    steps = []
    p = profile
    while not _is_pi_min(p):
        inv = _non_wrapping_inversions(p)
        i = inv[0]
        steps.append(i)
        p = _swap_profile(p, i)
    return steps


def compute_dependency_layers(steps, T):
    """
    Assign each step to the earliest layer it can be computed at,
    based on which partition positions have been modified by earlier steps.

    A step at position i depends on all earlier steps that modify
    positions (i-1)%T, i, or (i+1)%T.
    """
    modified_at_layer = {}  # position → layer it was last modified
    step_layers = []

    for step_idx, pos in enumerate(steps):
        left = (pos - 1) % T
        right = (pos + 1) % T
        neighbors = {left, pos, right}

        # This step must come after all modifications to its neighbors
        min_layer = 0
        for nb in neighbors:
            if nb in modified_at_layer:
                min_layer = max(min_layer, modified_at_layer[nb] + 1)

        step_layers.append(min_layer)
        modified_at_layer[pos] = min_layer

    return step_layers


# ── Model loading ─────────────────────────────────────────────────────────────

def load_cylindric_model(profile, max_label, device="cpu"):
    """Load trained cylindric CPP model from checkpoint."""
    prof_str = "".join(str(b) for b in profile)
    ckpt_path = CHECKPOINT_DIR / f"encoder_cyl_{prof_str}_m{max_label}" / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint at {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt["model_config"]

    model = RSKEncoder(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"Loaded cylindric model (profile={''.join(str(b) for b in profile)}, max_label={max_label})")
    print(f"  Epoch {ckpt['epoch']}, val={ckpt['val_metrics']}")
    print(f"  Parameters: {model.count_parameters():,}")
    print(f"  Config: seq_len={config.seq_len}, vocab_size={config.vocab_size}, "
          f"num_tokens={config.num_tokens}")

    return model, config


# ── Sample generation ─────────────────────────────────────────────────────────

def generate_cpp_samples(n_samples, seed=42):
    """Generate CPP samples with their ALCD labels."""
    rng = random.Random(seed)
    num_labels = _num_alcd_labels(PROFILE)
    max_parts = MAX_GAMMA_PARTS + num_labels

    samples = []
    for _ in range(n_samples):
        gamma = sample_gamma(MAX_GAMMA_PARTS, MAX_GAMMA_SIZE, rng)
        alcd = sample_alcd(PROFILE, MAX_LABEL, rng)
        cpp = growth_diagram_forward(PROFILE, gamma, alcd)

        values, positions = encode_cpp(cpp, T, max_parts)
        target = torch.tensor(alcd, dtype=torch.long)

        samples.append({
            "values": values,
            "positions": positions,
            "target": target,
            "cpp": cpp,
            "gamma": gamma,
            "alcd": alcd,
        })

    return samples


def batch_samples(samples):
    """Stack samples into batched tensors."""
    values = torch.stack([s["values"] for s in samples])
    positions = torch.stack([s["positions"] for s in samples])
    return values, positions


# ── Attention analysis ────────────────────────────────────────────────────────

def compute_partition_attention(attn, positions, T, max_parts):
    """
    Compute the T×T partition attention matrix from a (seq, seq) attention pattern.

    partition_attn[i, j] = mean attention from tokens of partition i to tokens of partition j.
    """
    seq_len = attn.shape[0]
    assert seq_len == T * max_parts

    # Token k belongs to partition k // max_parts
    part_attn = torch.zeros(T, T)
    for i in range(T):
        for j in range(T):
            # Tokens of partition i: indices [i*max_parts : (i+1)*max_parts]
            # Tokens of partition j: indices [j*max_parts : (j+1)*max_parts]
            block = attn[i*max_parts:(i+1)*max_parts, j*max_parts:(j+1)*max_parts]
            part_attn[i, j] = block.sum().item() / max_parts  # normalize by source tokens

    return part_attn


def analyze_batch(model, samples, config):
    """Compute per-head partition attention matrices for a batch."""
    values, positions = batch_samples(samples)
    device = next(model.parameters()).device
    values, positions = values.to(device), positions.to(device)

    result = extract_all_head_outputs(model, values, positions)

    batch_size = values.shape[0]
    max_parts = config.num_tokens // T

    # Accumulate partition attention matrices per head
    head_part_attn = {}
    for l in range(N_LAYERS):
        for h in range(N_HEADS):
            attn = result[f"L{l}H{h}_attn"]  # (batch, seq, seq)
            matrices = []
            for b in range(batch_size):
                m = compute_partition_attention(attn[b], positions[b], T, max_parts)
                matrices.append(m)
            head_part_attn[(l, h)] = torch.stack(matrices)  # (batch, T, T)

    return head_part_attn


def run_experiment():
    print(f"\n{'='*60}")
    print(f"  Experiment 06b — Cylindric Growth Diagram Attention")
    print(f"  Profile: {''.join(str(b) for b in PROFILE)}, T={T}")
    print(f"{'='*60}")

    model, config = load_cylindric_model(PROFILE, MAX_LABEL)
    max_parts = config.num_tokens // T

    # Compute growth diagram structure
    steps = compute_inversion_sequence(PROFILE)
    step_layers = compute_dependency_layers(steps, T)
    max_depth = max(step_layers)

    print(f"\n  Growth diagram structure:")
    print(f"    Inversion sequence: {steps}")
    print(f"    Step layers:        {step_layers}")
    print(f"    Max depth: {max_depth} (model has {N_LAYERS} layers)")

    # Group steps by layer
    layer_to_triples = {d: [] for d in range(max_depth + 1)}
    for step_idx, (pos, depth) in enumerate(zip(steps, step_layers)):
        left = (pos - 1) % T
        right = (pos + 1) % T
        layer_to_triples[depth].append((left, pos, right))

    print(f"\n  Predicted partition triples per depth:")
    for d in range(max_depth + 1):
        triples = layer_to_triples[d]
        print(f"    Depth {d}: {triples}")

    # Generate and analyze samples
    print(f"\n  Generating {N_SAMPLES} CPP samples...")
    samples = generate_cpp_samples(N_SAMPLES, seed=SEED)

    # Accumulate partition attention matrices
    all_part_attn = {(l, h): [] for l in range(N_LAYERS) for h in range(N_HEADS)}

    for i in range(0, N_SAMPLES, BATCH_SIZE):
        batch = samples[i : i + BATCH_SIZE]
        head_pa = analyze_batch(model, batch, config)
        for key in all_part_attn:
            all_part_attn[key].append(head_pa[key])
        done = min(i + BATCH_SIZE, N_SAMPLES)
        print(f"    {done}/{N_SAMPLES}")

    # Average partition attention matrices
    mean_part_attn = {}
    for key in all_part_attn:
        mean_part_attn[key] = torch.cat(all_part_attn[key]).mean(dim=0)  # (T, T)

    # ── Score: triple concentration ───────────────────────────────────────

    print(f"\n  Triple concentration scores:")
    print(f"  (How much attention goes to predicted partition triples at each depth)")
    print(f"\n  {'Head':>6s}", end="")
    for d in range(max_depth + 1):
        print(f"  {'D'+str(d):>6s}", end="")
    print(f"  {'Best':>6s}")
    print(f"  {'─'*60}")

    head_scores = {}
    for l in range(N_LAYERS):
        for h in range(N_HEADS):
            pa = mean_part_attn[(l, h)]  # (T, T)

            # For each depth level, compute concentration on its triples
            depth_scores = []
            for d in range(max_depth + 1):
                triples = layer_to_triples[d]
                if not triples:
                    depth_scores.append(0.0)
                    continue

                # Attention from center to left+right for each triple
                triple_attn = 0.0
                for left, center, right in triples:
                    triple_attn += pa[center, left].item() + pa[center, right].item()

                # Baseline: if attention were uniform, each cell gets 1/T
                # Each triple contributes 2 cells, so expected = 2*len(triples)/T
                n_cells = 2 * len(triples)
                baseline = n_cells / T  # expected attention to these n_cells under uniform distribution

                # Score = actual / expected
                score = triple_attn / max(baseline, 1e-10)
                depth_scores.append(score)

            head_scores[f"L{l}H{h}"] = depth_scores
            best_d = int(np.argmax(depth_scores))

            print(f"  L{l}H{h}", end="")
            for s in depth_scores:
                print(f"  {s:6.2f}", end="")
            print(f"  D{best_d:>3d}")

        if l < N_LAYERS - 1:
            print()

    # ── Layer-averaged analysis ───────────────────────────────────────────

    print(f"\n  Layer-averaged triple concentration:")
    print(f"  (If the model follows the dependency structure, layer L should")
    print(f"   score highest at depth L)")
    print(f"\n  {'Layer':>7s}", end="")
    for d in range(max_depth + 1):
        print(f"  {'D'+str(d):>6s}", end="")
    print()
    print(f"  {'─'*50}")

    layer_depth_scores = np.zeros((N_LAYERS, max_depth + 1))
    for l in range(N_LAYERS):
        for d in range(max_depth + 1):
            scores = [head_scores[f"L{l}H{h}"][d] for h in range(N_HEADS)]
            layer_depth_scores[l, d] = np.mean(scores)

        print(f"  Layer {l}", end="")
        for d in range(max_depth + 1):
            val = layer_depth_scores[l, d]
            marker = " *" if d == np.argmax(layer_depth_scores[l]) else "  "
            print(f"  {val:5.2f}{marker}", end="")
        print()

    # ── Figures ───────────────────────────────────────────────────────────

    prefix = "06b_cyl10101010"

    # 1. Partition attention matrices for each layer (averaged across heads)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for l in range(N_LAYERS):
        ax = axes[l // 3, l % 3]
        # Average across heads
        layer_avg = torch.stack([mean_part_attn[(l, h)] for h in range(N_HEADS)]).mean(dim=0)
        im = ax.imshow(layer_avg.numpy(), cmap="YlOrRd", aspect="equal", vmin=0)
        ax.set_title(f"Layer {l}", fontsize=11)
        ax.set_xlabel("Target partition")
        ax.set_ylabel("Source partition")
        ax.set_xticks(range(T))
        ax.set_yticks(range(T))

        # Mark predicted triples for this layer's depth
        if l <= max_depth:
            for left, center, right in layer_to_triples[l]:
                ax.plot(left, center, 's', color='blue', markersize=8, markerfacecolor='none', linewidth=2)
                ax.plot(right, center, 's', color='blue', markersize=8, markerfacecolor='none', linewidth=2)

        plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle(f"Cylindric CPP (10101010): Partition Attention by Layer\n"
                 f"Blue squares = predicted local rule triples at matching depth",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"{prefix}_partition_attention.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2. Layer × Depth concentration heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(layer_depth_scores, cmap="YlGnBu", aspect="auto")
    ax.set_xlabel("Dependency depth")
    ax.set_ylabel("Transformer layer")
    ax.set_xticks(range(max_depth + 1))
    ax.set_xticklabels([f"D{d}" for d in range(max_depth + 1)])
    ax.set_yticks(range(N_LAYERS))
    ax.set_yticklabels([f"Layer {l}" for l in range(N_LAYERS)])
    for l in range(N_LAYERS):
        for d in range(max_depth + 1):
            ax.text(d, l, f"{layer_depth_scores[l,d]:.2f}", ha="center", va="center", fontsize=9)
    ax.set_title("Triple Concentration: Layer × Depth\n"
                 "(Diagonal = model matches predicted dependency structure)")
    plt.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"{prefix}_layer_depth.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 3. Per-head heatmap: best matching depth
    best_depth = np.zeros((N_LAYERS, N_HEADS))
    max_score = np.zeros((N_LAYERS, N_HEADS))
    for l in range(N_LAYERS):
        for h in range(N_HEADS):
            scores = head_scores[f"L{l}H{h}"]
            best_depth[l, h] = np.argmax(scores)
            max_score[l, h] = np.max(scores)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    im0 = axes[0].imshow(best_depth, cmap="viridis", aspect="auto", vmin=0, vmax=max_depth)
    axes[0].set_title("Best matching depth per head")
    axes[0].set_xlabel("Head")
    axes[0].set_ylabel("Layer")
    axes[0].set_xticks(range(N_HEADS))
    axes[0].set_yticks(range(N_LAYERS))
    for l in range(N_LAYERS):
        for h in range(N_HEADS):
            axes[0].text(h, l, f"D{int(best_depth[l,h])}", ha="center", va="center",
                        fontsize=7, color="white" if best_depth[l,h] > 1 else "black")
    plt.colorbar(im0, ax=axes[0], shrink=0.8)

    im1 = axes[1].imshow(max_score, cmap="YlOrRd", aspect="auto", vmin=0.5)
    axes[1].set_title("Max triple concentration score")
    axes[1].set_xlabel("Head")
    axes[1].set_ylabel("Layer")
    axes[1].set_xticks(range(N_HEADS))
    axes[1].set_yticks(range(N_LAYERS))
    for l in range(N_LAYERS):
        for h in range(N_HEADS):
            axes[1].text(h, l, f"{max_score[l,h]:.1f}", ha="center", va="center", fontsize=7)
    plt.colorbar(im1, ax=axes[1], shrink=0.8)

    fig.suptitle("Cylindric CPP (10101010): Per-Head Depth Alignment", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"{prefix}_head_depth.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Save JSON ─────────────────────────────────────────────────────────

    results = {
        "profile": list(PROFILE),
        "T": T,
        "n_samples": N_SAMPLES,
        "inversion_sequence": steps,
        "step_layers": step_layers,
        "layer_to_triples": {str(k): v for k, v in layer_to_triples.items()},
        "layer_depth_scores": layer_depth_scores.tolist(),
        "head_scores": head_scores,
    }

    with open(RESULTS_DIR / f"{prefix}_growth_diagram.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Saved to {RESULTS_DIR}/{prefix}_*")
    return results


if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_experiment()
