"""
Octopus Streams — Experiment 1: Attention Head Correlations

Do attention heads in Pythia-70m form semi-independent processing "streams"
(like octopus arms), or are they deeply entangled?

Approach:
  1. Run diverse input sequences through the model
  2. Record each head's output norm (how much it writes to the residual stream)
  3. Compute head-to-head correlation matrix across inputs
  4. PCA to find principal axes of co-variation
  5. Visualize: do heads cluster into coherent groups?

Model: Pythia-70m (6 layers x 8 heads = 48 heads total)
"""

import torch
import numpy as np
from transformer_lens import HookedTransformer
from pathlib import Path
import json

# --- Config ---
MODEL_NAME = "pythia-70m"
N_SAMPLES = 200        # number of input sequences
SEQ_LEN = 128          # tokens per sequence
SEED = 42
OUT_DIR = Path(__file__).parent / "results"
OUT_DIR.mkdir(exist_ok=True)

# --- Load model ---
print(f"Loading {MODEL_NAME}...")
model = HookedTransformer.from_pretrained(MODEL_NAME)
print(f"  {model.cfg.n_layers} layers, {model.cfg.n_heads} heads/layer = {model.cfg.n_layers * model.cfg.n_heads} heads total")
print(f"  d_model={model.cfg.d_model}, d_head={model.cfg.d_head}")

# --- Generate diverse inputs ---
# Use random token sequences. Not natural language, but ensures diversity
# and avoids biasing toward any particular domain.
# (We can repeat with natural text later to compare.)
torch.manual_seed(SEED)
vocab_size = model.cfg.d_vocab
random_tokens = torch.randint(0, vocab_size, (N_SAMPLES, SEQ_LEN))

# --- Extract head output norms ---
# For each head, we measure the L2 norm of its output (what it writes to
# the residual stream). A head that writes a large vector is "active";
# a head that writes near-zero is dormant for that input.

print(f"\nExtracting head activations across {N_SAMPLES} inputs...")

n_layers = model.cfg.n_layers
n_heads = model.cfg.n_heads
total_heads = n_layers * n_heads

# head_norms[sample, head] = mean output norm across sequence positions
head_norms = np.zeros((N_SAMPLES, total_heads))

# Process in batches to manage memory
BATCH_SIZE = 10
for batch_start in range(0, N_SAMPLES, BATCH_SIZE):
    batch_end = min(batch_start + BATCH_SIZE, N_SAMPLES)
    batch = random_tokens[batch_start:batch_end]

    # run_with_cache returns (logits, cache_dict)
    # hook_z = each head's output before the output projection mixes them
    # Shape: (batch, seq_len, n_heads, d_head)
    _, cache = model.run_with_cache(
        batch,
        names_filter=lambda name: name.endswith(".attn.hook_z"),
    )

    for layer in range(n_layers):
        hook_name = f"blocks.{layer}.attn.hook_z"
        z = cache[hook_name].detach().cpu().numpy()  # (batch, seq, n_heads, d_head)

        for head in range(n_heads):
            head_idx = layer * n_heads + head
            # Mean L2 norm across sequence positions
            norms = np.linalg.norm(z[:, :, head, :], axis=-1)  # (batch, seq_len)
            head_norms[batch_start:batch_end, head_idx] = norms.mean(axis=1)

    del cache  # free memory
    if (batch_start // BATCH_SIZE) % 5 == 0:
        print(f"  processed {batch_end}/{N_SAMPLES}")

print(f"  head_norms shape: {head_norms.shape}")

# --- Correlation matrix ---
print("\nComputing head-to-head correlation matrix...")
# Pearson correlation across samples: which heads co-activate?
corr_matrix = np.corrcoef(head_norms.T)  # (total_heads, total_heads)

# --- PCA ---
print("Running PCA...")
from sklearn.decomposition import PCA

# Standardize
head_norms_std = (head_norms - head_norms.mean(axis=0)) / (head_norms.std(axis=0) + 1e-8)

pca = PCA()
pca.fit(head_norms_std)

explained = pca.explained_variance_ratio_
cumulative = np.cumsum(explained)

print(f"\n  Top 10 PCA components (explained variance):")
for i in range(min(10, len(explained))):
    print(f"    PC{i+1}: {explained[i]:.3f}  (cumulative: {cumulative[i]:.3f})")

# How many components for 80% variance?
n_80 = np.searchsorted(cumulative, 0.80) + 1
print(f"\n  Components for 80% variance: {n_80} out of {total_heads}")
print(f"  → {'Strong' if n_80 < total_heads * 0.3 else 'Weak'} evidence for low-dimensional structure")

# --- Save results ---
results = {
    "model": MODEL_NAME,
    "n_samples": N_SAMPLES,
    "seq_len": SEQ_LEN,
    "n_layers": n_layers,
    "n_heads": n_heads,
    "total_heads": total_heads,
    "pca_explained_variance": explained.tolist(),
    "pca_cumulative_variance": cumulative.tolist(),
    "n_components_80pct": int(n_80),
}

with open(OUT_DIR / "01_correlations.json", "w") as f:
    json.dump(results, f, indent=2)

np.save(OUT_DIR / "01_corr_matrix.npy", corr_matrix)
np.save(OUT_DIR / "01_head_norms.npy", head_norms)
np.save(OUT_DIR / "01_pca_components.npy", pca.components_)

print(f"\nResults saved to {OUT_DIR}/")

# --- Quick visualization (terminal-friendly) ---
print("\n" + "=" * 60)
print("CORRELATION MATRIX SUMMARY")
print("=" * 60)

# Show strongest positive correlations (excluding self-correlation)
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
upper = corr_matrix.copy()
upper[~mask] = 0

# Top 10 most correlated pairs
flat_idx = np.argsort(upper.ravel())[::-1]
print("\nTop 10 most correlated head pairs:")
for rank, idx in enumerate(flat_idx[:10]):
    i, j = divmod(idx, total_heads)
    li, hi = divmod(i, n_heads)
    lj, hj = divmod(j, n_heads)
    print(f"  {rank+1}. L{li}H{hi} ↔ L{lj}H{hj}  r={upper.ravel()[idx]:.3f}")

# Top 10 most anti-correlated pairs
print("\nTop 10 most anti-correlated head pairs:")
for rank, idx in enumerate(flat_idx[-10:][::-1]):
    i, j = divmod(idx, total_heads)
    li, hi = divmod(i, n_heads)
    lj, hj = divmod(j, n_heads)
    r = corr_matrix[i, j]
    if r < 0:
        print(f"  {rank+1}. L{li}H{hi} ↔ L{lj}H{hj}  r={r:.3f}")

print("\nDone! Next: run 02_visualize.py to see the heatmap and PCA plot.")
