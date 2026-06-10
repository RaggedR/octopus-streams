"""
Experiment 05: Sparse Autoencoder Feature Analysis.

Mirror of pythia/07_sae_features.py for the RSK encoder model.

Trains TopK sparse autoencoders on the residual stream at layers 0, 2, 5
to decompose the model's internal representations into monosemantic features.

Layer choice rationale (from ablation experiment 02):
  - Layer 0: feature extraction (reads Q)
  - Layer 2: critical computation (most important layer, ablation → 62%)
  - Layer 5: idle/aggregation (ablation barely hurts)

The question: what combinatorial features does each layer compute, and do they
differ across domains?
"""

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn

from load_model import load_rsk_model
from hooks import collect_residual_streams
from domains import generate_all_domains, generate_uniform, domain_batch, DOMAINS


# ── Config ────────────────────────────────────────────────────────────────────

D_MODEL = 128
EXPANSION = 8
N_FEATURES = D_MODEL * EXPANSION  # 1024

SAE_LAYERS = [0, 2, 5]

# SAE training
N_TRAIN_PERMS = 5000  # permutations for SAE training data
SAE_STEPS = 30_000
SAE_BATCH = 128  # activation vectors per SAE training step
SAE_LR = 3e-4
TOP_K = 20  # active features per input (proportional to d_model=128)
LOG_EVERY = 1000

# Domain analysis
N_SAMPLES = 200
BATCH_SIZE = 50
SEED = 42

RESULTS_DIR = Path(__file__).parent / "results"

DOMAIN_NAMES = list(DOMAINS.keys())
DOMAIN_COLOURS = {
    "uniform": "#1f77b4",
    "involution": "#e377c2",
    "wide": "#2ca02c",
    "tall": "#ff7f0e",
    "derangement": "#9467bd",
}

# ── Device ────────────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


# ── Sparse Autoencoder ───────────────────────────────────────────────────────

class SparseAutoencoder(nn.Module):
    """TopK sparse autoencoder (same architecture as Pythia experiment)."""

    def __init__(self, d_model, n_features, k=TOP_K):
        super().__init__()
        self.d_model = d_model
        self.n_features = n_features
        self.k = k

        self.encoder = nn.Linear(d_model, n_features)
        self.decoder = nn.Linear(n_features, d_model)

        # Tied initialisation
        nn.init.kaiming_uniform_(self.decoder.weight)
        self.encoder.weight.data = self.decoder.weight.data.T.clone()
        nn.init.zeros_(self.encoder.bias)
        nn.init.zeros_(self.decoder.bias)
        self._constrain_decoder()

    def _constrain_decoder(self):
        with torch.no_grad():
            norms = self.decoder.weight.norm(dim=0, keepdim=True).clamp(min=1e-8)
            self.decoder.weight.div_(norms)

    def forward(self, x):
        x_centered = x - self.decoder.bias
        pre_acts = self.encoder(x_centered)

        topk_values, topk_indices = pre_acts.topk(self.k, dim=-1)
        topk_values = torch.relu(topk_values)

        h = torch.zeros_like(pre_acts)
        h.scatter_(-1, topk_indices, topk_values)

        x_hat = self.decoder(h)
        return x_hat, h


# ── Helpers ──────────────────────────────────────────────────────────────────

def collect_activations(model, perms, layer, batch_size=BATCH_SIZE):
    """Collect residual stream activations at a specific layer.

    Returns: (n_perms * seq_len, d_model) tensor on CPU.
    """
    all_acts = []
    for i in range(0, len(perms), batch_size):
        batch_perms = perms[i : i + batch_size]
        values, positions = domain_batch(batch_perms)
        streams = collect_residual_streams(model, values, positions)
        acts = streams[layer]  # (batch, seq, d_model)
        all_acts.append(acts.reshape(-1, acts.shape[-1]).cpu())
    return torch.cat(all_acts, dim=0).float()


def get_feature_activations(sae, activation_vectors, act_scale=1.0):
    """Run activation vectors through SAE, return feature activations."""
    sae.eval()
    all_h = []
    chunk_size = 4096
    with torch.no_grad():
        for start in range(0, len(activation_vectors), chunk_size):
            end = min(start + chunk_size, len(activation_vectors))
            x = activation_vectors[start:end] / act_scale
            _, h = sae(x)
            all_h.append(h.cpu())
    return torch.cat(all_h, dim=0)


def train_sae(activation_pool, layer):
    """Train a TopK SAE on a pool of activation vectors."""
    act_scale = activation_pool.std().item()
    activation_pool = activation_pool / act_scale
    n_vectors = len(activation_pool)

    print(f"\n  Training SAE for layer {layer} "
          f"({n_vectors} vectors, {SAE_STEPS} steps, k={TOP_K})...")
    print(f"    Activation scale: {act_scale:.4f}")

    sae = SparseAutoencoder(D_MODEL, N_FEATURES, k=TOP_K).to(DEVICE)
    optimizer = torch.optim.Adam(sae.parameters(), lr=SAE_LR)

    history = {"loss": [], "l0": [], "step": []}
    t0 = time.time()

    for step in range(SAE_STEPS):
        idx = np.random.randint(0, n_vectors, size=SAE_BATCH)
        x = activation_pool[idx].to(DEVICE)

        x_hat, h = sae(x)
        mse = (x - x_hat).pow(2).mean()

        optimizer.zero_grad()
        mse.backward()
        optimizer.step()
        sae._constrain_decoder()

        if (step + 1) % LOG_EVERY == 0:
            l0 = (h > 0).float().sum(dim=1).mean().item()
            elapsed = time.time() - t0
            history["loss"].append(mse.item())
            history["l0"].append(l0)
            history["step"].append(step + 1)
            print(f"    Step {step+1:>6}/{SAE_STEPS}  "
                  f"MSE={mse.item():.6f}  L0={l0:.0f}  ({elapsed:.0f}s)")

    # Final evaluation
    sae.eval()
    with torch.no_grad():
        eval_idx = np.random.randint(0, n_vectors, size=min(4096, n_vectors))
        x = activation_pool[eval_idx].to(DEVICE)
        x_hat, h = sae(x)
        final_l0 = (h > 0).float().sum(dim=1).mean().item()
        n_dead = ((h > 0).float().sum(dim=0) == 0).sum().item()
        final_mse = (x - x_hat).pow(2).mean().item()

    print(f"    Final: L0={final_l0:.0f}  MSE={final_mse:.6f}  "
          f"dead={n_dead}/{N_FEATURES} ({n_dead/N_FEATURES:.1%})")

    return sae.cpu(), history, act_scale


def describe_token(token_idx, n):
    """Describe a token position in human-readable terms."""
    if token_idx < n:
        return f"P[{token_idx}]"
    else:
        return f"Q[{token_idx - n}]"


# ── Main ─────────────────────────────────────────────────────────────────────

def run_experiment(n: int):
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    print(f"\n{'='*60}")
    print(f"  Experiment 05 — SAE Features (n={n})")
    print(f"{'='*60}")

    model, config = load_rsk_model(n)
    seq_len = 2 * n

    # ── Phase 1: Train SAEs ──────────────────────────────────────────────
    print(f"\n  Phase 1: Training SAEs on {N_TRAIN_PERMS} uniform permutations")

    train_perms = generate_uniform(n, N_TRAIN_PERMS, seed=SEED + 100)
    trained_saes = {}
    all_histories = {}
    act_scales = {}

    for layer in SAE_LAYERS:
        print(f"\n  {'─'*50}")
        print(f"  Layer {layer}")
        act_pool = collect_activations(model, train_perms, layer, batch_size=100)
        print(f"    Activation pool: {act_pool.shape}")

        sae, history, act_scale = train_sae(act_pool, layer)

        prefix = f"05_n{n}"
        torch.save({
            "state_dict": sae.state_dict(),
            "d_model": D_MODEL,
            "n_features": N_FEATURES,
            "layer": layer,
            "top_k": TOP_K,
            "act_scale": act_scale,
        }, RESULTS_DIR / f"{prefix}_sae_layer{layer}_weights.pt")

        trained_saes[layer] = sae
        all_histories[layer] = history
        act_scales[layer] = act_scale
        del act_pool

    # ── Phase 2: Domain Feature Analysis ─────────────────────────────────
    print(f"\n  Phase 2: Domain feature analysis")

    all_domains = generate_all_domains(n, N_SAMPLES, seed=SEED)

    # domain_features[domain][layer] = mean feature activation (n_features,)
    domain_features = {d: {} for d in DOMAIN_NAMES}

    for layer in SAE_LAYERS:
        sae = trained_saes[layer]
        scale = act_scales[layer]
        print(f"\n  Layer {layer} features across domains:")

        for domain_name in DOMAIN_NAMES:
            perms = all_domains[domain_name]
            acts = collect_activations(model, perms, layer)
            feat_acts = get_feature_activations(sae, acts, act_scale=scale)
            mean_act = feat_acts.mean(dim=0).numpy()
            domain_features[domain_name][layer] = mean_act

            n_alive = ((feat_acts > 0).float().mean(dim=0) > 0.01).sum().item()
            print(f"    {domain_name:>12s}: {n_alive} features alive (>1%)")

    # ── Phase 3: Feature Census ──────────────────────────────────────────
    print(f"\n  Phase 3: Feature census")

    ALIVE_THRESHOLD = 1e-3
    census = {}

    for layer in SAE_LAYERS:
        acts_matrix = np.stack([domain_features[d][layer] for d in DOMAIN_NAMES])
        max_per_feat = acts_matrix.max(axis=0)

        dead = max_per_feat < ALIVE_THRESHOLD
        alive_mask = ~dead
        activated_in = (acts_matrix > ALIVE_THRESHOLD).sum(axis=0)

        universal = alive_mask & (activated_in == len(DOMAIN_NAMES))
        domain_specific = alive_mask & (activated_in == 1)
        shared = alive_mask & (activated_in >= 2) & (activated_in < len(DOMAIN_NAMES))

        n_alive = int(alive_mask.sum())
        census[layer] = {
            "total": N_FEATURES,
            "dead": int(dead.sum()),
            "alive": n_alive,
            "universal": int(universal.sum()),
            "shared": int(shared.sum()),
            "domain_specific": int(domain_specific.sum()),
        }

        c = census[layer]
        print(f"\n    Layer {layer}:")
        print(f"      Alive:           {c['alive']:>5} / {N_FEATURES}")
        print(f"      Dead:            {c['dead']:>5}")
        print(f"      Universal:       {c['universal']:>5}")
        print(f"      Shared (2–4):    {c['shared']:>5}")
        print(f"      Domain-specific: {c['domain_specific']:>5}")

    # ── Phase 4: Max-activating token analysis ───────────────────────────
    print(f"\n  Phase 4: Max-activating token analysis")

    # For each SAE layer, find which token positions maximally activate top features
    token_analysis = {}
    for layer in SAE_LAYERS:
        sae = trained_saes[layer]
        scale = act_scales[layer]

        # Collect per-token activations (keep token identity)
        perms = all_domains["uniform"]
        all_feat_acts = []
        all_token_info = []

        for i in range(0, len(perms), BATCH_SIZE):
            batch_perms = perms[i : i + BATCH_SIZE]
            values, positions = domain_batch(batch_perms)
            streams = collect_residual_streams(model, values, positions)
            acts = streams[layer]  # (batch, seq, d_model)
            batch_sz, seq, d = acts.shape

            feat_acts = get_feature_activations(
                sae, acts.reshape(-1, d).cpu(), act_scale=scale
            )
            all_feat_acts.append(feat_acts)

            # Record token metadata
            for b in range(batch_sz):
                for s in range(seq):
                    all_token_info.append({
                        "perm_idx": i + b,
                        "token_pos": s,
                        "token_label": describe_token(s, n),
                        "value": values[b, s].item(),
                        "row": positions[b, s, 0].item(),
                        "col": positions[b, s, 1].item(),
                        "tableau": "P" if positions[b, s, 2].item() == 0 else "Q",
                    })

        all_feat_acts = torch.cat(all_feat_acts, dim=0)
        mean_acts = all_feat_acts.mean(dim=0)
        top20 = mean_acts.argsort(descending=True)[:20]

        layer_analysis = {}
        for feat_idx in top20:
            fi = feat_idx.item()
            feat_vals = all_feat_acts[:, fi]
            top5 = feat_vals.argsort(descending=True)[:5]

            examples = []
            for pos in top5:
                p = pos.item()
                info = all_token_info[p]
                examples.append({
                    **info,
                    "activation": round(feat_vals[p].item(), 4),
                })

            # Summarise: what tableau positions does this feature prefer?
            top20_positions = feat_vals.argsort(descending=True)[:50]
            tableau_counts = {"P": 0, "Q": 0}
            pos_counts = {}
            for p in top20_positions:
                info = all_token_info[p.item()]
                tableau_counts[info["tableau"]] += 1
                label = info["token_label"]
                pos_counts[label] = pos_counts.get(label, 0) + 1

            top_pos = sorted(pos_counts.items(), key=lambda x: -x[1])[:3]

            layer_analysis[fi] = {
                "mean_activation": round(mean_acts[fi].item(), 6),
                "top_examples": examples,
                "tableau_preference": tableau_counts,
                "top_positions": top_pos,
            }

        token_analysis[layer] = layer_analysis

        print(f"\n    Layer {layer} — top 10 features:")
        for fi, data in list(layer_analysis.items())[:10]:
            pref = "P" if data["tableau_preference"]["P"] > data["tableau_preference"]["Q"] else "Q"
            top_pos_str = ", ".join(f"{p}({c})" for p, c in data["top_positions"])
            print(f"      F{fi:>4}: mean={data['mean_activation']:.4f}  "
                  f"prefers={pref}  positions=[{top_pos_str}]")

    # ── Phase 5: Figures ─────────────────────────────────────────────────
    print(f"\n  Phase 5: Figures")
    prefix = f"05_n{n}"

    # 1. Training curves
    fig, axes = plt.subplots(len(SAE_LAYERS), 2, figsize=(12, 3.5 * len(SAE_LAYERS)))
    for row, layer in enumerate(SAE_LAYERS):
        h = all_histories[layer]
        axes[row, 0].plot(h["step"], h["loss"], linewidth=1.5)
        axes[row, 0].set_title(f"Layer {layer} — MSE Loss")
        axes[row, 0].set_xlabel("Step")
        axes[row, 0].grid(alpha=0.3)

        axes[row, 1].plot(h["step"], h["l0"], color="green", linewidth=1.5)
        axes[row, 1].axhline(TOP_K, color="gray", linestyle="--", alpha=0.5)
        axes[row, 1].set_title(f"Layer {layer} — L0 Sparsity")
        axes[row, 1].set_xlabel("Step")
        axes[row, 1].grid(alpha=0.3)
    fig.suptitle(f"RSK n={n}: SAE Training Curves", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"{prefix}_sae_training.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2. Feature specificity (stacked bars)
    fig, ax = plt.subplots(figsize=(8, 5))
    categories = ["dead", "domain_specific", "shared", "universal"]
    cat_labels = ["Dead", "Domain-specific", "Shared (2–4)", "Universal"]
    cat_colors = ["#d62728", "#ff7f0e", "#1f77b4", "#2ca02c"]

    x = np.arange(len(SAE_LAYERS))
    bottoms = np.zeros(len(SAE_LAYERS))
    for cat, label, color in zip(categories, cat_labels, cat_colors):
        values = [census[layer][cat] for layer in SAE_LAYERS]
        ax.bar(x, values, 0.5, bottom=bottoms, label=label, color=color, alpha=0.85)
        for i, v in enumerate(values):
            if v > N_FEATURES * 0.03:
                ax.text(x[i], bottoms[i] + v / 2, str(v),
                        ha="center", va="center", fontsize=8,
                        fontweight="bold", color="white")
        bottoms += values

    ax.set_xticks(x)
    ax.set_xticklabels([f"Layer {l}" for l in SAE_LAYERS])
    ax.set_ylabel("Number of features")
    ax.set_title(f"RSK n={n}: Feature Specificity by Layer")
    ax.legend(loc="upper right")
    ax.set_ylim(0, N_FEATURES * 1.05)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"{prefix}_feature_specificity.png", dpi=150)
    plt.close(fig)

    # 3. Domain × feature heatmap
    N_TOP = 30
    fig, axes = plt.subplots(1, len(SAE_LAYERS), figsize=(6 * len(SAE_LAYERS), 8))
    for col, layer in enumerate(SAE_LAYERS):
        ax = axes[col]
        acts_matrix = np.stack([domain_features[d][layer] for d in DOMAIN_NAMES])
        max_per_feat = acts_matrix.max(axis=0)
        top_idx = max_per_feat.argsort()[::-1][:N_TOP]

        submatrix = acts_matrix[:, top_idx].T
        row_max = submatrix.max(axis=1, keepdims=True)
        row_max = np.where(row_max > 0, row_max, 1.0)
        submatrix_norm = submatrix / row_max

        im = ax.imshow(submatrix_norm, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(range(len(DOMAIN_NAMES)))
        ax.set_xticklabels(DOMAIN_NAMES, fontsize=8, rotation=45, ha="right")
        ax.set_yticks(range(N_TOP))
        ax.set_yticklabels([f"F{i}" for i in top_idx], fontsize=6)
        ax.set_title(f"Layer {layer} — Top {N_TOP}", fontsize=11)
        plt.colorbar(im, ax=ax, shrink=0.6)

    fig.suptitle(f"RSK n={n}: Domain × Feature Heatmap", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"{prefix}_domain_feature_heatmap.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    # 4. P vs Q feature preference by layer
    fig, axes = plt.subplots(1, len(SAE_LAYERS), figsize=(5 * len(SAE_LAYERS), 4))
    for col, layer in enumerate(SAE_LAYERS):
        ax = axes[col]
        analysis = token_analysis[layer]

        p_prefs = []
        q_prefs = []
        labels = []
        for fi, data in analysis.items():
            total = data["tableau_preference"]["P"] + data["tableau_preference"]["Q"]
            if total > 0:
                p_frac = data["tableau_preference"]["P"] / total
                p_prefs.append(p_frac)
                q_prefs.append(1 - p_frac)
                labels.append(f"F{fi}")

        y = range(len(labels))
        ax.barh(y, p_prefs, color="#1f77b4", label="P (insertion)")
        ax.barh(y, [-q for q in q_prefs], color="#ff7f0e", label="Q (recording)")
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlim(-1, 1)
        ax.axvline(0, color="black", linewidth=0.5)
        ax.set_title(f"Layer {layer}", fontsize=11)
        ax.set_xlabel("← Q preference | P preference →")
        if col == 0:
            ax.legend(fontsize=8)

    fig.suptitle(f"RSK n={n}: Feature P/Q Tableau Preference", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"{prefix}_pq_preference.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Save JSON ────────────────────────────────────────────────────────
    results = {
        "n": n,
        "sae_config": {
            "d_model": D_MODEL,
            "n_features": N_FEATURES,
            "top_k": TOP_K,
            "layers": SAE_LAYERS,
            "train_steps": SAE_STEPS,
            "n_train_perms": N_TRAIN_PERMS,
        },
        "census": {str(k): v for k, v in census.items()},
        "token_analysis": {
            str(layer): {
                str(fi): {
                    "mean_activation": data["mean_activation"],
                    "tableau_preference": data["tableau_preference"],
                    "top_positions": data["top_positions"],
                }
                for fi, data in analysis.items()
            }
            for layer, analysis in token_analysis.items()
        },
    }

    with open(RESULTS_DIR / f"{prefix}_sae_results.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(RESULTS_DIR / f"{prefix}_sae_training_history.json", "w") as f:
        json.dump({str(k): v for k, v in all_histories.items()}, f, indent=2)

    print(f"\n  Saved results to {RESULTS_DIR}/{prefix}_sae_*")
    return results


if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for n in [8, 10]:
        run_experiment(n)
