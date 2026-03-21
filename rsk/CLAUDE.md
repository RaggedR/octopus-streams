# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Mechanistic interpretability experiments on a trained encoder-only transformer that computes the **inverse RSK correspondence** — a bijection from pairs of Standard Young Tableaux (P, Q) back to permutations σ ∈ S_n. Part of the "Octopus Streams" project studying whether attention heads form semi-independent processing streams. The parallel analysis on Pythia-70m lives in `../pythia/`.

## Critical External Dependency

All experiment scripts depend on the **source model code** at `~/git/paul/rsk/`. This is added to `sys.path` at runtime by `load_model.py`. Key imports from there:

- `model.RSKEncoder`, `config.ModelConfig` — model architecture and config
- `data.encode_tableaux` — converts (P, Q) tableaux to (values, positions) tensors
- `rsk.rsk_forward`, `rsk.tableau_shape` — RSK algorithm implementation
- `train.masked_greedy_decode` — constrained decoding for accuracy evaluation

Checkpoints live at `~/git/paul/rsk/checkpoints/encoder_n{8,10,15}/best.pt`.

## Running Experiments

```bash
cd /Users/robin/git/octopus-streams/rsk
python 01_head_correlations.py   # ~2 min, runs n=8 and n=10
python 02_ablation.py            # ~5 min, runs n=8 and n=10
python 03_interpretation.py      # ~5 min, runs n=8 and n=10
python 04_phase_transition.py    # ~3 min, runs n=8, n=10, n=15
python 05_sae_features.py        # ~25 min, trains SAEs for n=8 and n=10
```

Scripts are standalone — run in any order. All output goes to `results/` with prefix `{exp_num}_n{n}_*` (PNG plots, JSON stats, NPY arrays, PT weights).

## Architecture

**Three shared utility modules:**
- `load_model.py` — loads RSKEncoder from checkpoint, returns (model, config)
- `hooks.py` — manual MHA forward-pass decomposition (see below)
- `domains.py` — five combinatorial domain generators (uniform, involution, wide, tall, derangement) with batch encoding

**Five experiment scripts** (01–05) that import the utilities.

## Why hooks.py Exists (Key Technical Detail)

PyTorch's `nn.MultiheadAttention` internally calls `F.multi_head_attention_forward`, which **bypasses module-level hooks**. To get per-head attention patterns and output vectors, `hooks.py` manually decomposes the forward pass:

1. Splits `in_proj_weight` (3×d_model, d_model) into W_Q, W_K, W_V via `.chunk(3, dim=0)`
2. Projects Q, K, V and reshapes to (batch, n_heads, seq, d_head)
3. Computes attention (no causal mask — this is an encoder)
4. Projects each head's output through its slice of `out_proj.weight[:, h*d_head:(h+1)*d_head]`
5. The model uses **pre-norm**: `x = x + self_attn(norm1(x)); x = x + ffn(norm2(x))`

`verify_decomposition()` confirms the manual pass matches native forward within float32 tolerance (<1e-4).

## Model Shape

6 layers × 8 heads = 48 heads. d_model=128, d_head=16, dim_feedforward=512. Input is (values, positions) where positions encode (row, col, tableau_id) for each token. P tokens come first (tableau_id=0), Q tokens second (tableau_id=1). Sequence length = 2n.

## Key Findings (for context)

- **Pipeline, not streams**: L0 parses P/Q → L1-L2 cross-reference → L3 refines → L4-L5 mostly idle (for n=8/10)
- **Phase transition**: sharp entropy boundary between structured and uniform attention; disappears at n=15 (pipeline fills all layers)
- **SAE**: near-zero domain-specific features; Layer 0 has monosemantic P/Q position detectors; features become tableau-agnostic by Layer 5

## Dependencies

torch, numpy, matplotlib, scikit-learn (PCA in exp 01 only). Uses the venv at `../.venv/`.
