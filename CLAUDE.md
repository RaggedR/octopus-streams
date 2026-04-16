# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Mechanistic interpretability of small transformers. Studies how attention heads organise computation in a 1.2M-parameter encoder-only transformer trained on the inverse RSK correspondence (a bijection from pairs of Young tableaux to permutations). Also includes Pythia-70m comparison experiments.

## Commands

```bash
# RSK experiments (run from rsk/ directory)
cd rsk
python 01_head_correlations.py   # ~2 min — head output correlation + PCA
python 02_ablation.py            # ~5 min — per-head/layer ablation
python 03_interpretation.py      # Attention pattern analysis, direct logit attribution
python 04_phase_transition.py    # Entropy phase transition across n=8,10,15
python 05_sae_features.py        # Sparse autoencoder features
python 06_growth_diagram.py      # Growth diagram vs bumping attention (permutation RSK)
python 06b_cylindric_growth_diagram.py  # Growth diagram attention (cylindric)
```

## Key Findings

- Pipeline structure (not streams): L0 parses P/Q, L1-L2 cross-reference, L3 refines, L4-L5 idle.
- Layer 2 critical — ablating drops accuracy to 62%.
- Growth diagram signal in cylindric CPP model: attention concentrates on Burge local rule triples (2.5-3.7x above baseline).

## Architecture

- `rsk/` — 7 experiment scripts, outputs to `results/` (PNG, JSON, NPY)
- `pythia/` — Parallel analysis on Pythia-70m for comparison
- Depends on training code at [RaggedR/rsk-transformer](https://github.com/RaggedR/rsk-transformer)

## Stack

Python, PyTorch, NumPy, matplotlib, scikit-learn.
