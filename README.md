# Octopus Streams

Mechanistic interpretability of small transformers, studying how attention heads organise computation.

## RSK Transformer

Interpretability experiments on a 1.2M-parameter encoder-only transformer that computes the **inverse RSK correspondence** — a bijection from pairs of Young tableaux to permutations — with 100% accuracy. The same architecture also achieves near-perfect accuracy on **cylindric plane partitions**, a problem with no known closed-form algorithm.

**Trained models**: [HuggingFace](https://huggingface.co/RobBobin/rsk-transformer)
**Training code**: [github.com/RaggedR/rsk-transformer](https://github.com/RaggedR/rsk-transformer)
**Thesis** (mathematical foundation): [arXiv:2110.12629](https://arxiv.org/abs/2110.12629)

### Experiments

| Script | What it does |
|--------|-------------|
| `rsk/01_head_correlations.py` | Head output correlation matrix and PCA |
| `rsk/02_ablation.py` | Per-head and per-layer ablation |
| `rsk/03_interpretation.py` | Attention pattern analysis and direct logit attribution |
| `rsk/04_phase_transition.py` | Entropy phase transition across n=8, 10, 15 |
| `rsk/05_sae_features.py` | Sparse autoencoder features across combinatorial domains |
| `rsk/06_growth_diagram.py` | Growth diagram vs bumping: attention analysis (permutation RSK) |
| `rsk/06b_cylindric_growth_diagram.py` | Growth diagram attention analysis (cylindric plane partitions) |

### Key findings

- **Pipeline, not streams**: L0 parses P/Q, L1-L2 cross-reference, L3 refines, L4-L5 idle (for n=8/10). Layer 2 is critical — ablating it drops accuracy to 62%.
- **Phase transition**: sharp entropy boundary between structured and uniform attention; disappears at n=15 as the pipeline fills all layers.
- **Growth diagram signal (partial result)**: on the cylindric CPP model, attention in Layer 2 concentrates on the partition triples where the Burge local rule operates (2.5-3.7x above baseline). On permutation RSK, the signal is weaker and inconclusive.

See [rsk/results/06_report.md](rsk/results/06_report.md) for the full interpretability report.

### What we can't say yet

Attention concentration on local rule triples is suggestive but not conclusive. We haven't shown the *content* of attention implements the Burge rule. Next steps: linear probing for intermediate partition states, content analysis of attention value vectors, cross-model comparison between permutation/Hillman-Grassl/cylindric models.

## Pythia-70m

Parallel analysis on Pythia-70m (70M parameters) for comparison — same experimental pipeline applied to a natural language model across prose, code, poetry, and Wikipedia domains.

## Dependencies

`torch`, `numpy`, `matplotlib`, `scikit-learn`. The RSK experiments depend on the training code at [RaggedR/rsk-transformer](https://github.com/RaggedR/rsk-transformer) (added to `sys.path` at runtime).

## Running

```bash
cd rsk
python 01_head_correlations.py   # ~2 min
python 02_ablation.py            # ~5 min
python 06b_cylindric_growth_diagram.py  # ~10 min
```

All output goes to `results/` as PNG plots, JSON stats, and NPY arrays.
