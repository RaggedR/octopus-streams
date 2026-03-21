# Can We See the Local Rules?

## Mechanistic Interpretability of an RSK Transformer

*Robin Langer, March 2026*
*Report for Claudius (and anyone else who's curious)*

---

## Background

We have a small transformer (1.2M parameters, 6 layers, 8 attention heads) that has learned to compute the **inverse RSK correspondence** with 100% accuracy on held-out data. Given a pair of Young tableaux (P, Q), it outputs the permutation that produced them. It also achieves near-perfect accuracy on **cylindric plane partitions** — a harder problem with no known closed-form algorithm.

The question: **what algorithm did it learn?**

There are two fundamentally different ways to compute RSK:

1. **Schensted's bumping algorithm** — sequential. Process entries one at a time, each time tracing a "bumping path" through the tableau. This is the classical description taught in textbooks.

2. **Fomin's growth diagrams** — parallel. Apply a single **local rule** at every cell of an n×n grid. The local rule is the same operation everywhere: it takes three partitions and determines the fourth. The entire bijection emerges from composing this one rule. See [Langer 2013, §2.1–2.2 and §4.2](https://arxiv.org/abs/2110.12629) for the full treatment.

The distinction matters because the transformer is encoder-only — all tokens attend to all others simultaneously, with no causal mask. It *cannot* do sequential bumping in the classical sense. Growth diagrams are naturally parallel. So architecturally, the local rule is the more natural fit.

## What We Did

### Experiment 1: Permutation RSK (n=10)

For the permutation model, we extracted per-head attention patterns from 500 random permutations and measured two things:

- **Growth diagram score**: when Q-token with value i attends to P-token with value σ(i), that corresponds to the "hit" at cell (i, σ(i)) of the growth diagram. We measured how much above baseline this attention is.

- **Bumping path score**: when Q-token i attends to P-tokens at the (row, col) positions along the reverse bumping path for step i.

**Results**: mixed. The strongest growth diagram signal was 2.05× baseline (L1H7). The strongest bumping signal was 3.36× (L0H0), but this is an artifact — Q and P tokens at the same (row, col) share position embeddings, inflating the score. Some heads showed *anti*-bumping (0.45×), actively avoiding bumping path targets.

Verdict: inconclusive for permutations. Neither algorithm dominates cleanly.

### Experiment 2: Cylindric Plane Partitions (profile 10101010, T=8)

This is the cleaner experiment. Cylindric plane partitions have **no alternative algorithm** — the bijection is *defined* by the Burge local rule applied recursively through a cylindric growth diagram. Whatever the model learned, it must functionally implement the local rule.

The cylindric model takes 8 partitions as input (104 tokens) and outputs 10 ALCD face labels. Each label corresponds to one application of the Burge local rule at a specific face of the growth diagram.

**The dependency structure**: the recursive composition processes 10 faces in a fixed order. Some can be computed in parallel (they involve non-adjacent partitions), others must wait for earlier results. This gives a minimum depth of 4 layers:

```
Depth 0: positions 0, 2, 4, 6  (independent — all from original CPP)
Depth 1: positions 1, 3, 5     (depend on depth 0 results)
Depth 2: positions 2, 4        (depend on depth 1)
Depth 3: position 3            (depends on depth 2)
```

Each local rule application connects three adjacent partitions: the center vertex and its two neighbors. For profile (1,0,1,0,1,0,1,0), even positions have π=1 ("big" vertices) and odd positions have π=0 ("small" vertices). The local rule always connects a big vertex with its small neighbors.

**What we measured**: for each attention head, we computed an 8×8 "partition attention matrix" — how much tokens from partition i attend to tokens from partition j. Then we checked whether attention concentrates on the specific partition triples predicted by the local rule at each depth level.

**Results**:

- **Layer 2 is the critical computation layer** (consistent with earlier ablation studies). Heads L2H0, L2H1, L2H3, L2H6 show 2.5–3.7× concentration on local rule triples.

- **Even/odd partition structure is visible**. The attention heatmaps show clear structure corresponding to the π=1 vs π=0 distinction — exactly the "big vertex vs small vertex" structure that the Burge local rule operates on.

- **The model does NOT map layers to dependency depths sequentially**. We predicted Layer 0 → Depth 0, Layer 1 → Depth 1, etc. The actual pattern is more complex — different heads within the same layer handle different depths. The model parallelises more aggressively than the sequential recursion requires, using its 8 heads per layer to handle multiple local rule applications simultaneously.

- **Layers 4–5 are idle**, with attention flattening out — the computation is complete by layer 3, consistent with the minimum depth of 4 being achievable in fewer than 6 layers when parallelised across heads.

## What We Can Say

The cylindric experiment gives the strongest evidence. The model's attention in its critical computation layers (especially Layer 2) concentrates on exactly the partition triples where the Burge local rule operates, at 2.5–3.7× above chance. The big/small vertex structure of the profile is reflected in the attention patterns.

This is consistent with the model having learned something that functions like Fomin's local rule — applied in parallel across multiple growth diagram faces using different attention heads.

## What We Can't Say (Yet)

- The signal is moderate (2–4× baseline), not overwhelming. There's a lot of "other" attention that doesn't correspond to local rule triples. The local rule explains *some* of the computation, not all of it.

- We measured attention *between partitions*, not the actual computation within each attention head. Knowing that partition 2 attends to partitions 1 and 3 is necessary but not sufficient — we'd need to show that the *content* of that attention implements the Burge rule (the specific partition arithmetic).

- The model may have found a different algorithm that happens to have similar attention patterns to the local rule. We can't rule this out from attention analysis alone.

## What Would Come Next

- **Larger profiles**: training cylindric models with T=10, 12 would give more faces and more structure to test against.

- **Content analysis**: instead of just measuring *where* attention goes, examine *what information* flows — do the attention value vectors encode partition differences?

- **Probing**: train linear probes at each layer to predict intermediate partition states in the growth diagram. If the model computes local rules, intermediate partitions should be linearly readable from the residual stream.

- **Cross-model comparison**: the permutation RSK model, the Hillman-Grassl model, and the cylindric model all use the same Burge local rule with different boundary conditions. If SAE features (or attention patterns) transfer across these models, that would be strong evidence for Fomin's framework as the learned algorithm.

---

*Code: [github.com/RaggedR/rsk-transformer](https://github.com/RaggedR/rsk-transformer)*
*Thesis: [arXiv:2110.12629](https://arxiv.org/abs/2110.12629) — see §2.2 for RSK via growth diagrams, §4.2 for the cylindric local rule as higher-order function*
