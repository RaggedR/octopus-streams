#!/usr/bin/env python3
"""Verify the growth diagram claims from the README."""
import json, pathlib

# Verify growth diagram per-head scores
gd_path = pathlib.Path("rsk/results/06b_cyl10101010_growth_diagram.json")
if not gd_path.exists():
    print(f"WARNING: {gd_path} not found — skipping growth diagram check")
else:
    d = json.loads(gd_path.read_text())

    # Per-head maxima for Layer 2
    head_scores = d["head_scores"]
    l2_heads = {k: max(v) for k, v in head_scores.items() if k.startswith("L2")}
    print("Layer 2 per-head maxima:")
    for k, v in sorted(l2_heads.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v:.3f}x")

    # Layer-averaged max from layer_depth_scores
    lds = d["layer_depth_scores"]
    l2_avg_max = max(lds["2"]) if "2" in lds else max(lds[2])
    print(f"\nLayer 2 averaged max: {l2_avg_max:.3f}x")
    assert 1.5 < l2_avg_max < 1.8, f"Expected ~1.65x, got {l2_avg_max:.3f}x"
    print("Growth diagram claims VERIFIED")

# Verify ablation
abl_path = pathlib.Path("rsk/results/02_n10_ablation.json")
if not abl_path.exists():
    print(f"WARNING: {abl_path} not found — skipping ablation check")
else:
    d = json.loads(abl_path.read_text())
    l2_acc = d["layer 2"]["greedy_exact_match"]
    print(f"\nLayer 2 ablation accuracy: {l2_acc:.4f}")
    assert abs(l2_acc - 0.62) < 0.01, f"Expected ~0.62, got {l2_acc}"
    print("Ablation claim VERIFIED")
