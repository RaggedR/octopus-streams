"""
Combinatorial domain generators for RSK mechanistic interpretability.

Five domains of permutations, each with distinct combinatorial properties:
    uniform     — random permutations (baseline)
    involution  — σ² = id (P = Q under RSK)
    wide        — long first row (large LIS)
    tall        — many rows (large LDS)
    derangement — no fixed points

Each domain generator returns a list of permutations as lists of ints (1-indexed).
The batch encoder converts permutations to model input tensors via RSK.
"""

import sys
from pathlib import Path
import random
import math

import torch

RSK_DIR = Path.home() / "git" / "paul" / "rsk"
if str(RSK_DIR) not in sys.path:
    sys.path.insert(0, str(RSK_DIR))

from rsk import rsk_forward, tableau_shape
from data import encode_tableaux


# ---------------------------------------------------------------------------
# Domain generators
# ---------------------------------------------------------------------------

def generate_uniform(n: int, count: int, seed: int = 42) -> list[list[int]]:
    """Random permutations of {1..n}."""
    rng = torch.Generator().manual_seed(seed)
    return [(torch.randperm(n, generator=rng) + 1).tolist() for _ in range(count)]


def generate_involution(n: int, count: int, seed: int = 42) -> list[list[int]]:
    """Involutions: σ(σ(i)) = i for all i (equivalently, P = Q under RSK)."""
    rng = torch.Generator().manual_seed(seed)
    results = []
    while len(results) < count:
        sigma = (torch.randperm(n, generator=rng) + 1).tolist()
        # Check involution property directly: σ(σ(i)) = i
        if all(sigma[sigma[i] - 1] == i + 1 for i in range(n)):
            results.append(sigma)
    return results


def generate_wide(n: int, count: int, seed: int = 42) -> list[list[int]]:
    """Permutations with long first row (LIS ≥ ceil(0.6n))."""
    threshold = math.ceil(0.6 * n)
    rng = torch.Generator().manual_seed(seed)
    results = []
    while len(results) < count:
        sigma = (torch.randperm(n, generator=rng) + 1).tolist()
        P, _ = rsk_forward(sigma)
        if len(P[0]) >= threshold:  # first row length = LIS
            results.append(sigma)
    return results


def generate_tall(n: int, count: int, seed: int = 42) -> list[list[int]]:
    """Permutations with many rows (LDS ≥ ceil(0.6n))."""
    threshold = math.ceil(0.6 * n)
    rng = torch.Generator().manual_seed(seed)
    results = []
    while len(results) < count:
        sigma = (torch.randperm(n, generator=rng) + 1).tolist()
        P, _ = rsk_forward(sigma)
        if len(P) >= threshold:  # number of rows = LDS
            results.append(sigma)
    return results


def generate_derangement(n: int, count: int, seed: int = 42) -> list[list[int]]:
    """Derangements: σ(i) ≠ i for all i."""
    rng = torch.Generator().manual_seed(seed)
    results = []
    while len(results) < count:
        sigma = (torch.randperm(n, generator=rng) + 1).tolist()
        if all(sigma[i] != i + 1 for i in range(n)):
            results.append(sigma)
    return results


DOMAINS = {
    "uniform": generate_uniform,
    "involution": generate_involution,
    "wide": generate_wide,
    "tall": generate_tall,
    "derangement": generate_derangement,
}


# ---------------------------------------------------------------------------
# Batch encoder
# ---------------------------------------------------------------------------

def domain_batch(
    perms: list[list[int]],
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert permutations to batched model input tensors.

    Applies RSK forward to get (P, Q), then encodes as (values, positions).

    Returns:
        values:    (batch, 2n) LongTensor
        positions: (batch, 2n, 3) LongTensor
    """
    vals_list, pos_list = [], []
    for sigma in perms:
        P, Q = rsk_forward(sigma)
        v, p = encode_tableaux(P, Q)
        vals_list.append(v)
        pos_list.append(p)

    values = torch.stack(vals_list).to(device)
    positions = torch.stack(pos_list).to(device)
    return values, positions


def generate_all_domains(
    n: int, count: int, seed: int = 42
) -> dict[str, list[list[int]]]:
    """Generate permutations for all five domains."""
    return {
        name: gen(n, count, seed=seed)
        for name, gen in DOMAINS.items()
    }


if __name__ == "__main__":
    # Verify each domain generator
    for n in [8, 10]:
        print(f"\n=== n={n} ===")
        for name, gen in DOMAINS.items():
            perms = gen(n, 20, seed=42)
            print(f"  {name:12s}: generated {len(perms)} permutations")

            # Verify all are valid permutations
            for sigma in perms:
                assert sorted(sigma) == list(range(1, n + 1)), f"Invalid perm: {sigma}"

            # Verify domain-specific properties
            if name == "involution":
                for sigma in perms:
                    assert all(sigma[sigma[i] - 1] == i + 1 for i in range(n))
            elif name == "derangement":
                for sigma in perms:
                    assert all(sigma[i] != i + 1 for i in range(n))
            elif name == "wide":
                threshold = math.ceil(0.6 * n)
                for sigma in perms:
                    P, _ = rsk_forward(sigma)
                    assert len(P[0]) >= threshold
            elif name == "tall":
                threshold = math.ceil(0.6 * n)
                for sigma in perms:
                    P, _ = rsk_forward(sigma)
                    assert len(P) >= threshold

        # Test batch encoding
        values, positions = domain_batch(perms)
        print(f"  Batch shapes: values={values.shape}, positions={positions.shape}")

    print("\nAll domain generators verified!")
