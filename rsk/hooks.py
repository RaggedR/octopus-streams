"""
Manual head decomposition for PyTorch's nn.TransformerEncoder.

PyTorch's nn.MultiheadAttention calls F.multi_head_attention_forward internally,
which bypasses module-level hooks. This module manually decomposes the forward
pass to extract per-head attention patterns and output vectors.

The RSK model uses pre-norm TransformerEncoderLayers (norm_first=True):
    x = x + self_attn(norm1(x))
    x = x + ffn(norm2(x))

MHA weights are packed as:
    in_proj_weight: (3*d_model, d_model)  — Q, K, V projections concatenated
    in_proj_bias:   (3*d_model,)
    out_proj.weight: (d_model, d_model)   — output projection
    out_proj.bias:   (d_model,)
"""

import torch
import torch.nn.functional as F


def extract_all_head_outputs(
    model,
    values: torch.Tensor,
    positions: torch.Tensor,
    zero_heads: set[tuple[int, int]] | None = None,
) -> dict:
    """
    Manually decompose the forward pass to extract per-head data.

    Args:
        model: RSKEncoder instance (must be in eval mode)
        values: (batch, 2n) entry values
        positions: (batch, 2n, 3) [row, col, tableau_id]
        zero_heads: set of (layer, head) tuples to ablate (zero out)

    Returns:
        dict with keys:
            "L{l}H{h}_attn":   (batch, seq, seq) attention patterns
            "L{l}H{h}_output": (batch, seq, d_model) per-head residual contribution
            "pooled":          (batch, d_model) mean-pooled final representation
            "logits":          (batch, n, n) classification logits
    """
    if zero_heads is None:
        zero_heads = set()

    d_model = model.config.d_model
    n_heads = model.config.nhead
    d_head = d_model // n_heads

    with torch.no_grad():
        # 1. Embed tokens
        x = model.embedding(values, positions)  # (batch, 2n, d_model)

        result = {}

        # 2. Process each encoder layer manually
        for l, layer in enumerate(model.encoder.layers):
            # --- Self-attention block (pre-norm) ---
            x_norm = layer.norm1(x)  # (batch, seq, d_model)

            # Unpack in_proj into Q, K, V projections
            W = layer.self_attn.in_proj_weight  # (3*d_model, d_model)
            b = layer.self_attn.in_proj_bias    # (3*d_model,)
            W_Q, W_K, W_V = W.chunk(3, dim=0)  # each (d_model, d_model)
            b_Q, b_K, b_V = b.chunk(3, dim=0)  # each (d_model,)

            # Project: (batch, seq, d_model)
            Q = F.linear(x_norm, W_Q, b_Q)
            K = F.linear(x_norm, W_K, b_K)
            V = F.linear(x_norm, W_V, b_V)

            batch, seq = Q.shape[:2]

            # Reshape to multi-head: (batch, n_heads, seq, d_head)
            Q = Q.view(batch, seq, n_heads, d_head).transpose(1, 2)
            K = K.view(batch, seq, n_heads, d_head).transpose(1, 2)
            V = V.view(batch, seq, n_heads, d_head).transpose(1, 2)

            # Scaled dot-product attention (no causal mask — encoder)
            scale = d_head ** -0.5
            attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
            attn_weights = torch.softmax(attn_scores, dim=-1)  # (batch, n_heads, seq, seq)

            # Per-head output in V-space
            head_v_out = torch.matmul(attn_weights, V)  # (batch, n_heads, seq, d_head)

            # Project each head through its slice of out_proj
            W_O = layer.self_attn.out_proj.weight  # (d_model, d_model)
            b_O = layer.self_attn.out_proj.bias     # (d_model,)

            attn_total = torch.zeros(batch, seq, d_model, device=x.device)

            for h in range(n_heads):
                # Store attention pattern
                result[f"L{l}H{h}_attn"] = attn_weights[:, h]  # (batch, seq, seq)

                # Per-head contribution through out_proj
                # W_O columns h*d_head:(h+1)*d_head correspond to head h's input
                W_O_h = W_O[:, h * d_head : (h + 1) * d_head]  # (d_model, d_head)
                head_out = F.linear(head_v_out[:, h], W_O_h)  # (batch, seq, d_model)

                # Zero out if ablating
                if (l, h) in zero_heads:
                    head_out = torch.zeros_like(head_out)

                result[f"L{l}H{h}_output"] = head_out
                attn_total = attn_total + head_out

            # Add out_proj bias once (shared across heads)
            attn_total = attn_total + b_O

            # Residual connection (dropout is identity in eval mode)
            x = x + attn_total

            # --- FFN block (pre-norm) ---
            x_norm2 = layer.norm2(x)
            ff_out = layer.linear2(F.gelu(layer.linear1(x_norm2)))
            x = x + ff_out

        # 3. Mean pool over all tokens
        pooled = x.mean(dim=1)  # (batch, d_model)
        result["pooled"] = pooled

        # 4. Classification logits
        logits = torch.stack([head(pooled) for head in model.heads], dim=1)
        result["logits"] = logits

    return result


def verify_decomposition(
    model, values: torch.Tensor, positions: torch.Tensor
) -> float:
    """
    Verify that the manual decomposition matches the native forward pass.

    Returns the max absolute difference in logits. Should be < 1e-4
    (float32 error accumulates across 6 layers with GELU activations).
    """
    model.eval()
    with torch.no_grad():
        native_logits = model(values, positions)
        manual = extract_all_head_outputs(model, values, positions)
        diff = (native_logits - manual["logits"]).abs().max().item()
    return diff


def collect_residual_streams(
    model,
    values: torch.Tensor,
    positions: torch.Tensor,
) -> dict[int, torch.Tensor]:
    """
    Collect residual stream activations after each encoder layer.

    Lighter than extract_all_head_outputs — skips per-head decomposition
    and just tracks the residual stream state using the model's native layers.

    Returns:
        dict: layer_index → (batch, seq, d_model) tensor
              Also key -1 for embedding output (before any layers).
    """
    model.eval()
    with torch.no_grad():
        x = model.embedding(values, positions)
        streams = {-1: x.clone()}

        for l, layer in enumerate(model.encoder.layers):
            # Use native forward (norm_first=True, no mask, eval mode)
            x = layer(x)
            streams[l] = x.clone()

    return streams


def head_output_norms(result: dict, n_layers: int, n_heads: int) -> torch.Tensor:
    """
    Extract per-head output norms from decomposition result.

    Returns:
        (batch, n_layers * n_heads) tensor of mean L2 norms over positions
    """
    norms = []
    for l in range(n_layers):
        for h in range(n_heads):
            # (batch, seq, d_model) → L2 norm over d_model, mean over seq
            head_out = result[f"L{l}H{h}_output"]
            norm = head_out.norm(dim=-1).mean(dim=-1)  # (batch,)
            norms.append(norm)
    return torch.stack(norms, dim=-1)  # (batch, total_heads)
