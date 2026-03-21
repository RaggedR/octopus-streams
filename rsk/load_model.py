"""
Shared model loading utility for RSK mechanistic interpretability experiments.

Loads a trained RSKEncoder from checkpoint and returns the model + config.
"""

import sys
from pathlib import Path

# Make the RSK training codebase importable
RSK_DIR = Path.home() / "git" / "paul" / "rsk"
if str(RSK_DIR) not in sys.path:
    sys.path.insert(0, str(RSK_DIR))

import torch
from model import RSKEncoder
from config import ModelConfig


CHECKPOINT_DIR = RSK_DIR / "checkpoints"


def load_rsk_model(
    n: int, device: str = "cpu"
) -> tuple[RSKEncoder, ModelConfig]:
    """
    Load trained RSK model from checkpoint.

    Args:
        n: permutation size (8, 10, or 15)
        device: torch device

    Returns:
        (model, config) — model in eval mode on the specified device
    """
    ckpt_path = CHECKPOINT_DIR / f"encoder_n{n}" / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint at {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt["model_config"]

    # Handle checkpoints saved before seq_len/vocab_size/task were added
    if getattr(config, "seq_len", None) is None:
        kwargs = {}
        for field in ("n", "d_model", "nhead", "num_layers", "dim_feedforward", "dropout"):
            if hasattr(config, field):
                kwargs[field] = getattr(config, field)
        config = ModelConfig(**kwargs)

    model = RSKEncoder(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"Loaded RSKEncoder n={n} from {ckpt_path.name}")
    print(f"  Val metrics: {ckpt['val_metrics']}")
    print(f"  Parameters: {model.count_parameters():,}")

    return model, config
