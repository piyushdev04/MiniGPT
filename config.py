"""
Central configuration for MiniGPT training.
Edit these values to scale up or down your model.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class TrainingConfig:
    # ── Data ──────────────────────────────────────
    data_path: str = "data/corpus.txt"
    vocab_path: str = "checkpoints/vocab.json"
    max_chars: int = 20_000_000      # read at most 20 MB of text
    val_split: float = 0.1           # 10% validation

    # ── Model ─────────────────────────────────────
    vocab_size: int = 12_000         # max vocabulary tokens
    embed_dim: int = 256             # embedding dimension
    num_heads: int = 8               # attention heads (embed_dim must be divisible)
    num_layers: int = 4              # transformer decoder blocks
    ff_dim: int = 1024               # feed-forward hidden size (usually 4 × embed_dim)
    seq_len: int = 256               # context window length
    dropout: float = 0.1

    # ── Training ──────────────────────────────────
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0           # max gradient norm
    resume: bool = False             # resume from checkpoint

    # ── Checkpointing ─────────────────────────────
    checkpoint_path: str = "checkpoints/model.pt"
    save_every: int = 2              # save checkpoint every N epochs
    log_every: int = 50              # print progress every N steps
    sample_every: int = 2            # generate sample text every N epochs

    # ── Generation samples ────────────────────────
    sample_prompts: List[str] = field(default_factory=lambda: [
        "User: Explain overfitting in machine learning.\nAssistant:",
        "User: Write a short Python function to reverse a string.\nAssistant:",
        "User: Summarize what gradient descent does.\nAssistant:",
    ])


# ── Preset configurations ───────────────────────────────────────────────── #

def tiny_config() -> TrainingConfig:
    """Fastest config for testing (runs in minutes on CPU)."""
    cfg = TrainingConfig()
    cfg.embed_dim = 64
    cfg.num_heads = 2
    cfg.num_layers = 2
    cfg.ff_dim = 256
    cfg.seq_len = 64
    cfg.vocab_size = 3000
    cfg.batch_size = 64
    cfg.epochs = 5
    return cfg


def small_config() -> TrainingConfig:
    """Balanced config for a 1–5 MB corpus."""
    return TrainingConfig()  # defaults are already 'small'


def medium_config() -> TrainingConfig:
    """Larger model for a 5–10 MB corpus with a decent GPU."""
    cfg = TrainingConfig()
    cfg.embed_dim = 256
    cfg.num_heads = 8
    cfg.num_layers = 4
    cfg.ff_dim = 1024
    cfg.seq_len = 256
    cfg.vocab_size = 12000
    cfg.batch_size = 16
    cfg.learning_rate = 1e-4
    cfg.epochs = 15
    return cfg
