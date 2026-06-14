"""
Dataset utilities for next-token prediction training.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
import random


class TextDataset(Dataset):
    """
    Sliding-window dataset for language model training.

    Given a long token ID sequence, creates overlapping windows of length
    `seq_len`. Each sample (x, y) is:
        x = tokens[i : i + seq_len]
        y = tokens[i+1 : i + seq_len + 1]   (shifted by 1 for next-token pred)
    """

    def __init__(self, token_ids: List[int], seq_len: int, stride: int = None):
        """
        Args:
            token_ids:  flat list of all token IDs from the corpus
            seq_len:    context window length
            stride:     step between windows (default = seq_len // 2)
        """
        self.seq_len = seq_len
        self.stride = stride or seq_len // 2
        self.data = torch.tensor(token_ids, dtype=torch.long)

        # Pre-compute start indices
        self.starts = list(range(0, len(self.data) - seq_len - 1, self.stride))
        print(f"[Dataset] {len(self.data):,} tokens → {len(self.starts):,} samples "
              f"(seq_len={seq_len}, stride={self.stride})")

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = self.starts[idx]
        x = self.data[start : start + self.seq_len]
        y = self.data[start + 1 : start + self.seq_len + 1]
        return x, y


def make_dataloaders(
    token_ids: List[int],
    seq_len: int,
    batch_size: int,
    val_split: float = 0.1,
    num_workers: int = 0,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """
    Split token_ids into train/val and return DataLoaders.

    Args:
        token_ids:   full tokenized corpus
        seq_len:     context window length
        batch_size:  mini-batch size
        val_split:   fraction of data for validation
        num_workers: DataLoader workers (0 = main process)

    Returns:
        (train_loader, val_loader)
    """
    # Split at token level to avoid data leakage
    split_point = int(len(token_ids) * (1 - val_split))
    train_ids = token_ids[:split_point]
    val_ids = token_ids[split_point:]

    train_ds = TextDataset(train_ids, seq_len, stride=seq_len // 2)
    val_ds = TextDataset(val_ids, seq_len, stride=seq_len)

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=g,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"[DataLoader] Train batches: {len(train_loader):,} | Val batches: {len(val_loader):,}")
    return train_loader, val_loader


def load_text_file(path: str, max_chars: int = None) -> str:
    """
    Load and lightly clean a text file.
    - Normalizes line endings
    - Collapses excessive blank lines
    - Optionally truncates to `max_chars`
    """
    import re
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    # Normalize
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)   # max 2 consecutive newlines
    text = re.sub(r"[ \t]+", " ", text)       # collapse spaces/tabs

    if max_chars:
        text = text[:max_chars]

    print(f"[Data] Loaded {len(text):,} characters from {path}")
    return text
