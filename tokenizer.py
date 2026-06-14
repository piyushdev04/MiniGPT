"""
Simple word-level and character-level tokenizer with vocabulary building.
"""

import re
import json
from pathlib import Path
from collections import Counter
from typing import List, Optional


class WordTokenizer:
    """
    Word-level tokenizer.
    - Lowercases and splits on whitespace/punctuation.
    - Builds a vocabulary from training text.
    - Maps tokens ↔ integer IDs.

    Special tokens:
        <PAD>  : padding (id=0)
        <UNK>  : unknown words (id=1)
        <BOS>  : beginning of sequence (id=2)
        <EOS>  : end of sequence (id=3)
    """
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    BOS_TOKEN = "<BOS>"
    EOS_TOKEN = "<EOS>"

    def __init__(self, max_vocab_size: int = 8000):
        self.max_vocab_size = max_vocab_size
        self.token2id = {}
        self.id2token = {}
        self._built = False

        # Reserve special token IDs
        self._special_tokens = [
            self.PAD_TOKEN, self.UNK_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN
        ]

    # ------------------------------------------------------------------ #
    #  Text processing                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _normalize(text: str) -> str:
        """Lowercase and normalize whitespace."""
        text = text.lower()
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @staticmethod
    def _tokenize_text(text: str) -> List[str]:
        """
        Split on word boundaries; keep punctuation as separate tokens.
        Example: "Hello, world!" → ["hello", ",", "world", "!"]
        """
        text = text.lower()
        # Insert spaces around punctuation
        text = re.sub(r"([.,!?;:\"\'()\[\]{}\-])", r" \1 ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text.split()

    # ------------------------------------------------------------------ #
    #  Vocabulary building                                                 #
    # ------------------------------------------------------------------ #

    def build_vocab(self, text: str) -> None:
        """
        Build vocabulary from raw text.
        Keeps the `max_vocab_size - 4` most common tokens.
        """
        tokens = self._tokenize_text(text)
        counts = Counter(tokens)
        most_common = counts.most_common(self.max_vocab_size - len(self._special_tokens))

        self.token2id = {tok: i for i, tok in enumerate(self._special_tokens)}
        for token, _ in most_common:
            self.token2id[token] = len(self.token2id)

        self.id2token = {i: tok for tok, i in self.token2id.items()}
        self._built = True

        print(f"[Tokenizer] Vocabulary size: {len(self.token2id):,}")
        print(f"[Tokenizer] Top-10 tokens: {[t for t, _ in most_common[:10]]}")

    # ------------------------------------------------------------------ #
    #  Encode / decode                                                     #
    # ------------------------------------------------------------------ #

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """Convert text → list of integer IDs."""
        assert self._built, "Call build_vocab() first."
        tokens = self._tokenize_text(text)
        ids = [self.token2id.get(t, self.token2id[self.UNK_TOKEN]) for t in tokens]
        if add_special_tokens:
            ids = [self.token2id[self.BOS_TOKEN]] + ids + [self.token2id[self.EOS_TOKEN]]
        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """Convert list of integer IDs → text."""
        assert self._built, "Call build_vocab() first."
        special_ids = {self.token2id[t] for t in self._special_tokens}
        tokens = []
        for i in ids:
            if skip_special and i in special_ids:
                continue
            tokens.append(self.id2token.get(i, self.UNK_TOKEN))
        return " ".join(tokens)

    # ------------------------------------------------------------------ #
    #  Properties                                                          #
    # ------------------------------------------------------------------ #

    @property
    def vocab_size(self) -> int:
        return len(self.token2id)

    @property
    def pad_id(self) -> int:
        return self.token2id[self.PAD_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.token2id[self.UNK_TOKEN]

    @property
    def bos_id(self) -> int:
        return self.token2id[self.BOS_TOKEN]

    @property
    def eos_id(self) -> int:
        return self.token2id[self.EOS_TOKEN]

    # ------------------------------------------------------------------ #
    #  Save / load                                                         #
    # ------------------------------------------------------------------ #

    def save(self, path: str) -> None:
        """Save vocabulary to JSON."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"token2id": self.token2id, "max_vocab_size": self.max_vocab_size}, f, indent=2)
        print(f"[Tokenizer] Saved to {path}")

    def load(self, path: str) -> None:
        """Load vocabulary from JSON."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.token2id = data["token2id"]
        self.max_vocab_size = data.get("max_vocab_size", self.max_vocab_size)
        self.id2token = {int(i): tok for tok, i in self.token2id.items()}
        self._built = True
        print(f"[Tokenizer] Loaded vocab of size {len(self.token2id):,} from {path}")


class CharTokenizer:
    """
    Character-level tokenizer. Smaller vocabulary, handles any text,
    but requires learning longer-range dependencies.
    """
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    BOS_TOKEN = "<BOS>"
    EOS_TOKEN = "<EOS>"

    def __init__(self):
        self.token2id = {}
        self.id2token = {}
        self._built = False

    def build_vocab(self, text: str) -> None:
        specials = [self.PAD_TOKEN, self.UNK_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN]
        chars = sorted(set(text))
        all_tokens = specials + chars
        self.token2id = {tok: i for i, tok in enumerate(all_tokens)}
        self.id2token = {i: tok for tok, i in self.token2id.items()}
        self._built = True
        print(f"[CharTokenizer] Vocabulary size: {len(self.token2id)}")

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        assert self._built
        ids = [self.token2id.get(c, self.token2id[self.UNK_TOKEN]) for c in text]
        if add_special_tokens:
            ids = [self.token2id[self.BOS_TOKEN]] + ids + [self.token2id[self.EOS_TOKEN]]
        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        assert self._built
        special_ids = {self.token2id[t] for t in [self.PAD_TOKEN, self.UNK_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN]}
        chars = []
        for i in ids:
            if skip_special and i in special_ids:
                continue
            chars.append(self.id2token.get(i, self.UNK_TOKEN))
        return "".join(chars)

    @property
    def vocab_size(self) -> int:
        return len(self.token2id)

    @property
    def pad_id(self) -> int:
        return self.token2id[self.PAD_TOKEN]

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"token2id": self.token2id}, f, indent=2)

    def load(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.token2id = data["token2id"]
        self.id2token = {int(i): tok for tok, i in self.token2id.items()}
        self._built = True
