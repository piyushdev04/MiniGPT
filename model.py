"""
GPT-style Transformer Model
A decoder-only transformer for next-token prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TokenEmbedding(nn.Module):
    """Converts token IDs to dense vectors."""
    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, x):
        # Scale embeddings by sqrt(d_model) as in original paper
        return self.embedding(x) * math.sqrt(self.embed_dim)


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding.
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    def __init__(self, embed_dim: int, max_seq_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_seq_len, embed_dim)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_seq_len, embed_dim)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (batch, seq_len, embed_dim)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention with causal (look-ahead) masking.
    Splits embedding into `num_heads` heads, computes scaled dot-product
    attention in parallel, then concatenates and projects.
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Single projection matrix for Q, K, V
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, T, C = x.size()  # batch, seq_len, embed_dim

        # Compute Q, K, V in one shot
        qkv = self.qkv_proj(x)  # (B, T, 3*C)
        q, k, v = qkv.split(self.embed_dim, dim=2)

        # Reshape to (B, num_heads, T, head_dim)
        def reshape(t):
            return t.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        q, k, v = reshape(q), reshape(k), reshape(v)

        # Scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale  # (B, heads, T, T)

        # Apply causal mask: positions can only attend to previous positions
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Weighted sum of values
        out = torch.matmul(attn_weights, v)  # (B, heads, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class FeedForwardNetwork(nn.Module):
    """
    Two-layer feed-forward network with GELU activation.
    Expands dimension by 4x (as in original Transformer paper).
    """
    def __init__(self, embed_dim: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerDecoderBlock(nn.Module):
    """
    A single Transformer decoder block:
      LayerNorm → Masked Multi-Head Attention → Residual
      LayerNorm → Feed-Forward Network → Residual
    Uses Pre-LN (normalize before sub-layer) for training stability.
    """
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ffn = FeedForwardNetwork(embed_dim, ff_dim, dropout)

    def forward(self, x, mask=None):
        # Residual connection around attention
        x = x + self.attn(self.ln1(x), mask)
        # Residual connection around FFN
        x = x + self.ffn(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    """
    GPT-style decoder-only transformer.

    Architecture:
      Token Embedding + Positional Encoding
      → N × TransformerDecoderBlock
      → LayerNorm
      → Linear (embed_dim → vocab_size)
    """
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
        ff_dim: int = 512,
        max_seq_len: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        self.token_emb = TokenEmbedding(vocab_size, embed_dim)
        self.pos_enc = PositionalEncoding(embed_dim, max_seq_len, dropout)

        self.blocks = nn.ModuleList([
            TransformerDecoderBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

        self.ln_final = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

        # Weight tying: share weights between embedding and output projection
        self.lm_head.weight = self.token_emb.embedding.weight

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small normal distribution."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _make_causal_mask(self, seq_len: int, device: torch.device):
        """Upper-triangular mask: token i cannot see token j > i."""
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)

    def forward(self, input_ids, targets=None):
        """
        Args:
            input_ids: (batch, seq_len) long tensor
            targets:   (batch, seq_len) long tensor (optional, for loss)
        Returns:
            logits: (batch, seq_len, vocab_size)
            loss:   scalar if targets provided, else None
        """
        B, T = input_ids.shape
        assert T <= self.max_seq_len, f"Sequence too long ({T} > {self.max_seq_len})"

        mask = self._make_causal_mask(T, input_ids.device)

        x = self.token_emb(input_ids)   # (B, T, embed_dim)
        x = self.pos_enc(x)             # (B, T, embed_dim)

        for block in self.blocks:
            x = block(x, mask)

        x = self.ln_final(x)            # (B, T, embed_dim)
        logits = self.lm_head(x)        # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # Flatten for cross-entropy: (B*T, vocab_size) vs (B*T,)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )

        return logits, loss

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        repetition_penalty: float = 1.2,
    ):
        """
        Autoregressive text generation with temperature, top-k, and top-p sampling.

        Args:
            input_ids:          (1, prompt_len) starting token IDs
            max_new_tokens:     how many tokens to generate
            temperature:        >1 = more random, <1 = more focused
            top_k:              keep only top-k logits (0 = disabled)
            top_p:              nucleus sampling threshold
            repetition_penalty: penalize already-generated tokens
        """
        self.eval()
        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            # Crop to max_seq_len if needed
            context = generated[:, -self.max_seq_len:]
            logits, _ = self(context)
            logits = logits[:, -1, :]  # Only last position: (1, vocab_size)

            # Repetition penalty
            if repetition_penalty != 1.0:
                for token_id in set(generated[0].tolist()):
                    if logits[0, token_id] < 0:
                        logits[0, token_id] *= repetition_penalty
                    else:
                        logits[0, token_id] /= repetition_penalty

            # Temperature scaling
            logits = logits / max(temperature, 1e-8)

            # Top-k filtering
            if top_k > 0:
                top_k = min(top_k, logits.size(-1))
                kth_val = torch.topk(logits, top_k).values[:, -1, None]
                logits = logits.masked_fill(logits < kth_val, float("-inf"))

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                # Remove tokens with cumulative probability above threshold
                sorted_logits[cumulative_probs - F.softmax(sorted_logits, dim=-1) > top_p] = float("-inf")
                logits = torch.zeros_like(logits).scatter_(1, sorted_idx, sorted_logits)

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)

        return generated
