"""
Learnable and sinusoidal positional embeddings (from Lab 8).
Used by GPT model for token and position embeddings.
"""
import math

import torch
import torch.nn as nn


class Embedding(nn.Module):
    """Learnable embedding from scratch: one vector per index, same interface as nn.Embedding."""

    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        super().__init__()
        self.weight: nn.Parameter = nn.Parameter(torch.randn(num_embeddings, embedding_dim) * 0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """idx: (B, T) or (T,) integer token ids. Returns (B, T, n_embd) or (T, n_embd)."""
        return self.weight[idx]


class SinusoidalPositionalEmbedding(nn.Module):
    """Fixed sinusoidal positional embeddings (Attention is All You Need): PE(pos,2i)=sin(pos/10000^(2i/d)), PE(pos,2i+1)=cos(...). No learnable parameters."""

    def __init__(self, block_size: int, n_embd: int) -> None:
        super().__init__()
        pe: torch.Tensor = torch.zeros(block_size, n_embd)
        position: torch.Tensor = torch.arange(block_size, dtype=torch.float).unsqueeze(1)
        # 10000^(2i/d) for i = 0,1,...; paper: PE(pos,2i)=sin(...), PE(pos,2i+1)=cos(...)
        div_term: torch.Tensor = torch.exp(torch.arange(0, n_embd, 2).float() * (math.log(10000.0) / n_embd))
        pe[:, 0::2] = torch.sin(position / div_term)   # even dims 0,2,4,... → sin
        pe[:, 1::2] = torch.cos(position / div_term[: (n_embd // 2)])  # odd dims 1,3,5,... → cos
        self.register_buffer("pe", pe)

    def forward(self, pos_idx: torch.Tensor) -> torch.Tensor:
        """pos_idx: (T,) integer positions. Returns (T, n_embd)."""
        return self.pe[pos_idx]
