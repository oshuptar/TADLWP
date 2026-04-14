"""
Mock GPT building blocks for Lab 7 part 3: FeedForward, Head, Block, and training / generation helpers.
Embedding layers live in ``part_3.py``.
"""
from types import SimpleNamespace
from typing import Any, Callable, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class FeedForward(nn.Module):
    """Two linear layers with ReLU and dropout: n_embd -> 4*n_embd -> n_embd."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.net: nn.Sequential = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.ReLU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Head(nn.Module):
    """Simple placeholder for one attention head (no Q,K,V yet). Single linear layer; later replaced by causal self-attention."""

    def __init__(self, config: Any, head_size: int) -> None:
        super().__init__()
        self.linear: nn.Linear = nn.Linear(config.n_embd, head_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class Block(nn.Module):
    """Transformer block: self-attention head + feed-forward, each with residual connection."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        head_size: int = config.n_embd  # match n_embd so residual x + head(x) has same shape
        self.self_attn_head: Head = Head(config, head_size)
        self.ffwd: FeedForward = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn_head(x)
        x = x + self.ffwd(x)
        return x


def _get_config(
    vocab_size: int,
    block_size: int = 32,
    n_embd: int = 64,
    n_head: int = 4,
    n_layer: int = 2,
    dropout: float = 0.1,
) -> SimpleNamespace:
    return SimpleNamespace(
        vocab_size=vocab_size,
        block_size=block_size,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        dropout=dropout,
    )


def train_language_model(
    name: str,
    model: nn.Module,
    train_loader: DataLoader,
    config: Any,
    device: Union[str, torch.device],
    steps: int,
    lr: float,
) -> float:
    """
    Train for ``steps`` epochs over ``train_loader`` (each step: one full pass over all batches).
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    last_loss = 0.0
    for epoch in range(steps):
        epoch_loss = 0.0
        n_batches = 0
        for x, y in train_loader:
            x = x.to(device=device, dtype=torch.long)
            y = y.to(device=device, dtype=torch.long)
            optimizer.zero_grad()
            _, loss = model(x, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        last_loss = epoch_loss / max(n_batches, 1)
        if epoch % 50 == 0:
            print(f"{name} Epoch {epoch} Loss: {last_loss:.4f}")
    return float(last_loss)


def generate_continuation(
    model: nn.Module,
    encode: Callable[[str], List[int]],
    decode: Callable[[List[int]], str],
    context_str: str,
    num_tokens: int,
    config: Any,
    device: Union[str, torch.device],
) -> str:
    """Generate num_tokens next token ids given context string; decode to string."""
    model.eval()
    ids: List[int] = list(encode(context_str))
    if len(ids) > config.block_size:
        ids = ids[-config.block_size:]
    with torch.no_grad():
        for _ in range(num_tokens):
            x: torch.Tensor = torch.tensor([ids[-config.block_size:]], dtype=torch.long, device=device)
            logits, _ = model(x, None)
            next_logits: torch.Tensor = logits[0, -1, :]
            next_id: int = next_logits.argmax().item()
            ids.append(next_id)
    return decode(ids[len(ids) - num_tokens:])
