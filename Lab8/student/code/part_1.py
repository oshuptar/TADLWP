"""
Part 1 (Lab 9): FeedForward and single AttentionHead (self-attention).
Task: FeedForward (linear–ReLU–linear–dropout), AttentionHead (Q,K,V + causal mask + softmax).
"""
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from types import SimpleNamespace

from Lab8.student.code.part_3 import GPTConfig


def _test_config():
    """Minimal config for testing modules (block_size, n_embd, n_head, dropout)."""
    return SimpleNamespace(block_size=8, n_embd=32, n_head=4, dropout=0.1)


class FeedForward(nn.Module):
    """Two linear layers with ReLU and dropout: n_embd -> 4*n_embd -> n_embd."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        """
        TODO: Implement the FeedForward module with two linear layers, ReLU and dropout.
        Order: Linear(n_embd, 4*n_embd) → ReLU → Linear(4*n_embd, n_embd) → Dropout.
        """
        self.net = nn.Sequential(
            nn.Linear(in_features=config.n_embd, out_features=config.n_embd),
            nn.ReLU(),
            nn.Linear(in_features=4*config.n_embd, out_features=config.n_embd),
            nn.Dropout())


    def forward(self, x):
        """
        TODO: Implement the forward pass: pass x through the network and return the result.
        """
        return self.net(x)


class AttentionHead(nn.Module):
    """Single self-attention head: Q, K, V linear projections; causal mask; softmax; output = attention @ V."""

    def __init__(self, config, head_size):
        super().__init__()
        """
        TODO: Implement the AttentionHead module: key, query, value (nn.Linear to head_size);
        register_buffer('tril', torch.tril(...)) for causal mask; dropout.
        """
        self.key = nn.Linear(in_features=config.n_embd, out_features=head_size, bias=False)
        self.query = nn.Linear(in_features=config.n_embd, out_features=head_size, bias=False)
        self.value = nn.Linear(in_features=config.n_embd, out_features=head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(config.block_size, config.block_size)))
        self.dropout = nn.Dropout(config.dropout)


    def forward(self, x):
        """
        TODO: Implement the forward pass: Q,K,V from x; wei = scaled Q@K^T; mask; softmax; dropout; out = wei @ V.
        """
        B,T,C = x.shape
        Q: torch.Tensor = self.key(x)
        K: torch.Tensor = self.key(x)
        V: torch.Tensor = self.key(x)
        attention_scores: torch.Tensor = torch.matmul(Q, K.transpose(-2, -1))/math.sqrt(K.shape[-1])
        attention_scores.masked_fill_(self.tril[:T, :T] == 0, float("-inf"))
        attention_weights = F.softmax(attention_scores, dim = -1)
        attention_weights = self.dropout(attention_weights)
        out = attention_weights @ V
        return out



# ------ Tests ------

def test_feedforward_shape_and_finite():
    """Given a small config and (B,T,C) input, when we run FeedForward, then output shape is (B,T,C) and has no NaN."""
    config = _test_config()
    B, T, C = 2, 4, config.n_embd
    x = torch.randn(B, T, C)
    ff = FeedForward(config)
    out = ff(x)
    assert out.shape == (B, T, C), "FeedForward shape"
    assert not torch.isnan(out).any(), "FeedForward output should not be NaN"
    print("  test_feedforward_shape_and_finite [OK]")


def test_head_shape_and_finite():
    """Given a small config and (B,T,C) input, when we run AttentionHead, then output shape is (B,T,head_size) and has no NaN."""
    config = _test_config()
    B, T, C = 2, 4, config.n_embd
    x = torch.randn(B, T, C)
    head_size = config.n_embd // config.n_head
    head = AttentionHead(config, head_size)
    out = head(x)
    assert out.shape == (B, T, head_size), "AttentionHead shape"
    assert not torch.isnan(out).any(), "AttentionHead output should not be NaN"
    print("  test_head_shape_and_finite [OK]")


def main():
    print("=== Part 1: FeedForward and single AttentionHead ===\n")
    test_feedforward_shape_and_finite()
    test_head_shape_and_finite()
    print("\nAll tests passed [OK]")


if __name__ == "__main__":
    main()
