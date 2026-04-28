"""
Part 3 (Lab 9): GPT-style language model.
Task: GPTConfig, GPTLanguageModel (token/position embedding, blocks, lm_head), forward, generate, _init_weights, count_parameters.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

from part_2 import Block
from embeddings import Embedding, SinusoidalPositionalEmbedding


@dataclass
class GPTConfig:
    """Configuration for GPT model and training."""
    vocab_size: int = 256
    block_size: int = 256
    n_embd: int = 384
    n_head: int = 6
    n_layer: int = 6
    dropout: float = 0.2
    batch_size: int = 64
    max_iters: int = 5000
    eval_interval: int = 500
    eval_iters: int = 500
    learning_rate: float = 3e-4
    batch_mode: str = "sequential"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class GPTLanguageModel(nn.Module):
    """GPT-style decoder-only LM: token + position embedding, stack of Blocks, final LayerNorm and linear to vocab."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

    def forward(self, idx):
        """
        TODO: Implement the forward pass: embed tokens and positions (add them), blocks, ln_f, lm_head.
        """
        raise NotImplementedError("TODO: Implement the forward pass of the GPTLanguageModel module.")

    def generate(self, idx, max_new_tokens):
        """
        TODO: Implement generate: loop max_new_tokens times; each step: last block_size tokens → forward → last logits → softmax → sample → append to idx; return idx.
        """
        raise NotImplementedError("TODO: Implement the generate method of the GPTLanguageModel module.")
        return idx

    def count_parameters(self):
        """Return number of trainable parameters in millions."""
        return sum(p.numel() for p in self.parameters()) / 1e6

# ------ Tests ------

def test_forward_logits_and_loss():
    """Given a tiny GPT and token indices (B,T), when we forward and compute CE loss outside the model, logits shape is (B,T,vocab) and loss is finite."""
    config = GPTConfig(vocab_size=50, block_size=8, n_embd=32, n_head=2, n_layer=2)
    model = GPTLanguageModel(config)
    x = torch.randint(0, 50, (2, 6))
    logits = model(x)
    assert logits.shape == (2, 6, 50), "logits shape"
    B, T, C = logits.shape
    loss = F.cross_entropy(logits.view(B * T, C), x.view(B * T))
    assert not torch.isnan(loss), "loss finite"
    print("  test_forward_logits_and_loss [OK]")


def test_generate_shape():
    """Given a tiny GPT and (B,T) indices, when we generate 3 new tokens, then output shape is (B, T+3)."""
    config = GPTConfig(vocab_size=50, block_size=8, n_embd=32, n_head=2, n_layer=2)
    model = GPTLanguageModel(config)
    x = torch.randint(0, 50, (2, 6))
    out = model.generate(x, max_new_tokens=3)
    assert out.shape == (2, 6 + 3), "generate shape"
    print("  test_generate_shape [OK]")


def test_count_parameters():
    """Given a small GPT config, when we build the model, then count_parameters returns a positive number (millions)."""
    config = GPTConfig(vocab_size=64, block_size=16, n_embd=48, n_head=4, n_layer=2)
    model = GPTLanguageModel(config)
    n = model.count_parameters()
    assert n > 0, "count_parameters > 0"
    print("  test_count_parameters [OK]")


def main():
    print("=== Part 3: GPT Language Model ===\n")
    test_forward_logits_and_loss()
    test_generate_shape()
    test_count_parameters()
    print("\nAll tests passed [OK]")


if __name__ == "__main__":
    main()
