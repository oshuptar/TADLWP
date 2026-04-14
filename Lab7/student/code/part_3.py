"""
Part 3: Very simple language model with token and position embeddings.
Task: FeedForward, Head (simple mock), Block, VerySimpleLanguageModel with nn.Embedding for token and position.
"""
import math
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from gpt_mock import Block, _get_config, generate_continuation, train_language_model
from helpers import load_text_and_context
from part_1 import create_lm_train_loader, char_tokenizer, word_tokenizer
from part_2 import bpe_tokenizer


class Embedding(nn.Module):
    """
    Learnable embedding from scratch: one vector per index, same interface as nn.Embedding.

    TODO: Implement Embedding. In __init__, store a learnable parameter of shape (num_embeddings, embedding_dim).
    In forward(idx), return the embedding vectors for the given indices (index into the parameter).
    """

    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        super().__init__()
        raise NotImplementedError("TODO: Implement Embedding __init__ (create self.weight parameter).")

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("TODO: Implement Embedding forward (return self.weight[idx]).")


class SinusoidalPositionalEmbedding(nn.Module):
    """
    Fixed sinusoidal positional embeddings (Attention is All You Need): PE(pos,2i)=sin(pos/10000^(2i/d)), PE(pos,2i+1)=cos(...). No learnable parameters.

    TODO: Implement in __init__: build a (block_size, n_embd) table with sin on even dims and cos on odd dims; register as buffer. In forward(pos_idx), return the table indexed by pos_idx.
    """

    def __init__(self, block_size: int, n_embd: int) -> None:
        super().__init__()
        raise NotImplementedError("TODO: Implement SinusoidalPositionalEmbedding __init__ (compute pe, register_buffer).")

    def forward(self, pos_idx: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("TODO: Implement SinusoidalPositionalEmbedding forward (return self.pe[pos_idx]).")


class VerySimpleLanguageModel(nn.Module):
    """Minimal LM: token embedding + position embedding, then blocks and final linear to vocab logits."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.config: Any = config
        self.token_embedding_table: Embedding = Embedding(config.vocab_size, config.n_embd)
        self.position_embedding_table: SinusoidalPositionalEmbedding = SinusoidalPositionalEmbedding(config.block_size, config.n_embd)
        self.blocks: nn.Sequential = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.final_head: nn.Linear = nn.Linear(config.n_embd, config.vocab_size)

    def forward(
        self,
        token_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size: int
        seq_len: int
        batch_size, seq_len = token_ids.shape
        """
        TODO: Full forward (blocks/head are mocked in gpt_mock; focus on embeddings).
        1. Token embeddings + positional embeddings for positions 0..seq_len-1, sum.
        2. self.blocks, then self.final_head -> logits (batch, seq, vocab_size).
        3. If targets: flatten logits/targets and cross_entropy.
        """
        raise NotImplementedError("TODO: Implement VerySimpleLanguageModel forward.")
        logits: torch.Tensor = None
        loss: Optional[torch.Tensor] = None
        if targets is not None:
            vocab_size: int = logits.shape[2]
            logits = logits.view(batch_size * seq_len, vocab_size)
            targets = targets.view(batch_size * seq_len)
            loss = F.cross_entropy(logits, targets)
        return logits, loss


def verify_part3():
    """Check that model runs forward and backward."""
    config = _get_config(vocab_size=50, block_size=16, n_embd=32, n_head=2, n_layer=1)
    model = VerySimpleLanguageModel(config)
    x = torch.randint(0, 50, (2, 10))
    logits, loss = model(x, x)
    assert logits.shape == (2 * 10, 50), f"expected (20, 50), got {logits.shape}"
    assert loss is not None
    loss.backward()
    print("Part 3: verification passed [OK]")


def part3():
    """Train VerySimpleLanguageModel on text from ``../input.txt``; generate a short continuation from a file prefix."""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_string = "the old man"
    text = load_text_and_context()
    block_size = 24
    batch_size = 32
    steps = 200
    lr = 1e-2
    num_generate = 10

    print("=== Part 3: Embeddings and Very Simple LM ===\n")
    preview_n = 120
    text_preview = text[:preview_n] + ("..." if len(text) > preview_n else "")
    print(f"Training text ({len(text)} chars from input.txt): {text_preview!r}")
    print(f"Context for generation ({len(input_string)} chars): {input_string!r}")
    print(f"Steps: {steps}  |  Block size: {block_size}  |  Batch size: {batch_size}  |  Device: {device}\n")

    results = []
    for name, get_tokenizer in [
        ("char", lambda: char_tokenizer(text)),
        ("word", lambda: word_tokenizer(text)),
        ("bpe", lambda: bpe_tokenizer(text, num_merges=50)),
    ]:
        print(f"--- {name} tokenizer ---")
        encode, decode, vocab, _ = get_tokenizer()
        vocab_size = len(vocab)
        data = torch.tensor(encode(text), dtype=torch.long)
        train_loader = create_lm_train_loader(data, block_size, batch_size, shuffle=True)
        n_windows = len(train_loader.dataset)
        print(
            f"  Training a model with {name} tokenizer (vocab size {vocab_size}, "
            f"{n_windows} windows, batch_size={batch_size}, {steps} epochs)..."
        )

        config = _get_config(vocab_size=vocab_size, block_size=block_size)
        model = VerySimpleLanguageModel(config).to(device)
        final_loss = train_language_model(name, model, train_loader, config, device, steps, lr)
        print(f"  Loss of the model after training: {final_loss:.4f}")
        gen = generate_continuation(model, encode, decode, input_string, num_generate, config, device)
        print(f"  Generated {num_generate} tokens after \"{input_string}\": \"{gen}\"")
        print()
        results.append((name, final_loss, vocab_size, gen))

    print("--- Summary ---")
    print(f"{'Tokenizer':>10} | {'Vocab size':>10} | {'Final loss':>10} | Generated continuation")
    print("-" * 70)
    for name, loss, vs, gen in results:
        print(f"{name:>10} | {vs:>10} | {loss:>10.4f} | {gen}")
    print()
    verify_part3()
    print("All tests passed [OK]")


if __name__ == "__main__":
    part3()
