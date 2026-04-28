"""
GPT-style language model and dependencies (self-contained for Lab 10).
Contains: FeedForward, AttentionHead, MultiHeadAttention, Block, GPTConfig, GPTLanguageModel.
Lab 10 is separate from Lab 9; this file duplicates the model code so Lab 10 has no reference to Lab 9.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

from embeddings import Embedding, SinusoidalPositionalEmbedding


# ------ Building blocks ------

class FeedForward(nn.Module):
    """Two linear layers with ReLU and dropout: n_embd -> 4*n_embd -> n_embd."""

    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.ReLU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)


class AttentionHead(nn.Module):
    """Single self-attention head: Q, K, V linear projections; causal mask; softmax; output = attention @ V."""

    def __init__(self, config, head_size):
        super().__init__()
        self.key = nn.Linear(config.n_embd, head_size, bias=False)
        self.query = nn.Linear(config.n_embd, head_size, bias=False)
        self.value = nn.Linear(config.n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(config.block_size, config.block_size)))
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    """Multiple AttentionHead modules in parallel; concat outputs then project back to n_embd."""

    def __init__(self, config, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(config, head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class Block(nn.Module):
    """Transformer block: LayerNorm -> MultiHeadAttention -> residual; LayerNorm -> FeedForward -> residual."""

    def __init__(self, config):
        super().__init__()
        head_size = config.n_embd // config.n_head
        self.sa = MultiHeadAttention(config, config.n_head, head_size)
        self.ffwd = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


# ------ GPT config and model ------

@dataclass
class GPTConfig:
    """Configuration for GPT model and training."""
    vocab_size: int = 256
    block_size: int = 256
    n_embd: int = 384
    n_head: int = 6
    n_layer: int = 6
    dropout: float = 0.5
    batch_size: int = 64
    epochs: int = 10
    eval_interval: int = 1
    eval_iters: int = 500
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    batch_mode: str = "sequential"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class GPTLanguageModel(nn.Module):
    """GPT-style decoder-only LM: token + position embedding, stack of Blocks, final LayerNorm and linear to vocab."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding_table = Embedding(config.vocab_size, config.n_embd)
        self.position_embedding_table = SinusoidalPositionalEmbedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

    def forward(self, idx):
        B, T = idx.shape
        device = idx.device
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = self.drop(tok_emb + pos_emb)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits = self(idx_cond)
            logits = logits[:, -1, :] # Place for analysis
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters()) / 1e6