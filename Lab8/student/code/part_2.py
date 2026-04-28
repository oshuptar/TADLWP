"""
Part 2 (Lab 9): MultiHeadAttention and Block.
Task: MultiHeadAttention (multiple AttentionHeads, concat, project), Block (LayerNorm -> MHA/FFN -> residual).
"""
import torch
import torch.nn as nn

from part_1 import FeedForward, AttentionHead, _test_config


class MultiHeadAttention(nn.Module):
    """Multiple AttentionHead modules in parallel; concat outputs then project back to n_embd."""

    def __init__(self, config, num_heads, head_size):
        super().__init__()
        """
        TODO: Implement MultiHeadAttention: ModuleList of AttentionHead(config, head_size); Linear(head_size*num_heads, n_embd); dropout.
        """
        self.attention_block = nn.ModuleList(
            AttentionHead(config, head_size)
            for _ in range(num_heads))
        self.linear = nn.Linear(in_features=head_size*config.n_embd, out_features=config.n_embd)
        self.dropout = nn.Dropout()

    def forward(self, x):
        """
        TODO: Implement the forward pass: run each head on x, concat on dim=-1, then project and dropout.
        """
        out = torch.cat([h(x) for h in self.attention_block])
        out = self.dropout(self.linear(out))
        return out
        


class Block(nn.Module):
    """Transformer block: LayerNorm -> MultiHeadAttention -> residual; LayerNorm -> FeedForward -> residual."""

    def __init__(self, config):
        super().__init__()
        """
        TODO: Implement the Block: two LayerNorms, one MultiHeadAttention, one FeedForward; head_size = n_embd // n_head.
        """
        raise NotImplementedError("TODO: Implement the Block module.")

    def forward(self, x):
        """
        TODO: Implement the forward pass: x = x + sa(ln1(x)); x = x + ffwd(ln2(x)); return x. (Pre-norm.)
        """
        raise NotImplementedError("TODO: Implement the forward pass of the Block module.")
        return x


# ------ Tests ------

def test_mha_shape_and_finite():
    """Given a small config and (B,T,C) input, when we run MultiHeadAttention, then output shape is (B,T,C) and has no NaN."""
    config = _test_config()
    B, T, C = 2, 4, config.n_embd
    x = torch.randn(B, T, C)
    head_size = config.n_embd // config.n_head
    mha = MultiHeadAttention(config, config.n_head, head_size)
    out = mha(x)
    assert out.shape == (B, T, C), "MultiHeadAttention shape"
    assert not torch.isnan(out).any(), "MultiHeadAttention output should not be NaN"
    print("  test_mha_shape_and_finite [OK]")


def test_block_shape_and_finite():
    """Given a small config and (B,T,C) input, when we run Block, then output shape is (B,T,C) and has no NaN."""
    config = _test_config()
    B, T, C = 2, 4, config.n_embd
    x = torch.randn(B, T, C)
    block = Block(config)
    out = block(x)
    assert out.shape == (B, T, C), "Block shape"
    assert not torch.isnan(out).any(), "Block output should not be NaN"
    print("  test_block_shape_and_finite [OK]")


def main():
    print("=== Part 2: MultiHeadAttention and Block ===\n")
    test_mha_shape_and_finite()
    test_block_shape_and_finite()
    print("\nAll tests passed [OK]")


if __name__ == "__main__":
    main()
