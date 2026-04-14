"""
Part 1: Implement character-level and word-level tokenizers.
Task: char_tokenizer(text) and word_tokenizer(text) each return (encode, decode, vocabulary).
"""
import re
from typing import Callable, Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

def char_tokenizer(
    text: str,
) -> Tuple[Callable[[str], List[int]], Callable[[List[int]], str], List[str]]:
    """
    Character-level tokenizer: each character is one token. Returns (encode, decode, vocabulary).
    
    TODO: Implement character-level tokenizer.
    """
    vocab = sorted(list(set(text)))
    stoi = {token: index for index, token in enumerate(vocab)}
    itos = {index: token for index, token in enumerate(vocab)}
    encode: Callable[[str], List[int]] = lambda str: [stoi[char] for char in str]
    decode: Callable[[List[int]], str] = lambda index_list: "".join([itos[index] for index in index_list])
    return encode, decode, vocab


def word_tokenizer(
    text: str,
) -> Tuple[Callable[[str], List[int]], Callable[[List[int]], str], List[str]]:
    """
    Word-level tokenizer: splits on words and punctuation (regex \\w+ and non-word chars).
    Returns (encode, decode, vocabulary).
        
    TODO: Implement character-level tokenizer.
    """
    reg_exp = r"\w+|[^\w]"
    tokens: List[str] = re.findall(reg_exp, text)
    stoi = {token: index for index, token in enumerate(tokens)}
    itos = {index: token for index, token in enumerate(tokens)}
    vocab: List[str] = sorted(list(set(tokens)))
    encode: Callable[[str], List[int]] = lambda text: [stoi[tok] for tok in re.findall(reg_exp, text)]
    decode: Callable[[List[int]], str] = lambda index_list: "".join([itos[index] for index in index_list])
    return encode, decode, vocab


class TokenWindowDataset(Dataset):
    """Stride-1 LM windows: each item is ``(x, y)`` of length ``block_size`` with next-token targets."""

    def __init__(self, data, block_size: int):
        self.data = data if isinstance(data, torch.Tensor) else torch.tensor(data, dtype=torch.long)
        self.block_size = block_size

    def __len__(self) -> int:
        return max(0, len(self.data) - self.block_size)

    def __getitem__(self, i: int):
        """
        TODO: Get the input x and target y for the i-th window.
        x is the current window of length block_size.
        y is the next window of length block_size + 1.
        """
        x = self.data[i : i + self.block_size]
        y = self.data[i + 1 : i + 1 + self.block_size]
        return x, y
    
def create_lm_train_loader(data: torch.Tensor, block_size: int, batch_size: int, *, shuffle: bool = True) -> DataLoader:
    train_ds = TokenWindowDataset(data, block_size)
    return DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def verify_part1() -> None:
    """Check that char and word tokenizers round-trip; LM batches have expected shapes and first window content."""
    sample = "Litwo ojczyzno moja"
    enc_c, dec_c, vocab_c = char_tokenizer(sample)
    enc_w, dec_w, vocab_w = word_tokenizer(sample)
    assert dec_c(enc_c(sample)) == sample, "Char tokenizer round-trip failed"
    assert dec_w(enc_w(sample)) == sample, "Word tokenizer round-trip failed"
    assert len(vocab_c) >= len(set(sample)) and len(vocab_w) >= 1

    block_size = 4
    batch_size = 5
    ids = enc_c(sample)
    n_windows = max(0, len(ids) - block_size)
    data = torch.tensor(ids, dtype=torch.long)
    train_loader = create_lm_train_loader(data, block_size, batch_size, shuffle=False)
    assert len(train_loader.dataset) == n_windows

    seen = 0
    for x, y in train_loader:
        assert x.shape == y.shape
        assert x.dim() == 2 and x.shape[1] == block_size
        assert x.dtype == torch.long and y.dtype == torch.long
        if seen == 0:
            assert torch.equal(x[0], torch.tensor(ids[:block_size]))
            assert torch.equal(y[0], torch.tensor(ids[1 : block_size + 1]))
        seen += x.shape[0]
    assert seen == n_windows

    print("Part 1: verification passed [OK]")


def part1() -> None:
    """Run tokenizer checks and a short DataLoader demo (shuffle=False so row 0 is the first window)."""
    print("=== Part 1: Char and Word Tokenizers ===\n")
    sample = "Litwo ojczyzno moja ty jestes jak zdrowie"
    print(f"Sample text: \"{sample}\"\n")

    enc_c, dec_c, vocab_c = char_tokenizer(sample)
    print("Character tokenizer:")
    print(f"  Vocabulary: {vocab_c}")
    print(f"  Vocab size: {len(vocab_c)}")
    print(f"  Encode -> Decode: \"{dec_c(enc_c(sample))}\"\n")

    enc_w, dec_w, vocab_w = word_tokenizer(sample)
    print("Word tokenizer:")
    print(f"  Vocabulary: {vocab_w}")
    print(f"  Vocab size: {len(vocab_w)}")
    print(f"  Encode -> Decode: \"{dec_w(enc_w(sample))}\"\n")

    block_size = 8
    batch_size = 8
    data = torch.tensor(enc_c(sample), dtype=torch.long)
    train_loader = create_lm_train_loader(data, block_size, batch_size, shuffle=False)
    w = block_size + 1
    print(f"TokenWindowDataset + DataLoader (batch_size={batch_size}, shuffle=False), window {w} tokens:")
    for batch_idx, (x, y) in enumerate(train_loader):
        assert x.shape == y.shape and x.shape[1] == block_size
        b = x.shape[0]
        for i in range(b):
            for j in range(i + 1, b):
                assert not (
                    torch.equal(x[i], x[j]) and torch.equal(y[i], y[j])
                ), "rows in a batch must be distinct (x, y) pairs"
        print(f"  batch {batch_idx}: x.shape={tuple(x.shape)}  y.shape={tuple(y.shape)}")
        if batch_idx == 0:
            for i in range(min(3, x.shape[0])):
                print(f"    row {i}  x={dec_c(x[i].tolist())!r}  y={dec_c(y[i].tolist())!r}  (y is x shifted +1)")
    print()

    verify_part1()
    print("All tests passed [OK]")


if __name__ == "__main__":
    part1()
