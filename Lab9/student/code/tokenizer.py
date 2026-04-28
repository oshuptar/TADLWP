"""
Lab 9 text pipeline: BPE tokenizer, sliding-window LM dataset, and DataLoaders.
"""
from collections import Counter
from pathlib import Path
import re
from typing import Callable, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

MergeList = List[Tuple[Tuple[str, str], str]]


class TokenWindowDataset(Dataset):
    """Stride-1 LM windows: each item is ``(x, y)`` of length ``block_size`` with next-token targets."""

    def __init__(self, data, block_size: int):
        self.data = data if isinstance(data, torch.Tensor) else torch.tensor(data, dtype=torch.long)
        self.block_size = block_size

    def __len__(self) -> int:
        return max(0, len(self.data) - self.block_size)

    def __getitem__(self, i: int):
        x = self.data[i : i + self.block_size]
        y = self.data[i + 1 : i + self.block_size + 1]
        return x, y


def create_lm_dataloaders(
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    block_size: int,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader]:
    """Build train (shuffle) and val (no shuffle) DataLoaders over sliding windows."""
    train_ds = TokenWindowDataset(train_data, block_size)
    val_ds = TokenWindowDataset(val_data, block_size)
    if len(train_ds) == 0:
        raise ValueError("Train split has no windows: need len(train_data) > block_size.")
    if len(val_ds) == 0:
        raise ValueError("Val split has no windows: need len(val_data) > block_size.")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader


def load_text_and_tokenize(input_path, encoding="utf-8", num_merges=100):
    path = Path(input_path)
    text = path.read_text(encoding=encoding)
    encode, _, vocab, merges = bpe_tokenizer(text, num_merges=num_merges)
    encoded = encode(text)
    data = torch.tensor(encoded, dtype=torch.long)
    print(
        "[load_text_and_tokenize] "
        f"Corpus: {path.name} — {len(text):,} chars → {len(encoded):,} BPE tokens "
        f"({data.element_size() * data.numel() / (1024**2):.2f} MiB int64)"
    )
    return data, vocab, merges


def encode_decode_from_bpe(vocab: List[str], merges: MergeList):
    """
    Rebuild ``encode`` / ``decode`` from a BPE vocabulary (ids follow ``sorted`` token order) and merge list.
    Use the same ``merges`` produced by :func:`bpe_tokenizer` when saving checkpoints.
    """
    stoi = {token: i for i, token in enumerate(vocab)}
    itos = {i: token for i, token in enumerate(vocab)}

    def encode(s: str) -> List[int]:
        tokens = list(s)
        for pair, merged_token in merges:
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
                    new_tokens.append(merged_token)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        return [stoi[token] for token in tokens]

    def decode(ids: List[int]) -> str:
        return "".join(itos[i] for i in ids)

    return encode, decode


def bpe_tokenizer(
    text: str,
    num_merges: int = 100,
    verbose: bool = False,
) -> Tuple[Callable[[str], List[int]], Callable[[List[int]], str], List[str], MergeList]:
    """
    Simple BPE:
    - learn merges from training ``text``
    - ``encode`` applies merges in order; ``decode`` concatenates token strings
    If ``verbose``, print corpus size, merge cap vs learned merges, vocab size, encoded token count.
    """
    tokens = list(text)
    merges: MergeList = []

    for iteration in range(num_merges):
        pair_counts = Counter()
        for i in range(len(tokens) - 1):
            pair_counts[(tokens[i], tokens[i + 1])] += 1

        if not pair_counts:
            break

        best_pair, count = pair_counts.most_common(1)[0]
        if count < 2:
            break

        merged_token = best_pair[0] + best_pair[1]
        merges.append((best_pair, merged_token))

        new_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best_pair:
                new_tokens.append(merged_token)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1

        tokens = new_tokens
        if verbose and iteration % 50 == 0:
            print(
                f"[bpe_tokenizer] Iteration {iteration}: {len(merges)} merges learned, "
                f"vocab size: {len(sorted(set(text) | {merged for _, merged in merges}))}"
            )

    vocab = sorted(set(text) | {merged for _, merged in merges})
    stoi = {token: i for i, token in enumerate(vocab)}
    itos = {i: token for i, token in enumerate(vocab)}

    def encode(s: str) -> List[int]:
        toks = list(s)
        for pair, merged_token in merges:
            new_tokens = []
            i = 0
            while i < len(toks):
                if i < len(toks) - 1 and (toks[i], toks[i + 1]) == pair:
                    new_tokens.append(merged_token)
                    i += 2
                else:
                    new_tokens.append(toks[i])
                    i += 1
            toks = new_tokens
        return [stoi[token] for token in toks]

    def decode(ids: List[int]) -> str:
        return "".join(itos[i] for i in ids)

    all_tokens_in_whole_text = len(encode(text))
    seperate_words = len(list(set(re.findall(r"\w+|[^\w]", text))))
    print(
        "[bpe_tokenizer] "
        f"corpus_chars={len(text):,}, num_merges_cap={num_merges}, "
        f"merges_learned={len(merges)}, vocab_size={len(vocab)}, "
        f"all_tokens_in_whole_text={all_tokens_in_whole_text:,}"
        f"seperate_words_in_whole_text={seperate_words}"
    )

    return encode, decode, vocab, merges
