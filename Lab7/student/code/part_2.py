"""
Part 2: Implement a simple Byte-Pair Encoding (BPE) tokenizer.
Task: bpe_tokenizer(text, num_merges) returns (encode, decode, vocabulary).
"""
from collections import Counter
import re
from typing import Callable, List, Tuple

MergeList = List[Tuple[Tuple[str, str], str]]

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
        if iteration % 50 == 0:
            print(f"[bpe_tokenizer] Iteration {iteration}: {len(merges)} merges learned, vocab size: {len(sorted(set(text) | {merged for _, merged in merges}))}")

    vocab = sorted(set(text) | {merged for _, merged in merges})
    stoi = {token: i for i, token in enumerate(vocab)}
    itos = {i: token for i, token in enumerate(vocab)}

    def encode(s: str) -> List[int]:
        """
        TODO:
        1. Split s into chars (list).
        2. Apply merges in order (same rule as training loop).
        3. Map each token string to id via stoi.
        """
        raise NotImplementedError("TODO: Implement encode.")

    def decode(ids: List[int]) -> str:
        """
        TODO: Decode a list of token ids back to a string.
        """
        raise NotImplementedError("TODO: Implement decode.")

    return encode, decode, vocab

def verify_part2() -> None:
    """Check BPE round-trip, vocab range, and merge list structure."""
    sample = "the cat sat on the mat. the dog ran."
    enc, dec, vocab, merges = bpe_tokenizer(sample, num_merges=20)
    assert len(merges) <= 20
    for pair, merged_token in merges:
        assert len(pair) == 2 and merged_token == pair[0] + pair[1]
    reconstructed = dec(enc(sample))
    assert reconstructed == sample, f"BPE round-trip failed: got '{reconstructed}'"
    assert len(vocab) >= 1 and len(vocab) <= len(sample) + 20
    print("Part 2: verification passed [OK]")


def part2() -> None:
    """Run BPE on sample text and verify."""
    print("=== Part 2: BPE Tokenizer ===\n")
    sample = "Litwo ojczyzno moja ty jestes jak zdrowie"
    enc, dec, vocab, merges = bpe_tokenizer(sample, num_merges=30)
    print(f"Sample: \"{sample}\"\nBPE vocabulary: {vocab}")
    print(f"Vocab size: {len(vocab)}  |  merges: {len(merges)}")
    ids = enc(sample)
    print(f"Encoded length: {len(ids)} (chars: {len(sample)})")
    print(f"Decode(encode(text)): \"{dec(ids)}\"\n")
    verify_part2()
    print("All tests passed [OK]")


if __name__ == "__main__":
    part2()
