"""
BPE tokenizer for Lab 9: learn merges on training text, encode/decode, reload from saved vocab + merges.
"""
from collections import Counter
import re
from typing import Callable, List, Tuple

MergeList = List[Tuple[Tuple[str, str], str]]


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
        if iteration % 50 == 0:
            print(f"[bpe_tokenizer] Iteration {iteration}: {len(merges)} merges learned, vocab size: {len(sorted(set(text) | {merged for _, merged in merges}))}")

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
