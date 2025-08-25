import re
from collections import Counter
from typing import Iterable, List, Tuple, Dict, Union, Set
from get_corpus import get_corpus


Token = str
Pair = Tuple[Token, Token]

# base token format like: a_16, b_5.34, OTA_8, etc.
BASE_TOKEN_RE = re.compile(r"^[A-Za-z]+_\d+(?:\.\d+)?$")


def learn_bpe_growing_vocab(
    corpus: Union[Iterable[str], str],
    desired_vocab_size: int,
    joiner: str = " ",
) -> Tuple[List[Pair], List[Token], Dict[Token, int]]:
    """
    Start with alphabet = unique base tokens from corpus (letter_float).
    Repeatedly merge the most frequent adjacent pair in the *current* sequence,
    add the merged token to vocab, and re-count, until vocab reaches desired size.

    Returns:
        merges        : list of (left, right) pairs merged, in order
        vocab         : final vocabulary list (alphabet first, then merged tokens)
        token_counts  : counts after applying all merges to the sequence
    """
    # normalize input to token list
    tokens: List[Token] = corpus.split() if isinstance(corpus, str) else list(corpus)

    # initial alphabet (only base tokens that match letter_float)
    alphabet: Set[Token] = {t for t in tokens if BASE_TOKEN_RE.match(t)}
    vocab: List[Token] = sorted(alphabet)

    def count_pairs(seq: List[Token]) -> Counter[Pair]:
        return Counter(zip(seq, seq[1:]))

    def merge_once(seq: List[Token], pair: Pair, new_tok: Token) -> List[Token]:
        out: List[Token] = []
        a, b = pair
        i = 0
        n = len(seq)
        while i < n:
            if i + 1 < n and seq[i] == a and seq[i + 1] == b:
                out.append(new_tok)
                i += 2
            else:
                out.append(seq[i])
                i += 1
        return out

    merges: List[Pair] = []
    seq: List[Token] = tokens[:]

    # if desired size is not larger than the alphabet, stop early
    if desired_vocab_size <= len(vocab):
        token_counts: Dict[Token, int] = dict(Counter(seq))
        return merges, vocab, token_counts

    while len(vocab) < desired_vocab_size:
        pair_freqs: Counter[Pair] = count_pairs(seq)
        if not pair_freqs:
            break

        # mypy-safe arg to key:
        most_frequent, _ = pair_freqs.most_common(1)[0]
        new_tok: Token = f"{most_frequent[0]}{joiner}{most_frequent[1]}"

        # apply the merge
        seq = merge_once(seq, most_frequent, new_tok)

        # grow vocab if this merged token is new
        if new_tok not in alphabet and new_tok not in vocab:
            merges.append(most_frequent)
            vocab.append(new_tok)

        # loop continues and counts pairs again on the updated sequence

    token_counts = dict(Counter(seq))
    return merges, vocab, token_counts


# ---- example usage with your get_corpus() ----

tokens, tempo = get_corpus(
    fundamentals_path="../mel_48000.json",
    file_path="../data/first_data/old1.wav",
)
merges, vocab, counts = learn_bpe_growing_vocab(tokens, desired_vocab_size=40)
print("merges (first 10):", merges[:10])
print("vocab size:", len(vocab))
print("sample counts:", list(counts.items())[:])
