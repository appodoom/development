import re
from collections import Counter
from typing import Iterable, List, Tuple, Dict, Union, Set

Token = str
Pair = Tuple[Token, Token]

BASE_TOKEN_RE = re.compile(r"^[A-Za-z]+_\d+_-?\d+(?:\.\d+)?_-?\d+(?:\.\d+)?_\d+$")
SPECIAL_TOKENS: Set[Token] = {"<EOF>"}


def creat_vocab(
    corpus: Union[Iterable[str], str],
    desired_vocab_size: int,
    joiner: str = " ",
) -> Tuple[List[Pair], List[Token], Dict[Token, int]]:
    tokens: List[Token] = corpus.split() if isinstance(corpus, str) else list(corpus)

    # initial alphabet: base tokens + special tokens
    alphabet: Set[Token] = {t for t in tokens if BASE_TOKEN_RE.match(t)}
    vocab: List[Token] = sorted(SPECIAL_TOKENS) + sorted(
        alphabet
    )  # keep specials in vocab

    def count_pairs(seq: List[Token]) -> Counter[Pair]:
        c: Counter[Pair] = Counter()
        for a, b in zip(seq, seq[1:]):
            # do NOT allow pairs that touch boundaries/special tokens
            if a in SPECIAL_TOKENS or b in SPECIAL_TOKENS:
                continue
            c[(a, b)] += 1
        return c

    def merge_once(seq: List[Token], pair: Pair, new_tok: Token) -> List[Token]:
        out: List[Token] = []
        a, b = pair
        i = 0
        n = len(seq)
        while i < n:
            if (
                i + 1 < n
                and seq[i] == a
                and seq[i + 1] == b
                and seq[i] not in SPECIAL_TOKENS
                and seq[i + 1] not in SPECIAL_TOKENS
            ):
                out.append(new_tok)
                i += 2
            else:
                out.append(seq[i])
                i += 1
        return out

    merges: List[Pair] = []
    seq: List[Token] = tokens[:]

    if desired_vocab_size <= len(vocab):
        return merges, vocab, dict(Counter(seq))

    while len(vocab) < desired_vocab_size:
        pair_freqs = count_pairs(seq)
        if not pair_freqs:
            break

        most_frequent, _ = pair_freqs.most_common(1)[0]
        new_tok: Token = f"{most_frequent[0]}{joiner}{most_frequent[1]}"

        seq = merge_once(seq, most_frequent, new_tok)

        if new_tok not in vocab:
            merges.append(most_frequent)
            vocab.append(new_tok)

    return merges, vocab, dict(Counter(seq))
