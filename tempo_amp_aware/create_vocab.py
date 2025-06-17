import re
from collections import Counter
from pathlib import Path


def get_stats(vocab):
    """
    Count frequency of each adjacent symbol pair in the vocab.
    vocab: dict mapping word string (space-separated tokens) to frequency
    """
    pairs = Counter()
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i + 1])] += freq
    return pairs


def merge_vocab(pair, vocab):
    """
    Merge all occurrences of `pair` in the vocab into a single symbol.
    """
    # Escape the pair tokens for regex (they may contain dots etc.)
    pat = re.escape(' '.join(pair))
    # New merged token: simply concatenate pair tokens without space
    rep = pair[0] + pair[1]
    merged = {}
    # Pattern matches the pair as whole tokens separated by space
    pattern = re.compile(rf'(?<!\S){pat}(?!\S)')
    for word, freq in vocab.items():
        # Replace all occurrences of the pair with the merged token
        new_word = pattern.sub(rep, word)
        merged[new_word] = freq
    return merged


def train_bpe(corpus_path, output_vocab_path, target_vocab_size):
    # 1) load and split into atomic tokens (tokens like d_2.67_0)
    text = Path(corpus_path).read_text(encoding='utf-8')
    tokens = text.strip().split()  # tokens are your full tokens, e.g. d_2.67_0

    # 2) initial vocab: treat entire token sequence as one "word"
    vocab = {' '.join(tokens): 1}

    # 3) initialize bpe_vocab with all unique tokens in order
    unique_tokens = list(dict.fromkeys(tokens))
    bpe_vocab = unique_tokens.copy()

    # 4) iterative merges until target vocab size
    while len(bpe_vocab) < target_vocab_size:
        pairs = get_stats(vocab)
        if not pairs:
            break
        best_pair, freq = pairs.most_common(1)[0]
        new_token = best_pair[0] + best_pair[1]  # merged token concatenated
        bpe_vocab.append(new_token)
        vocab = merge_vocab(best_pair, vocab)

    # Write merges to file, one per line
    Path(output_vocab_path).write_text("".join(tok + "\n" for tok in bpe_vocab), encoding='utf-8')
    print(f"✔️ Wrote {len(bpe_vocab)} tokens to {output_vocab_path}")


if __name__ == '__main__':
    train_bpe(
        corpus_path='./demo_with_amp/corpus_with_amp/merged_enhanced.txt',
        output_vocab_path='vocab_music_sheet_2500.txt',
        target_vocab_size=1201
    )
