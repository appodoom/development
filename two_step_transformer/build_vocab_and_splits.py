
# 00_build_vocab_and_splits.py
import os
from token_utils import (
    load_sequences_from_json, save_json, split_train_val_test,
    SPECIAL_TOKENS, dedup_preserve_order,
    is_skeleton_hit, hit_type
)

# ====== GLOBAL CONFIG ======
DATA_JSON_PATH = "data/all_tokens.json"
EOF_TOKEN = "<EOF>"

OUT_DIR = "out_method3"
SEED = 123

TRAIN_FRAC = 0.90
VAL_FRAC = 0.05
# ===========================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    seqs = load_sequences_from_json(DATA_JSON_PATH, eof_token=EOF_TOKEN)
    if not seqs:
        print("no sequences found 😭")
        return

    train_ids, val_ids, test_ids = split_train_val_test(
        len(seqs), seed=SEED, train=TRAIN_FRAC, val=VAL_FRAC
    )

    # gather all tokens from raw
    all_tokens = []
    for s in seqs:
        all_tokens.extend(s)

    # discover hit types (for adding *_S_0 tokens)
    hit_types = set()
    for t in all_tokens:
        if is_skeleton_hit(t):
            ht = hit_type(t)
            if ht:
                hit_types.add(ht)

    vocab_tokens = []
    vocab_tokens.extend(SPECIAL_TOKENS)

    # we will add silence and beat-start skeleton offset tokens
    vocab_tokens.append("silence_S_1")
    for ht in sorted(list(hit_types)):
        vocab_tokens.append(f"{ht}_S_0")  # allow skeleton at exact beat start

    # then add all raw tokens
    vocab_tokens.extend(all_tokens)

    vocab_tokens = dedup_preserve_order(vocab_tokens)

    vocab = {t:i for i, t in enumerate(vocab_tokens)}
    id2tok = {str(i):t for t, i in vocab.items()}

    save_json(f"{OUT_DIR}/vocab.json", vocab)
    save_json(f"{OUT_DIR}/id2tok.json", id2tok)
    save_json(f"{OUT_DIR}/splits.json", {
        "train_ids": train_ids,
        "val_ids": val_ids,
        "test_ids": test_ids,
        "num_sequences": len(seqs),
        "eof_token": EOF_TOKEN,
    })

    print("done ✅ vocab size:", len(vocab), "| sequences:", len(seqs))

if __name__ == "__main__":
    main()
