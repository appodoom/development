# 01_make_lm1_skeleton_dataset.py
import os
from token_utils import (
    load_sequences_from_json, load_json, write_jsonl,
    parse_beats_from_raw, extract_skeleton_delay_stream,
    skeleton_delays_to_beat_offsets, make_beat_skeleton_tokens
)

# ====== GLOBAL CONFIG ======
DATA_JSON_PATH = "data/all_tokens.json"
EOF_TOKEN = "<EOF>"

OUT_DIR = "out_method3"
SPLITS_PATH = f"{OUT_DIR}/splits.json"

LM1_TRAIN_JSONL = f"{OUT_DIR}/lm1_train.jsonl"
LM1_VAL_JSONL   = f"{OUT_DIR}/lm1_val.jsonl"
LM1_TEST_JSONL  = f"{OUT_DIR}/lm1_test.jsonl"
# ===========================

def make_lm1_example(seq_tokens):
    beats = parse_beats_from_raw(seq_tokens)
    num_beats = len(beats)
    if num_beats == 0:
        return None

    skel_stream = extract_skeleton_delay_stream(seq_tokens)
    per_beat = skeleton_delays_to_beat_offsets(skel_stream, num_beats)
    beat_skel_tokens = make_beat_skeleton_tokens(per_beat)

    out = ["BOS"]
    for bt in beat_skel_tokens:
        out.extend(bt)
    out.append("EOS")

    return {"tokens": out}

def build(split_ids, seqs):
    rows = []
    for i in split_ids:
        ex = make_lm1_example(seqs[i])
        if ex is not None:
            rows.append(ex)
    return rows

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    seqs = load_sequences_from_json(DATA_JSON_PATH, eof_token=EOF_TOKEN)
    splits = load_json(SPLITS_PATH)

    write_jsonl(LM1_TRAIN_JSONL, build(splits["train_ids"], seqs))
    write_jsonl(LM1_VAL_JSONL,   build(splits["val_ids"], seqs))
    write_jsonl(LM1_TEST_JSONL,  build(splits["test_ids"], seqs))

    print("done ✅ LM1 beat-skeleton datasets written")

if __name__ == "__main__":
    main()
