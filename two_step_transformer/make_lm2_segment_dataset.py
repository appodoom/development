# 03_make_lm2_segment_dataset.py
import os
import math
from token_utils import (
    load_sequences_from_json, load_json, write_jsonl,
    parse_beats_from_raw, extract_skeleton_delay_stream,
    skeleton_delays_to_beat_offsets, make_beat_skeleton_tokens,
    is_skeleton_hit, offset_of_skeleton_token, segment_len
)

# ====== GLOBAL CONFIG ======
DATA_JSON_PATH = "data/all_tokens.json"
EOF_TOKEN = "<EOF>"

OUT_DIR = "out_method3"
SPLITS_PATH = f"{OUT_DIR}/splits.json"

LM2_TRAIN_JSONL = f"{OUT_DIR}/lm2_train.jsonl"
LM2_VAL_JSONL   = f"{OUT_DIR}/lm2_val.jsonl"
LM2_TEST_JSONL  = f"{OUT_DIR}/lm2_test.jsonl"
# ===========================

def split_variations_by_offsets(var_tokens, offsets):
    """
    var_tokens: list length K
    offsets: sorted list of offsets in [0,1] for skeleton events inside beat (excluding silence placeholder ideally)

    We compute split points p_i = round(K * offset_i).
    Returns split indices list including 0 and K.
    """
    K = len(var_tokens)
    points = [0]
    for o in offsets:
        p = int(round(K * o))
        p = max(0, min(K, p))
        points.append(p)
    points.append(K)

    # make monotonic
    fixed = [points[0]]
    for p in points[1:]:
        if p < fixed[-1]:
            p = fixed[-1]
        fixed.append(p)
    return fixed

def make_segments_for_beat(beat_skel_tokens):
    """
    beat_skel_tokens like: ["<SOB>", doom_S_0.5, "<EOB>"] or ["<SOB>", silence_S_1, "<EOB>"]
    Returns ordered boundaries list:
      [("<SOB>", tok1), (tok1, "<EOB>")] etc
    For multiple skeletons inside beat: ("<SOB>", s1), (s1,s2), ..., (slast,"<EOB>")
    If only silence_S_1: single ("<SOB>", "silence_S_1") AND we keep <EOB> in src format for that case.
    """
    inner = [t for t in beat_skel_tokens if is_skeleton_hit(t)]
    if len(inner) == 1 and inner[0].startswith("silence_S_"):
        # whole-beat segment
        return [("<SOB>", inner[0], "<EOB>")]  # 3-boundary form for this special case

    # real skeleton(s)
    # sort by offset
    inner_sorted = sorted(inner, key=lambda x: offset_of_skeleton_token(x) if offset_of_skeleton_token(x) is not None else 0.0)

    segs = []
    # from SOB to first skeleton
    segs.append(("<SOB>", inner_sorted[0], None))
    # middle segments between skeleton hits
    for i in range(len(inner_sorted) - 1):
        segs.append((inner_sorted[i], inner_sorted[i+1], None))
    # last skeleton to EOB
    segs.append((inner_sorted[-1], "<EOB>", None))
    return segs

def build_src_for_segment(seg):
    """
    seg can be:
      ("<SOB>", "silence_S_1", "<EOB>")   -> BOS <SOB> silence_S_1 <EOB> EOS
      ("<SOB>", skel, None)              -> BOS <SOB> skel EOS
      (skel, "<EOB>", None)              -> BOS skel <EOB> EOS
      (skel1, skel2, None)               -> BOS skel1 skel2 EOS
    """
    if len(seg) == 3 and seg[2] == "<EOB>":
        return ["BOS", seg[0], seg[1], seg[2], "EOS"]

    left, right, _ = seg
    if left == "<SOB>" and right != "<EOB>":
        return ["BOS", "<SOB>", right, "EOS"]
    if right == "<EOB>" and left != "<SOB>":
        return ["BOS", left, "<EOB>", "EOS"]
    # middle
    return ["BOS", left, right, "EOS"]

def seg_length_from_boundaries(seg):
    if len(seg) == 3 and seg[2] == "<EOB>":
        # <SOB> -> silence_S_1 is full beat
        return 1.0

    left, right, _ = seg
    L = segment_len(left, right)
    return L

def make_examples_for_sequence(seq_tokens):
    beats_raw = parse_beats_from_raw(seq_tokens)
    num_beats = len(beats_raw)
    if num_beats == 0:
        return []

    skel_stream = extract_skeleton_delay_stream(seq_tokens)
    per_beat = skeleton_delays_to_beat_offsets(skel_stream, num_beats)
    beat_skel_tokens = make_beat_skeleton_tokens(per_beat)

    examples = []

    for bi in range(num_beats):
        subd_tok = beats_raw[bi]["subd_tok"]
        k = int(subd_tok.replace("SUBD", "")) if subd_tok.startswith("SUBD") else len(beats_raw[bi]["var_tokens"])
        var_tokens = beats_raw[bi]["var_tokens"]

        # safety: if data is inconsistent, trust actual list length
        K = len(var_tokens)
        if K == 0:
            continue

        # skeleton events inside beat (exclude silence placeholder)
        inner = [t for t in beat_skel_tokens[bi] if is_skeleton_hit(t)]
        real_skel = [t for t in inner if not t.startswith("silence_S_")]
        offsets = sorted([offset_of_skeleton_token(t) for t in real_skel if offset_of_skeleton_token(t) is not None])

        # split points proportional to offsets
        split_points = split_variations_by_offsets(var_tokens, offsets)

        segs = make_segments_for_beat(beat_skel_tokens[bi])

        # Build per-segment targets by slicing var_tokens.
        # We align segment order with split_points order.
        # For silence beat: single segment gets all K vars.
        if len(segs) == 1 and len(segs[0]) == 3 and segs[0][2] == "<EOB>":
            src = build_src_for_segment(segs[0])
            tgt = ["BOS", subd_tok] + var_tokens + ["EOS"]
            examples.append({"src": src, "tgt": tgt})
            continue

        # real skeleton: segments count = len(offsets)+2 (SOB->s1, between, last->EOB)
        # split_points length = len(offsets)+2 as well (includes 0 and K)
        # segments correspond to consecutive intervals [p_i, p_{i+1}]
        # If there are multiple skeletons, this will map properly.
        if len(split_points) < 2:
            continue

        seg_intervals = []
        for i in range(len(split_points) - 1):
            a = split_points[i]
            b = split_points[i + 1]
            seg_intervals.append((a, b))

        # segs includes the last (skel_last -> EOB), so count should match seg_intervals
        # sometimes rounding makes counts mismatch; clamp to min
        m = min(len(segs), len(seg_intervals))

        for si in range(m):
            a, b = seg_intervals[si]
            seg_vars = var_tokens[a:b]

            # If this segment duration is ~0, skip
            L = seg_length_from_boundaries(segs[si])
            if L is None or L <= 1e-9:
                continue
            if len(seg_vars) == 0:
                # allowed to be empty, but usually skip training empty segments
                continue

            src = build_src_for_segment(segs[si])
            tgt = ["BOS", subd_tok] + seg_vars + ["EOS"]

            examples.append({"src": src, "tgt": tgt})

    return examples

def build(split_ids, seqs):
    rows = []
    for i in split_ids:
        rows.extend(make_examples_for_sequence(seqs[i]))
    return rows

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    seqs = load_sequences_from_json(DATA_JSON_PATH, eof_token=EOF_TOKEN)
    splits = load_json(SPLITS_PATH)

    write_jsonl(LM2_TRAIN_JSONL, build(splits["train_ids"], seqs))
    write_jsonl(LM2_VAL_JSONL,   build(splits["val_ids"], seqs))
    write_jsonl(LM2_TEST_JSONL,  build(splits["test_ids"], seqs))

    print("done ✅ LM2 segment datasets written")

if __name__ == "__main__":
    main()
