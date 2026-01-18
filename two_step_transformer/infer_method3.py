# 05_infer_method3.py
import os
import torch
from transformers import GPT2LMHeadModel, BartForConditionalGeneration

from token_utils import (
    load_json, encode, decode,
    is_skeleton_hit, is_variation_hit, is_subd, subd_k,
    offset_of_skeleton_token, segment_len
)

# ====== GLOBAL CONFIG ======
OUT_DIR = "out_method3"

VOCAB_PATH = f"{OUT_DIR}/vocab.json"
ID2TOK_PATH = f"{OUT_DIR}/id2tok.json"

LM1_MODEL_DIR = f"{OUT_DIR}/lm1_model"
LM2_MODEL_DIR = f"{OUT_DIR}/lm2_model"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LM1_MAX_NEW = 512
LM1_TOPK = 40
LM1_TEMP = 1.0

LM2_TOPK = 40
LM2_TEMP = 1.0

MAX_BEATS_AT_INFER = 16  # how many beats you want LM1 to generate

OUT_TXT = f"{OUT_DIR}/generated_beats.txt"
# ===========================

def top_k_sample(logits, k=40, temperature=1.0):
    if temperature <= 0:
        temperature = 1.0
    logits = logits / temperature
    if k is not None and k > 0:
        vals, idxs = torch.topk(logits, k)
        probs = torch.softmax(vals, dim=-1)
        pick = torch.multinomial(probs, 1).item()
        return idxs[pick].item()
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1).item()

def gen_lm1_beats(model, vocab, id2tok):
    """
    LM1 outputs: BOS <SOB> ... <EOB> <SOB> ... <EOB> ... EOS
    We stop after EOS or after MAX_BEATS_AT_INFER beats.
    """
    bos = vocab["BOS"]
    eos = vocab["EOS"]

    ids = [bos]
    x = torch.tensor([ids], dtype=torch.long, device=DEVICE)

    for _ in range(LM1_MAX_NEW):
        out = model(input_ids=x)
        logits = out.logits[0, -1, :]
        nxt = top_k_sample(logits, k=LM1_TOPK, temperature=LM1_TEMP)
        ids.append(nxt)
        x = torch.tensor([ids], dtype=torch.long, device=DEVICE)

        tok = id2tok[str(nxt)]
        if tok == "EOS":
            break

        # optional: stop after enough beats
        toks = decode(ids, id2tok)
        beat_count = toks.count("<EOB>")
        if beat_count >= MAX_BEATS_AT_INFER:
            # force EOS next by breaking (we'll just stop)
            break

    toks = decode(ids, id2tok)
    return toks

def parse_lm1_beats(tokens):
    """
    Takes LM1 generated tokens and extracts beat blocks:
      <SOB> ... <EOB>
    """
    beats = []
    i = 0
    while i < len(tokens):
        if tokens[i] == "<SOB>":
            j = i + 1
            while j < len(tokens) and tokens[j] != "<EOB>":
                j += 1
            if j < len(tokens) and tokens[j] == "<EOB>":
                beats.append(tokens[i:j+1])
                i = j + 1
            else:
                break
        else:
            i += 1
    return beats

def segments_from_beat(beat_tokens):
    """
    beat_tokens: ["<SOB>", ..., "<EOB>"]
    If beat contains only silence_S_1 -> one segment (<SOB>, silence_S_1, <EOB>)
    Else:
      (<SOB>, first_skel)
      (skel_i, skel_{i+1}) ...
      (last_skel, <EOB>)
    """
    inner = [t for t in beat_tokens if is_skeleton_hit(t)]
    if len(inner) == 1 and inner[0].startswith("silence_S_"):
        return [("<SOB>", inner[0], "<EOB>")]

    # real skeleton tokens sorted by offset
    real = [t for t in inner if not t.startswith("silence_S_")]
    real.sort(key=lambda x: offset_of_skeleton_token(x) if offset_of_skeleton_token(x) is not None else 0.0)

    if not real:
        # if LM1 forgot skeleton token, treat as silent beat
        return [("<SOB>", "silence_S_1", "<EOB>")]

    segs = []
    segs.append(("<SOB>", real[0], None))
    for i in range(len(real)-1):
        segs.append((real[i], real[i+1], None))
    segs.append((real[-1], "<EOB>", None))
    return segs

def build_src(seg):
    if len(seg) == 3 and seg[2] == "<EOB>":
        return ["BOS", "<SOB>", seg[1], "<EOB>", "EOS"]

    left, right, _ = seg
    if left == "<SOB>":
        return ["BOS", "<SOB>", right, "EOS"]
    if right == "<EOB>":
        return ["BOS", left, "<EOB>", "EOS"]
    return ["BOS", left, right, "EOS"]

def get_seg_len(seg):
    if len(seg) == 3 and seg[2] == "<EOB>":
        return 1.0
    left, right, _ = seg
    return segment_len(left, right)

def constrained_decode_segment(lm2, vocab, id2tok, src_tokens, seg_len_value):
    """
    Forces:
      1) first generated token must be SUBDk
      2) then exactly n = round(k * seg_len_value) variation hits
      3) then stop
    Returns tokens excluding BOS/EOS.
    """
    pad_id = vocab["PAD"]
    bos_id = vocab["BOS"]

    # allowed sets
    subd_ids = [i for t, i in vocab.items() if t.startswith("SUBD")]
    var_ids  = [i for t, i in vocab.items() if "_V_" in t]

    src_ids = encode(src_tokens, vocab)
    src = torch.tensor([src_ids], dtype=torch.long, device=DEVICE)
    attn = (src != pad_id).long()

    dec_ids = [bos_id]

    # step 1: pick SUBDk
    dec = torch.tensor([dec_ids], dtype=torch.long, device=DEVICE)
    out = lm2(input_ids=src, attention_mask=attn, decoder_input_ids=dec)
    logits = out.logits[0, -1, :]

    masked = torch.full_like(logits, float("-inf"))
    for sid in subd_ids:
        masked[sid] = logits[sid]

    subd_id = top_k_sample(masked, k=LM2_TOPK, temperature=LM2_TEMP)
    subd_tok = id2tok[str(subd_id)]
    k = subd_k(subd_tok)
    if k is None:
        return []

    dec_ids.append(subd_id)

    # compute how many variation hits allowed in this segment
    n = int(round(k * seg_len_value))
    if n < 0:
        n = 0

    # step 2: generate exactly n variation tokens
    for _ in range(n):
        dec = torch.tensor([dec_ids], dtype=torch.long, device=DEVICE)
        out = lm2(input_ids=src, attention_mask=attn, decoder_input_ids=dec)
        logits = out.logits[0, -1, :]

        masked = torch.full_like(logits, float("-inf"))
        for vid in var_ids:
            masked[vid] = logits[vid]

        vid = top_k_sample(masked, k=LM2_TOPK, temperature=LM2_TEMP)
        dec_ids.append(vid)

    # decode excluding initial BOS
    out_tokens = decode(dec_ids[1:], id2tok)
    return out_tokens

def main():
    vocab = load_json(VOCAB_PATH)
    id2tok = load_json(ID2TOK_PATH)

    lm1 = GPT2LMHeadModel.from_pretrained(LM1_MODEL_DIR).to(DEVICE)
    lm2 = BartForConditionalGeneration.from_pretrained(LM2_MODEL_DIR).to(DEVICE)
    lm1.eval()
    lm2.eval()

    lm1_tokens = gen_lm1_beats(lm1, vocab, id2tok)
    beats = parse_lm1_beats(lm1_tokens)
    if not beats:
        print("LM1 produced no complete beats 😭")
        return

    final = []
    for beat in beats:
        segs = segments_from_beat(beat)

        # reconstruct beat: <SOB> ... <EOB>
        beat_out = ["<SOB>"]

        # find skeleton tokens in beat in sorted order for inserting between segments
        inner_skel = [t for t in beat if is_skeleton_hit(t) and not t.startswith("silence_S_")]
        inner_skel.sort(key=lambda x: offset_of_skeleton_token(x) if offset_of_skeleton_token(x) is not None else 0.0)

        skel_insert_idx = 0

        for seg in segs:
            L = get_seg_len(seg)
            if L is None or L <= 1e-9:
                continue

            src = build_src(seg)
            seg_fill = constrained_decode_segment(lm2, vocab, id2tok, src, L)

            # append generated SUBDk + vars
            beat_out.extend(seg_fill)

            # if this segment ends at a skeleton token, insert it
            if len(seg) == 3 and seg[2] == "<EOB>":
                # silent beat, no real skeleton insertion
                pass
            else:
                left, right, _ = seg
                # segment ending at skeleton token
                if right != "<EOB>" and is_skeleton_hit(right) and not right.startswith("silence_S_"):
                    beat_out.append(right)

        beat_out.append("<EOB>")
        final.extend(beat_out)

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write(" ".join(final) + "\n")

    print("done ✅ wrote ->", OUT_TXT)

if __name__ == "__main__":
    main()
