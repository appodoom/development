# token_utils.py
import json
import random
import math

# ===== Special tokens we will actually use now =====
SPECIAL_TOKENS = [
    "PAD", "UNK", "BOS", "EOS",
    "<SOB>", "<EOB>",
]

def set_seed(seed):
    random.seed(seed)

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                rows.append(json.loads(ln))
    return rows

def is_subd(tok):
    return tok.startswith("SUBD")

def subd_k(tok):
    try:
        return int(tok.replace("SUBD", ""))
    except:
        return None

def is_skeleton_hit(tok):
    return "_S_" in tok

def is_variation_hit(tok):
    return "_V_" in tok

def hit_type(tok):
    # doom_S_0.5 -> doom
    try:
        return tok.split("_")[0]
    except:
        return None

def hit_float(tok):
    # doom_S_1.5 -> 1.5
    try:
        return float(tok.split("_")[-1])
    except:
        return None

def fmt_float(x):
    # 0.5 -> "0.5", 1.0 -> "1", 0.0 -> "0"
    if x is None:
        return "0"
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    s = f"{x:.6f}".rstrip("0").rstrip(".")
    return s

def split_by_eof(token_stream, eof_token="<EOF>"):
    seqs = []
    cur = []
    for t in token_stream:
        if t == eof_token:
            if cur:
                seqs.append(cur)
            cur = []
        else:
            cur.append(t)
    if cur:
        seqs.append(cur)
    return seqs

def load_sequences_from_json(data_json_path, eof_token="<EOF>"):
    """
    Supports:
      1) {"tokens": [..., "<EOF>", ...]}
      2) [..., "<EOF>", ...]
      3) {"files":[{"tokens":[...]}, ...]}
    Returns: List[List[str]]
    """
    data = load_json(data_json_path)

    try:
        if isinstance(data, dict) and "tokens" in data and isinstance(data["tokens"], list):
            return split_by_eof(data["tokens"], eof_token=eof_token)

        if isinstance(data, list):
            return split_by_eof(data, eof_token=eof_token)

        if isinstance(data, dict) and "files" in data and isinstance(data["files"], list):
            seqs = []
            for fobj in data["files"]:
                if isinstance(fobj, dict) and "tokens" in fobj and isinstance(fobj["tokens"], list):
                    if fobj["tokens"]:
                        seqs.append(fobj["tokens"])
            return seqs
    except:
        pass

    print("brooo JSON format not recognized 😭")
    return []

def split_train_val_test(n, seed=123, train=0.9, val=0.05):
    set_seed(seed)
    idxs = list(range(n))
    random.shuffle(idxs)

    n_train = int(n * train)
    n_val = int(n * val)

    train_ids = idxs[:n_train]
    val_ids = idxs[n_train:n_train + n_val]
    test_ids = idxs[n_train + n_val:]
    return train_ids, val_ids, test_ids

def dedup_preserve_order(lst):
    seen = set()
    out = []
    for x in lst:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def encode(tokens, vocab):
    unk = vocab["UNK"]
    return [vocab.get(t, unk) for t in tokens]

def decode(ids, id2tok):
    return [id2tok[str(i)] for i in ids]

# ---------- Parsing raw tokens into beats ----------
def parse_beats_from_raw(seq_tokens):
    """
    Raw assumption:
      SUBDk begins a beat
      beat continues until next SUBDk or end of sequence
    We store:
      - subd_tok (e.g. "SUBD8")
      - var_tokens: variation hit tokens inside that beat (in order)
      - raw_tokens: all tokens in that beat (optional debugging)
    """
    beats = []
    cur = None

    for t in seq_tokens:
        if is_subd(t):
            # new beat
            if cur is not None:
                beats.append(cur)
            cur = {"subd_tok": t, "var_tokens": [], "raw_tokens": [t]}
            continue

        if cur is None:
            # tokens before first SUBD, ignore
            continue

        cur["raw_tokens"].append(t)
        if is_variation_hit(t):
            cur["var_tokens"].append(t)

    if cur is not None:
        beats.append(cur)

    return beats

def extract_skeleton_delay_stream(seq_tokens):
    """
    Skeleton tokens in raw are treated as a stream of events with delays in beats.
    Returns list of (hitType, delay_float).
    """
    out = []
    for t in seq_tokens:
        if is_skeleton_hit(t):
            ht = hit_type(t)
            dt = hit_float(t)
            if ht is None or dt is None:
                continue
            out.append((ht, dt))
    return out

def skeleton_delays_to_beat_offsets(skel_delay_stream, num_beats):
    """
    Converts skeleton delays (in beats) to beat-indexed events with offsets in [0,1).
    We interpret delays cumulatively from time 0 (start of beat 0).

    Example:
      doom_S_1.5 => time=1.5 => beat_idx=1 offset=0.5

    Returns: list length num_beats, each entry is list of (offset, hitType)
    """
    per_beat = [[] for _ in range(num_beats)]
    t = 0.0

    for ht, dt in skel_delay_stream:
        if dt is None:
            continue
        t += dt
        if t < 0:
            continue

        beat_idx = int(math.floor(t + 1e-9))
        offset = t - beat_idx

        # if exactly on boundary, treat as offset 0 in that beat
        if abs(offset) < 1e-9:
            offset = 0.0

        if beat_idx < 0 or beat_idx >= num_beats:
            # event outside the beat grid, ignore
            continue

        per_beat[beat_idx].append((offset, ht))

    # sort within beat
    for b in range(num_beats):
        per_beat[b].sort(key=lambda x: x[0])

    return per_beat

def make_beat_skeleton_tokens(per_beat_offsets):
    """
    Creates beat-level skeleton representation using <SOB> ... <EOB>.
    Rule:
      - if beat has no skeleton events => <SOB> silence_S_1 <EOB>
      - else => <SOB> <hit_S_offset> ... <EOB>
    Note: offset is within-beat position (0.. <1). We keep as float token.
    """
    beats = []
    for events in per_beat_offsets:
        beat_toks = ["<SOB>"]
        if not events:
            beat_toks.append("silence_S_1")
        else:
            for off, ht in events:
                beat_toks.append(f"{ht}_S_{fmt_float(off)}")
        beat_toks.append("<EOB>")
        beats.append(beat_toks)
    return beats

def parse_beat_skeleton_events(beat_tokens):
    """
    Input beat_tokens includes <SOB> ... <EOB>
    Returns skeleton event tokens inside beat (excluding silence placeholder unless you want it).
    """
    inner = [t for t in beat_tokens if is_skeleton_hit(t)]
    return inner

def offset_of_skeleton_token(tok):
    """
    tok: doom_S_0.5 -> 0.5
    silence_S_1 -> 1.0
    """
    return hit_float(tok)

def segment_len(left_boundary, right_boundary):
    """
    Boundaries are either "<SOB>", "<EOB>", or skeleton tokens "*_S_*".
    We treat:
      <SOB> -> offset 0
      <EOB> -> offset 1
      skel token -> offset parsed
    """
    def boundary_offset(b):
        if b == "<SOB>":
            return 0.0
        if b == "<EOB>":
            return 1.0
        # skeleton token
        off = offset_of_skeleton_token(b)
        if off is None:
            return None
        return off

    lo = boundary_offset(left_boundary)
    ro = boundary_offset(right_boundary)
    if lo is None or ro is None:
        return None
    return max(0.0, ro - lo)
