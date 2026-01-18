# convert_to_pipeline_tokens.py
# Turns your current "sequence" format into the raw token stream expected by the beat/segment pipeline:
#   - SUBDk tokens (beat boundaries)  -> "SUBD4", "SUBD8", ...
#   - variation hits                 -> "<HIT>_V_1"
#   - skeleton hits w/ delay (beats) -> "<HIT>_S_<delay>"
# Output JSON format:
#   {"tokens": [ ...tokens for file1..., "<EOF>", ...tokens for file2..., "<EOF>", ... ]}

import os
import json
import math

# ===================== GLOBAL CONFIG =====================
INPUT_PATH = "../sequence_khara.json"          # can be a single .json file OR a directory containing many .json files
OUTPUT_JSON_PATH = "all_tokens.json"

EOF_TOKEN = "<EOF>"

DEFAULT_SUBD = 4                      # used only if we see variation hits before any SUBD (shouldn't happen)
SKELETON_DELAY_QUANT = 0.25           # quantize skeleton delays to nearest 0.25 beats (matches your earlier spec)
VAR_TOKEN_FLOAT = "1"                 # we keep variation hit tokens as HITNAME_V_1 (float not used by pipeline)
# =========================================================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def quantize(x, q):
    if q is None or q <= 0:
        return x
    return round(x / q) * q

def fmt_float(x):
    if x is None:
        return "0"
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    s = f"{x:.6f}".rstrip("0").rstrip(".")
    return s

def is_tempo(tok):
    return tok.startswith("TEMPO_")

def is_subd(tok):
    return tok.startswith("SUBD_") or tok.startswith("SUBD")

def parse_subd_k(tok):
    # accepts "SUBD_4" or "SUBD4"
    if tok.startswith("SUBD_"):
        try:
            return int(tok.split("_", 1)[1])
        except:
            return None
    if tok.startswith("SUBD"):
        try:
            return int(tok.replace("SUBD", ""))
        except:
            return None
    return None

def is_hit(tok):
    return tok.startswith("HIT_")

def hit_name(tok):
    # "HIT_OTA" -> "OTA"
    return tok.replace("HIT_", "", 1)

def is_dev(tok):
    return tok.startswith("DEV_")

def is_amp(tok):
    return tok.startswith("AMP_")

def convert_one_sequence(seq_tokens):
    """
    Walk the token stream and emit:
      - SUBDk at beat boundaries
      - variation tokens for HIT + AMP
      - skeleton tokens (with computed delay) for HIT + DEV

    Timing model:
      - each beat has length 1.0
      - within a beat, each variation hit consumes 1/subd beats
      - skeleton hits occur at the current within-beat position (instantaneous)
      - skeleton token stores delay (in beats) since previous skeleton
    """
    out = []

    in_beat = False
    beat_start_time = 0.0          # absolute time of current beat start
    beat_pos = 0.0                 # progress within beat [0,1]
    current_subd = None

    last_skel_time = 0.0
    have_seen_skeleton = False

    i = 0
    n = len(seq_tokens)

    while i < n:
        tok = seq_tokens[i]

        # ignore tempo
        if is_tempo(tok):
            i += 1
            continue

        # new beat
        if is_subd(tok):
            k = parse_subd_k(tok)
            if k is None:
                i += 1
                continue

            # close previous beat (advance to next beat boundary)
            if in_beat:
                beat_start_time += 1.0
                beat_pos = 0.0
            else:
                in_beat = True
                beat_start_time = 0.0
                beat_pos = 0.0

            current_subd = k
            out.append(f"SUBD{k}")
            i += 1
            continue

        # hits
        if is_hit(tok):
            name = hit_name(tok)

            # need a marker token right after: DEV_* => skeleton, AMP_* => variation
            if i + 1 < n and is_dev(seq_tokens[i + 1]):
                # skeleton hit at current absolute time
                abs_t = (beat_start_time + beat_pos) if in_beat else 0.0

                if not have_seen_skeleton:
                    delay = abs_t
                    have_seen_skeleton = True
                else:
                    delay = abs_t - last_skel_time

                delay = max(0.0, delay)
                delay = quantize(delay, SKELETON_DELAY_QUANT)

                out.append(f"{name}_S_{fmt_float(delay)}")
                last_skel_time = abs_t

                i += 2
                continue

            if i + 1 < n and is_amp(seq_tokens[i + 1]):
                # variation hit consumes one subdivision slot
                out.append(f"{name}_V_{VAR_TOKEN_FLOAT}")

                k = current_subd if current_subd is not None else DEFAULT_SUBD
                if in_beat and k and k > 0:
                    beat_pos += 1.0 / float(k)
                    # if more hits than slots, clamp (data should normally align)
                    if beat_pos > 1.0:
                        beat_pos = 1.0

                i += 2
                continue

            # HIT without AMP/DEV right after -> ignore safely
            i += 1
            continue

        # ignore DEV/AMP if they appear alone
        if is_dev(tok) or is_amp(tok):
            i += 1
            continue

        # ignore anything else
        i += 1

    return out

def list_input_files(path):
    if os.path.isfile(path) and path.lower().endswith(".json"):
        return [path]
    if os.path.isdir(path):
        files = []
        for fn in sorted(os.listdir(path)):
            if fn.lower().endswith(".json"):
                files.append(os.path.join(path, fn))
        return files
    return []

def main():
    files = list_input_files(INPUT_PATH)
    if not files:
        print("brooo no json files found at:", INPUT_PATH)
        return

    all_tokens = []
    kept = 0

    for fp in files:
        obj = load_json(fp)

        # expected: {"sequence":[...]}
        seq = obj.get("sequence", None)
        if not isinstance(seq, list):
            # allow also {"tokens":[...]} just in case
            seq = obj.get("tokens", None)

        if not isinstance(seq, list):
            print("skipping (no sequence list):", fp)
            continue

        converted = convert_one_sequence(seq)

        if len(converted) == 0:
            print("skipping (empty after convert):", fp)
            continue

        all_tokens.extend(converted)
        all_tokens.append(EOF_TOKEN)
        kept += 1

    save_json(OUTPUT_JSON_PATH, {"tokens": all_tokens})
    print("done ✅ converted files:", kept, "| wrote:", OUTPUT_JSON_PATH)

if __name__ == "__main__":
    main()
