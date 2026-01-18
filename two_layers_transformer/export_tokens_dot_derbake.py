# export_to_original_derbake.py
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Tuple, Optional

RE_TEMPO = re.compile(r"^TEMPO_(-?\d+(?:\.\d+)?)$")
RE_SUBD = re.compile(r"^SUBD_(\d+)$")
RE_DELAY = re.compile(r"^DELAY_-?\d+(?:\.\d+)?$")
RE_HIT = re.compile(r"^HIT_[A-Z0-9_]+$")
RE_DEV = re.compile(r"^DEV_-?\d+(?:\.\d+)?$")
RE_AMP = re.compile(r"^AMP_-?\d+(?:\.\d+)?$")


def toks(text: str) -> List[str]:
    return [t for t in text.strip().split() if t]


def split_beats(tokens: List[str]) -> List[List[str]]:
    """Extract complete beats: <SOB> ... <EOB>"""
    beats: List[List[str]] = []
    i = 0
    n = len(tokens)
    while i < n:
        while i < n and tokens[i] != "<SOB>":
            i += 1
        if i >= n:
            break
        j = i
        while j < n and tokens[j] != "<EOB>":
            j += 1
        if j >= n:
            break  # incomplete tail
        beats.append(tokens[i : j + 1])
        i = j + 1
    return beats


def tempo_value_from_token(tok: str) -> Optional[str]:
    m = RE_TEMPO.match(tok)
    if not m:
        return None
    s = m.group(1)
    # match your file style: ensure at least one decimal place
    return s if "." in s else (s + ".0")


def parse_beat(beat: List[str]) -> Tuple[str, str, List[str], List[str]]:
    """
    Returns:
      tempo_value_str (e.g. "120.0")
      subd_token (e.g. "SUBD_4")
      event_tokens flat: [DELAY_*, HIT_*, DEV_*, ...]
      vars_tokens raw inside <VARS> ... <VARE>
    """
    tempo_val = "120.0"
    subd_tok = "SUBD_1"

    for t in beat:
        tv = tempo_value_from_token(t)
        if tv is not None:
            tempo_val = tv
        if RE_SUBD.match(t):
            subd_tok = t

    event_tokens: List[str] = []
    vars_tokens: List[str] = []

    i = 0
    end = len(beat)
    while i < end:
        t = beat[i]

        if t == "<VARS>":
            i += 1
            while i < end and beat[i] != "<VARE>":
                vars_tokens.append(beat[i])
                i += 1
            if i < end and beat[i] == "<VARE>":
                i += 1
            continue

        # (DELAY HIT DEV?) triples
        if RE_DELAY.match(t):
            if i + 1 < end and RE_HIT.match(beat[i + 1]):
                event_tokens.append(t)
                event_tokens.append(beat[i + 1])
                if i + 2 < end and RE_DEV.match(beat[i + 2]):
                    event_tokens.append(beat[i + 2])
                    i += 3
                else:
                    # keep DEV_0 default if missing
                    event_tokens.append("DEV_0")
                    i += 2
                continue

        i += 1

    return tempo_val, subd_tok, event_tokens, vars_tokens


def normalize_vars_for_subd(subd_tok: str, vars_tokens: List[str]) -> List[str]:
    """
    Output exactly: SUBD_k then k*(HIT AMP) pairs.
    If missing, fill with HIT_S + AMP_1.
    If AMP missing after a HIT, fill AMP_1.
    """
    m = RE_SUBD.match(subd_tok)
    k = int(m.group(1)) if m else 1

    out: List[str] = [subd_tok]

    # build pairs from vars_tokens
    pairs: List[str] = []
    i = 0
    while i < len(vars_tokens):
        if RE_HIT.match(vars_tokens[i]):
            pairs.append(vars_tokens[i])
            i += 1
            if i < len(vars_tokens) and RE_AMP.match(vars_tokens[i]):
                pairs.append(vars_tokens[i])
                i += 1
            else:
                pairs.append("AMP_1")
        else:
            i += 1

    # enforce exactly 2*k tokens
    need = 2 * k
    if len(pairs) < need:
        pairs.extend(
            (["HIT_S", "AMP_1"] * ((need - len(pairs) + 1) // 2))[: need - len(pairs)]
        )
    pairs = pairs[:need]

    out.extend(pairs)
    return out


def tokens_to_original_derbake(all_tokens: List[str]) -> str:
    # stop early on <EOF> if present
    if "<EOF>" in all_tokens:
        all_tokens = all_tokens[: all_tokens.index("<EOF>")]

    beats = split_beats(all_tokens)
    if not beats:
        raise ValueError("No complete <SOB>...<EOB> beats found in token stream.")

    tempo_list: List[str] = []
    event_stream: List[str] = []
    vars_stream: List[str] = []

    for beat in beats:
        tempo_val, subd_tok, ev, vraw = parse_beat(beat)
        tempo_list.append(tempo_val)
        event_stream.extend(ev)
        vars_stream.extend(normalize_vars_for_subd(subd_tok, vraw))

    initial_tempo = tempo_list[0]  # matches the style of your example file

    # 4-line output (like your uploaded file)
    lines = [
        initial_tempo,
        " ".join(tempo_list),
        " ".join(event_stream),
        " ".join(vars_stream),
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in", dest="inp", required=True, help="Text file containing final tokens"
    )
    ap.add_argument("--out", dest="out", required=True, help="Output .derbake path")
    args = ap.parse_args()

    inp = Path(args.inp)
    out = Path(args.out)

    all_tokens = toks(inp.read_text(encoding="utf-8"))
    out_text = tokens_to_original_derbake(all_tokens)
    out.write_text(out_text, encoding="utf-8")

    print(f"[ok] wrote {out} (beats={out_text.count('SUBD_')})")


if __name__ == "__main__":
    main()
