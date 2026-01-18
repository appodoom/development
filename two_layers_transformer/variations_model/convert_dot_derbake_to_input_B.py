from __future__ import annotations
import re
from dataclasses import dataclass
from decimal import Decimal, getcontext
from pathlib import Path
from typing import List, Optional, Tuple
from defaults_paths import INPUT_FOLDER_PATH, OUTPUT_FOLDER_PATH, ALLOWED_SUFFIXES

getcontext().prec = 28

DELAY_RE = re.compile(r"^DELAY_(-?\d+(?:\.\d+)?)$")
HIT_RE = re.compile(r"^HIT_[A-Z0-9_]+$")
DEV_RE = re.compile(r"^DEV_-?\d+(?:\.\d+)?$")

SUBD_RE = re.compile(r"^SUBD_(\d+)$")
AMP_RE = re.compile(r"^AMP_-?\d+(?:\.\d+)?$")  # variations amplitude


@dataclass
class Event:
    delay: Decimal
    hit: str
    dev: Optional[str] = None


def tokenize_stream(text: str) -> List[str]:
    return text.strip().split()


def parse_delay(tok: str) -> Optional[Decimal]:
    m = DELAY_RE.match(tok)
    return Decimal(m.group(1)) if m else None


def fmt_decimal(x: Decimal) -> str:
    x = x.normalize()
    s = format(x, "f")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s if s else "0"


def fmt_tempo_token(bpm: Decimal) -> str:
    return f"TEMPO_{fmt_decimal(bpm)}"


def read_input_file(path: Path) -> Tuple[Decimal, List[Decimal], str]:
    text = path.read_text(encoding="utf-8")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) < 3:
        raise ValueError(
            f"{path.name}: expected >= 3 non-empty lines:\n"
            "1) initial tempo\n2) tempo per beat list\n3+) event stream + SUBD/variations stream"
        )
    initial_tempo = Decimal(lines[0])
    tempo_list = [Decimal(x) for x in lines[1].split()]
    stream_text = " ".join(lines[2:])
    return initial_tempo, tempo_list, stream_text


def parse_events_prefix(tokens: List[str]) -> Tuple[List[Event], int]:
    """Parse initial prefix: (DELAY HIT DEV?)+ . Return (events, stop_index)."""
    events: List[Event] = []
    i = 0
    n = len(tokens)

    while i < n:
        d = parse_delay(tokens[i])
        if d is None:
            break
        if i + 1 >= n or not HIT_RE.match(tokens[i + 1]):
            break

        hit = tokens[i + 1]
        dev = None
        if i + 2 < n and DEV_RE.match(tokens[i + 2]):
            dev = tokens[i + 2]
            i += 3
        else:
            i += 2

        events.append(Event(delay=d, hit=hit, dev=dev))

    return events, i


def split_into_beats(
    events: List[Event],
    beat_len: Decimal = Decimal("1.0"),
    pad_hit: str = "HIT_S",
    pad_dev: str = "DEV_0",
    eps: Decimal = Decimal("0.0000000001"),
) -> List[List[Event]]:
    beats: List[List[Event]] = []
    cur: List[Event] = []
    remaining = beat_len

    def close_beat_pad_if_needed() -> None:
        nonlocal cur, remaining
        if remaining > eps:
            cur.append(Event(delay=remaining, hit=pad_hit, dev=pad_dev))
            remaining = Decimal("0")
        beats.append(cur)
        cur = []
        remaining = beat_len

    for ev in events:
        d = ev.delay
        if d < 0:
            raise ValueError(f"Negative delay not supported: {ev.delay}")

        while d > remaining + eps:
            cur.append(Event(delay=remaining, hit=pad_hit, dev=pad_dev))
            beats.append(cur)
            cur = []
            d = d - remaining
            remaining = beat_len

        cur.append(Event(delay=d, hit=ev.hit, dev=ev.dev))
        remaining = remaining - d

        if remaining <= eps:
            beats.append(cur)
            cur = []
            remaining = beat_len

    if cur:
        close_beat_pad_if_needed()

    return beats


def parse_variations_stream(tokens: List[str]) -> List[Tuple[str, List[str]]]:
    """
    Parse:
      SUBD_k (HIT_* AMP_*) * k  SUBD_k (HIT_* AMP_*) * k ...
    Return per beat: (subd_token, [HIT, AMP, HIT, AMP, ...])  length = 2*k
    """
    # find first SUBD
    start = None
    for j, t in enumerate(tokens):
        if SUBD_RE.match(t):
            start = j
            break
    if start is None:
        return []

    out: List[Tuple[str, List[str]]] = []
    i = start
    n = len(tokens)

    while i < n:
        m = SUBD_RE.match(tokens[i])
        if not m:
            break
        k = int(m.group(1))
        subd_tok = tokens[i]
        i += 1

        beat_vars: List[str] = []
        for _ in range(k):
            # HIT
            if i < n and HIT_RE.match(tokens[i]):
                beat_vars.append(tokens[i])
                i += 1
            else:
                # if stream breaks, stop cleanly
                return out

            # AMP (if missing, default)
            if i < n and AMP_RE.match(tokens[i]):
                beat_vars.append(tokens[i])
                i += 1
            else:
                beat_vars.append("AMP_1")

        out.append((subd_tok, beat_vars))

    return out


def beats_to_text_b(
    beats: List[List[Event]],
    tempo_per_beat: List[Decimal],
    vars_per_beat: List[Tuple[str, List[str]]],
    pad_dev_default: str = "DEV_0",
) -> str:
    lines: List[str] = ["<BOS>"]

    def get_tempo(i: int) -> Decimal:
        if not tempo_per_beat:
            return Decimal("120")
        return tempo_per_beat[i] if i < len(tempo_per_beat) else tempo_per_beat[-1]

    def get_subd_and_vars(i: int) -> Tuple[str, List[str]]:
        if not vars_per_beat:
            return ("SUBD_1", [])
        if i < len(vars_per_beat):
            return vars_per_beat[i]
        return vars_per_beat[-1]

    for i, beat in enumerate(beats):
        tempo_tok = fmt_tempo_token(get_tempo(i))
        subd_tok, vtokens = get_subd_and_vars(i)

        lines.append(f"<SOB> {tempo_tok} {subd_tok}")
        for ev in beat:
            d = f"DELAY_{fmt_decimal(ev.delay)}"
            dev = ev.dev if ev.dev is not None else pad_dev_default
            lines.append(f"{d} {ev.hit} {dev}")

        # ✅ VARS is INSIDE the beat
        lines.append("<VARS> " + " ".join(vtokens) + " <VARE>")
        lines.append("<EOB>")

    lines.append("<EOS>")
    return "\n".join(lines)


def convert_file(infile: Path, outfile: Path) -> None:
    _, tempo_per_beat, stream_text = read_input_file(infile)
    tokens = tokenize_stream(stream_text)

    events, stop_i = parse_events_prefix(tokens)
    if not events:
        raise ValueError(f"{infile.name}: could not parse (DELAY HIT DEV) prefix.")

    trailing = tokens[stop_i:]
    vars_per_beat = parse_variations_stream(trailing)

    beats = split_into_beats(events)

    if vars_per_beat and len(vars_per_beat) < len(beats):
        print(
            f"[warn] {infile.name}: vars beats={len(vars_per_beat)} < skeleton beats={len(beats)}; "
            f"reusing last VARS for remaining beats."
        )
    if vars_per_beat and len(vars_per_beat) > len(beats):
        print(
            f"[warn] {infile.name}: vars beats={len(vars_per_beat)} > skeleton beats={len(beats)}; "
            f"extra VARS will be ignored."
        )

    out_text = beats_to_text_b(beats, tempo_per_beat, vars_per_beat)
    outfile.write_text(out_text, encoding="utf-8")


def convert_folder(
    infolder: Path = INPUT_FOLDER_PATH,
    outfolder: Path = OUTPUT_FOLDER_PATH,
) -> None:
    if not infolder.exists():
        raise FileNotFoundError(f"Input folder not found: {infolder.resolve()}")
    outfolder.mkdir(parents=True, exist_ok=True)

    files = sorted(
        [p for p in infolder.iterdir() if p.is_file()], key=lambda p: p.name.lower()
    )
    for p in files:
        if ALLOWED_SUFFIXES and p.suffix.lower() not in ALLOWED_SUFFIXES:
            continue
        out_path = outfolder / f"{p.stem}_converted{p.suffix}"
        convert_file(p, out_path)
        print(f"[ok] {p.name} -> {out_path}")


if __name__ == "__main__":
    convert_folder()
