from __future__ import annotations
import re
from dataclasses import dataclass
from decimal import Decimal, getcontext
from pathlib import Path
from typing import List, Optional, Tuple

INPUT_FOLDER_PATH = Path("../fixed")
OUTPUT_FOLDER_PATH = Path("./fixed_converted")

ALLOWED_SUFFIXES = {".derbake"}

getcontext().prec = 28

DELAY_RE = re.compile(r"^DELAY_(-?\d+(?:\.\d+)?)$")
HIT_RE = re.compile(r"^HIT_[A-Z0-9_]+$")
DEV_RE = re.compile(r"^DEV_-?\d+(?:\.\d+)?$")

# NEW:
SUBD_RE = re.compile(r"^SUBD_(\d+)$")
AMP_RE = re.compile(r"^AMP_-?\d+(?:\.\d+)?$")  # in your raw files it seems AMP_* exists


@dataclass
class Event:
    delay: Decimal
    hit: str
    dev: Optional[str] = None


def parse_delay(tok: str) -> Optional[Decimal]:
    m = DELAY_RE.match(tok)
    if not m:
        return None
    return Decimal(m.group(1))


def fmt_decimal(x: Decimal) -> str:
    x = x.normalize()
    s = format(x, "f")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s if s else "0"


def fmt_tempo_token(bpm: Decimal) -> str:
    return f"TEMPO_{fmt_decimal(bpm)}"


def tokenize_stream(text: str) -> List[str]:
    return text.strip().split()


def parse_events_prefix(tokens: List[str]) -> Tuple[List[Event], int]:
    """
    Parse only the initial prefix that matches:
      (DELAY_x HIT_y DEV_z?)+
    Returns (events, stop_index).
    """
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


# NEW:
def parse_subd_stream(tokens: List[str]) -> List[str]:
    """
    Parses a stream like:
      SUBD_4 HIT_X AMP_Y HIT_X AMP_Y HIT_X AMP_Y HIT_X AMP_Y SUBD_2 ...
    Returns ["SUBD_4", "SUBD_4", "SUBD_2", ...]
    We ignore the HIT/AMP content; we only need SUBD per beat for Model A.
    """
    # find first SUBD token
    start = None
    for j, t in enumerate(tokens):
        if SUBD_RE.match(t):
            start = j
            break
    if start is None:
        return []

    out: List[str] = []
    i = start
    n = len(tokens)

    while i < n:
        m = SUBD_RE.match(tokens[i])
        if not m:
            break
        k = int(m.group(1))
        out.append(tokens[i])
        i += 1

        # consume k "slots" of (HIT_*, AMP_* or DEV_*) if present
        for _ in range(k):
            if i < n and HIT_RE.match(tokens[i]):
                i += 1
            else:
                # pattern broken; stop safely
                return out
            if i < n and (AMP_RE.match(tokens[i]) or DEV_RE.match(tokens[i])):
                i += 1
            # if no AMP/DEV, still continue (be permissive)

    return out


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


def read_input_file(path: Path) -> Tuple[Decimal, List[Decimal], str]:
    text = path.read_text(encoding="utf-8")
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]

    if len(lines) < 3:
        raise ValueError(
            f"{path.name}: Input must have at least 3 non-empty lines:\n"
            "1) initial tempo\n2) tempo per beat list\n3+) event stream"
        )

    initial_tempo = Decimal(lines[0])
    tempo_list = [Decimal(x) for x in lines[1].split()]
    stream_text = " ".join(
        lines[2:]
    )  # includes delay stream + any trailing streams (SUBD...)

    return initial_tempo, tempo_list, stream_text


def beats_to_text(
    beats: List[List[Event]],
    tempo_per_beat: List[Decimal],
    subd_per_beat: List[str],  # NEW
    pad_dev_default: str = "DEV_0",
    pretty: bool = True,
    include_bos_eos: bool = True,
    include_eob: bool = True,
) -> str:
    lines: List[str] = []
    if include_bos_eos:
        lines.append("<BOS>")

    n_tempos = len(tempo_per_beat)
    n_subd = len(subd_per_beat)

    def get_tempo(i: int) -> Decimal:
        if n_tempos == 0:
            return Decimal("120")
        if i < n_tempos:
            return tempo_per_beat[i]
        return tempo_per_beat[-1]

    def get_subd(i: int) -> str:
        if n_subd == 0:
            return "SUBD_1"
        if i < n_subd:
            return subd_per_beat[i]
        return subd_per_beat[-1]

    for i, beat in enumerate(beats):
        tempo_tok = fmt_tempo_token(get_tempo(i))
        subd_tok = get_subd(i)
        header = f"<SOB> {tempo_tok} {subd_tok}"  # NEW
        lines.append(header)

        if pretty:
            for ev in beat:
                d = f"DELAY_{fmt_decimal(ev.delay)}"
                dev = ev.dev if ev.dev is not None else pad_dev_default
                lines.append(f"{d} {ev.hit} {dev}")
            if include_eob:
                lines.append("<EOB>")
        else:
            toks = header.split()
            for ev in beat:
                toks.append(f"DELAY_{fmt_decimal(ev.delay)}")
                toks.append(ev.hit)
                toks.append(ev.dev if ev.dev is not None else pad_dev_default)
            if include_eob:
                toks.append("<EOB>")
            lines.append(" ".join(toks))

    if include_bos_eos:
        lines.append("<EOS>")

    return "\n".join(lines)


def convert_file(infile: Path, outfile: Path, pretty: bool = True) -> None:
    _, tempo_per_beat, stream_text = read_input_file(infile)

    tokens = tokenize_stream(stream_text)
    events, stop_i = parse_events_prefix(tokens)

    if not events:
        raise ValueError(
            f"{infile.name}: Could not parse any (DELAY HIT DEV) events from the stream."
        )

    trailing = tokens[stop_i:]
    subd_per_beat = parse_subd_stream(trailing)  # NEW

    beats = split_into_beats(events)

    out_text = beats_to_text(
        beats=beats,
        tempo_per_beat=tempo_per_beat,
        subd_per_beat=subd_per_beat,  # NEW
        pretty=pretty,
        include_bos_eos=True,
    )

    if len(tempo_per_beat) < len(beats):
        print(
            f"[warn] {infile.name}: tempo_per_beat has {len(tempo_per_beat)} entries "
            f"but produced {len(beats)} beats. Reusing last tempo for remaining beats."
        )

    if subd_per_beat and len(subd_per_beat) < len(beats):
        print(
            f"[warn] {infile.name}: subd_per_beat has {len(subd_per_beat)} entries "
            f"but produced {len(beats)} beats. Reusing last SUBD for remaining beats."
        )

    if not subd_per_beat:
        print(
            f"[warn] {infile.name}: No SUBD stream found; using SUBD_1 for all beats."
        )

    outfile.write_text(out_text, encoding="utf-8")


def convert_folder(
    infolder: Path = INPUT_FOLDER_PATH,
    outfolder: Path = OUTPUT_FOLDER_PATH,
    pretty: bool = True,
) -> None:
    if not infolder.exists():
        raise FileNotFoundError(f"Input folder not found: {infolder.resolve()}")

    outfolder.mkdir(parents=True, exist_ok=True)

    files = [p for p in infolder.iterdir() if p.is_file()]
    files.sort(key=lambda p: p.name.lower())

    for p in files:
        if ALLOWED_SUFFIXES is not None and p.suffix.lower() not in ALLOWED_SUFFIXES:
            continue

        out_path = outfolder / f"{p.stem}_converted{p.suffix}"
        convert_file(p, out_path, pretty=pretty)
        print(f"[ok] {p.name} -> {out_path}")


def main() -> None:
    convert_folder(pretty=True)


if __name__ == "__main__":
    main()
