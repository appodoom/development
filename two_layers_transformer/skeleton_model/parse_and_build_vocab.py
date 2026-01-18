from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
from defaults_paths import FILE_GLOB, DATA_DIR, OUT_DIR
from typing import Counter as TCounter

SPECIAL_TOKENS = ["<BOS>", "<EOS>", "<SOB>", "<EOB>", "<EOF>"]


def read_tokens(path: Path) -> List[str]:
    """Read a .derbake file and return a flat list of whitespace-separated tokens."""
    toks: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        toks.extend(line.split())
    return toks


def encode(
    tokens: List[str], token_to_id: Dict[str, int], *, where: str = ""
) -> List[int]:
    ids: List[int] = []
    for t in tokens:
        if t not in token_to_id:
            raise ValueError(
                f"Out-of-vocab token {t!r} {('in ' + where) if where else ''}"
            )
        ids.append(token_to_id[t])
    return ids


def decode(ids: List[int], id_to_token: List[str]) -> List[str]:
    return [id_to_token[i] for i in ids]


def build_vocab(
    token_lists: Iterable[List[str]],
) -> Tuple[Dict[str, int], List[str], Counter]:
    """Return token_to_id, id_to_token, and raw token counts."""
    counts: TCounter[str] = Counter()
    for toks in token_lists:
        counts.update(toks)

    id_to_token: List[str] = []
    seen = set()
    for s in SPECIAL_TOKENS:
        if s not in seen:
            id_to_token.append(s)
            seen.add(s)

    for tok, _ in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
        if tok in seen:
            continue
        id_to_token.append(tok)
        seen.add(tok)

    token_to_id = {tok: i for i, tok in enumerate(id_to_token)}
    return token_to_id, id_to_token, counts


def main():
    if not (DATA_DIR.exists() and DATA_DIR.is_dir()):
        raise SystemExit(
            f"DATA_DIR {DATA_DIR.resolve()} does not exist. "
            "Edit DATA_DIR at the top of this script (no argparse)."
        )

    paths = sorted(DATA_DIR.glob(FILE_GLOB))
    if not paths:
        raise SystemExit(f"No files matched {FILE_GLOB} in {DATA_DIR.resolve()}")

    print(f"Found {len(paths)} file(s).")
    all_token_lists: List[List[str]] = []

    discovered = defaultdict(set)

    for i, p in enumerate(paths):
        tokens = read_tokens(p)
        print(f"path {i}: {len(tokens)}")
        tokens.append("<EOF>")

        all_token_lists.append(tokens)

        for t in set(tokens):
            if t.startswith("TEMPO_"):
                discovered["tempo"].add(t)
            elif t.startswith("DELAY_"):
                discovered["delay"].add(t)
            elif t.startswith("HIT_"):
                discovered["hit"].add(t)
            elif t.startswith("DEV_"):
                discovered["dev"].add(t)
            elif t.startswith("SUBD_"):
                discovered["subd"].add(t)

    token_to_id, id_to_token, counts = build_vocab(all_token_lists)

    total_tokens = sum(len(seq) for seq in all_token_lists)
    unique_tokens = len(id_to_token)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    vocab_path = OUT_DIR / "vocab.json"
    stats_path = OUT_DIR / "stats.json"
    vocab_txt_path = OUT_DIR / "vocab.txt"

    vocab_obj = {
        "special_tokens": SPECIAL_TOKENS,
        "token_to_id": token_to_id,
        "id_to_token": id_to_token,
    }

    stats_obj = {
        "num_files": len(paths),
        "total_tokens": total_tokens,
        "vocab_size": unique_tokens,
        "top_tokens": counts.most_common(50),
        "discovered_groups": {k: sorted(list(v)) for k, v in discovered.items()},
    }

    vocab_path.write_text(json.dumps(vocab_obj, indent=2), encoding="utf-8")
    stats_path.write_text(json.dumps(stats_obj, indent=2), encoding="utf-8")
    vocab_txt_path.write_text("\n".join(id_to_token) + "\n", encoding="utf-8")

    print("\nSaved:")
    print(" -", vocab_path.resolve())
    print(" -", stats_path.resolve())
    print(" -", vocab_txt_path.resolve())

    print("\nQuick sanity:")
    print(f"  vocab_size={unique_tokens} total_tokens={total_tokens}")
    print("  first 20 vocab tokens:", id_to_token[:20])

    example = all_token_lists[0][:40]
    encoded = encode(example, token_to_id, where=str(paths[0]))
    decoded = decode(encoded, id_to_token)
    print("\nExample tokens:", example)
    print("Example ids   :", encoded)
    assert decoded == example


if __name__ == "__main__":
    main()
