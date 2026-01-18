from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
from typing import Counter as TCounter

from defaults_paths import FILE_GLOB, OUTPUT_FOLDER_PATH, OUT_DIR

SPECIAL_TOKENS = ["<BOS>", "<EOS>", "<SOB>", "<EOB>", "<VARS>", "<VARE>", "<EOF>"]


def read_tokens(path: Path) -> List[str]:
    toks: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        toks.extend(line.split())
    return toks


def build_vocab(
    token_lists: Iterable[List[str]],
) -> Tuple[Dict[str, int], List[str], TCounter[str]]:
    counts: TCounter[str] = Counter()
    for toks in token_lists:
        counts.update(toks)

    id_to_token: List[str] = []
    seen = set()

    # special tokens first (stable ids)
    for s in SPECIAL_TOKENS:
        if s not in seen:
            id_to_token.append(s)
            seen.add(s)

    # then all other tokens by frequency
    for tok, _ in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
        if tok in seen:
            continue
        id_to_token.append(tok)
        seen.add(tok)

    token_to_id = {tok: i for i, tok in enumerate(id_to_token)}
    return token_to_id, id_to_token, counts


def main() -> None:
    if not (OUTPUT_FOLDER_PATH.exists() and OUTPUT_FOLDER_PATH.is_dir()):
        raise SystemExit(
            f"OUTPUT_FOLDER_PATH {OUTPUT_FOLDER_PATH.resolve()} does not exist."
        )

    paths = sorted(OUTPUT_FOLDER_PATH.glob(FILE_GLOB))
    if not paths:
        raise SystemExit(
            f"No files matched {FILE_GLOB} in {OUTPUT_FOLDER_PATH.resolve()}"
        )

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
            elif t.startswith("AMP_"):
                discovered["amp"].add(t)

    token_to_id, id_to_token, counts = build_vocab(all_token_lists)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    vocab_obj = {
        "special_tokens": SPECIAL_TOKENS,
        "token_to_id": token_to_id,
        "id_to_token": id_to_token,
    }

    stats_obj = {
        "num_files": len(paths),
        "total_tokens": sum(len(seq) for seq in all_token_lists),
        "vocab_size": len(id_to_token),
        "top_tokens": counts.most_common(50),
        "discovered_groups": {k: sorted(list(v)) for k, v in discovered.items()},
    }

    (OUT_DIR / "vocab.json").write_text(
        json.dumps(vocab_obj, indent=2), encoding="utf-8"
    )
    (OUT_DIR / "stats.json").write_text(
        json.dumps(stats_obj, indent=2), encoding="utf-8"
    )
    (OUT_DIR / "vocab.txt").write_text("\n".join(id_to_token) + "\n", encoding="utf-8")

    print("\nSaved:")
    print(" -", (OUT_DIR / "vocab.json").resolve())
    print(" -", (OUT_DIR / "stats.json").resolve())
    print(" -", (OUT_DIR / "vocab.txt").resolve())


if __name__ == "__main__":
    main()
