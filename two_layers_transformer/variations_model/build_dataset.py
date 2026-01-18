from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from defaults_paths import FILE_GLOB, OUTPUT_FOLDER_PATH, OUT_DIR


def read_tokens(path: Path) -> List[str]:
    toks: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        toks.extend(line.split())
    return toks


def load_vocab(vocab_path: Path) -> Tuple[Dict[str, int], List[str]]:
    obj = json.loads(vocab_path.read_text(encoding="utf-8"))
    token_to_id: Dict[str, int] = {k: int(v) for k, v in obj["token_to_id"].items()}
    id_to_token: List[str] = list(obj["id_to_token"])
    return token_to_id, id_to_token


def encode_tokens(
    tokens: List[str], token_to_id: Dict[str, int], *, where: str
) -> List[int]:
    ids: List[int] = []
    for t in tokens:
        if t not in token_to_id:
            raise ValueError(f"Out-of-vocab token {t!r} in {where}")
        ids.append(token_to_id[t])
    return ids


@dataclass(frozen=True)
class SplitConfig:
    val_ratio: float = 0.1
    seed: int = 1337
    add_bos_per_file: bool = False  # IMPORTANT: converter already writes <BOS>
    ensure_eof: bool = True


def pick_dtype(vocab_size: int) -> np.dtype:
    if vocab_size - 1 <= np.iinfo(np.uint16).max:
        return np.dtype(np.uint16)
    if vocab_size - 1 <= np.iinfo(np.int32).max:
        return np.dtype(np.int32)
    return np.dtype(np.int64)


def build_stream(
    paths: List[Path], token_to_id: Dict[str, int], cfg: SplitConfig
) -> np.ndarray:
    all_ids: List[int] = []

    for p in paths:
        toks = read_tokens(p)

        if cfg.add_bos_per_file:
            toks = ["<BOS>"] + toks

        if cfg.ensure_eof and (not toks or toks[-1] != "<EOF>"):
            toks = toks + ["<EOF>"]

        ids = encode_tokens(toks, token_to_id, where=str(p))
        all_ids.extend(ids)

    dtype = pick_dtype(len(token_to_id))
    return np.asarray(all_ids, dtype=dtype)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    vocab_path = OUT_DIR / "vocab.json"
    if not vocab_path.exists():
        raise SystemExit(
            f"Missing {vocab_path.resolve()} — run parse_and_build_vocab_b.py first."
        )

    token_to_id, _ = load_vocab(vocab_path)

    if "<BOS>" not in token_to_id or "<EOF>" not in token_to_id:
        raise SystemExit("vocab.json must contain <BOS> and <EOF>.")

    if not (OUTPUT_FOLDER_PATH.exists() and OUTPUT_FOLDER_PATH.is_dir()):
        raise SystemExit(
            f"OUTPUT_FOLDER_PATH {OUTPUT_FOLDER_PATH.resolve()} does not exist."
        )

    paths = sorted(OUTPUT_FOLDER_PATH.glob(FILE_GLOB))
    if not paths:
        raise SystemExit(
            f"No files matched {FILE_GLOB} in {OUTPUT_FOLDER_PATH.resolve()}"
        )

    cfg = SplitConfig()
    rng = random.Random(cfg.seed)
    rng.shuffle(paths)

    n_val = max(1, int(len(paths) * cfg.val_ratio))
    val_paths = paths[:n_val]
    train_paths = paths[n_val:]

    print(f"Files: total={len(paths)} train={len(train_paths)} val={len(val_paths)}")

    train_ids = build_stream(train_paths, token_to_id, cfg)
    val_ids = build_stream(val_paths, token_to_id, cfg)

    (OUT_DIR / "train.bin").write_bytes(train_ids.tobytes())
    (OUT_DIR / "val.bin").write_bytes(val_ids.tobytes())

    meta = {
        "dtype": str(train_ids.dtype),
        "train_tokens": int(train_ids.size),
        "val_tokens": int(val_ids.size),
        "val_ratio": cfg.val_ratio,
        "seed": cfg.seed,
        "add_bos_per_file": cfg.add_bos_per_file,
        "ensure_eof": cfg.ensure_eof,
    }
    (OUT_DIR / "dataset_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    print("Saved:")
    print(" -", (OUT_DIR / "train.bin").resolve())
    print(" -", (OUT_DIR / "val.bin").resolve())
    print(" -", (OUT_DIR / "dataset_meta.json").resolve())
    print("Meta:", meta)


if __name__ == "__main__":
    main()
