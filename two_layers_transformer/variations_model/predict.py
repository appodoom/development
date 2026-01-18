from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from model_gpt import GPT, GPTConfig
from defaults_paths import OUT_DIR


def load_vocab(out_dir: Path) -> Tuple[Dict[str, int], List[str]]:
    obj = json.loads((out_dir / "vocab.json").read_text(encoding="utf-8"))
    token_to_id: Dict[str, int] = {k: int(v) for k, v in obj["token_to_id"].items()}
    id_to_token: List[str] = list(obj["id_to_token"])
    return token_to_id, id_to_token


def encode_tokens(
    tokens: List[str], token_to_id: Dict[str, int], *, where: str = ""
) -> List[int]:
    ids: List[int] = []
    for t in tokens:
        if t not in token_to_id:
            raise ValueError(f"OOV token {t!r}" + (f" in {where}" if where else ""))
        ids.append(token_to_id[t])
    return ids


def split_beats(tokens: List[str]) -> Tuple[List[str], List[List[str]], List[str]]:
    """
    Expect Model A output:
      <BOS> (<SOB> ... <EOB>)+ <EOS>
    Returns: prefix, beats (each includes <SOB>.. <EOB>), suffix (includes <EOS> ...)
    """
    i = 0
    while i < len(tokens) and tokens[i] != "<SOB>":
        i += 1
    prefix = tokens[:i]

    beats: List[List[str]] = []
    while i < len(tokens):
        if tokens[i] == "<EOS>":
            break
        if tokens[i] != "<SOB>":
            i += 1
            continue
        j = i
        while j < len(tokens) and tokens[j] != "<EOB>":
            j += 1
        if j >= len(tokens):
            break
        beats.append(tokens[i : j + 1])
        i = j + 1

    suffix = tokens[i:]
    return prefix, beats, suffix


def get_subd_k(beat_tokens: List[str]) -> int:
    for t in beat_tokens:
        if t.startswith("SUBD_"):
            try:
                return int(t.split("_", 1)[1])
            except Exception:
                return 1
    return 1


@torch.no_grad()
def sample_next_id(
    model: GPT,
    idx: torch.Tensor,
    *,
    temperature: float,
    top_k: Optional[int],
    allowed_ids: List[int],
    forbid_ids: List[int],
) -> int:
    model.eval()
    device = idx.device

    idx_cond = (
        idx[:, -model.cfg.block_size :] if idx.size(1) > model.cfg.block_size else idx
    )
    logits = model(idx_cond)[:, -1, :]  # (1, V)
    logits = logits / max(1e-8, temperature)

    # forbid
    for tid in forbid_ids:
        if 0 <= tid < logits.size(-1):
            logits[:, tid] = float("-inf")

    # allowlist mask
    allow = set(int(x) for x in allowed_ids)
    mask = torch.full_like(logits, float("-inf"))
    mask[:, list(allow)] = logits[:, list(allow)]
    logits = mask

    # top-k within allowed
    if top_k is not None and top_k > 0:
        v, _ = torch.topk(logits, k=min(top_k, logits.size(-1)))
        cutoff = v[:, -1].unsqueeze(-1)
        logits = torch.where(
            logits < cutoff, torch.tensor(float("-inf"), device=device), logits
        )

    probs = torch.softmax(logits, dim=-1)
    nxt = torch.multinomial(probs, num_samples=1).item()
    return int(nxt)


def main() -> None:
    out_dir = Path(OUT_DIR)
    token_to_id, id_to_token = load_vocab(out_dir)

    ckpt = torch.load(out_dir / "ckpt.pt", map_location="cpu")
    cfg = GPTConfig(**ckpt["cfg"])
    model = GPT(cfg)
    model.load_state_dict(ckpt["model_state"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # forbid sampling <EOF>
    forbid_ids: List[int] = []
    eof_id = token_to_id.get("<EOF>")
    if eof_id is not None:
        forbid_ids.append(int(eof_id))

    # allowlists
    hit_ids = [tid for tok, tid in token_to_id.items() if tok.startswith("HIT_")]
    amp_ids = [tid for tok, tid in token_to_id.items() if tok.startswith("AMP_")]

    vars_id = token_to_id["<VARS>"]
    vare_id = token_to_id["<VARE>"]

    # ---- paste Model A output here (no VARS) ----
    skeleton = (
        "<BOS> <SOB> TEMPO_120 SUBD_4 DELAY_1 HIT_S DEV_0 <EOB> "
        "<SOB> TEMPO_120 SUBD_4 DELAY_1 HIT_D DEV_0 <EOB> "
        "<SOB> TEMPO_120 SUBD_4 DELAY_1 HIT_OTI DEV_0 <EOB> "
        "<SOB> TEMPO_120 SUBD_4 DELAY_1 HIT_S DEV_0 <EOB> "
        "<SOB> TEMPO_120 SUBD_4 DELAY_1 HIT_D DEV_0 <EOB> "
        "<SOB> TEMPO_120 SUBD_2 DELAY_0.25 HIT_S DEV_0 DELAY_0.75 HIT_D DEV_0 <EOB> "
        "<EOS>"
    )

    tokens = [t for t in skeleton.split() if t]
    prefix, beats, suffix = split_beats(tokens)

    out_tokens: List[str] = []
    out_tokens.extend(prefix)

    # running context ids
    idx = torch.tensor(
        [encode_tokens(out_tokens, token_to_id, where="prefix")],
        dtype=torch.long,
        device=device,
    )

    for beat in beats:
        if not beat or beat[0] != "<SOB>" or beat[-1] != "<EOB>":
            raise ValueError("Bad beat format. Expected <SOB> ... <EOB> per beat.")

        beat_body = beat[:-1]  # exclude <EOB>
        k = get_subd_k(beat_body)

        # append beat body
        beat_body_ids = encode_tokens(beat_body, token_to_id, where="beat_body")
        idx = torch.cat(
            [idx, torch.tensor([beat_body_ids], dtype=torch.long, device=device)], dim=1
        )
        out_tokens.extend(beat_body)

        # insert <VARS>
        idx = torch.cat(
            [idx, torch.tensor([[vars_id]], dtype=torch.long, device=device)], dim=1
        )
        out_tokens.append("<VARS>")

        # generate exactly k pairs: HIT then AMP
        for _ in range(k):
            hid = sample_next_id(
                model,
                idx,
                temperature=0.9,
                top_k=50,
                allowed_ids=hit_ids,
                forbid_ids=forbid_ids,
            )
            idx = torch.cat(
                [idx, torch.tensor([[hid]], dtype=torch.long, device=device)], dim=1
            )
            out_tokens.append(id_to_token[hid])

            aid = sample_next_id(
                model,
                idx,
                temperature=0.9,
                top_k=50,
                allowed_ids=amp_ids,
                forbid_ids=forbid_ids,
            )
            idx = torch.cat(
                [idx, torch.tensor([[aid]], dtype=torch.long, device=device)], dim=1
            )
            out_tokens.append(id_to_token[aid])

        # close VARS + beat end (forced)
        idx = torch.cat(
            [idx, torch.tensor([[vare_id]], dtype=torch.long, device=device)], dim=1
        )
        out_tokens.append("<VARE>")

        out_tokens.append("<EOB>")
        eob_id = token_to_id["<EOB>"]
        idx = torch.cat(
            [idx, torch.tensor([[eob_id]], dtype=torch.long, device=device)], dim=1
        )

    out_tokens.extend(suffix)

    print("=== MODEL B OUTPUT (skeleton + filled variations) ===")
    print(" ".join(out_tokens))


if __name__ == "__main__":
    main()
