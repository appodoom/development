from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch

from model_gpt import GPT, GPTConfig


def get_default_out_dir() -> Path:
    try:
        from defaults_paths import OUT_DIR as D_OUT

        return Path(D_OUT)
    except Exception:
        return Path("./out")


def load_vocab(out_dir: Path) -> Tuple[Dict[str, int], List[str]]:
    obj = json.loads((out_dir / "vocab.json").read_text(encoding="utf-8"))
    token_to_id: Dict[str, int] = {k: int(v) for k, v in obj["token_to_id"].items()}
    id_to_token: List[str] = list(obj["id_to_token"])
    return token_to_id, id_to_token


def encode_prompt(prompt: str, token_to_id: Dict[str, int]) -> List[int]:
    # prompt is whitespace-separated tokens
    toks = [t for t in prompt.strip().split() if t]
    ids: List[int] = []
    for t in toks:
        if t not in token_to_id:
            raise ValueError(f"Prompt has OOV token: {t!r}")
        ids.append(token_to_id[t])
    return ids


@torch.no_grad()
def generate(
    model: GPT,
    idx: torch.Tensor,
    *,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: Optional[int] = 50,
    stop_id: Optional[int] = None,
    forbid_ids: Optional[List[int]] = None,
) -> torch.Tensor:
    model.eval()
    device = idx.device

    forbid = set(int(x) for x in (forbid_ids or []))

    for _ in range(max_new_tokens):
        # crop context if too long
        if idx.size(1) > model.cfg.block_size:
            idx_cond = idx[:, -model.cfg.block_size :]
        else:
            idx_cond = idx

        logits = model(idx_cond)  # (B, T, V)
        logits = logits[:, -1, :]  # (B, V)

        logits = logits / max(1e-8, temperature)

        # HARD BAN certain tokens (e.g., <EOF>) so they can never be sampled
        if forbid:
            for tid in forbid:
                if 0 <= tid < logits.size(-1):
                    logits[:, tid] = float("-inf")

        if top_k is not None and top_k > 0:
            v, _ = torch.topk(logits, k=min(top_k, logits.size(-1)))
            cutoff = v[:, -1].unsqueeze(-1)
            logits = torch.where(
                logits < cutoff, torch.tensor(float("-inf"), device=device), logits
            )

        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)  # (B, 1)

        idx = torch.cat([idx, next_id], dim=1)

        if stop_id is not None and int(next_id.item()) == int(stop_id):
            break

    return idx


def main() -> None:
    out_dir = get_default_out_dir()

    token_to_id, id_to_token = load_vocab(out_dir)

    ckpt = torch.load(out_dir / "ckpt.pt", map_location="cpu")
    cfg = GPTConfig(**ckpt["cfg"])
    model = GPT(cfg)
    model.load_state_dict(ckpt["model_state"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    eof_id = token_to_id.get("<EOF>", None)
    eos_id = token_to_id.get("<EOS>", None)

    prompt = "<BOS> <SOB> TEMPO_120 SUBD_4 DELAY_1 HIT_S DEV_0 <EOB> <SOB> TEMPO_120 SUBD_4 DELAY_1 HIT_D DEV_0 <EOB> <SOB> TEMPO_120 SUBD_4 DELAY_1 HIT_OTI DEV_0 <EOB> <SOB> TEMPO_120 SUBD_4 DELAY_1 HIT_S DEV_0 <EOB> <SOB> TEMPO_120 SUBD_4 DELAY_1 HIT_D DEV_0 <EOB> <SOB> TEMPO_120 SUBD_2 DELAY_0.25 HIT_S DEV_0 DELAY_0.75 HIT_D DEV_0 <EOB>"

    prompt_ids = encode_prompt(prompt, token_to_id)
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    forbid_ids: List[int] = []
    if eof_id is not None:
        forbid_ids.append(int(eof_id))
    out = generate(
        model,
        idx,
        max_new_tokens=400,
        temperature=0.9,
        top_k=50,
        stop_id=eos_id,
        forbid_ids=forbid_ids,
    )

    out_ids = out[0].tolist()
    out_toks = [id_to_token[i] for i in out_ids]

    print("=== GENERATED TOKENS ===")
    print(" ".join(out_toks))


if __name__ == "__main__":
    main()
