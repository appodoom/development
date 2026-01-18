# predict_full_model.py
# User enters FULL prompt (can include <VARS> ... <VARE> and AMP_*)
# -> Model A gets the SAME prompt but with VARS removed
# -> Model A generates skeleton continuation
# -> Model B takes (original full prompt WITH VARS) + (A continuation)
#    and generates VARS only for beats that don't already contain VARS.

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from model_gpt import GPT, GPTConfig


# =========================
# USER VARIABLES (EDIT ME)
# =========================
SAVE_INPUT_TXT = Path("./input.txt")  # <-- add this

SKELETON_DIR = Path(r"./skeleton_model/model")
VARIATIONS_DIR = Path(r"./variations_model/model")

# Generation hyperparams
SKELETON_MAX_NEW_TOKENS = 500
SKELETON_TEMPERATURE = 0.7
SKELETON_TOP_K: Optional[int] = 80

VARIATIONS_TEMPERATURE = 0.9
VARIATIONS_TOP_K: Optional[int] = 50

# Save outputs
SAVE_SKELETON_TXT = Path("./generated_skeleton.txt")
SAVE_FINAL_TXT = Path("./generated_with_vars.txt")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# VOCAB / MODEL HELPERS
# =========================


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


def load_model(
    model_dir: Path, device: torch.device
) -> Tuple[GPT, Dict[str, int], List[str]]:
    token_to_id, id_to_token = load_vocab(model_dir)
    ckpt = torch.load(model_dir / "ckpt.pt", map_location=device)
    cfg = GPTConfig(**ckpt["cfg"])
    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, token_to_id, id_to_token


@torch.no_grad()
def sample_next_id(
    model: GPT,
    idx: torch.Tensor,
    *,
    temperature: float,
    top_k: Optional[int],
    allowed_ids: Optional[List[int]] = None,  # None => allow all
    forbid_ids: Optional[List[int]] = None,
) -> int:
    model.eval()
    device = idx.device

    idx_cond = (
        idx[:, -model.cfg.block_size :] if idx.size(1) > model.cfg.block_size else idx
    )
    logits = model(idx_cond)[:, -1, :]  # (1, V)
    logits = logits / max(1e-8, temperature)

    # forbid
    if forbid_ids:
        for tid in forbid_ids:
            if 0 <= tid < logits.size(-1):
                logits[:, tid] = float("-inf")

    # allowlist (optional)
    if allowed_ids is not None:
        allow = set(int(x) for x in allowed_ids)
        masked = torch.full_like(logits, float("-inf"))
        if allow:
            masked[:, list(allow)] = logits[:, list(allow)]
        logits = masked

    # top-k
    if top_k is not None and top_k > 0:
        v, _ = torch.topk(logits, k=min(top_k, logits.size(-1)))
        cutoff = v[:, -1].unsqueeze(-1)
        logits = torch.where(
            logits < cutoff, torch.tensor(float("-inf"), device=device), logits
        )

    probs = torch.softmax(logits, dim=-1)
    nxt = torch.multinomial(probs, num_samples=1).item()
    return int(nxt)


# =========================
# PROMPT TRANSFORMS
# =========================


def strip_vars_for_skeleton(full_tokens: List[str]) -> List[str]:
    """
    Remove <VARS> ... <VARE> blocks completely (including AMP_* tokens inside),
    so the remaining tokens can be fed into Model A (skeleton) safely.
    """
    out: List[str] = []
    i = 0
    while i < len(full_tokens):
        t = full_tokens[i]

        if t == "<VARS>":
            # skip until (and including) <VARE>
            i += 1
            while i < len(full_tokens) and full_tokens[i] != "<VARE>":
                i += 1
            if i < len(full_tokens) and full_tokens[i] == "<VARE>":
                i += 1
            continue

        # If someone pasted AMP_ outside by mistake, drop it too
        if t.startswith("AMP_") or t == "<VARE>":
            i += 1
            continue

        out.append(t)
        i += 1

    return out


def merge_full_prompt_with_A_continuation(
    full_prompt_tokens: List[str],
    a_out_tokens: List[str],
    a_prompt_tokens: List[str],
) -> List[str]:
    """
    Build the input sequence for Model B:
      (original full prompt WITH VARS) + (continuation from A output)
    Continuation = A_out minus the initial A_prompt prefix (if it matches).
    """
    cont: List[str]
    if (
        len(a_out_tokens) >= len(a_prompt_tokens)
        and a_out_tokens[: len(a_prompt_tokens)] == a_prompt_tokens
    ):
        cont = a_out_tokens[len(a_prompt_tokens) :]
    else:
        # fallback: if for some reason prefix doesn't match, just append full A output
        cont = a_out_tokens

    merged = list(full_prompt_tokens)

    # If user already ended their prompt with <EOS>, remove it to allow continuation after it
    if merged and merged[-1] == "<EOS>":
        merged.pop()

    merged.extend(cont)
    return merged


# =========================
# BEAT PARSING
# =========================


def split_beats(tokens: List[str]) -> Tuple[List[str], List[List[str]], List[str]]:
    """
    Expect:
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


def beat_has_vars(beat_tokens: List[str]) -> bool:
    return "<VARS>" in beat_tokens


# =========================
# STAGE 1: MODEL A (SKELETON)
# =========================


@torch.no_grad()
def generate_skeleton_tokens(
    sk_model: GPT,
    sk_token_to_id: Dict[str, int],
    sk_id_to_token: List[str],
    prompt_tokens: List[str],
    *,
    max_new_tokens: int,
    temperature: float,
    top_k: Optional[int],
) -> List[str]:
    device = next(sk_model.parameters()).device

    eos_id = sk_token_to_id.get("<EOS>")
    if eos_id is None:
        raise ValueError("Skeleton vocab must contain <EOS>")

    forbid: List[int] = []
    eof_id = sk_token_to_id.get("<EOF>")
    if eof_id is not None:
        forbid.append(int(eof_id))

    prompt_ids = encode_tokens(
        prompt_tokens, sk_token_to_id, where="skeleton prompt (vars removed)"
    )
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    out_ids = prompt_ids[:]
    hit_eos = False

    for _ in range(max_new_tokens):
        nxt = sample_next_id(
            sk_model,
            idx,
            temperature=temperature,
            top_k=top_k,
            allowed_ids=None,
            forbid_ids=forbid,
        )
        out_ids.append(nxt)
        idx = torch.cat(
            [idx, torch.tensor([[nxt]], dtype=torch.long, device=device)], dim=1
        )

        if nxt == int(eos_id):
            hit_eos = True
            break

    out_tokens = [sk_id_to_token[i] for i in out_ids]
    if not hit_eos:
        print("[warn] Model A did not hit <EOS> within max_new_tokens.")
    return out_tokens


# =========================
# STAGE 2: MODEL B (VARIATIONS)
# =========================


@torch.no_grad()
def add_variations_only_missing(
    var_model: GPT,
    var_token_to_id: Dict[str, int],
    var_id_to_token: List[str],
    tokens_for_B: List[str],
    *,
    temperature: float,
    top_k: Optional[int],
) -> List[str]:
    device = next(var_model.parameters()).device

    # Ensure Model B can read everything
    oov = sorted({t for t in tokens_for_B if t not in var_token_to_id})
    if oov:
        raise ValueError(
            "Some tokens are OOV for Model B vocab:\n"
            + "\n".join(oov[:80])
            + ("" if len(oov) <= 80 else f"\n... and {len(oov) - 80} more")
        )

    forbid_ids: List[int] = []
    eof_id = var_token_to_id.get("<EOF>")
    if eof_id is not None:
        forbid_ids.append(int(eof_id))

    hit_ids = [tid for tok, tid in var_token_to_id.items() if tok.startswith("HIT_")]
    amp_ids = [tid for tok, tid in var_token_to_id.items() if tok.startswith("AMP_")]

    vars_id = var_token_to_id["<VARS>"]
    vare_id = var_token_to_id["<VARE>"]

    prefix, beats, suffix = split_beats(tokens_for_B)

    out_tokens: List[str] = []
    out_tokens.extend(prefix)

    idx = torch.tensor(
        [encode_tokens(out_tokens, var_token_to_id, where="B prefix")],
        dtype=torch.long,
        device=device,
    )

    for beat in beats:
        if not beat or beat[0] != "<SOB>" or beat[-1] != "<EOB>":
            raise ValueError("Bad beat format. Expected <SOB> ... <EOB> per beat.")

        # If user already provided VARS in this beat, KEEP IT AS-IS (no changes).
        if beat_has_vars(beat):
            beat_ids = encode_tokens(
                beat, var_token_to_id, where="beat (already has vars)"
            )
            idx = torch.cat(
                [idx, torch.tensor([beat_ids], dtype=torch.long, device=device)], dim=1
            )
            out_tokens.extend(beat)
            continue

        # Otherwise, generate <VARS> ... <VARE> based on SUBD_k
        beat_body = beat[:-1]  # exclude <EOB>
        k = get_subd_k(beat_body)

        beat_body_ids = encode_tokens(
            beat_body, var_token_to_id, where="beat_body (no vars)"
        )
        idx = torch.cat(
            [idx, torch.tensor([beat_body_ids], dtype=torch.long, device=device)], dim=1
        )
        out_tokens.extend(beat_body)

        # insert <VARS>
        idx = torch.cat(
            [idx, torch.tensor([[vars_id]], dtype=torch.long, device=device)], dim=1
        )
        out_tokens.append("<VARS>")

        # generate exactly k (HIT, AMP) pairs
        for _ in range(k):
            hid = sample_next_id(
                var_model,
                idx,
                temperature=temperature,
                top_k=top_k,
                allowed_ids=hit_ids,
                forbid_ids=forbid_ids,
            )
            idx = torch.cat(
                [idx, torch.tensor([[hid]], dtype=torch.long, device=device)], dim=1
            )
            out_tokens.append(var_id_to_token[hid])

            aid = sample_next_id(
                var_model,
                idx,
                temperature=temperature,
                top_k=top_k,
                allowed_ids=amp_ids,
                forbid_ids=forbid_ids,
            )
            idx = torch.cat(
                [idx, torch.tensor([[aid]], dtype=torch.long, device=device)], dim=1
            )
            out_tokens.append(var_id_to_token[aid])

        # close VARS
        idx = torch.cat(
            [idx, torch.tensor([[vare_id]], dtype=torch.long, device=device)], dim=1
        )
        out_tokens.append("<VARE>")

        # close beat
        eob_id = var_token_to_id["<EOB>"]
        idx = torch.cat(
            [idx, torch.tensor([[eob_id]], dtype=torch.long, device=device)], dim=1
        )
        out_tokens.append("<EOB>")

    out_tokens.extend(suffix)
    return out_tokens


# =========================
# PIPELINE
# =========================


def run_pipeline(full_prompt_text: str) -> Tuple[List[str], List[str]]:
    # Load models
    sk_model, sk_t2i, sk_i2t = load_model(SKELETON_DIR, DEVICE)
    var_model, var_t2i, var_i2t = load_model(VARIATIONS_DIR, DEVICE)

    full_prompt_tokens = [t for t in full_prompt_text.split() if t]
    if not full_prompt_tokens:
        raise ValueError("Empty prompt.")

    # 1) Build skeleton prompt for Model A (strip VARS)
    a_prompt_tokens = strip_vars_for_skeleton(full_prompt_tokens)

    # 2) Run Model A (skeleton continuation)
    a_out_tokens = generate_skeleton_tokens(
        sk_model,
        sk_t2i,
        sk_i2t,
        a_prompt_tokens,
        max_new_tokens=SKELETON_MAX_NEW_TOKENS,
        temperature=SKELETON_TEMPERATURE,
        top_k=SKELETON_TOP_K,
    )

    # 3) Build Model B input = (original full prompt WITH VARS) + (A continuation)
    b_input_tokens = merge_full_prompt_with_A_continuation(
        full_prompt_tokens=full_prompt_tokens,
        a_out_tokens=a_out_tokens,
        a_prompt_tokens=a_prompt_tokens,
    )

    # 4) Run Model B: generate VARS only for beats missing them (keep provided VARS unchanged)
    final_tokens = add_variations_only_missing(
        var_model,
        var_t2i,
        var_i2t,
        tokens_for_B=b_input_tokens,
        temperature=VARIATIONS_TEMPERATURE,
        top_k=VARIATIONS_TOP_K,
    )

    return a_out_tokens, final_tokens


if __name__ == "__main__":
    user_prompt = input("Enter your model prompt: ").strip()

    # Save the exact input (no trailing newline)
    if user_prompt:
        SAVE_INPUT_TXT.write_text(user_prompt, encoding="utf-8")

    skeleton, final = run_pipeline(user_prompt)

    sk_text = " ".join(skeleton)
    final_text = " ".join(final)

    print("\n=== MODEL A OUTPUT (SKELETON) ===")
    print(sk_text)

    print("\n=== FINAL (PROMPT VARS PRESERVED + A CONTINUATION + B VARS GENERATED) ===")
    print(final_text)

    SAVE_SKELETON_TXT.write_text(sk_text + "\n", encoding="utf-8")
    SAVE_FINAL_TXT.write_text(final_text + "\n", encoding="utf-8")
    print(f"\nSaved:\n- {SAVE_INPUT_TXT}\n- {SAVE_SKELETON_TXT}\n- {SAVE_FINAL_TXT}")
