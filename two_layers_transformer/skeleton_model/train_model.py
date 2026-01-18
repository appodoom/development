from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from lm_dataset import make_loaders
from model_gpt import GPT, GPTConfig


def load_vocab(out_dir: Path) -> Tuple[Dict[str, int], List[str]]:
    obj = json.loads((out_dir / "vocab.json").read_text(encoding="utf-8"))
    token_to_id: Dict[str, int] = {k: int(v) for k, v in obj["token_to_id"].items()}
    id_to_token: List[str] = list(obj["id_to_token"])
    return token_to_id, id_to_token


def compute_loss(
    logits: torch.Tensor, y: torch.Tensor, eof_id: int | None
) -> torch.Tensor:
    """
    Cross-entropy loss, but if eof_id is provided, we IGNORE positions where target == <EOF>.
    This makes <EOF> act like a pure separator: present in the stream but not a learnable output.
    """
    if eof_id is None:
        return F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

    return F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        y.view(-1),
        ignore_index=int(eof_id),
    )


@torch.no_grad()
def estimate_loss(
    model: GPT,
    val_loader,
    device: torch.device,
    eof_id: int | None,
    max_batches: int = 50,
) -> float:
    model.eval()
    losses: List[float] = []
    for i, (x, y) in enumerate(val_loader):
        if i >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)  # (B, T, V)
        loss = compute_loss(logits, y, eof_id)
        losses.append(float(loss.item()))
    model.train()
    return float(sum(losses) / max(1, len(losses)))


def get_default_out_dir() -> Path:
    # Prefer your defaults_paths.OUT_DIR if available
    try:
        from defaults_paths import OUT_DIR as D_OUT  # type: ignore

        return Path(D_OUT)
    except Exception:
        return Path("./out")


def save_checkpoint(
    out_dir: Path,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    step: int,
    cfg: GPTConfig,
    best_val: float,
) -> None:
    ckpt = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "step": step,
        "best_val": best_val,
        "cfg": asdict(cfg),
    }
    tmp = out_dir / "ckpt_tmp.pt"
    final = out_dir / "ckpt.pt"
    torch.save(ckpt, tmp)
    tmp.replace(final)


def main() -> None:
    out_dir = get_default_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    token_to_id, _ = load_vocab(out_dir)
    eof_id = token_to_id.get("<EOF>", None)
    vocab_size = len(token_to_id)

    # ---- training hyperparams (edit freely) ----
    block_size = 256
    batch_size = 32
    max_steps = 1000
    eval_interval = 500
    log_interval = 50

    lr = 3e-4
    weight_decay = 0.1
    grad_clip = 1.0

    # model size (2 layers by default)
    cfg = GPTConfig(
        vocab_size=vocab_size,
        block_size=block_size,
        n_layer=2,
        n_head=4,
        n_embd=256,
        dropout=0.1,
    )

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin = device.type == "cuda"

    # Windows-safe: num_workers=0
    train_loader, val_loader = make_loaders(
        out_dir,
        block_size=block_size,
        batch_size=batch_size,
        stride=block_size,
        num_workers=0,
        pin_memory=pin,
    )

    model = GPT(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # optional AMP on CUDA
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # resume if checkpoint exists
    ckpt_path = out_dir / "ckpt.pt"
    step = 0
    best_val = float("inf")
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        step = int(ckpt.get("step", 0))
        best_val = float(ckpt.get("best_val", float("inf")))
        print(f"Resumed from {ckpt_path} at step={step}, best_val={best_val:.4f}")

    print(f"Device: {device} | vocab_size={vocab_size} | block_size={block_size}")
    if eof_id is not None:
        print(f"Loss masking enabled: ignoring <EOF> as target (eof_id={eof_id})")
    else:
        print("Warning: <EOF> not found in vocab; no masking applied.")

    print("Training...")

    t0 = time.time()
    train_iter = iter(train_loader)

    while step < max_steps:
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(x)
            loss = compute_loss(logits, y, eof_id)

        scaler.scale(loss).backward()
        if grad_clip is not None:
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()

        step += 1

        if step % log_interval == 0:
            dt = time.time() - t0
            tok_per_step = batch_size * block_size
            tps = (tok_per_step * log_interval) / max(1e-9, dt)
            print(f"step {step:6d} | loss {loss.item():.4f} | tokens/s ~{tps:,.0f}")
            t0 = time.time()

        if step % eval_interval == 0:
            val_loss = estimate_loss(
                model, val_loader, device=device, eof_id=eof_id, max_batches=50
            )
            print(
                f"[eval] step {step:6d} | val_loss {val_loss:.4f} (best {best_val:.4f})"
            )

            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(out_dir, model, optimizer, step, cfg, best_val)
                print(f"Saved checkpoint: {out_dir / 'ckpt.pt'}")

    # final save
    save_checkpoint(out_dir, model, optimizer, step, cfg, best_val)
    print("Done.")


if __name__ == "__main__":
    main()
