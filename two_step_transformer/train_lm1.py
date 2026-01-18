# 02_train_lm1.py
import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Config, GPT2LMHeadModel
from tqdm import tqdm

from token_utils import load_json, read_jsonl, encode

# ====== GLOBAL CONFIG ======
OUT_DIR = "out_method3"
VOCAB_PATH = f"{OUT_DIR}/vocab.json"

LM1_TRAIN_JSONL = f"{OUT_DIR}/lm1_train.jsonl"
LM1_VAL_JSONL   = f"{OUT_DIR}/lm1_val.jsonl"

LM1_MODEL_DIR = f"{OUT_DIR}/lm1_model"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_LEN = 512
BATCH_SIZE = 8
EPOCHS = 5
LR = 3e-4

N_LAYER = 6
N_HEAD  = 8
N_EMBD  = 512
# ===========================

class LmDataset(Dataset):
    def __init__(self, jsonl_path, vocab, max_len):
        self.rows = read_jsonl(jsonl_path)
        self.vocab = vocab
        self.max_len = max_len
        self.pad_id = vocab["PAD"]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        toks = self.rows[idx]["tokens"]
        ids = encode(toks, self.vocab)[:self.max_len]
        attn = [1] * len(ids)

        if len(ids) < self.max_len:
            pad_n = self.max_len - len(ids)
            ids += [self.pad_id] * pad_n
            attn += [0] * pad_n

        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }

def main():
    os.makedirs(LM1_MODEL_DIR, exist_ok=True)

    vocab = load_json(VOCAB_PATH)
    pad_id = vocab["PAD"]
    bos_id = vocab["BOS"]
    eos_id = vocab["EOS"]

    train_ds = LmDataset(LM1_TRAIN_JSONL, vocab, MAX_LEN)
    val_ds   = LmDataset(LM1_VAL_JSONL, vocab, MAX_LEN)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    cfg = GPT2Config(
        vocab_size=len(vocab),
        n_layer=N_LAYER,
        n_head=N_HEAD,
        n_embd=N_EMBD,
        bos_token_id=bos_id,
        eos_token_id=eos_id,
    )
    model = GPT2LMHeadModel(cfg).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)

    best_val = 1e9

    for ep in range(EPOCHS):
        model.train()
        pbar = tqdm(train_dl, desc=f"LM1 train ep {ep+1}/{EPOCHS}")

        for batch in pbar:
            input_ids = batch["input_ids"].to(DEVICE)
            attn = batch["attention_mask"].to(DEVICE)

            labels = input_ids.clone()
            labels[input_ids == pad_id] = -100

            loss = model(input_ids=input_ids, attention_mask=attn, labels=labels).loss
            opt.zero_grad()
            loss.backward()
            opt.step()

            pbar.set_postfix(loss=float(loss.detach().cpu()))

        model.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for batch in val_dl:
                input_ids = batch["input_ids"].to(DEVICE)
                attn = batch["attention_mask"].to(DEVICE)
                labels = input_ids.clone()
                labels[input_ids == pad_id] = -100
                loss = model(input_ids=input_ids, attention_mask=attn, labels=labels).loss
                tot += float(loss.detach().cpu())
                n += 1

        val_loss = tot / max(1, n)
        print("LM1 val_loss:", val_loss)

        if val_loss < best_val:
            best_val = val_loss
            model.save_pretrained(LM1_MODEL_DIR)
            print("saved best LM1 ✅", LM1_MODEL_DIR)

if __name__ == "__main__":
    main()
