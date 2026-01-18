# 04_train_lm2.py
import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BartConfig, BartForConditionalGeneration
from tqdm import tqdm

from token_utils import load_json, read_jsonl, encode

# ====== GLOBAL CONFIG ======
OUT_DIR = "out_method3"
VOCAB_PATH = f"{OUT_DIR}/vocab.json"

LM2_TRAIN_JSONL = f"{OUT_DIR}/lm2_train.jsonl"
LM2_VAL_JSONL   = f"{OUT_DIR}/lm2_val.jsonl"

LM2_MODEL_DIR = f"{OUT_DIR}/lm2_model"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_SRC_LEN = 128
MAX_TGT_LEN = 128
BATCH_SIZE = 16
EPOCHS = 5
LR = 3e-4

D_MODEL = 512
ENC_LAYERS = 6
DEC_LAYERS = 6
ENC_HEADS = 8
DEC_HEADS = 8
# ===========================

class Seq2SeqDataset(Dataset):
    def __init__(self, jsonl_path, vocab, max_src_len, max_tgt_len):
        self.rows = read_jsonl(jsonl_path)
        self.vocab = vocab
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.pad_id = vocab["PAD"]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        src_toks = r["src"]
        tgt_toks = r["tgt"]

        src_ids = encode(src_toks, self.vocab)[:self.max_src_len]
        tgt_ids = encode(tgt_toks, self.vocab)[:self.max_tgt_len]

        src_attn = [1] * len(src_ids)
        if len(src_ids) < self.max_src_len:
            pad_n = self.max_src_len - len(src_ids)
            src_ids += [self.pad_id] * pad_n
            src_attn += [0] * pad_n

        labels = tgt_ids[:]
        if len(labels) < self.max_tgt_len:
            labels += [self.pad_id] * (self.max_tgt_len - len(labels))
        labels = [(-100 if x == self.pad_id else x) for x in labels]

        return {
            "input_ids": torch.tensor(src_ids, dtype=torch.long),
            "attention_mask": torch.tensor(src_attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

def main():
    os.makedirs(LM2_MODEL_DIR, exist_ok=True)

    vocab = load_json(VOCAB_PATH)
    pad_id = vocab["PAD"]
    bos_id = vocab["BOS"]
    eos_id = vocab["EOS"]

    train_ds = Seq2SeqDataset(LM2_TRAIN_JSONL, vocab, MAX_SRC_LEN, MAX_TGT_LEN)
    val_ds   = Seq2SeqDataset(LM2_VAL_JSONL, vocab, MAX_SRC_LEN, MAX_TGT_LEN)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    cfg = BartConfig(
        vocab_size=len(vocab),
        d_model=D_MODEL,
        encoder_layers=ENC_LAYERS,
        decoder_layers=DEC_LAYERS,
        encoder_attention_heads=ENC_HEADS,
        decoder_attention_heads=DEC_HEADS,
        pad_token_id=pad_id,
        bos_token_id=bos_id,
        eos_token_id=eos_id,
        decoder_start_token_id=bos_id,
    )

    model = BartForConditionalGeneration(cfg).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)

    best_val = 1e9

    for ep in range(EPOCHS):
        model.train()
        pbar = tqdm(train_dl, desc=f"LM2 train ep {ep+1}/{EPOCHS}")

        for batch in pbar:
            input_ids = batch["input_ids"].to(DEVICE)
            attn = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

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
                labels = batch["labels"].to(DEVICE)
                loss = model(input_ids=input_ids, attention_mask=attn, labels=labels).loss
                tot += float(loss.detach().cpu())
                n += 1

        val_loss = tot / max(1, n)
        print("LM2 val_loss:", val_loss)

        if val_loss < best_val:
            best_val = val_loss
            model.save_pretrained(LM2_MODEL_DIR)
            print("saved best LM2 ✅", LM2_MODEL_DIR)

if __name__ == "__main__":
    main()
