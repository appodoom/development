import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from defaults import VOCAB_JSON_PATH, TOKENS_JSON_PATH, MODEL_PTH_PATH
from utils import load_json

root = "/content"
save_dir = root

vocab_path = os.path.join(root, VOCAB_JSON_PATH)
tokens_path = os.path.join(root, TOKENS_JSON_PATH)

EOF_TOKEN = "<EOF>"
CONTEXT_SIZE = 1024
BATCH_SIZE = 16
EPOCHS = 5
LR = 1e-4

# ----------------------------
# Load vocab (JSON)
# ----------------------------
vocab_data = load_json(json_file_path=vocab_path)
vocab_list = vocab_data["vocab"] if isinstance(vocab_data, dict) else vocab_data

string_to_index = {tok: i for i, tok in enumerate(vocab_list)}
index_to_string = {i: tok for tok, i in string_to_index.items()}
vocab_size = len(vocab_list)
print("vocab_size:", vocab_size)


# ----------------------------
# Split by <EOF> as delimiter
# ----------------------------
def split_on_eof_tokens(token_list, eof_token=EOF_TOKEN):
    sequences = []
    cur = []
    for t in token_list:
        if t == eof_token:
            if cur:
                sequences.append(cur)
                cur = []
        else:
            cur.append(t)
    if cur:
        sequences.append(cur)
    return sequences


def tokens_to_ids_strict(seq_tokens, where=""):
    for i, t in enumerate(seq_tokens):
        if t not in string_to_index:
            raise ValueError(f"Token not in vocab: {t} (index {i}) in {where}")
    return [string_to_index[t] for t in seq_tokens]


all_sequences = []
token_list = load_json(json_file_path=tokens_path)
seqs = split_on_eof_tokens(token_list)  # <EOF> removed here
all_sequences.extend([tokens_to_ids_strict(s, where=tokens_path) for s in seqs])

print("Loaded sequences:", len(all_sequences))


class SegmentedTextDataset(Dataset):
    def __init__(self, sequences, context_size=1024):
        self.context_size = context_size
        # keep only sequences long enough for (x,y) of length context_size
        # need at least context_size + 1 tokens
        self.seqs = [
            torch.tensor(s, dtype=torch.long)
            for s in sequences
            if len(s) > context_size + 1
        ]

        self.samples_per_seq = [len(s) - context_size - 1 for s in self.seqs]

        # prefix sums for indexing across sequences
        self.cum = []
        total = 0
        for k in self.samples_per_seq:
            total += k
            self.cum.append(total)

    def __len__(self):
        return self.cum[-1] if self.cum else 0

    def __getitem__(self, idx):
        import bisect

        si = bisect.bisect_right(self.cum, idx)
        prev = 0 if si == 0 else self.cum[si - 1]
        offset = idx - prev

        data = self.seqs[si]
        x = data[offset : offset + self.context_size]
        y = data[offset + 1 : offset + 1 + self.context_size]
        return x, y


dataset = SegmentedTextDataset(all_sequences, context_size=CONTEXT_SIZE)
if len(dataset) == 0:
    raise RuntimeError(
        "Dataset is empty. Check: (1) you split correctly on <EOF>, "
        "(2) sequences are longer than CONTEXT_SIZE+1, (3) tokens match vocab."
    )

train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
print("Total training samples:", len(dataset))
print("Total batches before training:", len(train_loader))


# ----------------------------
# Model (Relative attention)
# ----------------------------
class RelativeMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, max_len=512, dropout=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.max_len = max_len
        self.rel_bias = nn.Parameter(torch.zeros(num_heads, 2 * max_len - 1))

    def forward(self, x, attn_mask=None):
        B, L, E = x.size()
        H, D = self.num_heads, self.head_dim

        q = self.q_proj(x).view(B, L, H, D).transpose(1, 2)
        k = self.k_proj(x).view(B, L, H, D).transpose(1, 2)
        v = self.v_proj(x).view(B, L, H, D).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D)

        positions = torch.arange(L, device=x.device)
        rel_idx = positions.unsqueeze(1) - positions.unsqueeze(0) + (self.max_len - 1)
        bias = self.rel_bias[:, rel_idx]  # (H, L, L)
        scores = scores + bias.unsqueeze(0)  # (B, H, L, L)

        if attn_mask is not None:
            scores = scores.masked_fill(
                attn_mask.unsqueeze(0).unsqueeze(1), float("-inf")
            )

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, v)  # (B, H, L, D)

        context = context.transpose(1, 2).contiguous().view(B, L, E)
        return self.out_proj(context)


class RelativeTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.1, max_len=512):
        super().__init__()
        self.self_attn = RelativeMultiheadAttention(
            d_model, nhead, max_len=max_len, dropout=dropout
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        src2 = self.self_attn(src, attn_mask=src_mask)
        src = self.norm1(src + self.dropout1(src2))

        ff = F.relu(self.linear1(src))
        ff2 = self.linear2(self.dropout2(ff))
        src = self.norm2(src + self.dropout2(ff2))
        return src


class MusicTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=256,
        nhead=4,
        num_layers=4,
        dim_feedforward=1024,
        max_len=1024,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [
                RelativeTransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=0.1,
                    max_len=max_len,
                )
                for _ in range(num_layers)
            ]
        )
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        emb = self.token_emb(x)  # (B, L, D)
        L = x.size(1)
        # causal mask: True above diagonal
        mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()

        h = emb
        for layer in self.layers:
            h = layer(h, src_mask=mask)
        return self.out(h)


# ----------------------------
# Train
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MusicTransformer(vocab_size, max_len=CONTEXT_SIZE).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()  # no <EOF> in data, so no ignore_index needed

for ep in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    seen_batches = 0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)

        logits = model(xb)  # (B, L, V)
        loss = criterion(logits.reshape(-1, vocab_size), yb.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        seen_batches += 1

        if seen_batches % 10 == 0:
            print(
                f"Epoch {ep} batch {seen_batches} loss {total_loss / seen_batches:.4f}",
                flush=True,
            )

    avg_loss = total_loss / max(seen_batches, 1)
    print(f"Epoch {ep:2d} — loss: {avg_loss:.4f}", flush=True)

    ckpt = {
        "epoch": ep,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": avg_loss,
    }
    torch.save(ckpt, os.path.join(save_dir, f"checkpoint_ep{ep}.pt"))

torch.save(model.state_dict(), os.path.join(save_dir, MODEL_PTH_PATH))
print("✅ Training complete! Saved to:", os.path.join(save_dir, MODEL_PTH_PATH))
