import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from defaults import VOCAB_JSON_PATH, MODEL_PTH_PATH
from utils import load_json, save_json  # if you don't have save_json, remove it

# ----------------------------
# Config (MUST match training)
# ----------------------------
vocab_path = VOCAB_JSON_PATH
model_path = MODEL_PTH_PATH

EOF_TOKEN = "<EOF>"
CONTEXT_SIZE = 1024

# These MUST match your training model constructor:
D_MODEL = 256
NHEAD = 4
NUM_LAYERS = 4
DIM_FF = 1024

# ----------------------------
# Load vocab
# ----------------------------
vocab_data = load_json(json_file_path=vocab_path)
vocab_list = vocab_data["vocab"] if isinstance(vocab_data, dict) else vocab_data

string_to_index = {tok: i for i, tok in enumerate(vocab_list)}
index_to_string = {i: tok for tok, i in string_to_index.items()}
vocab_size = len(vocab_list)

eof_id = string_to_index.get(EOF_TOKEN, None)


# ----------------------------
# Model (same as training)
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

        q = self.q_proj(x).view(B, L, H, D).transpose(1, 2)  # (B,H,L,D)
        k = self.k_proj(x).view(B, L, H, D).transpose(1, 2)
        v = self.v_proj(x).view(B, L, H, D).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D)  # (B,H,L,L)

        # L must be <= max_len, so we keep a sliding window in generation
        positions = torch.arange(L, device=x.device)
        rel_idx = positions.unsqueeze(1) - positions.unsqueeze(0) + (self.max_len - 1)
        bias = self.rel_bias[:, rel_idx]  # (H,L,L)
        scores = scores + bias.unsqueeze(0)  # (B,H,L,L)

        if attn_mask is not None:
            scores = scores.masked_fill(
                attn_mask.unsqueeze(0).unsqueeze(1), float("-inf")
            )

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, v)  # (B,H,L,D)

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
        emb = self.token_emb(x)  # (B,L,D)
        L = x.size(1)
        mask = torch.triu(
            torch.ones(L, L, device=x.device), diagonal=1
        ).bool()  # causal
        h = emb
        for layer in self.layers:
            h = layer(h, src_mask=mask)
        return self.out(h)  # (B,L,V)


# ----------------------------
# Load weights (state_dict OR checkpoint)
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MusicTransformer(
    vocab_size=vocab_size,
    d_model=D_MODEL,
    nhead=NHEAD,
    num_layers=NUM_LAYERS,
    dim_feedforward=DIM_FF,
    max_len=CONTEXT_SIZE,
).to(device)

loaded = torch.load(model_path, map_location=device)

# if you saved just model.state_dict():
if isinstance(loaded, dict) and "model_state_dict" in loaded:
    model.load_state_dict(loaded["model_state_dict"])
else:
    model.load_state_dict(loaded)

model.eval()


# ----------------------------
# Helpers
# ----------------------------
def tokens_to_ids_strict(tokens, where="prompt"):
    ids = []
    for i, t in enumerate(tokens):
        if t not in string_to_index:
            raise ValueError(f"Token not in vocab: {t} (index {i}) in {where}")
        ids.append(string_to_index[t])
    return ids


@torch.no_grad()
def generate(
    prompt_ids,
    max_new_tokens=256,
    temperature=1.0,
    top_k=50,
    do_sample=True,
    stop_id=None,
):
    ids = list(prompt_ids)

    for _ in range(max_new_tokens):
        ctx = ids[-CONTEXT_SIZE:]  # keep L <= CONTEXT_SIZE to avoid rel-bias OOB
        x = torch.tensor(ctx, dtype=torch.long, device=device).unsqueeze(0)  # (1,L)

        logits = model(x)  # (1,L,V)
        next_logits = logits[0, -1, :]  # (V,)

        # temperature
        if temperature is None or temperature <= 0:
            temperature = 1.0
        next_logits = next_logits / temperature

        # top-k sampling / greedy
        if top_k is not None and top_k > 0:
            k = min(top_k, next_logits.size(-1))
            vals, idxs = torch.topk(next_logits, k=k)
            probs = torch.softmax(vals, dim=-1)
            if do_sample:
                pick = torch.multinomial(probs, 1).item()
                next_id = idxs[pick].item()
            else:
                next_id = idxs[torch.argmax(probs).item()].item()
        else:
            probs = torch.softmax(next_logits, dim=-1)
            next_id = (
                torch.multinomial(probs, 1).item()
                if do_sample
                else torch.argmax(probs).item()
            )

        ids.append(next_id)

        if stop_id is not None and next_id == stop_id:
            break

    return ids


# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    # Option A: start from a manual prompt (list of token-strings)
    # prompt_tokens = ["<BOS>", "NOTE_ON_60", "TIME_SHIFT_10", ...]
    # prompt_ids = tokens_to_ids_strict(prompt_tokens)

    # Option B: start from a JSON prompt file containing a list of token-strings
    # prompt_tokens = load_json("/content/prompt_tokens.json")
    # prompt_ids = tokens_to_ids_strict(prompt_tokens, where="prompt_tokens.json")

    # Minimal fallback: start from a single token (only if it exists in your vocab)
    prompt_tokens = [
        "OTA_4_0.0_1.5_0 OTA_4_0.0_1.5_0 OTA_4_0.0_1.5_0",
        "S_4_0.0_1.5_0",
        "D_0_0.0_3_0 S_4_0.0_1.5_0 S_4_0.0_1.5_0",
        "OTA_4_0.0_1.5_0 S_4_0.0_1.5_0",
    ]
    prompt_ids = tokens_to_ids_strict(prompt_tokens)

    out_ids = generate(
        prompt_ids,
        max_new_tokens=200,
        temperature=1.0,
        top_k=50,
        do_sample=True,
        stop_id=eof_id,
    )
    out_tokens = [index_to_string[i] for i in out_ids]

    print("Generated token count:", len(out_tokens))
    print("First 50 tokens:", out_tokens[:50])

    # Save if you want
    try:
        save_json("./generated.json", out_tokens)
        print("Saved:", "./generated.json")
    except Exception as e:
        print("Could not save ./generated.json:", e)
