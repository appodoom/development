import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.io import wavfile

FUNDAMENTALS_DIR = "fundamentals"   # folder with a.wav, b.wav, etc.
VOCAB_FILE = "vocab_music_sheet_250.txt"
WEIGHTS = "music_sheet_transformer_weights_20_epoch_originaldata.pth"

SAMPLE_RATE = 48000
TEMPO = 93
QUARTER_DURATION = 60 / TEMPO

SEED_TOKENS = [
  "c_2.67",
  "g_16g_16",
  "c_16g_16",
  "a_10.67b_8",
  "b_12",
  "g_16",
  "a_10.67a_10.67",
  "a_21.33a_32",
  "a_24",
  "g_16b_16",
  "d_21.33d_32",
  "d_24",
  "g_16",
  "a_10.67",
  "d_8",
  "g_10.67g_16",
  "a_10.67a_10.67a_16",
  "a_24",
  "d_32d_32d_32d_32b_32",
  "g_21.33",
  "a_16",
  "d_24d_32d_32",
  "d_32d_32b_32",
  "a_16",
  "d_8",
  "b_10.67",
  "g_16a_16a_16",
  "a_16",
  "a_24",
  "a_32",
  "d_24",
  "g_16g_16",
  "a_24",
  "a_32d_32",
  "d_24",
  "g_16a_16a_16",
  "g_16",
  "a_24",
  "d_32d_32a_24",
  "g_16a_16a_16",
  "g_16",
  "a_24",
  "d_32d_32",
  "c_32",
  "a_24",
  "g_16g_16g_16",
  "g_16",
  "a_10.67",
  "g_10.67",
  "a_21.33",
  "d_32b_32",
  "g_24",
  "g_16a_16a_16a_16a_16",
  "a_24",
  "c_32",
  "d_32d_32d_32d_32",
  "d_16",
  "b_12",
  "e_32",
  "d_32d_32d_32d_32",
  "a_16",
  "g_10.67a_16",
  "d_24d_32d_32",
  "d_32d_32",
  "a_12a_10.67a_16",
  "a_24",
  "a_32d_32d_32",
  "d_32d_32a_24",
  "g_16g_16g_16",
  "a_16a_16",
  "g_16",
  "a_21.33a_32",
  "d_32d_24",
  "g_16a_16a_16",
  "a_16",
  "a_24",
  "a_32a_32a_32",
  "c_32",
  "d_32d_32d_32d_32",
  "a_32b_32",
  "d_32d_32d_32d_32",
  "d_32d_32d_32",
  "c_32",
  "d_32a_32",
  "d_32d_32a_32",
  "d_32",
  "c_32",
  "d_32d_32d_32d_32",
  "d_32a_32",
  "d_32d_32",
  "c_32",
  "g_12",
  "e_8",
  "e_10.67",
  "g_16g_16g_16",
  "a_10.67",
  "d_8",
  "g_10.67g_16",
  "a_16a_16",
  "d_24d_32",
  "a_32d_32d_32",
  "b_32a_24",
  "g_16",
  "d_24d_32d_32",
  "d_32d_32d_32",
  "a_24",
  "g_16c_16c_16",
  "g_12a_16",
  "d_32",
  "g_24",
  "c_16a_16a_16a_16",
  "a_24",
  "a_32d_32d_32",
  "g_16g_16",
  "a_24",
  "d_32d_32",
  "d_24",
  "g_16g_16g_16",
  "g_16",
  "a_21.33",
  "d_32d_32a_24",
  "g_16a_16a_16",
  "a_16",
  "a_24",
  "d_32d_32a_24",
  "a_24",
  "d_32d_32",
  "d_24",
  "c_16c_16g_16",
  "a_16a_16",
  "g_16",
  "a_24",
  "d_32d_32",
  "d_24",
  "g_16a_16a_16",
  "g_16",
  "a_21.33",
  "d_32d_32a_32",
  "d_32d_32d_32d_32",
  "d_32",
  "c_32",
  "d_32d_32d_32d_32",
  "a_12a_10.67a_16",
  "d_24b_32",
  "d_32d_32",
  "a_12a_12",
  "d_32d_32d_32d_32",
  "e_32",
  "d_32d_32d_32a_32",
  "b_12",
  "b_10.67",
  "g_16",
  "a_10.67",
  "b_10.67",
  "d_21.33d_32",
  "g_21.33",
  "g_16a_16a_16",
  "d_24d_32d_32",
  "d_32d_32a_32",
  "a_24",
  "g_16",
  "d_24a_32",
  "d_32d_32a_32",
  "d_32a_24",
  "g_16g_16g_16",
  "g_16",
  "a_21.33",
  "d_32d_32a_24",
  "g_16a_16a_16",
  "c_21.33d_32",
  "a_32d_32d_32",
  "d_32d_32b_32",
  "d_32d_32d_32d_32",
  "a_32d_32d_32",
  "d_32b_32",
  "a_24",
  "g_16",
  "e_16",
  "c_16a_21.33",
  "a_24",
  "c_24c_24",
  "c_24c_24",
  "c_24c_24",
  "c_24",
  "d_12",
  "e_10.67",
  "f_16",
  "c_16",
  "d_32d_32d_32d_32",
  "d_32a_32",
  "d_32d_32d_32",
  "a_12",
  "g_10.67",
  "a_21.33",
  "d_32b_32",
  "a_24",
  "g_16a_16a_16a_16a_16",
  "a_21.33",
  "d_32a_32",
  "d_32d_32d_32b_32",
  "g_21.33",
  "c_21.33d_32",
  "d_32d_32d_32d_32",
  "d_32",
  "e_21.33",
  "d_10.67",
  "g_10.67g_16",
  "a_16a_16a_16",
  "a_24",
  "a_32a_32",
  "a_24",
  "g_16g_16",
  "c_21.33d_32",
  "c_21.33c_21.33",
  "c_32",
  "c_21.33",
  "g_16",
  "g_12",
  "g_16",
  "d_24",
  "g_16a_16a_16",
  "a_16",
  "a_24",
  "a_32d_32",
  "a_32",
  "g_16g_16",
  "a_16",
  "g_16c_16",
  "c_21.33",
  "g_32",
  "d_32d_32b_32",
  "d_16",
  "f_12",
  "c_32",
  "d_32d_32d_32",
  "a_12a_10.67",
  "c_16",
  "d_32d_32a_32",
  "d_32",
  "d_16",
  "f_12",
  "d_32d_32d_32b_32",
  "a_24",
  "a_10.67",
  "g_10.67",
  "a_21.33",
  "d_32a_24",
  "g_16a_16a_16",
  "a_16",
  "a_24",
  "a_32d_32",
  "a_32",
  "g_16g_16",
  "d_24d_32",
  "b_32",
  "d_32",
  "g_16g_16g_16",
  "g_16",
  "a_21.33a_32",
  "d_32a_24",
  "g_16a_16a_16",
  "a_16",
  "a_24",
  "d_32d_32a_32",
  "d_32d_32d_32",
  "a_24",
  "e_24",
  "d_32b_32",
  "d_32a_24",
  "c_24",
  "a_21.33a_16",
  "g_16",
  "a_21.33",
  "d_32d_32",
  "d_24",
  "g_16a_16",
  "g_16a_16",
  "a_24",
  "a_32a_32",
  "d_32",
  "g_16b_16",
  "a_21.33",
  "d_32",
  "g_21.33",
  "g_16",
  "a_21.33a_32",
  "c_32",
  "d_32",
  "c_32",
  "d_32d_32d_32d_32",
  "d_32d_32d_32",
  "d_21.33",
  "g_10.67g_12",
  "d_32d_21.33",
  "d_24d_32",
  "g_21.33",
  "d_24d_32",
  "g_21.33",
  "c_24",
  "c_32",
  "a_12",
  "g_10.67",
  "a_24",
  "d_32d_32",
  "d_24",
  "g_16a_16a_16",
  "d_24a_32",
  "d_32d_32d_32d_32",
  "d_32a_32",
  "d_32d_32d_32b_32",
  "d_32",
  "g_16g_16",
  "c_16a_16a_16",
  "b_12",
  "a_21.33",
  "d_32a_24",
  "c_16a_16a_16",
  "d_24d_32d_32",
  "d_32d_32d_32d_32d_32d_32d_32d_32",
  "a_24",
  "c_16g_16g_16",
  "a_16a_16",
  "b_12",
  "a_21.33",
  "d_32a_24",
  "c_16a_16a_16",
  "d_24a_32",
  "d_32a_32",
  "d_32a_32",
  "d_32d_32d_32d_32",
  "a_32",
  "d_24",
  "g_16g_16g_16",
  "e_16",
  "g_16g_16g_16",
  "g_16g_16",
  "a_16",
  "g_16a_16",
  "a_32",
  "d_24",
  "g_16g_16g_16",
  "g_16g_16g_16",
  "a_10.67g_8",
  "g_10.67g_16",
  "a_10.67a_10.67",
  "a_21.33a_32",
  "a_32a_32",
  "g_16g_16",
  "a_24",
  "d_32d_32b_32",
  "g_16a_16a_16",
  "g_16",
  "a_21.33a_32",
  "b_32",
  "d_32",
  "g_16a_16a_16a_16a_16",
  "a_24",
  "d_32a_32",
  "d_32d_32d_32",
  "g_24",
  "a_16",
  "c_24",
  "d_32d_32d_32a_32",
  "a_12",
  "g_10.67a_16",
  "d_24",
  "c_32",
  "d_32d_32d_32",
  "a_16",
  "a_12",
  "d_32d_32d_32d_32",
  "g_21.33",
  "g_16g_16",
  "c_16c_16g_16",
  "a_21.33",
  "d_32d_32",
  "c_32",
  "g_16c_16",
  "c_24",
  "c_32",
  "d_32d_32",
  "g_16c_16",
  "c_21.33",
  "a_32",
  "d_24",
  "g_16c_16",
  "c_24",
  "c_32",
  "d_32b_32",
  "b_16",
  "c_16",
  "d_32d_32a_32",
  "d_32d_32a_32",
  "a_32",
  "g_16",
  "e_16",
  "g_16",
  "d_24d_32d_32",
  "d_32b_32",
  "a_24",
  "g_16g_16g_16",
  "d_24b_32",
  "d_32d_32d_32",
  "b_12b_8",
  "g_10.67a_16",
  "e_10.67",
  "e_8",
  "a_8a_10.67",
  "a_21.33a_32",
  "g_21.33",
  "g_16c_16",
  "g_16"
]

NEXT_N = 100
TEMPERATURE = 0.8
TOP_K = 50
NO_REPEAT_NGRAM = 4
OUTPUT_WAV = "generated_continuation.wav"

# Duration mapping
DURATION_MAP = {
    '16': QUARTER_DURATION / 4,
    '8': QUARTER_DURATION / 2,
    '4': QUARTER_DURATION,
    '2': QUARTER_DURATION * 2,
    '1': QUARTER_DURATION * 4,
    '24': QUARTER_DURATION / 6,
    '32': QUARTER_DURATION / 8,
    '12': QUARTER_DURATION / 3,
    '2.67': QUARTER_DURATION * (3/2),
    '5.34': QUARTER_DURATION * (3/4),
    '10.67': QUARTER_DURATION * (3/8),
    '21.33': QUARTER_DURATION * (3/16),
    '1.33': QUARTER_DURATION * 3
}

fundamentals = {}
for note in ['a', 'b', 'c', 'd', 'e', 'f', 'g']:
    wav_path = os.path.join(FUNDAMENTALS_DIR, f"{note}.wav")
    if os.path.exists(wav_path):
        _, audio = wavfile.read(wav_path)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)  # mono
        audio = audio.astype(np.float32)
        fundamentals[note] = audio
    else:
        raise FileNotFoundError(f"Missing fundamental: {wav_path}")

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
        bias = self.rel_bias[:, rel_idx]
        scores = scores + bias.unsqueeze(0)
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask.unsqueeze(0).unsqueeze(1), float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(B, L, E)
        return self.out_proj(context)

class RelativeTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.1, max_len=512):
        super().__init__()
        self.self_attn = RelativeMultiheadAttention(d_model, nhead, max_len=max_len, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        src2 = self.self_attn(src, attn_mask=src_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        ff = F.relu(self.linear1(src))
        ff2 = self.linear2(self.dropout2(ff))
        src = src + self.dropout2(ff2)
        return self.norm2(src)

class MusicTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=4, num_layers=4, dim_feedforward=1024, max_len=1024):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        layers = []
        for _ in range(num_layers):
            layers.append(RelativeTransformerEncoderLayer(
                d_model=d_model, nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=0.1, max_len=max_len
            ))
        self.transformer = nn.Sequential(*layers)
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        emb = self.token_emb(x)
        seq_len = x.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        h = emb
        for layer in self.transformer:
            h = layer(h, src_mask=mask)
        return self.out(h)

def load_vocabulary(vocab_file):
    with open(vocab_file, "r") as f:
        tokens = [line.strip() for line in f if line.strip()]
    index_to_string = {i: tok for i, tok in enumerate(tokens)}
    string_to_index = {tok: i for i, tok in enumerate(tokens)}
    return tokens, index_to_string, string_to_index

def load_model(vocab_size, weights_path):
    model = MusicTransformer(vocab_size=vocab_size)
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def predict_next_tokens(seed_tokens, model, string_to_index, index_to_string,
                        next_n=100, temperature=0.8, top_k=0, no_repeat_ngram=0):
    ids = [string_to_index[tok] for tok in seed_tokens if tok in string_to_index]
    generated_ids = []

    def apply_no_repeat_ngram_constraint(seq_ids, logits_row, n):
        if n <= 1 or len(seq_ids) < n-1:
            return logits_row
        n1 = n - 1
        tail = tuple(seq_ids[-n1:]) if n1 > 0 else tuple()
        seen = set()
        for i in range(len(seq_ids) - n + 1):
            if tuple(seq_ids[i:i+n1]) == tail:
                seen.add(seq_ids[i+n1])
        logits_row[list(seen)] = -float('inf')
        return logits_row

    for _ in range(next_n):
        ctx = torch.tensor([ids[-128:]], dtype=torch.long)
        with torch.no_grad():
            logits = model(ctx)[0, -1]
        if no_repeat_ngram > 1:
            logits = apply_no_repeat_ngram_constraint(ids, logits.clone(), no_repeat_ngram)
        logits = logits / max(1e-6, temperature)
        if top_k and top_k > 0 and top_k < logits.numel():
            topk_vals, topk_idx = torch.topk(logits, top_k)
            probs = torch.zeros_like(logits).scatter_(0, topk_idx, torch.softmax(topk_vals, dim=0))
        else:
            probs = torch.softmax(logits, dim=0)
        next_id = torch.multinomial(probs, 1).item()
        ids.append(next_id)          # new token becomes part of next input
        generated_ids.append(next_id)
    return [index_to_string[i] for i in generated_ids]

def tokens_to_wav(tokens, output_file="output.wav"):
    def parse_token(token):
        parts = []
        i = 0
        while i < len(token):
            note = ""
            while i < len(token) and token[i].isalpha():
                note += token[i]
                i += 1
            if i >= len(token) or token[i] != '_':
                break
            i += 1
            duration = ""
            while i < len(token) and (token[i].isdigit() or token[i] == '.'):
                duration += token[i]
                i += 1
            if note and duration:
                parts.append((note, duration))
        return parts

    total_duration = 0.0
    for token in tokens:
        for note, duration in parse_token(token):
            if duration in DURATION_MAP:
                total_duration += DURATION_MAP[duration]
    total_samples = int(SAMPLE_RATE * total_duration)
    output_audio = np.zeros(total_samples)
    current_position = 0

    for token in tokens:
        for note_part, duration_str in parse_token(token):
            duration = DURATION_MAP.get(duration_str)
            if duration is None:
                continue
            num_samples = int(SAMPLE_RATE * duration)
            for note in note_part:
                if note in fundamentals:
                    fundamental = fundamentals[note]
                    if len(fundamental) < num_samples:
                        repeated = np.tile(fundamental, num_samples // len(fundamental) + 1)
                        note_audio = repeated[:num_samples]
                    else:
                        note_audio = fundamental[:num_samples]
                    end_pos = current_position + num_samples
                    if end_pos > len(output_audio):
                        output_audio = np.pad(output_audio, (0, end_pos - len(output_audio)))
                    output_audio[current_position:end_pos] += note_audio * 0.5
            current_position += num_samples

    output_audio = np.int16(output_audio / np.max(np.abs(output_audio)) * 32767)
    wavfile.write(output_file, SAMPLE_RATE, output_audio)

# ========================
# MAIN
# ========================
if __name__ == "__main__":
    tokens, index_to_string, string_to_index = load_vocabulary(VOCAB_FILE)
    model = load_model(len(tokens), WEIGHTS)

    # Autoregressive prediction
    next_tokens = predict_next_tokens(
        seed_tokens=SEED_TOKENS,
        model=model,
        string_to_index=string_to_index,
        index_to_string=index_to_string,
        next_n=NEXT_N,
        temperature=TEMPERATURE,
        top_k=TOP_K,
        no_repeat_ngram=NO_REPEAT_NGRAM
    )

    combined_tokens = next_tokens
    print("Generated tokens:")
    print(next_tokens)

    tokens_to_wav(combined_tokens, OUTPUT_WAV)
    print(f"Saved generated audio to {OUTPUT_WAV}")