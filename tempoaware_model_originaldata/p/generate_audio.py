import soundfile as sf
import librosa
import numpy as np
import json
import re
import os
import random
from pathlib import Path

# === CONFIGURATION ===
vocab_file = "vocab_music_sheet_250.txt"     # top tokens to process
audio_folder = "./soundgeneration"
output_dir = "./regenerated_top50random"
sr = 48000  # or use the real sampling rate if you know it
tempo = 120  # fallback default tempo if unknown

os.makedirs(output_dir, exist_ok=True)

# === NOTE DURATIONS ===
quarter_duration = 60.0 / tempo
note_durations = {
    16: quarter_duration / 4,
    8: quarter_duration / 2,
    4: quarter_duration,
    2: quarter_duration * 2,
    1: quarter_duration * 4,
    24: quarter_duration / 6,
    32: quarter_duration / 8,
    12: quarter_duration / 3,
    2.67: quarter_duration * (3 / 2),
    5.34: quarter_duration * (3 / 4),
    10.67: quarter_duration * (3 / 8),
    21.33: quarter_duration * (3 / 16),
    1.33: quarter_duration * 3
}

# === TOKEN PARSING ===
token_re = re.compile(r'([A-Za-z]+)_([0-9]+(?:\.[0-9]+)?)')

def generate_audio_from_token(token, output_name):
    subs = token_re.findall(token)
    chosen_paths = []
    durations = []

    for letter, dur_str in subs:
        dur_key = float(dur_str)
        if dur_key not in note_durations:
            print(f"⚠️  Duration {dur_key} not found, skipping token: {token}")
            return
        dur_seconds = note_durations[dur_key]
        wav_file = os.path.join(audio_folder, f"{letter}.wav")
        if not os.path.isfile(wav_file):
            print(f"⚠️  Missing sample for {letter}, skipping.")
            return
        chosen_paths.append(wav_file)
        durations.append(dur_seconds)

    segments = []
    for path, dur in zip(chosen_paths, durations):
        data, fs = sf.read(path)
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        target_len = int(dur * fs)
        if data.shape[0] < target_len:
            pad = np.zeros((target_len - data.shape[0]), dtype=data.dtype)
            data = np.concatenate([data, pad])
        else:
            data = data[:target_len]
        segments.append(data)

    combined = np.concatenate(segments)
    out_path = os.path.join(output_dir, f"{output_name}.wav")
    sf.write(out_path, combined, sr)
    print(f"✅ {out_path} ({len(combined)} samples)")

# === MAIN ===
with open(vocab_file, "r", encoding="utf-8") as f:
    all_tokens = [line.strip() for line in f if line.strip()]

# Take tokens starting from index 74
tokens_from_74 = all_tokens[74:]

# Randomly select 50 tokens
random_tokens = random.sample(tokens_from_74, 50)

# Generate and save with custom names
for idx, token in enumerate(random_tokens, start=1):
    output_name = f"toptoken_random_{idx}"
    generate_audio_from_token(token, output_name)
