import soundfile as sf
import librosa
import numpy as np
import json
import re
import os
from pathlib import Path



input_txt_file = "./corpus_no_amp/corpus_tempo_orig/corpus0.json"
wav_path = "../../samples/old_samples/sample1.wav"
audio_folder = "../soundGeneration/resampled"
output_path = "./regenerated/sample1_regenerated_t.wav"

import json

with open(input_txt_file[:-4]+"txt", "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]

with open(input_txt_file, "w", encoding="utf-8") as f:
    json.dump(lines, f, indent=2)

y, sr = librosa.load(wav_path, sr=None)
tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
tempo = tempo[0]
# def adjust_tempo(tempo):
#     while tempo > 150:
#         tempo /= 2
#     while tempo < 50:
#         tempo*=2
#     return tempo
# tempo = adjust_tempo(tempo)
# amplitude_bins = np.array([0.15, 0.225, 0.25, 0.275, 0.3, 0.4,0.5, 0.6,1])

def map_to_nearest_bin(val, bins):
    idx = np.abs(bins - val).argmin()
    return bins[idx]

def generate_audio_from_tokens(input_txt_file, sr, audio_folder, tempo, output_path):
    with open(input_txt_file, encoding="utf-8") as f:
        list_of_tokens = json.load(f)
    # list_of_tokens =['c_32d_32', 'b_12g_12g_12g_12g_12', 'a_32a_32d_32d_12', 'a_12a_12a_10.67', 'a_24d_32', 'b_16', 'g_12g_10.67', 'a_32a_32d_32', 'a_12g_12g_12', 'd_32a_32']
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
        # 28:quarter_duration*(1/7),
        # 56:quarter_duration*(1/14),
        # 14:quarter_duration*(2/7),j
        # 9.33:quarter_duration*(3/7),
        # 7:quarter_duration*(4/7),
        # 5.6:quarter_duration*(5/7),
        # 4.67:quarter_duration*(6/7),
        # 20:quarter_duration*(1/5),
        # 40:quarter_duration*(1/10),
        # 10:quarter_duration*(2/5),
        # 6.67:quarter_duration*(3/5),
        # 5:quarter_duration*(4/5),
        # 36:quarter_duration*(1/9),
        # 72:quarter_duration*(1/18),
        # 18:quarter_duration*(2/9),
        # 9:quarter_duration*(4/9),
        # 7.2:quarter_duration*(5/9),
        # 6:quarter_duration*(6/9),
        # 5.1:quarter_duration*(7/9),
        # 4.5:quarter_duration*(8/9)
    }
    token_re = re.compile(r'([A-Za-z]+)_([0-9]+(?:\.[0-9]+)?)')


    chosen_paths = []
    durations = []
    # gains = []

    for token in list_of_tokens:
        subs = token_re.findall(token)
        for letter, dur_str in subs:
            dur_key = float(dur_str)
            # amp_key = int(amp_str)
            # gain=amplitude_bins[amp_key]
            dur_seconds = note_durations[dur_key]
            wav_file = os.path.join(audio_folder, f"{letter}.wav")
            
            chosen_paths.append(wav_file)
            durations.append(dur_seconds)
            # gains.append(gain)
            print(f"→ will play {letter}.wav for {dur_seconds:.3f}s")

    segments = []
    for path, dur in zip(chosen_paths, durations):
        data, fs = sf.read(path)
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        # data = data * gain

        target_len = int(dur * fs)
        if data.shape[0] < target_len:
            pad = np.zeros((target_len - data.shape[0]), dtype=data.dtype)
            data = np.concatenate([data, pad])
        else:
            data = data[:target_len]
        # pad = np.zeros(sr*,dtype=data.dtype)
        segments.append(data)

    combined = np.concatenate(segments)
    sf.write(output_path, combined, sr)
    print(f"✅ wrote {output_path}, {combined.shape[0]} samples @ {fs} Hz")

generate_audio_from_tokens(input_txt_file=input_txt_file, sr=sr, audio_folder=audio_folder, tempo=tempo, output_path=output_path)
