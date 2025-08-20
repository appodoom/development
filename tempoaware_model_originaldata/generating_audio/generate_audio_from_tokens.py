import soundfile as sf
import librosa
import numpy as np
import json
import re
import os

input_json_file = "modeloutput.json"
wav_path = "../p/audios/sample1.wav"          # reference audio for tempo
audio_folder = "../p/soundgeneration"        # folder containing note WAVs (c.wav, d.wav, ...)
output_path = "./modeoutputsound.wav"  # output audio

# --- LOAD REFERENCE AUDIO AND ESTIMATE TEMPO ---
y, sr = librosa.load(wav_path, sr=None)
tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
tempo = tempo if isinstance(tempo, float) else tempo[0]
tempo = 93 #tempo of sample1
print(f"Detected tempo: {tempo:.2f} BPM")

# --- HELPER FUNCTION ---
def generate_audio_from_tokens(input_json_file, sr, audio_folder, tempo, output_path):
    # Load token list from JSON
    with open(input_json_file, encoding="utf-8") as f:
        list_of_tokens = json.load(f)

    # Map musical note durations to seconds
    quarter_duration = 60.0 / tempo
    note_durations = {
        16: quarter_duration / 4,
        12: quarter_duration / 3,
        8: quarter_duration / 2,
        4: quarter_duration,
        2: quarter_duration * 2,
        1: quarter_duration * 4,
        24: quarter_duration / 6,
        32: quarter_duration / 8,
        2.67: quarter_duration * (3 / 2),
        5.34: quarter_duration * (3 / 4),
        10.67: quarter_duration * (3 / 8),
        21.33: quarter_duration * (3 / 16),
        1.33: quarter_duration * 3
    }

    # Regex to parse note tokens like "c_32"
    token_re = re.compile(r'([A-Za-z]+)_([0-9]+(?:\.[0-9]+)?)')

    chosen_paths = []
    durations = []

    # Parse tokens and prepare WAV paths + durations
    for token in list_of_tokens:
        subs = token_re.findall(token)
        for letter, dur_str in subs:
            dur_key = float(dur_str)
            dur_seconds = note_durations.get(dur_key)
            if dur_seconds is None:
                raise ValueError(f"Unknown duration {dur_key} in token {token}")
            wav_file = os.path.join(audio_folder, f"{letter}.wav")
            if not os.path.isfile(wav_file):
                raise FileNotFoundError(f"Note WAV not found: {wav_file}")
            chosen_paths.append(wav_file)
            durations.append(dur_seconds)
            print(f"→ will play {letter}.wav for {dur_seconds:.3f}s")

    # Read WAVs, pad/truncate to target durations, and add silence
    segments = []

    for path, dur in zip(chosen_paths, durations):
        data, fs = sf.read(path)
        if data.ndim > 1:
            data = np.mean(data, axis=1)  # convert stereo to mono

        target_len = int(dur * fs)
        if data.shape[0] < target_len:
            pad = np.zeros(target_len - data.shape[0], dtype=data.dtype)
            data = np.concatenate([data, pad])
        else:
            data = data[:target_len]

        segments.append(data)

    # Concatenate all segments and save
    combined = np.concatenate(segments)
    sf.write(output_path, combined, sr)
    print(f"✅ wrote {output_path}, {combined.shape[0]} samples @ {sr} Hz")

# --- RUN THE GENERATOR ---
generate_audio_from_tokens(
    input_json_file=input_json_file,
    sr=sr,
    audio_folder=audio_folder,
    tempo=tempo,
    output_path=output_path,
)
