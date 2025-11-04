import soundfile as sf
import numpy as np
import re
import os

# import random
from get_corpus import get_note_duration, get_corpus


def generate_audio_from_tokens(
    list_of_tokens,
    tempo,
    audio_folder="../data/fundemental_hits",
    # variations_folder="./data",
):
    quarter_duration = 60.0 / tempo[0]
    note_durations = get_note_duration(quarter_duration)

    token_re = re.compile(r"^([A-Za-z0-9]+)_(\d+(?:\.\d+)?)$")

    chosen_paths = []
    durations = []
    for token in list_of_tokens:
        subs = token_re.findall(token)
        for letter, dur_str in subs:
            dur_key = float(dur_str)
            dur_seconds = note_durations[dur_key]
            wav_file = os.path.join(audio_folder, f"{letter}.wav")

            chosen_paths.append(wav_file)
            durations.append(dur_seconds)
            print(f"→ will play {letter}.wav for {dur_seconds:.3f}s")

    segments = []
    for path, dur in zip(chosen_paths, durations):
        data, fs = sf.read(path)
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        # data = data * 3

        target_len = int(dur * fs)  # in samples
        if data.shape[0] < target_len:
            pad = np.zeros(target_len - data.shape[0], dtype=data.dtype)
            data = np.concatenate([data, pad])
        else:
            data = data[:target_len]
        segments.append(data)

    combined = np.concatenate(segments)
    sf.write("old1_fund_regen_new.wav", data=combined, samplerate=fs)


classified_hits_with_model_pred, tempo = get_corpus(
    fundamentals_path="../mel.json",
    file_path="../data/first_data/old1.wav",
    model_pred=True,
    log_mel=False,
)
print(classified_hits_with_model_pred)
generate_audio_from_tokens(list_of_tokens=classified_hits_with_model_pred, tempo=tempo)
