import os
import json
import glob
import numpy as np
import librosa

SAMPLES_DIR = "fundemental_hits"
OUTPUT_JSON = "log_mels.json"
SR = 48000


def get_log_mel(wav_path):
    y, sr = librosa.load(wav_path, sr=SR, mono=True)
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    log_mel = librosa.power_to_db(mel, ref=np.max, top_db=80)
    return log_mel


def main():
    wavs = sorted(glob.glob(os.path.join(SAMPLES_DIR, "*.wav")))
    if not wavs:
        raise SystemExit(f"No .wav files found in {SAMPLES_DIR}/")

    out = {}
    for w in wavs:
        name = os.path.splitext(os.path.basename(w))[0]
        log_mel = get_log_mel(w)
        out[name] = log_mel.tolist()

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(out)} items to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
