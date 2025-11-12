AUDIO_PATH   = r"C:\path\to\your\file.wav"  # <- your wav
AMP_THRESH   = 0.05                          # drop onsets with peak |amp| < this
WINDOW_SEC   = 0.05                          # half-window around each onset (seconds)
HOP_LENGTH   = 512                           # librosa hop for onset detection
BACKTRACK    = True                          # snap onsets to preceding peak
RETURN_MODE  = "times"                       # "times" | "frames" | "samples"

import numpy as np
import librosa

def load_audio(path):
    try:
        y, sr = librosa.load(path, sr=None, mono=True)
        return y.astype(np.float32), sr
    except Exception as e:
        print("Brooo… couldn't load audio:", e)
        return None, None

def detect_onsets(y, sr, hop_length, backtrack):
    onset_frames = librosa.onset.onset_detect(
        y=y, sr=sr, hop_length=hop_length, backtrack=backtrack
    )
    onset_samples = librosa.frames_to_samples(onset_frames, hop_length=hop_length)
    onset_times   = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)
    return onset_frames, onset_samples, onset_times

def local_peak_amplitude(y, center, half_win):
    a = max(0, center - half_win)
    b = min(len(y), center + half_win)
    if a >= b: 
        return 0.0
    return float(np.max(np.abs(y[a:b])))

def measure_onset_amplitudes(y, sr, onset_samples, window_sec):
    half_win = int(round(window_sec * sr))
    amps = [local_peak_amplitude(y, s, half_win) for s in onset_samples]
    return np.array(amps, dtype=np.float32)

def get_clean_onsets():
    y, sr = load_audio(AUDIO_PATH)
    if y is None:
        return np.array([])

    onset_frames, onset_samples, onset_times = detect_onsets(
        y, sr, HOP_LENGTH, BACKTRACK
    )
    amps = measure_onset_amplitudes(y, sr, onset_samples, WINDOW_SEC)

    keep = amps >= AMP_THRESH
    print(f"found {len(onset_frames)} onsets → keeping {int(keep.sum())}, dropping {int((~keep).sum())} (thresh={AMP_THRESH})")

    if RETURN_MODE == "frames":
        return onset_frames[keep]
    if RETURN_MODE == "samples":
        return onset_samples[keep]
    # default: times
    return onset_times[keep]

if __name__ == "__main__":
    clean_onsets = get_clean_onsets()
    print("clean_onsets:", np.round(clean_onsets[:16], 4))
