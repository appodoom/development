import librosa
import numpy as np
import soundfile as sf
from pathlib import Path

wav_path = "./data/new_variations/interference.wav"
saving_path = "./output"
y, sr = librosa.load(path=wav_path, sr=None)
onset_frames = librosa.onset.onset_detect(y=y, sr=sr, backtrack=True)
onset_samples = librosa.frames_to_samples(onset_frames)
onset_samples = np.append(onset_samples, len(y))


def get_single_onset_segments(y, onset_samples):
    """Return a list of y-segments, one per onset interval."""
    segments = []
    for i in range(len(onset_samples) - 1):
        start = onset_samples[i]
        end = onset_samples[i + 1]
        segments.append(y[start:end])
    return segments


segments = get_single_onset_segments(y, onset_samples)


def make_pairs(segments, start_index=0):
    """
    Group segments into pairs:
    - start_index = 0 -> (0,1), (2,3), (4,5), ...
    - start_index = 1 -> (1,2), (3,4), (5,6), ...
    """
    pairs = []
    i = start_index
    while i + 1 < len(segments):
        pairs.append((segments[i], segments[i + 1]))
        i += 2
    return pairs


pairs_even = make_pairs(segments, start_index=0)
pairs_odd = make_pairs(segments, start_index=1)

out_dir = Path("./data/pairs")
out_dir.mkdir(exist_ok=True)


def save_pairs(pairs, prefix):
    for idx, (y1, y2) in enumerate(pairs):
        pair_audio = np.concatenate([y1, y2])
        out_file = out_dir / f"{prefix}_pair_{idx:03d}.wav"
        sf.write(out_file, pair_audio, sr)


save_pairs(pairs_even, "even")
save_pairs(pairs_odd, "odd")
