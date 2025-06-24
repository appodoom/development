import librosa
import numpy as np
from matplotlib import pyplot as plt

y, sr = librosa.load("../../sample8.wav", sr=None)

onsets = librosa.onset.onset_detect(y=y, sr=sr, units="samples")

# (start, end) in samples
onset_intervals = [(0, onsets[0]//2)]
for i in range(len(onsets)-1):
    start = onset_intervals[i][1]
    end = (onsets[i] + onsets[i+1])//2
    onset_intervals.append((start, end))

bpm, beats = librosa.beat.beat_track(y=y, sr=sr, units="samples")

silent_onsets = []

for interval in onset_intervals:
    is_silent = True
    for beat in beats:
        if interval[0] <= beat <= interval[1]:
            is_silent = False
            break
    if is_silent: silent_onsets.append(interval)


new_y = np.array([])
for interval in onset_intervals:
    if interval not in silent_onsets:
        new_y = np.concatenate((new_y, y[interval[0]:interval[1]]))
    else:
        new_y = np.concatenate((new_y, np.zeros(interval[1] - interval[0])))

def sliding_cross_correlation(X, Y):
    # Ensure both X and Y are 2D arrays
    if X.ndim == 1:
        X = X.reshape(1, -1)  # Reshape X to be 2D (1 row, multiple columns)
    if Y.ndim == 1:
        Y = Y.reshape(1, -1)  # Reshape Y to be 2D (1 row, multiple columns)

    # Ensure X is the larger matrix
    if Y.shape[1] > X.shape[1]:
        X, Y = Y, X

    best_score = -np.inf
    best_offset = None

    # Compute the norms of X and Y
    dim_X = np.linalg.norm(X)
    dim_Y = np.linalg.norm(Y)

    # Iterate over possible offsets
    for offset in range(X.shape[1] - Y.shape[1] + 1):
        # Slice the window of X to match the size of Y
        X_slider = X[:, offset:offset + Y.shape[1]]

        # Skip the loop if either X or Y has zero norm (to avoid division by zero)
        if dim_X == 0 or dim_Y == 0:
            continue
        else:
            # Compute the correlation score (dot product between the sliding window and Y)
            score = np.tensordot(X_slider, Y, axes=(1, 1)) / (dim_X * dim_Y)

        # Keep track of the best score and corresponding offset
        if score > best_score:
            best_score = score
            best_offset = offset

    return best_score


def find_cycle_beat_indices(
    y,
    sr,
    tempo,
    beat_samples,
    min_beats: int = 3,
    max_beats: int = 16,
    n_mels: int = 128,
    hop_length: int = 512
) -> np.ndarray:
    beat_frames = librosa.samples_to_frames(beat_samples)
    y2 = np.concatenate((y, y))
    mel1 = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
    mel2 = librosa.feature.melspectrogram(y=y2, sr=sr, n_mels=n_mels, hop_length=hop_length)
    mel1 = librosa.util.normalize(mel1, axis=1)
    mel2 = librosa.util.normalize(mel2, axis=1)
    frames_per_beat = int(sr * 60 / (tempo * hop_length))
    corrs = []
    for b in range(min_beats, max_beats + 1):
        shift = b * frames_per_beat
        if shift + mel1.shape[1] > mel2.shape[1]:
            break
        corr_val = sum(
            sliding_cross_correlation(mel1[i], mel2[i, shift:shift + mel1.shape[1]])
            for i in range(n_mels)
        )
        corrs.append(corr_val)
    corrs = np.array(corrs)
    best_beats = np.argmax(corrs) + min_beats
    cycle_indices = np.arange(0, len(beat_frames), best_beats)
    print("Best beats:", best_beats)
    return cycle_indices, beat_frames

find_cycle_beat_indices(y=new_y, sr=sr, tempo=bpm, beat_samples=beats)



