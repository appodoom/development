import numpy as np

# Workaround for librosa compatibility with newer NumPy versions:
if not hasattr(np, "complex"):
    np.complex = complex

import librosa
import matplotlib.pyplot as plt

def plot(i):
    y, sr = librosa.load(f"../samples/sample{i}.wav", sr=None)

    # Compute onset strength envelope
    hop_length = 512
    oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    win_length=len(y)/sr
    # print(win_length)
    # Estimate instantaneous tempo per frame
    tempos = librosa.beat.tempo(
        onset_envelope=oenv,
        sr=sr,
        hop_length=hop_length,
        aggregate=None,
        ac_size= win_length

    )
    # for i in range(len(tempos)):
    #     print(tempos[i])
    # Convert frame indices to time in seconds
    times = librosa.frames_to_time(np.arange(len(tempos)), sr=sr, hop_length=hop_length)

    # Plot tempo variation over time
    plt.figure()
    plt.plot(times, tempos)
    plt.xlabel("Time (s)")
    plt.ylabel("Tempo (BPM)")
    plt.title("Tempo Variation Over Time")
    plt.show()
plot(8)