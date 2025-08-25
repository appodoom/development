import librosa
import numpy as np
import matplotlib.pyplot as plt


def plot_wav(path, start_sec=None, end_sec=None):
    y, sr = librosa.load(path, sr=None)
    times = np.arange(len(y)) / sr
    if start_sec is not None or end_sec is not None:
        s = int(start_sec * sr) if start_sec is not None else 0
        e = int(end_sec * sr) if end_sec is not None else len(y)
        times = times[s:e]
        y = y[s:e]
    plt.figure(figsize=(10, 4))
    plt.plot(times, y, linewidth=0.8)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(f"Waveform: {path}")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


plot_wav("./data/first_data/old1.wav")
