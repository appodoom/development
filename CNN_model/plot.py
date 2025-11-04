import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


def plot_mel_segments(y_segments, sr=22050, n_mels=128):
    for i, y in enumerate(y_segments):
        # Compute mel spectrogram
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        S_dB = librosa.power_to_db(S, ref=np.max)

        # Plot
        plt.figure(figsize=(8, 4))
        librosa.display.specshow(S_dB, sr=sr, x_axis="time", y_axis="mel", cmap="magma")
        plt.title(f"Segment {i + 1}")
        plt.colorbar(format="%+2.0f dB")
        plt.tight_layout()
        plt.show()


# Example usage:
# Suppose you have a list of NumPy arrays (each is a segment)
# y_segments = [y1, y2, y3, ...]
# plot_mel_segments(y_segments, sr=44100)
