import numpy as np
import soundfile as sf
import librosa
import matplotlib.pyplot as plt
from scipy.stats import uniform, norm
# Workaround for librosa compatibility
if not hasattr(np, "complex"):
    np.complex = complex

def plot_and_listen(i):
    # Load audio
    uni=uniform(loc=0,scale=320)
    y, sr = librosa.load(f"../../samples/sample{i}.wav", sr=None)
    tempo = librosa.feature.tempo(y=y, sr=sr )
    low, high = 0.25,0.75
    denom = norm.ppf(high) - norm.ppf(low)
    q3=np.log2(tempo+(5.5/100)*tempo)
    q1=np.log2(tempo-(5.5/100)*tempo)
    iqr_log2 = q3-q1
    std_log2 = iqr_log2 / denom
    print(std_log2)
    # Compute tempo and beats

    # Compute tempo and beats
    # tempo = librosa.feature.tempo(y=y, sr=sr)
    tempos = librosa.feature.tempo(
        y=y,
        sr=sr,
        aggregate=None,
        ac_size=12,
        std_bpm=0.12,
        start_bpm=tempo
    )

    _, beats = librosa.beat.beat_track(y=y, sr=sr, bpm=tempos)
    beat_times = librosa.frames_to_time(beats, sr=sr)

    # Create and save new audio with clicks
    click_track = librosa.clicks(times=beat_times, sr=sr, length=len(y))
    y_new = 0.8 * y + click_track
    output_path = f"../../samples_with_beats/sample{i}.wav"
    sf.write(output_path, y_new, sr)

    # Create figure with two subplots
    plt.figure(figsize=(14, 8))

    # First subplot: Tempo variation
    plt.subplot(2, 1, 1)
    tempo_times = librosa.frames_to_time(np.arange(len(tempos)), sr=sr)
    plt.plot(tempo_times, tempos, label="Tempo", color="green")
    for bt in beat_times:
        plt.axvline(x=bt, color="red", alpha=0.3, linestyle="--", linewidth=1)
    plt.ylabel("Tempo (BPM)")
    plt.title("Tempo Variation Over Time with Beat Positions")
    plt.grid(alpha=0.3)
    plt.legend()

    # Second subplot: Waveform with beats
    plt.subplot(2, 1, 2)
    time = np.arange(len(y)) / sr
    plt.plot(time, y, alpha=0.7, label='Waveform', color='blue', linewidth=0.8)
    for bt in beat_times:
        plt.axvline(x=bt, color='red', alpha=0.5, linestyle='--', linewidth=1.2)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Waveform with Beat Markers")
    plt.grid(alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.show()

for j in range (1, 14):
    plot_and_listen(j)