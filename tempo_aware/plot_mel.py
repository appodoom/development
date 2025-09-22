import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import soundfile as sf


def plot_mel(
    index: int,
    mel: np.ndarray,
    sr: int = 48000,
    hop_length: int = 512,
    out_path: str = "./detected",
    is_db: bool = False,
    fmin: float = 0.0,
    fmax: float | None = None,
    title: str = "Mel Spectrogram (dB)",
) -> None:
    out_path = out_path[-1] + str(index) + ".png"
    """
    Plot and save a Mel spectrogram.

    Parameters
    ----------
    mel : np.ndarray
        Mel spectrogram matrix. Shape (n_mels, n_frames).
        If is_db=False, this is power (or magnitude^2). If is_db=True, it’s in dB.
    sr : int
        Audio sample rate used to generate `mel` (for time axis).
    hop_length : int
        Hop length used when generating `mel` (for time axis).
    out_path : str
        Where to save the image (e.g., 'mel.png').
    is_db : bool
        Set True if `mel` is already in dB. If False, will convert with power_to_db.
    fmin, fmax : float | None
        Min/max Mel (Hz) for y-axis scaling. Keep None to let librosa infer.
    title : str
        Plot title.
    """
    M_db = mel if is_db else librosa.power_to_db(mel, ref=np.max)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(
        M_db,
        x_axis="time",
        y_axis="mel",
        sr=sr,
        hop_length=hop_length,
        fmin=fmin,
        fmax=fmax,
    )
    plt.colorbar(format="%+2.0f dB", label="Amplitude (dB)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


count = 0
count = 1
hits = ["../fundemental_hits/doum.wav", "../fundemental_hits/ra.wav"]
tempo = 122
quarter_duration = 60.0 / tempo
durations = [quarter_duration / 21.33, quarter_duration / 5.34]

for path, dur in zip(hits, durations):
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
    mel = librosa.feature.melspectrogram(y=data, sr=48000)  # TODO revisit

    plot_mel(mel=mel, index=count)
    count += 1
