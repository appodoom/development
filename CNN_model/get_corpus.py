import librosa
import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# def get_amp_bin_values(y, onsets, amp_bin_values):
#     onset_amps = np.abs(y[onsets - 1])
#     onset_amps = onset_amps / np.max(onset_amps)
#     bin_indices = [int(np.argmin(np.abs(amp_bin_values - amp))) for amp in onset_amps]
#     return bin_indices


# def get_closest_amplitude_bin(y, amplitude_bins, mode="rms", normalize=True, eps=1e-12):
#     """
#     For a single onset waveform y, compute its amplitude and return
#     the closest value from amp_bin_values.

#     Parameters
#     ----------
#     y : np.ndarray
#         Audio samples for one onset (1D).
#     amp_bin_values : list/np.ndarray
#         Candidate amplitude bins (typically in [0, 1]).
#     mode : {'rms', 'peak'}
#         How to measure amplitude. 'rms' is more robust; 'peak' is max |y|.
#     normalize : bool
#         If True, peak-normalize y before measuring amplitude (recommended
#         if bins are on [0,1]).
#     eps : float
#         Tiny constant to avoid divide-by-zero on silent segments.

#     Returns
#     -------
#     float
#         The closest amplitude **bin value**.
#     """
#     y = np.asarray(y, dtype=float)
#     bins = np.asarray(amplitude_bins, dtype=float)

#     if normalize:
#         peak = np.max(np.abs(y))
#         if peak > eps:
#             y = y / peak  # bring into [−1, 1]

#     if mode == "rms":
#         amp = np.sqrt(np.mean(y**2))
#     elif mode == "peak":
#         amp = np.max(np.abs(y))
#     else:
#         raise ValueError("mode must be 'rms' or 'peak'")

#     idx = int(np.argmin(np.abs(bins - amp)))
#     return float(bins[idx]), amp


class SmallAudioCNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


CKPT_PATH = "best_model_finetuned.pt"
ckpt = torch.load(CKPT_PATH, map_location="cpu")

SAMPLE_RATE = 48000
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 64
TARGET_FRAMES = 256
CLASSES = ckpt["classes"]
mean = float(ckpt["mean"])
std = float(ckpt["std"])

MODEL = SmallAudioCNN(n_classes=len(CLASSES))
MODEL.load_state_dict(ckpt["model_state"])
MODEL.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL.to(device)


def pad_or_crop_time(mel: np.ndarray, target_frames: int) -> np.ndarray:
    if target_frames is None:
        return mel
    T = mel.shape[1]
    if T == target_frames:
        return mel
    if T < target_frames:
        pad = target_frames - T
        return np.pad(mel, ((0, 0), (0, pad)), mode="constant")
    start = (T - target_frames) // 2
    end = start + target_frames
    return mel[:, start:end]


def get_mel(y, sr, target_frames: int = TARGET_FRAMES) -> torch.Tensor:
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)  # shape: (n_mels, T)
    mel_db = pad_or_crop_time(mel_db, target_frames)
    mel_db = (mel_db - mean) / (std + 1e-8)
    mel_t = torch.from_numpy(mel_db.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    return mel_t.to(device)


@torch.no_grad()
def predict_hit(model: nn.Module, y, sr):
    xb = get_mel(y=y, sr=sr)
    logits = model(xb)
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred_idx = int(np.argmax(probs))
    return CLASSES[pred_idx], probs


def get_note_duration(quarter_duration):
    return {
        16: quarter_duration / 4,
        8: quarter_duration / 2,
        4: quarter_duration,
        2: quarter_duration * 2,
        1: quarter_duration * 4,
        24: quarter_duration / 6,
        32: quarter_duration / 8,
        12: quarter_duration / 3,
        2.67: quarter_duration * (3 / 2),
        5.34: quarter_duration * (3 / 4),
        10.67: quarter_duration * (3 / 8),
        21.33: quarter_duration * (3 / 16),
        1.33: quarter_duration * (3),
        # 28:quarter_duration*(1/7),
        # 56:quarter_duration*(1/14),
        # 14:quarter_duration*(2/7),
        # 9.33:quarter_duration*(3/7),
        # 7:quarter_duration*(4/7),
        # 5.6:quarter_duration*(5/7),
        # 4.67:quarter_duration*(6/7),
        # 20:quarter_duration*(1/5),
        # 40:quarter_duration*(1/10),
        # 10:quarter_duration*(2/5),
        # 6.67:quarter_duration*(3/5),
        # 5:quarter_duration*(4/5),
        # 36:quarter_duration*(1/9),
        # 72:quarter_duration*(1/18),
        # 18:quarter_duration*(2/9),
        # 9:quarter_duration*(4/9),
        # 7.2:quarter_duration*(5/9),
        # 6:quarter_duration*(6/9),
        # 5.1:quarter_duration*(7/9),
        # 4.5:quarter_duration*(8/9)
    }


def sliding_cross_correlation(X, Y):
    if X.shape[1] > Y.shape[1]:
        X, Y = Y, X

    n_freq, n_time_X = X.shape
    _, n_time_Y = Y.shape

    best_score = -np.inf
    best_offset = None
    norm_X = np.linalg.norm(X)
    for offset in range(n_time_Y - n_time_X + 1):
        Y_slice = Y[:, offset : offset + n_time_X]
        if norm_X != 0 and np.linalg.norm(Y_slice) != 0:
            score = np.tensordot(X, Y_slice, axes=2) / (
                norm_X * np.linalg.norm(Y_slice)
            )
        else:
            continue

        if score > best_score:
            best_score = score
            best_offset = offset

    return best_score, best_offset


def load_json(file_path):
    with open(file_path, "r") as f:
        mels = json.load(f)
    return {name: np.array(mel) for name, mel in mels.items()}


def load_file(file_path):
    y, sr = librosa.load(file_path, sr=None)
    y = librosa.util.normalize(y)
    return y, sr


def adjust_tempo(y, sr):
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    # while tempo > 150:
    #     tempo = tempo // 2
    # while tempo < 50:
    #     tempo *= 2
    print(f"tempo : {tempo}")
    return tempo


def get_onsets(y, sr, hop_length=512):
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop_length)

    if len(onset_frames) == 0:
        n_frames = 1 + (len(y) // hop_length)
        return librosa.frames_to_samples([0, n_frames], hop_length=hop_length)

    if len(onset_frames) == 1:
        n_frames = 1 + (len(y) // hop_length)
        return librosa.frames_to_samples(
            [0, onset_frames[0], n_frames], hop_length=hop_length
        )
    mids = [
        (onset_frames[i] + onset_frames[i + 1]) // 2
        for i in range(len(onset_frames) - 1)
    ]
    n_frames = int(np.ceil(len(y) / hop_length))
    boundaries_frames = [0] + mids + [n_frames]
    return librosa.frames_to_samples(boundaries_frames, hop_length=hop_length)


def get_intervals(y, sr, hop_length=512):
    boundaries = get_onsets(y, sr, hop_length=hop_length)  # in samples
    intervals = [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]
    return intervals


def get_intervals_for_duration(y, sr, hop_length=512):
    boundaries = librosa.onset.onset_detect(
        y=y, sr=sr, hop_length=hop_length, units="samples"
    )
    intervals = [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]
    intervals.append((boundaries[-1], len(y)))
    print("onset_duration intervals", len(intervals))
    return intervals


def plot_mel(
    index: int,
    mel: np.ndarray,
    sr: int = 48000,
    hop_length: int = 512,
    out_path: str = "./mel1.png",
    is_db: bool = False,
    fmin: float = 0.0,
    fmax: float | None = None,
    title: str = "Mel Spectrogram (dB)",
) -> None:
    out_path = out_path[-1] + str(index)
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


def get_corpus(fundamentals_path, file_path, model_pred, log_mel):
    y, sr = load_file(file_path)
    y = librosa.resample(y, orig_sr=sr, target_sr=48000)
    sr = 48000
    y = np.concatenate([np.zeros(48000), y])
    fundamentals = load_json(fundamentals_path)
    intervals_duration = get_intervals_for_duration(y, sr)
    intervals = get_intervals(y, sr)
    tempo = adjust_tempo(y, sr)
    quarter_duration = 60.0 / tempo  # in seconds
    note_durations = get_note_duration(quarter_duration)  # in seconds
    classified_hits = []
    # count = 1

    if model_pred:
        for i, (interval, interval_dur) in enumerate(
            zip(intervals, intervals_duration)
        ):
            segment = y[interval[0] : interval[1]]
            mel = librosa.feature.melspectrogram(y=segment, sr=sr)  # TODO revisit
            if log_mel:
                mel = librosa.power_to_db(mel, ref=np.max, top_db=80)
            # if count == 9 or count == 8 or count == 2:
            #     plot_mel(index=count, mel=mel)
            best_hit, _ = predict_hit(model=MODEL, y=segment, sr=sr)  # _ : probs
            # print(f"Choosen fundemental is: {best_hit} with max corr score = {best_score}")
            # print("")
            hit_duration = (interval_dur[1] - interval_dur[0]) / sr  # in seconds
            # best_amp_bin = str(
            #     get_closest_amplitude_bin(y=segment, amplitude_bins=amp_bin_values)
            # )
            min_diff = np.inf
            best_note = ""
            for note in note_durations:
                diff = abs(note_durations[note] - hit_duration)
                if diff < min_diff:
                    min_diff = diff
                    best_note = str(note)

            classified_hits.append(best_hit + "_" + best_note)
            # count += 1

    else:
        for i, (interval, interval_dur) in enumerate(
            zip(intervals, intervals_duration)
        ):
            segment = y[interval[0] : interval[1]]
            mel = librosa.feature.melspectrogram(y=segment, sr=sr)  # TODO revisit

            best_score = -np.inf
            best_hit = ""
            for fundamental_hit in fundamentals:
                score, _ = sliding_cross_correlation(mel, fundamentals[fundamental_hit])
                # print(
                #     f"Current fundemental is: {fundamental_hit} with corr score = {score}"
                # )
                if score > best_score:
                    best_score = score
                    best_hit = fundamental_hit
            # print(f"Choosen fundemental is: {best_hit} with max corr score = {best_score}")
            # print("")
            hit_duration = (interval_dur[1] - interval_dur[0]) / sr  # in seconds
            min_diff = np.inf
            best_note = ""
            # best_amp_bin = str(
            #     get_closest_amplitude_bin(y=segment, amplitude_bins=amp_bin_values)
            # )
            for note in note_durations:
                diff = abs(note_durations[note] - hit_duration)
                if diff < min_diff:
                    min_diff = diff
                    best_note = str(note)
            classified_hits.append(best_hit + "_" + best_note)

    return classified_hits, tempo


# classified_hits_with_model_pred, _ = get_corpus(
#     fundamentals_path="../mel.json",
#     file_path="../data/first_data/old1.wav",
#     model_pred=True,
#     log_mel=False,
# )
# classified_hits_sliding_window, _ = get_corpus(
#     fundamentals_path="../mel.json",
#     file_path="../data/first_data/old1.wav",
#     model_pred=False,
#     log_mel=False,
# )

# print(classified_hits_with_model_pred)
# print(classified_hits_sliding_window)
# print("")
# classified_hits_old, _ = get_corpus_old(
#     fundamentals_path="../mel.json", file_path="../first_data/old1.wav"
# )
# print("")


# print("old approach : ", classified_hits_old)
# print("")
# print("new approach: ", classified_hits)
