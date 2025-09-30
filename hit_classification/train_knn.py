import os
import re
import numpy as np
import joblib
import librosa
from typing import List, Tuple, Dict, Optional
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# -------------------
# Configuration
# -------------------
DATA_DIR     = "../../fundamentals"
OUT_PATH     = "stroke_knn_model.joblib"
SR           = 48000
N_FFT        = 1024
HOP          = 256
WINDOW       = "hann"
LOGF_WEIGHT  = True
K            = 1
VAL_RATIO    = 0.20           # per-class 80/20 split
RNG_SEED     = 348392675849320
LABELS=("pa2", "tak", "tik")

# Type aliases
Pair = Tuple[str, str]
PerLabelStats = Dict[str, Tuple[float, int, int]]  # label -> (accuracy, correct, total)


def _stft_mag(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute magnitude STFT and the corresponding frequency axis.

    Args:
        x: 1D audio waveform (mono), shape (n_samples,).

    Returns:
        (S_mag, freqs):
            - S_mag: |STFT|, shape (n_freq_bins, n_frames).
            - freqs: center frequencies of STFT bins, shape (n_freq_bins,).
    """
    D = librosa.stft(x, n_fft=N_FFT, hop_length=HOP, window=WINDOW, center=True)
    return np.abs(D), librosa.fft_frequencies(sr=SR, n_fft=N_FFT)


def _moments(Scol: np.ndarray, freqs: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Compute spectral moments for a single STFT column (frame).

    Args:
        Scol: magnitude spectrum at one frame, shape (n_freq_bins,).
        freqs: frequency bins in Hz, shape (n_freq_bins,).

    Returns:
        (centroid, spread, skewness, kurtosis).
    """
    f = freqs
    w = Scol.astype(float)
    if LOGF_WEIGHT:
        w = w * np.log1p(f)

    wsum = float(np.sum(w))
    if wsum <= 0:
        return 0.0, 0.0, 0.0, 0.0

    mu  = float(np.sum(f * w) / wsum)
    var = float(np.sum(w * (f - mu) ** 2) / wsum)
    sd  = float(np.sqrt(max(var, 1e-20)))  # numerical guard
    skew = float(np.sum(w * (f - mu) ** 3) / (wsum * (sd ** 3) + 1e-20))
    kurt = float(np.sum(w * (f - mu) ** 4) / (wsum * (sd ** 4) + 1e-20))
    return mu, sd, skew, kurt


def _zcr_per_s(x: np.ndarray, sr: int) -> float:
    """
    Zero-crossing rate per second (sign changes per second).

    Args:
        x: 1D audio waveform (mono), shape (n_samples,).
        sr: sampling rate in Hz.

    Returns:
        Zero-crossing rate measured in crossings per second.
    """
    if x.size < 2:
        return 0.0
    x = x - float(np.mean(x))  # remove DC
    s1 = x[:-1] >= 0
    s2 = x[1:]  >= 0
    crossings = int(np.count_nonzero(s1 != s2))
    return crossings * (sr / float(len(x) - 1))


def _feat_vec(x: np.ndarray) -> np.ndarray:
    """
    Compute feature vector for one audio clip:
    [spectral centroid, spread, skewness, kurtosis, RMS, ZCR].

    Args:
        x: 1D audio waveform (mono), shape (n_samples,).

    Returns:
        Feature vector of shape (6,) as float64.
    """
    S, freqs = _stft_mag(x)
    if S.shape[1] == 0:  # very short audio
        return np.zeros(6, dtype=float)

    cs, ss, sks, ks = [], [], [], []
    for i in range(S.shape[1]):
        c, s, sk, k = _moments(S[:, i], freqs)
        cs.append(c); ss.append(s); sks.append(sk); ks.append(k)

    centroid = float(np.mean(cs))
    spread   = float(np.mean(ss))
    skew     = float(np.mean(sks))
    kurt     = float(np.mean(ks))
    rms      = float(np.sqrt(np.mean(x ** 2))) if x.size > 0 else 0.0
    zcr      = float(_zcr_per_s(x, SR))
    return np.array([centroid, spread, skew, kurt, rms, zcr], dtype=float)


def _label_from_name(fname: str) -> Optional[str]:
    """
    Parse a filename to infer its label (one of: 'pa2', 'doum', 'tak', 'tik').

    Args:
        fname: file name or path; only the basename is considered.

    Returns:
        The label string if recognized, otherwise None.
    """
    b = re.sub(r"^[\s._-]+", "", os.path.basename(fname).lower())
    for p in LABELS:
        if b.startswith(p):
            return p
    return None


def split_train_val_per_class(
    pairs: List[Pair], val_ratio: float = 0.2, rng_seed: int = 42
) -> Tuple[List[Pair], List[Pair]]:
    """
    Split (filepath,label) pairs into train/val with a per-class ratio.

    Args:
        pairs: list of (filepath, label) pairs.
        val_ratio: fraction per class to allocate to validation (e.g., 0.2 => 80/20).
        rng_seed: random seed for reproducible shuffling.

    Returns:
        (train_pairs, val_pairs) where each is a list of (filepath, label).
        Safeguards avoid empty splits when class size is tiny.
    """
    by_lab: Dict[str, List[str]] = defaultdict(list)
    for p, lab in pairs:
        by_lab[lab].append(p)

    rng = np.random.default_rng(rng_seed)
    train_pairs: List[Pair] = []
    val_pairs: List[Pair] = []

    for lab, files in by_lab.items():
        lst = list(files)
        rng.shuffle(lst)  # shuffle within class
        n = len(lst)

        # Nominal train size (floor); clamp to keep both non-empty when possible
        n_train = int(np.floor((1.0 - val_ratio) * n))
        if n >= 2:
            n_train = max(1, min(n - 1, n_train))
        else:
            n_train = 1  # only one sample in class -> must be train

        tr, va = lst[:n_train], lst[n_train:]
        train_pairs.extend((p, lab) for p in tr)
        val_pairs.extend((p, lab) for p in va)

    return train_pairs, val_pairs


def _extract_features(p: str) -> Optional[np.ndarray]:
    """
    Load an audio file and compute its 6-D feature vector.

    Args:
        p: path to a .wav file.

    Returns:
        Feature vector of shape (6,) if successful and finite; otherwise None.
    """
    x, _ = librosa.load(p, sr=SR, mono=True)
    if x.size == 0:
        return None
    v = _feat_vec(x)
    return v if np.all(np.isfinite(v)) else None


def per_label_accuracy(
    y_true: np.ndarray, y_pred: np.ndarray
) -> PerLabelStats:
    """
    Compute accuracy for each label present in y_true.

    Args:
        y_true: ground-truth labels, shape (N,).
        y_pred: predicted labels, shape (N,).

    Returns:
        Dict mapping label -> (accuracy_in_[0,1], correct_count, total_count).
    """
    stats: PerLabelStats = {}
    labels = np.unique(y_true)
    for lab in labels:
        mask = (y_true == lab)
        total = int(np.sum(mask))
        correct = int(np.sum(y_pred[mask] == lab))
        acc = float(correct) / float(total) if total > 0 else float("nan")
        stats[str(lab)] = (acc, correct, total)
    return stats


# -------------------
# Gather files
# -------------------
pairs: List[Pair] = []
for dp, _, fns in os.walk(DATA_DIR):
    for fn in fns:
        if fn.lower().endswith(".wav"):
            lab = _label_from_name(fn)
            if lab is not None:
                pairs.append((os.path.join(dp, fn), lab))

if not pairs:
    raise RuntimeError(f"No labeled .wav files found under {DATA_DIR}")

# -------------------
# Split using the function
# -------------------
train_paths, val_paths = split_train_val_per_class(
    pairs, val_ratio=VAL_RATIO, rng_seed=RNG_SEED
)

# -------------------
# Feature extraction
# -------------------
X_train_list: List[np.ndarray] = []
y_train_list: List[str] = []
for p, lab in train_paths:
    v = _extract_features(p)
    if v is not None:
        X_train_list.append(v)
        y_train_list.append(lab)

X_val_list: List[np.ndarray] = []
y_val_list: List[str] = []
for p, lab in val_paths:
    v = _extract_features(p)
    if v is not None:
        X_val_list.append(v)
        y_val_list.append(lab)

if not X_train_list or not X_val_list:
    raise RuntimeError("Empty train or validation set after feature extraction.")

X_train = np.vstack(X_train_list)
y_train = np.array(y_train_list)
X_val   = np.vstack(X_val_list)
y_val   = np.array(y_val_list)

# -------------------
# Train
# -------------------
n_neighbors = max(1, min(K, len(y_train)))  # cap K to #train samples
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance", metric="euclidean")),
])
pipe.fit(X_train, y_train)
joblib.dump(pipe, OUT_PATH)

# -------------------
# Evaluate: overall + per-label accuracy
# -------------------
y_pred = pipe.predict(X_val)
overall_acc: float = accuracy_score(y_val, y_pred)

# Print overall (0..1)
print(f"overall: {overall_acc:.6f}")

# Print per-label (accuracy and counts)
stats = per_label_accuracy(y_val, y_pred)
for lab in sorted(stats.keys()):
    acc, correct, total = stats[lab]
    print(f"{lab}: {acc:.6f}  ({correct}/{total})")
