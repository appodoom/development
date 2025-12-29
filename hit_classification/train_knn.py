import os
import re
import shutil
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
DATA_DIR       = "../../data/Tracks_with_onsets"
OUT_PATH       = "stroke_knn_model.joblib"
OUT_ERRORS_DIR = "../../data/misclassified_by_pred"   # <--- added: root folder for misclassified files
SR             = 48000
N_FFT          = 1024
HOP            = 256
WINDOW         = "hann"
LOGF_WEIGHT    = True
K              = 1
VAL_RATIO      = 0.20           # per-class 80/20 split
RNG_SEED       = 39874839
LABELS         = ("PAA", "doum", "tak", "tik")

# Type aliases
Pair = Tuple[str, str]
PerLabelStats = Dict[str, Tuple[float, int, int]]  # label -> (accuracy, correct, total)


def _stft_mag(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    D = librosa.stft(x, n_fft=N_FFT, hop_length=HOP, window=WINDOW, center=True)
    return np.abs(D), librosa.fft_frequencies(sr=SR, n_fft=N_FFT)


def _moments(Scol: np.ndarray, freqs: np.ndarray) -> Tuple[float, float, float, float]:
    f = freqs
    w = Scol.astype(float)
    if LOGF_WEIGHT:
        w = w * np.log1p(f)

    wsum = float(np.sum(w))
    if wsum <= 0:
        return 0.0, 0.0, 0.0, 0.0

    mu  = float(np.sum(f * w) / wsum)
    var = float(np.sum(w * (f - mu) ** 2) / wsum)
    sd  = float(np.sqrt(max(var, 1e-20)))
    skew = float(np.sum(w * (f - mu) ** 3) / (wsum * (sd ** 3) + 1e-20))
    kurt = float(np.sum(w * (f - mu) ** 4) / (wsum * (sd ** 4) + 1e-20))
    return mu, sd, skew, kurt


def _zcr_per_s(x: np.ndarray, sr: int) -> float:
    if x.size < 2:
        return 0.0
    x = x - float(np.mean(x))
    s1 = x[:-1] >= 0
    s2 = x[1:]  >= 0
    crossings = int(np.count_nonzero(s1 != s2))
    return crossings * (sr / float(len(x) - 1))


def _feat_vec(x: np.ndarray) -> np.ndarray:
    S, freqs = _stft_mag(x)
    if S.shape[1] == 0:
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
    b = re.sub(r"^[\s._-]+", "", os.path.basename(fname).lower())
    for p in LABELS:
        if b.startswith(p):
            return p
    return None


def split_train_val_per_class(
    pairs: List[Pair], val_ratio: float = 0.2, rng_seed: int = 42
) -> Tuple[List[Pair], List[Pair]]:
    by_lab: Dict[str, List[str]] = defaultdict(list)
    for p, lab in pairs:
        by_lab[lab].append(p)

    rng = np.random.default_rng(rng_seed)
    train_pairs: List[Pair] = []
    val_pairs: List[Pair] = []

    for lab, files in by_lab.items():
        lst = list(files)
        rng.shuffle(lst)
        n = len(lst)

        n_train = int(np.floor((1.0 - val_ratio) * n))
        if n >= 2:
            n_train = max(1, min(n - 1, n_train))
        else:
            n_train = 1

        tr, va = lst[:n_train], lst[n_train:]
        train_pairs.extend((p, lab) for p in tr)
        val_pairs.extend((p, lab) for p in va)

    return train_pairs, val_pairs


def _extract_features(p: str) -> Optional[np.ndarray]:
    x, _ = librosa.load(p, sr=SR, mono=True)
    if x.size == 0:
        return None
    v = _feat_vec(x)
    return v if np.all(np.isfinite(v)) else None


def per_label_accuracy(
    y_true: np.ndarray, y_pred: np.ndarray
) -> PerLabelStats:
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
# FN breakdown helpers
# -------------------
def false_negative_breakdown(y_true: np.ndarray, y_pred: np.ndarray, labels=LABELS):
    breakdown = {}
    for L in labels:
        mask_L = (y_true == L)
        mask_wrong = mask_L & (y_pred != L)
        total_wrong = int(np.sum(mask_wrong))
        per_pred = {}
        for P in labels:
            if P == L:
                continue
            count = int(np.sum(mask_L & (y_pred == P)))
            pct = (100.0 * count / total_wrong) if total_wrong > 0 else 0.0
            per_pred[P] = (pct, count, total_wrong)
        breakdown[L] = per_pred
    return breakdown


def print_false_negative_breakdown(y_true: np.ndarray, y_pred: np.ndarray, labels=LABELS):
    print("\nFalse-negative breakdown (percent of errors per true label):")
    b = false_negative_breakdown(y_true, y_pred, labels)
    for L in labels:
        entries = b[L]
        totals = [tot for (_, _, tot) in entries.values()] or [0]
        total_wrong = totals[0]
        if total_wrong == 0:
            print(f"{L}: perfect (no misclassifications)")
            continue
        for P in labels:
            if P == L:
                continue
            pct, cnt, _ = entries[P]
            print(f"{L} misclassified as {P}: {pct:.2f}%  ({cnt}/{total_wrong})")


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
# Split
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
val_kept_paths: List[str] = []   # <--- added: keep the path for each kept val sample
for p, lab in val_paths:
    v = _extract_features(p)
    if v is not None:
        X_val_list.append(v)
        y_val_list.append(lab)
        val_kept_paths.append(p)  # <--- added

if not X_train_list or not X_val_list:
    raise RuntimeError("Empty train or validation set after feature extraction.")

X_train = np.vstack(X_train_list)
y_train = np.array(y_train_list)
X_val   = np.vstack(X_val_list)
y_val   = np.array(y_val_list)

# -------------------
# Train
# -------------------
n_neighbors = max(1, min(K, len(y_train)))
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance", metric="euclidean")),
])
pipe.fit(X_train, y_train)
joblib.dump(pipe, OUT_PATH)

# -------------------
# Evaluate: overall + per-label accuracy + FN breakdown
# -------------------
y_pred = pipe.predict(X_val)
overall_acc: float = accuracy_score(y_val, y_pred)

print(f"overall: {overall_acc:.6f}")

stats = per_label_accuracy(y_val, y_pred)
for lab in sorted(stats.keys()):
    acc, correct, total = stats[lab]
    print(f"{lab}: {acc:.6f}  ({correct}/{total})")

print_false_negative_breakdown(y_val, y_pred, LABELS)

# -------------------
# Copy misclassified files into folders named by PREDICTED label
# Example: if tak009 -> predicted 'tik', copy to misclassified_by_pred/tik/tak009.wav
# -------------------
os.makedirs(OUT_ERRORS_DIR, exist_ok=True)
for lab in LABELS:
    os.makedirs(os.path.join(OUT_ERRORS_DIR, lab), exist_ok=True)

mis_counts = {lab: 0 for lab in LABELS}
for true_lab, pred_lab, src_path in zip(y_val, y_pred, val_kept_paths):
    if pred_lab != true_lab and pred_lab in LABELS:
        dst_path = os.path.join(OUT_ERRORS_DIR, pred_lab, os.path.basename(src_path))
        try:
            shutil.copy2(src_path, dst_path)
            mis_counts[pred_lab] += 1
        except Exception as e:
            # keep going on copy errors
            print(f"[warn] could not copy {src_path} -> {dst_path}: {e}")

print("\nMisclassified files copied to:", OUT_ERRORS_DIR)
for lab in LABELS:
    print(f"  {lab}: {mis_counts[lab]} files")
