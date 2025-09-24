import numpy as np
import joblib
import librosa
from typing import List

MODEL_PATH   = "stroke_knn_model.joblib"
SR           = 48000
POST_MS      = 50.0        
N_FFT        = 1024
HOP          = 256
WINDOW       = "hann"
LOGF_WEIGHT  = True        
MIN_GAP_MS   = 80.0


def _stft_mag(x: np.ndarray):
    D = librosa.stft(x, n_fft=N_FFT, hop_length=HOP, window=WINDOW, center=True)
    return np.abs(D), librosa.fft_frequencies(sr=SR, n_fft=N_FFT)

def _moments(Scol: np.ndarray, freqs: np.ndarray):
    f = freqs
    w = Scol.astype(float)
    if LOGF_WEIGHT:
        w = w * np.log1p(f)
    wsum = float(np.sum(w))
    if wsum <= 0.0:
        return 0.0, 0.0, 0.0, 0.0
    mu   = float(np.sum(f*w) / wsum)
    var  = float(np.sum(w * (f - mu) ** 2) / wsum)
    sd   = float(np.sqrt(max(var, 1e-20)))
    skew = float(np.sum(w * (f - mu) ** 3) / (wsum * (sd**3) + 1e-20))
    kurt = float(np.sum(w * (f - mu) ** 4) / (wsum * (sd**4) + 1e-20))
    return mu, sd, skew, kurt

def _zcr_per_s(x: np.ndarray, sr: int) -> float:
    if x.size < 2:
        return 0.0
    x = x - np.mean(x)
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
    rms      = float(np.sqrt(np.mean(x**2))) if x.size > 0 else 0.0
    zcr      = float(_zcr_per_s(x, SR))
    return np.array([centroid, spread, skew, kurt, rms, zcr], dtype=float)

def classify_onsets(wav_path: str) -> List[str]:
    pipe = pipe = joblib.load(MODEL_PATH)
    x, _ = librosa.load(wav_path, sr=SR, mono=True)
    if x.size == 0:
        return []

    wait_frames = int(round((MIN_GAP_MS / 1000.0) * SR / HOP))
    onsets = librosa.onset.onset_detect(
            y=x, sr=SR, hop_length=HOP, backtrack=True, wait=wait_frames, units="samples"
        )

    if onsets is None or len(onsets) == 0:
        return []

    seg_len = int(POST_MS * SR / 1000.0)
    feats = []
    for oi in onsets:
        start = int(max(0, oi))
        end   = int(min(len(x), start + seg_len))
        seg   = x[start:end]
        if len(seg) < seg_len:
            seg = np.pad(seg, (0, seg_len - len(seg)))
        feats.append(_feat_vec(seg))

    if not feats:
        return []
    X = np.vstack(feats)
    preds = pipe.predict(X)
    return list(preds)
