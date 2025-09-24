import os, re, numpy as np, joblib, librosa
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

DATA_DIR   = "../../fundamentals"
OUT_PATH   = "stroke_knn_model.joblib"
SR         = 48000
N_FFT      = 1024
HOP        = 256
WINDOW     = "hann"
LOGF_WEIGHT= True
K          = 5

def _stft_mag(x):
    # This function gets the stft of our hit
    D = librosa.stft(x, n_fft=N_FFT, hop_length=HOP, window=WINDOW, center=True)
    return np.abs(D), librosa.fft_frequencies(sr=SR, n_fft=N_FFT)

def _moments(Scol, freqs):
    #This function gets the features we want for each frame
    f = freqs
    w = Scol.astype(float)
    if LOGF_WEIGHT: w = w * np.log1p(f)
    wsum = np.sum(w)
    if wsum <= 0: return 0.0, 0.0, 0.0, 0.0
    mu  = float(np.sum(f*w)/wsum)
    var = float(np.sum(w*(f-mu)**2)/wsum)
    sd  = float(np.sqrt(max(var,1e-20)))
    skew= float(np.sum(w*(f-mu)**3)/(wsum*(sd**3)+1e-20))
    kurt= float(np.sum(w*(f-mu)**4)/(wsum*(sd**4)+1e-20))
    return mu, sd, skew, kurt

def _zcr_per_s(x, sr):
    #This function gets the zero cross rating of our hit
    if x.size < 2: return 0.0
    x = x - np.mean(x)
    s1 = x[:-1] >= 0; s2 = x[1:] >= 0
    crossings = np.count_nonzero(s1 != s2)
    return crossings * (sr/float(len(x)-1))

def _feat_vec(x):
    #This function gets the features for each frame and averages them to get the features for our
    #hit and creates the vector of features
    S, freqs = _stft_mag(x)
    if S.shape[1]==0:
        return np.zeros(6, dtype=float)
    cs, ss, sks, ks = [], [], [], []
    for i in range(S.shape[1]):
        c, s, sk, k = _moments(S[:,i], freqs)
        cs.append(c)
        ss.append(s)
        sks.append(sk)
        ks.append(k)
    centroid = float(np.mean(cs))
    spread = float(np.mean(ss))
    skew = float(np.mean(sks))
    kurt  = float(np.mean(ks))
    rms  = float(np.sqrt(np.mean(x**2))) if x.size>0 else 0.0
    zcr  = float(_zcr_per_s(x, SR))
    return np.array([centroid, spread, skew, kurt, rms, zcr], dtype=float)

def _label_from_name(fname):
    b = re.sub(r"^[\s._-]+","", os.path.basename(fname).lower())
    for p in ("pa2","doom","tak","tik"):
        if b.startswith(p): 
            return p
    return None

#Creates an array of tuples (path of wav file, label i.e doum, tek...)
pairs = []
for dp, _, fns in os.walk(DATA_DIR):
    for fn in fns:
        if fn.lower().endswith(".wav"):
            lab = _label_from_name(fn)
            if lab is not None:
                pairs.append((os.path.join(dp, fn), lab))

#Extract the features vectors for each hit and put them in X, and put the corresponding label in Y
X, y = [], []
for p, lab in pairs:
    x, _ = librosa.load(p, sr=SR, mono=True)
    if x.size==0: continue
    v = _feat_vec(x)
    if np.all(np.isfinite(v)):
        X.append(v); y.append(lab)

X = np.vstack(X); y = np.array(y)
n_neighbors = max(1, min(K, len(y)))

# Train and save the model
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance", metric="euclidean")),
])
pipe.fit(X, y)
joblib.dump(pipe, OUT_PATH)
