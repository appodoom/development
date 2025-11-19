import os
import glob
import math
import random
import numpy as np
import torch
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import librosa

DATA_DIR = "../data/pairs"
CLASSES = ["doum", "tak"]

CLASS_WEIGHTS = {
    "doum": 1.0,
    "tak": 1.0,
}

VAL_RATIO = 0.15
TEST_RATIO = 0.15
BATCH_SIZE = 16
LR = 1e-3
MAX_EPOCHS = 50
PATIENCE = 8
SEED = 42
NUM_WORKERS = 0
MODEL_PATH = "./pairs_model/pairs_model.pt"
MISCLASS_CSV = "./pairs_model/misclassified_doum_tak_pairs.csv"

# If mels have variable time length, we pad/crop to this many frames (time axis).
FIXED_MAX_FRAMES = 256  # set None to disable padding/cropping

# Audio / mel-spectrogram parameters (you can tune these)
SAMPLE_RATE = 48000
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 256
# ================================================================

# -------------------- Reproducibility --------------------
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# -------------------- Helpers for WAV pairs --------------------
def extract_leading_int_from_name(path):
    parity = path.split("_")[0]
    if parity == "even":
        return 2
    else:
        return 3


def wav_to_mel_pair(path, target_sr=SAMPLE_RATE):
    """
    Load a WAV containing TWO hits and split it into (hit1, hit2)
    by cutting waveform in half, then compute mel spectrograms.

    Returns:
        mel1: (n_mels, T1)
        mel2: (n_mels, T2)
    """
    y, sr = librosa.load(path, sr=target_sr, mono=True)

    if len(y) < 2:
        # Very short, just duplicate
        y1 = y
        y2 = y
    else:
        mid = len(y) // 2
        y1 = y[:mid]
        y2 = y[mid:]

    mel1 = librosa.feature.melspectrogram(
        y=y1, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    mel2 = librosa.feature.melspectrogram(
        y=y2, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )

    mel1 = librosa.power_to_db(mel1, ref=np.max)
    mel2 = librosa.power_to_db(mel2, ref=np.max)

    return mel1.astype(np.float32), mel2.astype(np.float32)


def pad_or_crop_time(mel, target_frames):
    """mel: (n_mels, T). Pads with zeros (right) or center-crops to target_frames."""
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


# Split dataset into training, validation, testing (same logic as before)
def stratified_split(per_class_files, val_ratio, test_ratio, seed=SEED):
    rng = random.Random(seed)
    train, val, test = [], [], []
    for c, files in per_class_files.items():
        files = files[:]  # copy
        rng.shuffle(files)
        n = len(files)
        n_test = int(round(n * test_ratio))
        n_val = int(round(n * val_ratio))
        n_train = n - n_test - n_val
        n_train = max(0, n_train)
        n_val = max(0, n_val)
        n_test = max(0, n - n_train - n_val)

        train_files = files[:n_train]
        val_files = files[n_train : n_train + n_val]
        test_files = files[n_train + n_val :]

        ci = CLASSES.index(c)
        train += [(f, ci) for f in train_files]
        val += [(f, ci) for f in val_files]
        test += [(f, ci) for f in test_files]

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def build_pairs_per_class(root_dir, classes):
    """
    Scan DATA_DIR and assign each WAV to the class of the SECOND hit
    using your rule:

    - filename starts with 'even' -> first is doum, second is tak -> label = 'tak'
    - filename starts with 'odd'  -> first is tak,  second is doum -> label = 'doum'
    """
    per_class = {c: [] for c in classes}

    wav_files = sorted(
        glob.glob(os.path.join(root_dir, "*.wav"))
        + glob.glob(os.path.join(root_dir, "*.WAV"))
    )

    print(f"[build_pairs_per_class] root_dir={os.path.abspath(root_dir)}")
    print(f"[build_pairs_per_class] found {len(wav_files)} wav files")

    if not wav_files:
        return per_class  # all empty → will be caught later

    for path in wav_files:
        base = os.path.splitext(os.path.basename(path))[0].lower()

        if base.startswith("even"):
            label_name = "tak"
        elif base.startswith("odd"):
            label_name = "doum"
        else:
            raise ValueError(
                f"Filename '{base}' does not start with 'even' or 'odd', "
                "which is required for the labeling rule."
            )

        if label_name not in per_class:
            raise ValueError(
                f"Label '{label_name}' from filename {path} not in CLASSES={classes}"
            )

        per_class[label_name].append(path)

    for c in classes:
        print(f"[build_pairs_per_class] class '{c}' has {len(per_class[c])} files")

    return per_class


# -------------------- Dataset for (prev_hit, current_hit) --------------------
class MelPairDataset(Dataset):
    """
    Each sample:
        x: (2, n_mels, T)  -> [prev_hit_mel, current_hit_mel]
        y: label index of current hit (0=doum, 1=tak)
    samples = list of (wav_path, class_idx) where class_idx is SECOND hit.
    """

    def __init__(self, samples, mean=None, std=None):
        self.samples = samples
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label_idx = self.samples[idx]

        mel1, mel2 = wav_to_mel_pair(path)
        mel1 = pad_or_crop_time(mel1, FIXED_MAX_FRAMES)
        mel2 = pad_or_crop_time(mel2, FIXED_MAX_FRAMES)

        if (self.mean is not None) and (self.std is not None):
            mel1 = (mel1 - self.mean) / (self.std + 1e-8)
            mel2 = (mel2 - self.mean) / (self.std + 1e-8)

        # stack as channels: (2, n_mels, T)
        mel_pair = np.stack([mel1, mel2], axis=0).astype(np.float32)
        x = torch.from_numpy(mel_pair)
        y = torch.tensor(label_idx, dtype=torch.long)

        return x, y, path


# -------------------- Train-set mean/std --------------------
def compute_mean_std_pairs(train_samples):
    """
    Compute global mean/std over train set of mel pairs (both hits).
    Uses Welford's algorithm over all elements of both mel1 and mel2.
    """
    count = 0
    mean = 0.0
    M2 = 0.0

    for path, _ in train_samples:
        mel1, mel2 = wav_to_mel_pair(path)
        mel1 = pad_or_crop_time(mel1, FIXED_MAX_FRAMES).astype(np.float64)
        mel2 = pad_or_crop_time(mel2, FIXED_MAX_FRAMES).astype(np.float64)

        arr = np.concatenate([mel1.ravel(), mel2.ravel()])
        for v in arr:
            count += 1
            delta = v - mean
            mean += delta / count
            M2 += delta * (v - mean)

    variance = M2 / max(1, (count - 1))
    std = math.sqrt(variance) if variance > 0 else 1.0
    return float(mean), float(std)


# -------------------- Model --------------------
class SmallAudioCNN(nn.Module):
    """
    Lightweight CNN with AdaptiveAvgPool for variable-length time axis.
    Input: (B, 2, n_mels, T)  # 2 channels: [previous_hit, current_hit]
    """

    def __init__(self, n_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),  # CHANGED: in_channels=2
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # (n_mels/2, T/2)
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # (n_mels/4, T/4)
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


# -------------------- Training / Evaluation --------------------
def train_one_epoch(model, loader, criterion, optimizer, device="cpu"):
    model.train()
    running_loss, total, correct = 0.0, 0, 0
    for xb, yb, _ in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * xb.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == yb).sum().item()
        total += xb.size(0)
    return running_loss / max(1, total), correct / max(1, total)


def evaluate(model, loader, criterion, device="cpu"):
    model.eval()
    running_loss, total, correct = 0.0, 0, 0
    all_true, all_pred = [], []
    with torch.no_grad():
        for xb, yb, _ in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            running_loss += loss.item() * xb.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += xb.size(0)
            all_true.extend(yb.cpu().numpy().tolist())
            all_pred.extend(pred.cpu().numpy().tolist())
    return (
        running_loss / max(1, total),
        correct / max(1, total),
        np.array(all_true),
        np.array(all_pred),
    )


def confusion_matrix(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def per_class_accuracy(cm):
    per_cls = []
    for i in range(cm.shape[0]):
        denom = cm[i, :].sum()
        acc = (cm[i, i] / denom) if denom > 0 else 0.0
        per_cls.append(acc)
    return per_cls


def save_misclassified(model, loader, classes, out_csv, device="cpu", mapping_csv=None):
    """
    Save misclassified samples.

    Here paths are WAV pair paths already, so if mapping_csv is None,
    we just write the same path in both columns.
    """
    mapping_dict = {}
    if mapping_csv is not None and os.path.exists(mapping_csv):
        df = pd.read_csv(mapping_csv)
        mapping_dict = dict(zip(df["mel_path"], df["wav_path"]))

    model.eval()
    rows = []
    softmax = nn.Softmax(dim=1)

    with torch.no_grad():
        for xb, yb, paths in loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = softmax(logits).cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()
            yb = yb.numpy()

            for path, t, p, pr in zip(paths, yb, preds, probs):
                if p != t:
                    wav_path = mapping_dict.get(path.replace("\\", "/"), path)
                    rows.append(
                        {
                            "pair_path": path,
                            "wav_path": wav_path,
                            "true_label": classes[t],
                            "pred_label": classes[p],
                            "confidence": float(pr[p]),
                        }
                    )

    import csv as _csv

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = _csv.DictWriter(
            f,
            fieldnames=[
                "pair_path",
                "wav_path",
                "true_label",
                "pred_label",
                "confidence",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)


# -------------------- Main --------------------
def main():
    # 1) index data from WAV pairs
    per_class_files = build_pairs_per_class(DATA_DIR, CLASSES)

    # 2) stratified split
    train_s, val_s, test_s = stratified_split(
        per_class_files, VAL_RATIO, TEST_RATIO, SEED
    )
    print(
        f"Samples (pairs) -> train: {len(train_s)}, val: {len(val_s)}, test: {len(test_s)}"
    )

    # 3) compute train mean/std for normalization
    print("Computing train mean/std over mel pairs ...")
    mean, std = compute_mean_std_pairs(train_s)
    print(f"Train mean: {mean:.4f}, std: {std:.4f}")

    # 4) datasets & loaders
    train_ds = MelPairDataset(train_s, mean=mean, std=std)
    val_ds = MelPairDataset(val_s, mean=mean, std=std)
    test_ds = MelPairDataset(test_s, mean=mean, std=std)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    # 5) model / loss / optim
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # build class-weight tensor in CLASSES order
    try:
        weight_vec = torch.tensor(
            [float(CLASS_WEIGHTS[c]) for c in CLASSES],
            dtype=torch.float32,
            device=device,
        )
    except KeyError as e:
        missing = str(e).strip("'")
        raise ValueError(
            f"CLASS_WEIGHTS is missing an entry for label '{missing}'. Provide a number for every label in CLASSES."
        ) from None

    print(
        "Class weights (in CLASSES order):",
        [float(w) for w in weight_vec.cpu().numpy()],
    )

    model = SmallAudioCNN(n_classes=len(CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight_vec)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # 6) train with early stopping
    best_val_loss = float("inf")
    best_epoch = -1
    epochs_no_improve = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        va_loss, va_acc, _, _ = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={tr_loss:.4f} acc={tr_acc:.3f} | "
            f"val_loss={va_loss:.4f} acc={va_acc:.3f}"
        )

        if va_loss < best_val_loss - 1e-6:
            best_val_loss = va_loss
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "classes": CLASSES,
                    "mean": mean,
                    "std": std,
                    "class_weights": [
                        float(w) for w in weight_vec.detach().cpu().numpy()
                    ],
                },
                MODEL_PATH,
            )
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch} (best epoch: {best_epoch})")
                break

    # 7) load best & evaluate on test
    if os.path.exists(MODEL_PATH):
        ckpt = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(ckpt["model_state"])

    te_loss, te_acc, y_true, y_pred = evaluate(model, test_loader, criterion, device)
    print(f"\nTest: loss={te_loss:.4f}  acc={te_acc:.3f}")

    # 8) confusion matrix + per-class accuracy
    cm = confusion_matrix(y_true, y_pred, num_classes=len(CLASSES))
    print("\nConfusion Matrix (rows=true, cols=pred):")
    for i, row in enumerate(cm):
        print(f"{CLASSES[i]:>6}: {row}")

    per_cls_acc = per_class_accuracy(cm)
    print("\nPer-class accuracy:")
    for c, a in zip(CLASSES, per_cls_acc):
        print(f"{c:>6}: {a:.3f}")

    # 9) save misclassified
    n_mis = save_misclassified(model, test_loader, CLASSES, MISCLASS_CSV, device)
    print(f"\nMisclassified examples saved to {MISCLASS_CSV} (count={n_mis})")


if __name__ == "__main__":
    main()
