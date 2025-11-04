import os
import glob
import math
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import librosa

# ======================= CONFIG =======================
CHECKPOINT_PATH = "best_model.pt"  # your existing .pt
NEW_WAV_DIR = "../../data/interference_data"  # new data root (subfolders = class names)
FINETUNED_PATH = "best_model_finetuned.pt"

VAL_RATIO = 0.15
TEST_RATIO = 0.15
BATCH_SIZE = 16
LR_FEATURES = 2e-4  # lower LR for conv "feature extractor"
LR_HEAD = 1e-3  # higher LR for classifier head
MAX_EPOCHS = 70
PATIENCE = 5
SEED = 42
NUM_WORKERS = 0

# Keep the same time cap you used before
FIXED_MAX_FRAMES = 256  # set None to disable pad/crop

# Mel settings — try to match original preprocessing
SR = 48000  # sampling rate
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 128
FMIN = 30
FMAX = None  # set (e.g., 8000) if you used a capped band

# Normalization strategy for new data:
# - True  => use mean/std from checkpoint (recommended for continuity)
# - False => recompute on new train split before fine-tune
USE_CHECKPOINT_NORM = True

# If you want to save misclassified in test set
MISCLASS_CSV = "misclassified_finetune.csv"

# ---- NEW: Per-label class weights override (order: [doum, tak, tik, pa2]) ----
# Set to None to use the weights stored in the checkpoint.
WEIGHTS = [1,1,1,1]  # e.g., WEIGHTS = [2.0, 1.0, 1.5, 3.0]
# ======================================================


# -------------------- Utils --------------------
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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


def melspec_from_wav(path):
    y, sr = librosa.load(path, sr=None, mono=True)
    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        power=2.0,  # power mel
    )
    S_db = librosa.power_to_db(S, ref=np.max)  # log-mel in dB
    return S_db.astype(np.float32)  # (n_mels, T)


def list_wavs_per_class(root_dir, allowed_classes):
    per_class = {}
    for c in allowed_classes:
        class_dir = os.path.join(root_dir, c)
        files = sorted(glob.glob(os.path.join(class_dir, "*.wav")))
        per_class[c] = files
    return per_class


def stratified_split(
    per_class_files, val_ratio, test_ratio, seed=SEED, class_to_idx=None
):
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

        if class_to_idx is None:
            raise ValueError("class_to_idx mapping must be provided")
        ci = class_to_idx[c]

        train += [(f, ci) for f in train_files]
        val += [(f, ci) for f in val_files]
        test += [(f, ci) for f in test_files]

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


# -------------------- Dataset --------------------
class WaveToMelDataset(Dataset):
    """
    Loads WAVs, converts to log-mel (dB), normalizes with mean/std,
    pads/crops time to FIXED_MAX_FRAMES.
    """

    def __init__(self, samples, mean=None, std=None):
        self.samples = samples
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        mel = melspec_from_wav(path)  # (n_mels, T)
        mel = pad_or_crop_time(mel, FIXED_MAX_FRAMES)
        if (self.mean is not None) and (self.std is not None):
            mel = (mel - self.mean) / (self.std + 1e-8)
        mel = torch.from_numpy(mel).unsqueeze(0)  # (1, n_mels, T)
        label = torch.tensor(label, dtype=torch.long)
        return mel, label, path


def compute_mean_std_over_mels(samples):
    """
    If you choose to recompute mean/std on new train split.
    Uses Welford's algorithm across ALL elements.
    """
    count = 0
    mean = 0.0
    M2 = 0.0
    for path, _ in samples:
        mel = melspec_from_wav(path).astype(np.float64)
        mel = pad_or_crop_time(mel, FIXED_MAX_FRAMES)
        x = mel.ravel()
        for v in x:
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
    Must match the architecture used in pretraining.
    Input: (B, 1, n_mels, T)
    """

    def __init__(self, n_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
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
            nn.Linear(64, 4),  # <-- keep 4 outputs (doum,tak,tik,pa2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


# -------------------- Train / Eval --------------------
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


@torch.no_grad()
def evaluate(model, loader, criterion, device="cpu"):
    model.eval()
    running_loss, total, correct = 0.0, 0, 0
    all_true, all_pred = [], []
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


@torch.no_grad()
def save_misclassified(model, loader, classes, out_csv, device="cpu"):
    model.eval()
    rows = []
    softmax = nn.Softmax(dim=1)
    for xb, yb, paths in loader:
        xb = xb.to(device)
        logits = model(xb)
        probs = softmax(logits).cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()
        yb = yb.numpy()
        for path, t, p, pr in zip(paths, yb, preds, probs):
            if p != t:
                rows.append(
                    {
                        "wav_path": path,
                        "true_label": classes[t],
                        "pred_label": classes[p],
                        "confidence": float(pr[p]),
                    }
                )
    import csv as _csv

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = _csv.DictWriter(
            f,
            fieldnames=["wav_path", "true_label", "pred_label", "confidence"],
        )
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


# -------------------- Fine-tune --------------------
def main():
    set_seed()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Load checkpoint (model state + metadata)
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)

    # classes & weights from checkpoint (fallback to defaults)
    ckpt_classes = ckpt.get("classes", ["doum", "tak", "tik", "pa2"])
    if "class_weights" in ckpt:
        ckpt_class_weights = [float(x) for x in ckpt["class_weights"]]
    else:
        # neutral weights if not saved
        ckpt_class_weights = [1.0] * len(ckpt_classes)

    # mean/std for normalization
    ckpt_mean = ckpt.get("mean", 0.0)
    ckpt_std = ckpt.get("std", 1.0)

    print("Checkpoint classes:", ckpt_classes)
    print("Checkpoint mean/std:", ckpt_mean, ckpt_std)
    print("Checkpoint class weights (from ckpt):", ckpt_class_weights)

    # ---- NEW: OPTIONAL override of class weights using global WEIGHTS ----
    if WEIGHTS is not None:
        if len(WEIGHTS) != 4:
            raise ValueError("WEIGHTS must have 4 values for [doum, tak, tik, pa2].")
        provided_order = ["doum", "tak", "tik", "pa2"]
        label2w = {k: float(v) for k, v in zip(provided_order, WEIGHTS)}
        ckpt_class_weights = [label2w.get(c, 1.0) for c in ckpt_classes]
        print("Overriding class weights with WEIGHTS (reordered to ckpt classes):",
              ckpt_class_weights)

    # 2) Build model and load weights
    num_classes = len(ckpt_classes)  # should be 4
    model = SmallAudioCNN(n_classes=num_classes).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)

    # 3) Prepare NEW dataset (WAVs)
    # Only use class folders that exist in NEW_WAV_DIR and are known to the checkpoint
    present_classes = [
        c for c in ckpt_classes if os.path.isdir(os.path.join(NEW_WAV_DIR, c))
    ]
    if not present_classes:
        raise RuntimeError(
            f"No class folders from {ckpt_classes} were found under {NEW_WAV_DIR}. "
            f"Create subfolders like {NEW_WAV_DIR}/doum, {NEW_WAV_DIR}/tak, ..."
        )
    print("New data classes found:", present_classes)

    class_to_idx = {c: ckpt_classes.index(c) for c in present_classes}
    per_class_files = list_wavs_per_class(NEW_WAV_DIR, present_classes)

    train_s, val_s, test_s = stratified_split(
        per_class_files, VAL_RATIO, TEST_RATIO, seed=SEED, class_to_idx=class_to_idx
    )

    print(f"New data -> train: {len(train_s)}, val: {len(val_s)}, test: {len(test_s)}")

    # 4) Normalization for new data
    if USE_CHECKPOINT_NORM:
        mean, std = ckpt_mean, ckpt_std
    else:
        print("Computing mean/std on new train split ...")
        mean, std = compute_mean_std_over_mels(train_s)
        print(f"New mean/std: {mean:.4f}, {std:.4f}")

    # 5) Datasets / loaders
    train_ds = WaveToMelDataset(train_s, mean=mean, std=std)
    val_ds = WaveToMelDataset(val_s, mean=mean, std=std)
    test_ds = WaveToMelDataset(test_s, mean=mean, std=std)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    # 6) Loss & optimizer (reuse weights in checkpoint order or overridden WEIGHTS)
    weight_vec = torch.tensor(ckpt_class_weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weight_vec)

    # Two parameter groups: lower LR for features, higher for classifier head
    optimizer = torch.optim.Adam(
        [
            {"params": model.features.parameters(), "lr": LR_FEATURES},
            {"params": model.classifier.parameters(), "lr": LR_HEAD},
        ]
    )

    # 7) Train with early stopping
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
                    "classes": ckpt_classes,
                    "mean": mean,
                    "std": std,
                    "class_weights": ckpt_class_weights,  # persisted (ckpt or overridden)
                    "mel_params": {
                        "SR": SR,
                        "N_FFT": N_FFT,
                        "HOP_LENGTH": HOP_LENGTH,
                        "N_MELS": N_MELS,
                        "FMIN": FMIN,
                        "FMAX": FMAX,
                        "FIXED_MAX_FRAMES": FIXED_MAX_FRAMES,
                    },
                },
                FINETUNED_PATH,
            )
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch} (best epoch: {best_epoch})")
                break

    # 8) Load best & evaluate on test
    if os.path.exists(FINETUNED_PATH):
        best = torch.load(FINETUNED_PATH, map_location=device)
        model.load_state_dict(best["model_state"])

    te_loss, te_acc, y_true, y_pred = evaluate(model, test_loader, criterion, device)
    print(f"\nTest: loss={te_loss:.4f}  acc={te_acc:.3f}")

    # 9) Confusion matrix & per-class acc (over full label space)
    cm = confusion_matrix(y_true, y_pred, num_classes=len(ckpt_classes))
    print(
        "\nConfusion Matrix (rows=true, cols=pred) [only rows for classes present in new data will be non-zero]:"
    )
    for i, row in enumerate(cm):
        print(f"{ckpt_classes[i]:>6}: {row}")
    per_cls_acc = per_class_accuracy(cm)
    print("\nPer-class accuracy:")
    for c, a in zip(ckpt_classes, per_cls_acc):
        print(f"{c:>6}: {a:.3f}")

    # 10) Save misclassified (test split)
    n_mis = save_misclassified(model, test_loader, ckpt_classes, MISCLASS_CSV, device)
    print(f"\nMisclassified examples saved to {MISCLASS_CSV} (count={n_mis})")


if __name__ == "__main__":
    main()
