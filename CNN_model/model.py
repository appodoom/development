import os
import glob
import math
import random
import numpy as np
import torch
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
# from torch.utils.data.sampler import WeightedRandomSampler  # optional if you want sampler-based balancing

# ======================= CONFIG (edit me) =======================
DATA_DIR = "./mels_data"
CLASSES = ["doum", "tak", "tik", "pa2"]

# give a number (hyperparameter) for each label
# these act as class weights in the loss: larger number => that class’ errors are penalized more
CLASS_WEIGHTS = {
    "doum": 0.8,
    "tak": 0.35,
    "tik": 0.76,
    "pa2": 1.0,
}

VAL_RATIO = 0.15
TEST_RATIO = 0.15
BATCH_SIZE = 16
LR = 1e-3
MAX_EPOCHS = 50
PATIENCE = 8  # early-stopping patience (epochs)
SEED = 42
NUM_WORKERS = 0  # set >0 if your OS benefits (Linux); Windows often 0
MODEL_PATH = "best_model.pt"
MISCLASS_CSV = "misclassified.csv"

# If mels have variable time length, we pad/crop to this many frames (time axis).
# You can set None to keep variable length; model uses AdaptiveAvgPool so it's okay to vary.
# Padding to a modest cap can help batch efficiency.
FIXED_MAX_FRAMES = 256  # set to None to disable padding/cropping
# ================================================================

# -------------------- Reproducibility --------------------
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# -------------------- Dataset --------------------
def list_files_per_class(root_dir, classes):
    per_class = {}
    for c in classes:
        per_class[c] = sorted(glob.glob(os.path.join(root_dir, c, "*.npy")))
    return per_class


# Split dataset into training, validation, testing
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


class MelDataset(Dataset):
    def __init__(self, samples, compute_stats=False, mean=None, std=None):
        """
        samples: list of (filepath, class_idx)
        If compute_stats=True, you can iterate to compute mean/std beforehand.
        """
        self.samples = samples
        self.compute_stats = compute_stats
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        mel = np.load(path)  # (n_mels, T)
        mel = pad_or_crop_time(mel, FIXED_MAX_FRAMES)
        mel = mel.astype(np.float32)
        if (self.mean is not None) and (self.std is not None):
            mel = (mel - self.mean) / (self.std + 1e-8)
        mel = torch.from_numpy(mel).unsqueeze(0)  # (1, n_mels, T)
        label = torch.tensor(label, dtype=torch.long)
        return mel, label, path


# -------------------- Train-set mean/std --------------------
def compute_mean_std(train_samples):
    """
    Compute global mean/std over train set without loading all in RAM.
    Uses Welford's algorithm over all elements of mel arrays.
    """
    count = 0
    mean = 0.0
    M2 = 0.0
    for path, _ in train_samples:
        mel = np.load(path).astype(np.float64)  # (n_mels, T)
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
    Lightweight CNN with AdaptiveAvgPool for variable-length time axis.
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
    Save misclassified samples with original WAV filenames instead of .npy paths.
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
                            "mel_path": path,
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
                "mel_path",
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
    # 1) index data
    per_class_files = list_files_per_class(DATA_DIR, CLASSES)

    # 2) stratified split
    train_s, val_s, test_s = stratified_split(
        per_class_files, VAL_RATIO, TEST_RATIO, SEED
    )
    print(f"Samples -> train: {len(train_s)}, val: {len(val_s)}, test: {len(test_s)}")

    # 3) compute train mean/std for normalization
    print("Computing train mean/std ...")
    mean, std = compute_mean_std(train_s)
    print(f"Train mean: {mean:.4f}, std: {std:.4f}")

    # 4) datasets & loaders
    train_ds = MelDataset(train_s, mean=mean, std=std)
    val_ds = MelDataset(val_s, mean=mean, std=std)
    test_ds = MelDataset(test_s, mean=mean, std=std)

    # OPTIONAL: if you prefer sampling based on your numbers, build a WeightedRandomSampler
    # Here we interpret CLASS_WEIGHTS as "desired sampling weight per class"
    # (commented out by default; loss weights below are usually enough)
    # weights_per_sample = [CLASS_WEIGHTS[CLASSES[label]] for _, label in train_s]
    # sampler = WeightedRandomSampler(weights=weights_per_sample, num_samples=len(weights_per_sample), replacement=True)
    # train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS)

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
    criterion = nn.CrossEntropyLoss(
        weight=weight_vec
    )  # <<— per-label numbers applied here
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
        # if you ever want to rehydrate the criterion weights from the checkpoint:
        # if "class_weights" in ckpt:
        #     weight_vec = torch.tensor(ckpt["class_weights"], dtype=torch.float32, device=device)
        #     criterion = nn.CrossEntropyLoss(weight=weight_vec)

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
