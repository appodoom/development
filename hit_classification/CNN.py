# train_cnn_audio.py
# CNN training on doum/tak/tik/pa2 audio files using librosa + PyTorch.
# No argparse; all configuration is via the GLOBALS below.
# Stratified split: per-label TRAIN_RATIO / (1 - TRAIN_RATIO).

from __future__ import annotations
import os
import random
from typing import List, Tuple, Dict, Sequence, Optional

import numpy as np
import librosa
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# =========================
# ====== CONFIG (GLOBAL)
# =========================
DATA_DIR: str = "../../fundamentals"             # Folder containing your audio files
ALLOWED_EXTS: Tuple[str, ...] = (".wav")

# Audio / feature params
SAMPLE_RATE: int = 48_000
DURATION_S: float = 1.0              # Each clip is padded/truncated to this duration
N_MELS: int = 64
N_FFT: int = 1024
HOP_LENGTH: int = 256

# Train params
BATCH_SIZE: int = 32
EPOCHS: int = 20
LEARNING_RATE: float = 1e-3
WEIGHT_DECAY: float = 1e-4
TRAIN_RATIO: float = 0.80            # per-label ratio for train; remainder goes to val
RANDOM_SEED: int = 1337
NUM_WORKERS: int = 0                 # increase if you want parallel data loading

# Output
MODEL_OUT_PATH: str = "./audio_cnn_state.pt"
LABELMAP_OUT_PATH: str = "./label_map.txt"  # human-readable label mapping

# Device
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

# Label mapping by filename prefix (lowercased)
LABELS_BY_PREFIX: Dict[str, int] = {
    "doum": 0,
    "tak": 1,
    "tik": 2,
    "pa2": 3,
}
IDX_TO_LABEL: Dict[int, str] = {v: k for k, v in LABELS_BY_PREFIX.items()}

# =========================
# ====== MODEL SIZE KNOBS
# =========================
# Number of conv layers = len(CHANNEL). Each entry is the out_channels of that layer.
CHANNEL: Tuple[int, ...] = (32, 64, 128)    # e.g., () -> 0 layers; (32,) -> 1 layer; etc.
KERNEL_SIZE: int = 3                        # odd numbers 3/5/7 etc.
USE_DEPTHWISE: bool = False                 # True: depthwise-separable convs (fewer params)
POOL_KERNEL: int = 2                        # maxpool kernel/stride
DROPOUT_P: float = 0.5                      # dropout in the head

# MLP head neurons (after global pooling). Empty tuple => single Linear to n_classes.
HEAD_HIDDEN_LAYERS: Tuple[int, ...] = ()    # e.g., (128,) or (256,128)

# =========================
# ====== UTILITIES
# =========================
def infer_label_from_name(fname: str) -> Optional[int]:
    """
    Brief: Infer the class index from a filename by its prefix among {doum,tak,tik,pa2}.
    Args:
        fname (str): Basename of the file (e.g., 'doum_001.wav').
    Returns:
        Optional[int]: Class index if a known prefix is found, else None.
    """
    base = os.path.basename(fname).lower()
    for pref, idx in LABELS_BY_PREFIX.items():
        if base.startswith(pref):
            return idx
    return None


def discover_files(data_dir: str, exts: Sequence[str]) -> List[Tuple[str, int]]:
    """
    Brief: Recursively collect (filepath, label_idx) pairs for audio files under data_dir.
    Args:
        data_dir (str): Root directory containing audio files.
        exts (Sequence[str]): Allowed file extensions.
    Returns:
        List[Tuple[str, int]]: List of (absolute_path, class_index) pairs.
    """
    out: List[Tuple[str, int]] = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if any(f.lower().endswith(e) for e in exts):
                lbl = infer_label_from_name(f)
                if lbl is not None:
                    out.append((os.path.join(root, f), lbl))
    return out


def set_all_seeds(seed: int) -> None:
    """
    Brief: Set random seeds for Python, NumPy, and PyTorch for reproducibility.
    Args:
        seed (int): The seed value.
    Returns:
        None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_audio_fixed(path: str, sr: int, duration_s: float) -> np.ndarray:
    """
    Brief: Load mono audio at sample rate sr and pad/trim to duration_s seconds.
    Args:
        path (str): File path to the audio.
        sr (int): Target sampling rate.
        duration_s (float): Target duration (seconds).
    Returns:
        np.ndarray: Audio waveform of shape (samples,), float32.
    """
    y, _ = librosa.load(path, sr=sr, mono=True)
    target_len = int(round(duration_s * sr))
    if y.shape[0] < target_len:
        pad = target_len - y.shape[0]
        y = np.pad(y, (0, pad), mode="constant")
    elif y.shape[0] > target_len:
        y = y[:target_len]
    return y.astype(np.float32)


def melspec_db(y: np.ndarray, sr: int, n_mels: int, n_fft: int, hop_len: int) -> np.ndarray:
    """
    Brief: Compute log-mel spectrogram in decibels.
    Args:
        y (np.ndarray): 1D audio waveform (float32).
        sr (int): Sample rate.
        n_mels (int): Number of Mel bands.
        n_fft (int): FFT window size.
        hop_len (int): Hop length in samples.
    Returns:
        np.ndarray: 2D array (n_mels, time_frames) float32 of log-mel (dB).
    """
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_len, power=2.0
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    return S_db.astype(np.float32)


def spec_to_tensor(spec_db: np.ndarray) -> torch.Tensor:
    """
    Brief: Convert (H,W) spectrogram to a 3D torch tensor with a channel dim.
    Args:
        spec_db (np.ndarray): Log-mel spectrogram (n_mels, time).
    Returns:
        torch.Tensor: Tensor of shape (1, n_mels, time), dtype float32.
    """
    x = torch.from_numpy(spec_db)  # (H, W)
    x = (x - x.mean()) / (x.std() + 1e-6)  # per-sample normalization
    return x.unsqueeze(0).float()

# =========================
# ====== DATASET
# =========================
class AudioSpecDataset(Dataset):
    """
    Brief: Dataset that loads audio files, converts to log-mel spectrograms, and yields tensors + labels.
    Args:
        files (List[Tuple[str, int]]): List of (path, class_idx) items.
        sr (int): Sampling rate.
        duration_s (float): Seconds to pad/trim audio to.
        n_mels (int): Number of mel bands.
        n_fft (int): FFT size.
        hop_len (int): Hop length.
    Returns:
        torch.utils.data.Dataset: Yields (tensor, int) where tensor=(1, n_mels, T).
    """
    def __init__(
        self,
        files: List[Tuple[str, int]],
        sr: int = SAMPLE_RATE,
        duration_s: float = DURATION_S,
        n_mels: int = N_MELS,
        n_fft: int = N_FFT,
        hop_len: int = HOP_LENGTH,
    ) -> None:
        self.files = files
        self.sr = sr
        self.duration_s = duration_s
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_len = hop_len

    def __len__(self) -> int:
        """Brief: Number of samples. Returns: int: dataset length."""
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Brief: Load one example and return (log-mel tensor, label).
        Args:
            idx (int): Index of the sample.
        Returns:
            Tuple[torch.Tensor, int]: (x, y) where x has shape (1, n_mels, T) and y is class index.
        """
        path, label = self.files[idx]
        y = load_audio_fixed(path, self.sr, self.duration_s)
        spec = melspec_db(y, self.sr, self.n_mels, self.n_fft, self.hop_len)
        x = spec_to_tensor(spec)
        return x, label

# =========================
# ====== SPLIT & COUNTS
# =========================
def stratified_split(
    files: List[Tuple[str, int]],
    train_ratio: float,
    seed: int,
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    """
    Brief: Stratify by label and split each label's files into train/val by train_ratio.
    Args:
        files (List[Tuple[str,int]]): Items to split.
        train_ratio (float): Fraction for training set in [0,1].
        seed (int): RNG seed for shuffling within each label.
    Returns:
        Tuple[List[Tuple[str,int]], List[Tuple[str,int]]]: (train_files, val_files)
    """
    by_label: Dict[int, List[Tuple[str, int]]] = {}
    for p, y in files:
        by_label.setdefault(y, []).append((p, y))

    rng = random.Random(seed)
    train_files: List[Tuple[str, int]] = []
    val_files: List[Tuple[str, int]] = []

    for _, items in by_label.items():
        items = items[:]  # copy
        rng.shuffle(items)
        n = len(items)
        if n <= 1:
            train_files.extend(items)
            continue
        n_train = int(round(train_ratio * n))
        n_train = max(1, min(n - 1, n_train))
        train_files.extend(items[:n_train])
        val_files.extend(items[n_train:])

    rng.shuffle(train_files)
    rng.shuffle(val_files)
    return train_files, val_files


def counts_by_label(files: List[Tuple[str, int]]) -> Dict[str, int]:
    """
    Brief: Count samples per label for a file list.
    Args:
        files (List[Tuple[str,int]]): Items to count.
    Returns:
        Dict[str,int]: Mapping from label name to count.
    """
    c: Dict[int, int] = {}
    for _, y in files:
        c[y] = c.get(y, 0) + 1
    return {IDX_TO_LABEL[k]: v for k, v in sorted(c.items())}


def make_loaders(
    train_files: List[Tuple[str, int]],
    val_files: List[Tuple[str, int]],
    batch_size: int,
) -> Tuple[DataLoader, DataLoader]:
    """
    Brief: Create PyTorch DataLoaders for train and validation splits.
    Args:
        train_files (List[Tuple[str,int]]): Training file list.
        val_files (List[Tuple[str,int]]): Validation file list.
        batch_size (int): Batch size.
    Returns:
        Tuple[DataLoader, DataLoader]: (train_loader, val_loader)
    """
    train_ds = AudioSpecDataset(train_files)
    val_ds = AudioSpecDataset(val_files)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)
    return train_loader, val_loader

# =========================
# ====== MODEL BUILDING
# =========================
class DWSeparableConv(nn.Module):
    """
    Brief: Depthwise (groups=in_ch) + pointwise (1x1) conv block with BN+ReLU.
    Args:
        in_ch (int): Input channels.
        out_ch (int): Output channels.
        k (int): Kernel size (odd).
        padding (int): Padding for depthwise conv.
    Returns:
        torch.nn.Module: A lightweight conv block.
    """
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, padding: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=k, padding=padding, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch), nn.ReLU(),
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


def make_conv_block(in_ch: int, out_ch: int, k: int, use_depthwise: bool, pool_k: int) -> nn.Sequential:
    """
    Brief: Construct one convolutional block (conv->BN->ReLU->MaxPool).
    Args:
        in_ch (int): Input channels.
        out_ch (int): Output channels.
        k (int): Kernel size (odd).
        use_depthwise (bool): Use depthwise-separable conv if True, else standard conv.
        pool_k (int): MaxPool2d kernel/stride size.
    Returns:
        nn.Sequential: A convolutional block.
    """
    pad = k // 2
    if use_depthwise:
        conv = DWSeparableConv(in_ch, out_ch, k=k, padding=pad)
    else:
        conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=pad, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(),
        )
    return nn.Sequential(conv, nn.MaxPool2d(kernel_size=pool_k, stride=pool_k))


def build_head(in_ch: int, hidden: Sequence[int], n_classes: int, dropout_p: float) -> nn.Sequential:
    """
    Brief: Build the classification head as MLP after global pooling.
    Args:
        in_ch (int): Input feature dim after pooling.
        hidden (Sequence[int]): Hidden layer sizes (can be empty).
        n_classes (int): Number of output classes.
        dropout_p (float): Dropout probability inserted before each Linear.
    Returns:
        nn.Sequential: Head mapping pooled features to logits.
    """
    layers: List[nn.Module] = [nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten()]
    prev = in_ch
    for h in hidden:
        layers += [nn.Dropout(dropout_p), nn.Linear(prev, h), nn.ReLU()]
        prev = h
    layers += [nn.Dropout(dropout_p), nn.Linear(prev, n_classes)]
    return nn.Sequential(*layers)


class AudioCNN(nn.Module):
    """
    Brief: Scalable CNN for log-mel spectrograms with configurable depth, width, and head.
    Args:
        n_classes (int): Number of output classes.
        in_ch (int): Input channels (1 for spectrograms).
    Returns:
        torch.nn.Module: Model whose forward returns logits (B, n_classes).
    """
    def __init__(self, n_classes: int, in_ch: int = 1) -> None:
        super().__init__()
        blocks: List[nn.Module] = []
        prev_ch = in_ch
        for out_ch in CHANNEL:
            blocks.append(make_conv_block(prev_ch, out_ch, KERNEL_SIZE, USE_DEPTHWISE, POOL_KERNEL))
            prev_ch = out_ch
        self.feat = nn.Sequential(*blocks) if blocks else nn.Identity()
        last_ch = CHANNEL[-1] if len(CHANNEL) > 0 else in_ch
        self.head = build_head(last_ch, HEAD_HIDDEN_LAYERS, n_classes, DROPOUT_P)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Brief: Forward pass from spectrogram batch to logits.
        Args:
            x (torch.Tensor): Input of shape (B, 1, n_mels, T).
        Returns:
            torch.Tensor: Logits of shape (B, n_classes).
        """
        z = self.feat(x)
        out = self.head(z)
        return out


def count_params(model: nn.Module) -> int:
    """
    Brief: Count trainable parameters in a model.
    Args:
        model (nn.Module): PyTorch module.
    Returns:
        int: Number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# =========================
# ====== TRAIN / EVAL
# =========================
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    opt: torch.optim.Optimizer,
    device: str
) -> float:
    """
    Brief: Train the model for one epoch.
    Args:
        model (nn.Module): The neural network.
        loader (DataLoader): Training data loader.
        opt (torch.optim.Optimizer): Optimizer.
        device (str): 'cuda' or 'cpu'.
    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    running = 0.0
    crit = nn.CrossEntropyLoss()
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        opt.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = crit(logits, yb)
        loss.backward()
        opt.step()
        running += loss.item() * xb.size(0)
    return running / len(loader.dataset)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: str
) -> Tuple[float, float]:
    """
    Brief: Evaluate average loss and accuracy on a data loader.
    Args:
        model (nn.Module): The neural network (in eval mode).
        loader (DataLoader): Data loader for validation.
        device (str): 'cuda' or 'cpu'.
    Returns:
        Tuple[float, float]: (avg_loss, accuracy_in_[0,1]).
    """
    model.eval()
    crit = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    n = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = crit(logits, yb)
            total_loss += loss.item() * xb.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == yb).sum().item()
            n += xb.size(0)
    return total_loss / n, correct / n


def save_model_state(model: nn.Module, path: str) -> None:
    """
    Brief: Save model's state_dict to disk.
    Args:
        model (nn.Module): Trained model.
        path (str): Output file path (e.g., './audio_cnn_state.pt').
    Returns:
        None
    """
    torch.save(model.state_dict(), path)


def save_label_map(idx_to_label: Dict[int, str], out_path: str) -> None:
    """
    Brief: Save a simple text file mapping class indices to labels.
    Args:
        idx_to_label (Dict[int,str]): Mapping like {0: 'doum', 1: 'tak', ...}.
        out_path (str): Destination text file path.
    Returns:
        None
    """
    with open(out_path, "w", encoding="utf-8") as f:
        for k in sorted(idx_to_label):
            f.write(f"{k}\t{idx_to_label[k]}\n")

# =========================
# ====== MAIN
# =========================
def main() -> None:
    """
    Brief: Orchestrate discovery, stratified split, loaders, training, validation, and saving.
    Args:
        None
    Returns:
        None
    """
    print(f"Using device: {DEVICE}")
    set_all_seeds(RANDOM_SEED)

    files = discover_files(DATA_DIR, ALLOWED_EXTS)
    if not files:
        raise RuntimeError(f"No labeled files found in {DATA_DIR}. "
                           f"Expect names starting with one of {list(LABELS_BY_PREFIX.keys())}")

    # Overall class counts
    overall = counts_by_label(files)
    print("Overall counts:", overall)

    # Stratified split per label
    train_files, val_files = stratified_split(files, TRAIN_RATIO, RANDOM_SEED)
    print("Train counts:", counts_by_label(train_files))
    print("Val   counts:", counts_by_label(val_files))
    print(f"Train/Val sizes: {len(train_files)} / {len(val_files)}")

    # Loaders
    train_loader, val_loader = make_loaders(train_files, val_files, BATCH_SIZE)

    # Model/opt
    n_classes = len(IDX_TO_LABEL)
    model = AudioCNN(n_classes=n_classes).to(DEVICE)
    print(f"Trainable params: {count_params(model):,}")
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    best_val_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        tr_loss = train_one_epoch(model, train_loader, opt, DEVICE)
        val_loss, val_acc = evaluate(model, val_loader, DEVICE)
        print(f"Epoch {epoch:02d} | train_loss={tr_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.3f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model_state(model, MODEL_OUT_PATH)
            save_label_map(IDX_TO_LABEL, LABELMAP_OUT_PATH)
            print(f"  ✔ Saved checkpoint to {MODEL_OUT_PATH} (val_acc={val_acc:.3f})")

    print(f"Done. Best val_acc={best_val_acc:.3f}. Model: {MODEL_OUT_PATH} | labels: {LABELMAP_OUT_PATH}")


if __name__ == "__main__":
    main()
