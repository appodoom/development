import torch
import numpy as np
import torch.nn as nn
import librosa


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


CKPT_PATH = "best_model.pt"
ckpt = torch.load(CKPT_PATH, map_location="cpu")

SAMPLE_RATE = 48000
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 64
TARGET_FRAMES = 256
CLASSES = ckpt["classes"]
mean = float(ckpt["mean"])
std = float(ckpt["std"])

model = SmallAudioCNN(n_classes=len(CLASSES))
model.load_state_dict(ckpt["model_state"])
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


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


def wav_to_mel(wav_path: str, target_frames: int = TARGET_FRAMES) -> torch.Tensor:
    y, sr = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)  # shape: (n_mels, T)
    mel_db = pad_or_crop_time(mel_db, target_frames)
    mel_db = (mel_db - mean) / (std + 1e-8)
    mel_t = torch.from_numpy(mel_db.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    return mel_t.to(device)


@torch.no_grad()
def predict_hit(model: nn.Module, wav_path: str):
    xb = wav_to_mel(wav_path)
    logits = model(xb)
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred_idx = int(np.argmax(probs))
    return CLASSES[pred_idx], probs


if __name__ == "__main__":
    wav_path = "../data/fundemental_hits/tak.wav"
    pred_label, probs = predict_hit(model, wav_path)
    print(f"Predicted class: {pred_label}")
    print("Class probabilities:")
    for c, p in zip(CLASSES, probs):
        print(f"  {c:>6}: {p:.3f}")
