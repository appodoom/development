import os
import glob
import numpy as np
import librosa

SAMPLE_RATE = 48000
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 64
DATA_DIR = "../../data/interference_data"
SAVE_DIR = "./mels_data_doum_tak"

def compute_and_save_mels(root_dir):
    os.makedirs(SAVE_DIR, exist_ok=True)
    for label in ["doum", "tak"]:
        class_dir = os.path.join(root_dir, label)
        save_subdir = os.path.join(SAVE_DIR, label)
        os.makedirs(save_subdir, exist_ok=True)

        for i, wav_path in enumerate(glob.glob(f"{class_dir}/*.wav")):
            y, sr = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
            mel = librosa.feature.melspectrogram(
                y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
            )
            mel_db = librosa.power_to_db(mel, ref=np.max)
            np.save(os.path.join(save_subdir, f"{i:04d}.npy"), mel_db)


compute_and_save_mels(root_dir=DATA_DIR)
