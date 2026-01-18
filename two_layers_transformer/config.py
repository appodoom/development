import librosa
import os
import random

# this file contains configuration for generator
paths = {
    "D": "./sounds/doums",
    "OTA": "./sounds/taks",
    "OTI": "./sounds/tiks",
    "PAA": "./sounds/pa2s",
    # "RA": "./sounds/ra.wav",
    # "T1": "./sounds/tik1.wav",
    # "T2": "./sounds/tik2.wav",
    "S": "./sounds/silence",
}


def get_audio_data(symbol, sr=48000):
    directory = paths.get(symbol)
    files = os.listdir(directory)
    chosen_file = random.choice(files)
    full_path = os.path.join(directory, chosen_file)
    y, _ = librosa.load(full_path, sr=sr)
    return y
