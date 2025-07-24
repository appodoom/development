import librosa

# this file contains configuration for generator
paths = {
    "D": "../data/fundemental_hits/doum.wav",
    "OTA": "../data/fundemental_hits/open_tak.wav",
    "OTI": "../data/fundemental_hits/open_tik.wav",
    "PA2": "../data/fundemental_hits/pa2.wav",
    "RA": "../data/fundemental_hits/ra.wav",
    "T1": "../data/fundemental_hits/tik1.wav",
    "T2": "../data/fundemental_hits/tik2.wav",
}

def get_audio_data(symbol, sr=48000):
    path = paths.get(symbol)
    y, _ = librosa.load(path, sr=sr)
    return y