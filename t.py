import librosa
import soundfile as sf

y, sr = librosa.load(path="./data/fundemental_hits/doum.wav", sr=None)
y1 = [0] * len(y)
sf.write("./data/fundemental_hits/S.wav", y1, sr)
