import librosa as lb
import numpy as np
import soundfile as sf
from config import paths

initial_y, sr = lb.load(paths["D"], sr=None)
y = np.zeros(len(initial_y))
sf.write("silence.wav", y, samplerate=sr)
