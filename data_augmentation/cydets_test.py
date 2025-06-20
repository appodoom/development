from cydets.algorithm import detect_cycles
import librosa
import pandas as pd
y,sr = librosa.load("../../sample8.wav", sr=None)
y = pd.Series(y)
cycles = detect_cycles(y)
print(cycles)