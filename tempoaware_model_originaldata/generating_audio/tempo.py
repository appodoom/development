import librosa

y, sr = librosa.load("modeoutputsound.wav", sr=None)
tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
print(tempo,sr)