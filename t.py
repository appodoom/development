import numpy as np
# Monkey-patch for compatibility if using numpy >=1.24
np.complex = complex

import librosa
import matplotlib.pyplot as plt

# Load audio file (replace with your file path)
y, sr = librosa.load('../samples/sample8.wav', sr=None)

# Compute onset strength envelope
hop_length = 512
oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)

# Detect beats
tempo, beat_frames = librosa.beat.beat_track(
    y=y, 
    sr=sr, 
    hop_length=hop_length,
    tightness=100
)
beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)

# Create time axis for envelope
frame_times = librosa.frames_to_time(range(len(oenv)), sr=sr, hop_length=hop_length)

# Plot onset strength envelope and beats
plt.figure(figsize=(8, 4))
plt.plot(frame_times, oenv)
plt.vlines(beat_times, ymin=0, ymax=oenv.max(), linestyle='--',colors='r')
plt.xlabel('Time (s)')
plt.ylabel('Onset Strength')
plt.title('Onset Strength Envelope with Beat Times')
plt.tight_layout()
plt.show()
