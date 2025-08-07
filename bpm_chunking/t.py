import numpy as np
import librosa
from scipy.stats import uniform, median_abs_deviation, norm

# 1) Load your audio
filename = '../../samples/sample1.wav'
y, sr = librosa.load(filename, sr=None)

# 2) Compute onset strength envelope
onset_env = librosa.onset.onset_strength(y=y, sr=sr)
# 4) Compute log₂-scaled IQR-based σ estimate
low, high = 0.25,0.75
denom = norm.ppf(high) - norm.ppf(low)
q3=np.log2(66)
q1=np.log2(54)
iqr_log2 = q3-q1
std_log2 = iqr_log2 / denom  # convert IQR to σ (normal distribution)
#5.5% +- you log2 and then you divide
#1.648721271


print(f"Log₂-scale IQR (σ): {std_log2:.4f} octaves")
