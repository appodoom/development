"""
This code intends to try different tightness values
to experiment and choose the one that fits khaled and cycle detection.
==================================================
"""

import librosa
from onset_correlation import cycle_length
from matplotlib import pyplot as plt
import numpy as np
y,sr=librosa.load("../data/samples/sample10.wav", sr=None)
X=[]
Y=[]
for i in range(5, 601, 5):
    X.append(i)
    Y.append(cycle_length(y=y,sr=sr,window=12000,tightness=i))
plt.plot(np.array(X),np.array(Y))
plt.xlabel("Tightness")
plt.ylabel("Cycle length")
plt.show()