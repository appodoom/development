import os
import glob

DATA_DIR = "../data/pairs"

print("DATA_DIR exists:", os.path.isdir(DATA_DIR))
files = glob.glob(os.path.join(DATA_DIR, "*.wav"))
print("Number of .wav files:", len(files))
print("First few files:", files[:5])
