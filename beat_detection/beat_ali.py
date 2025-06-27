import librosa
import numpy as np

def get_beat_timestamps(y,sr, window):
    tempo, beat_frames=librosa.beat.beat_track(y=y,sr=sr)
    beat_duration= int(sr*60/tempo[0])
    beat_samples=librosa.frames_to_samples(beat_frames)
    beats=[beat_samples[0]] #we can initialize the first beat to be beat_duration if we don't want to use librosa
    previous_beat=beats[-1]
    onsets=librosa.onset.onset_detect(y=y,sr=sr)
    onsets=librosa.frames_to_samples(onsets)
    while(previous_beat+beat_duration<len(y)):
        expected_beat=previous_beat+beat_duration
        for onset in onsets:
            if onset>=expected_beat-window and onset<=expected_beat+window:
                expected_beat=onset
            break
        beats.append(int(expected_beat))
        previous_beat=beats[-1]
    return beats



