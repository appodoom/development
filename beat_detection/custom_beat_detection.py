import librosa

def get_beat_timestamps(y,sr, window, units="samples"):
    tempo, _ =librosa.beat.beat_track(y=y,sr=sr)
    beat_duration= int(sr*60/tempo[0]) #in samples
    onsets=librosa.onset.onset_detect(y=y,sr=sr)
    onsets=librosa.frames_to_samples(onsets)
    beats=[onsets[0]] #we can initialize the first beat to be beat_duration if we don't want to use librosa
    previous_beat=beats[-1]
    while(previous_beat+beat_duration<len(y)):
        expected_beat=previous_beat+beat_duration
        for onset in onsets:
            if onset>=expected_beat-window and onset<=expected_beat+window:
                expected_beat=onset
                break
        beats.append(expected_beat)
        previous_beat=beats[-1]
        
    if units=="samples":
        return beats
    elif units=="frames":
        return librosa.samples_to_frames(beats)
