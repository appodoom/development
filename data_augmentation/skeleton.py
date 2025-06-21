import librosa
import numpy as np

def sliding_cross_correlation(X,Y):
    if Y.shape[1]>X.shape[1]:
        X, Y= Y,X
    best_score=-np.inf
    best_offset=None
    dim_X=np.linalg.norm(X)
    dim_Y=np.linalg.norm(Y)

    for offset in range(X.shape[1]-Y.shape[1]+1):
        X_slider=X[ : , offset:Y.shape[1]+offset]
        if np.linalg.norm(X_slider)==0 or dim_Y==0:
            continue
        else:
            score=np.tensordot(X_slider,Y,axes=2)/(dim_Y*np.linalg.norm(X_slider))
        if score>best_score:
            best_score=score
            best_offset=offset
    return best_score

def get_onsets(y,sr):
    onsets=librosa.onset.onset_detect(y=y , sr=sr)
    onsets_avg=[onsets[0]//2]
    for i in range(len(onsets)-1):
        onsets_avg.append((onsets[i]+onsets[i+1])//2)
    onsets_avg.append(onsets[-1]+(onsets[-1]-onsets_avg[-1]))
    onsets_avg=librosa.frames_to_samples(onsets_avg)
    onsets=librosa.frames_to_samples(onsets)
    return onsets_avg, onsets

def get_filtered_audio(y,sr):
    new_y=y
    onsets_avg, onsets=get_onsets(y=y,sr=sr)
    intervals=[]
    for i in range(len(onsets_avg)-1):
        interval=(onsets_avg[i],onsets_avg[i+1])
        intervals.append(interval)
    tempo, beat_frames=librosa.beat.beat_track(y=y,sr=sr)
    beat_samples=librosa.frames_to_samples(beat_frames)
    beat_in_sec=60/tempo
    beat_in_samples=int(sr*beat_in_sec)
    window=beat_in_samples/8
    for i in range(len(intervals)):
        zero_interval=True
        for beat in beat_samples:
            start=intervals[i][0]
            end=intervals[i][1]
            if (beat>=start and beat<=end and (beat-onsets[i]<=window)):
                zero_interval=False
        if zero_interval:
            slope=(y[start]-y[end])/(start-end)
            b=y[start]-slope*start
            for j in range(start,end+1):
                new_y[j]= j*slope+b
    return new_y

def find_cycle_beat_indices(
    y,
    sr,
    min_beats: int = 3,
    max_beats: int = 16,
    n_mels: int = 128,
    hop_length: int = 512
) -> np.ndarray:
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    y2 = np.concatenate((y, y))
    mel1 = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
    mel2 = librosa.feature.melspectrogram(y=y2, sr=sr, n_mels=n_mels, hop_length=hop_length)
    mel1 = librosa.util.normalize(mel1, axis=1)
    mel2 = librosa.util.normalize(mel2, axis=1)
    corrs = []
    intervals=np.diff(beat_frames)
    frames_per_beat=int(np.round(np.mean(intervals)))
    for b in range(min_beats, max_beats+1):
        shift=b*frames_per_beat
        if b + mel1.shape[1] > mel2.shape[1]:
            break
        corr_val = sliding_cross_correlation(mel1, mel2[:,shift:shift+mel1.shape[1]])
    # print(corrs[:20])
        corrs.append(corr_val)
    # print(corrs, len(corrs))
    corrs = np.array(corrs)
    best_beats=np.argmax(corrs)+min_beats
    cycle_indices = np.arange(0, len(beat_frames), best_beats)
    return best_beats

def get_cycle_length(y,sr):
    new_y=get_filtered_audio(y=y,sr=sr)
    return find_cycle_beat_indices(y=new_y,sr=sr)
y,sr=librosa.load("../../samples/sample8.wav", sr=None)
print(get_cycle_length(y=y,sr=sr))