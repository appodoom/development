import librosa
import json
import numpy as np


def get_note_duration(quarter_duration):
    return {
        16: quarter_duration / 4,
        8: quarter_duration / 2,
        4: quarter_duration,
        2: quarter_duration * 2,
        1: quarter_duration * 4,
        24: quarter_duration/6,
        32:quarter_duration/8,
        12:quarter_duration/3,
        2.67:quarter_duration*(3/2),
        5.34:quarter_duration*(3/4),
        10.67:quarter_duration*(3/8),
        21.33:quarter_duration*(3/16),
        1.33:quarter_duration*(3),
        28:quarter_duration*(1/7),
        56:quarter_duration*(1/14),
        14:quarter_duration*(2/7),
        9.33:quarter_duration*(3/7),
        7:quarter_duration*(4/7),
        5.6:quarter_duration*(5/7),
        4.67:quarter_duration*(6/7),
        20:quarter_duration*(1/5),
        40:quarter_duration*(1/10),
        10:quarter_duration*(2/5),
        6.67:quarter_duration*(3/5),
        5:quarter_duration*(4/5),
        36:quarter_duration*(1/9),
        72:quarter_duration*(1/18),
        18:quarter_duration*(2/9),
        9:quarter_duration*(4/9),
        7.2:quarter_duration*(5/9),
        6:quarter_duration*(6/9),
        5.1:quarter_duration*(7/9),
        4.5:quarter_duration*(8/9)
    }

def sliding_cross_correlation(X,Y):
    if Y.shape[1]>X.shape[1]:
        X, Y= Y,X
    best_score=-np.inf
    best_offset=None
    dim_X=np.linalg.norm(X)
    dim_Y=np.linalg.norm(Y)

    for offset in range(X.shape[1]-Y.shape[1]+1):
        X_slider=X[ : , offset:Y.shape[1]+offset]
        if dim_X==0 or dim_Y==0:
            continue
        else:
            score=np.tensordot(X_slider,Y,axes=2)/(dim_Y*np.linalg.norm(X_slider))
        if score>best_score:
            best_score=score
            best_offset=offset
    return best_score, best_offset


def load_json(file_directory):
    with open(file_directory, 'r') as f:
        mels = json.load(f)
    return {name:np.array(mel) for name,mel in mels.items()}


def load_file(data_direc):
    y ,sr =librosa.load(data_direc, sr=None)
    y=librosa.util.normalize(y=y)
    return y, sr


def get_onsets(y,sr):
    onsets=librosa.onset.onset_detect(y=y , sr=sr)
    onsets_avg=[onsets[0]//2]
    for i in range(len(onsets)-1):
        onsets_avg.append((onsets[i]+onsets[i+1])//2)
    onsets_avg=librosa.frames_to_samples(onsets_avg)
    return onsets_avg


def adjust_tempo(y, sr):
    tempo,_=librosa.beat.beat_track(y=y,sr=sr)
    while tempo>150:
        tempo=tempo//2
    while tempo<50:
        tempo*=2
    return tempo


def get_intervals(y , sr):
    onsets=get_onsets(y,sr)
    intervals=[]
    for i in range(len(onsets)-1):
        intervals.append((onsets[i],onsets[i+1]))
    return intervals


def get_corpus(fundamentals_path, data_path):
    y,sr=load_file(data_path)
    fundamentals=load_json(fundamentals_path)
    intervals=get_intervals(y,sr)
    tempo=adjust_tempo(y,sr)
    quarter_duration=60.0/tempo
    note_durations=get_note_duration(quarter_duration)
    classified_hits=[]
    for interval in intervals:
        segment=y[interval[0]:interval[1]]
        mel=librosa.feature.melspectrogram(y=segment, sr=sr)
        best_score=-np.inf
        best_hit=''
        for fundamental_hit in fundamentals:
            score,_=sliding_cross_correlation(mel,fundamentals[fundamental_hit])
            if score>best_score:
                best_score=score
                best_hit=fundamental_hit
        hit_duration=len(segment)/sr
        min_diff=np.inf
        best_note=''
        for note in note_durations:
            diff=abs(note_durations[note]-hit_duration)
            if diff<min_diff:
                min_diff=diff
                best_note=str(note)
        classified_hits.append(best_hit+'_'+best_note)
    return classified_hits
        




