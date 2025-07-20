import librosa
import numpy as np
# from beat_detection.custom_beat_detection import get_beat_timestamps
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
    return best_score

def cycle_length(y,sr, window):
    window = librosa.samples_to_frames([window])[0]
    tempo, beat_frames=librosa.beat.beat_track(y=y,sr=sr, tightness=100)
    # beat_frames=get_beat_timestamps(y,sr,2048)
    print(tempo)
    correlation_scores=[0]*14
    mel=librosa.feature.melspectrogram(y=y,sr=sr)
    for cycle_length in range(3,17):
        total_score=0
        for i in range(len(beat_frames)-16):
            total_score+=sliding_cross_correlation(mel[:,beat_frames[i]-window:beat_frames[i]+window+1],mel[:,beat_frames[i+cycle_length]-window:beat_frames[i+cycle_length]+window+1])
        correlation_scores[cycle_length-3]=total_score
    best_cycle=np.argmax(correlation_scores)+3
    cycle_indices=[]
    if tempo>150 and best_cycle%2==0:
        best_cycle=best_cycle//2
        tempo/=2
    for i in range(0,len(beat_frames),best_cycle):
        cycle_indices.append(beat_frames[i])
    return best_cycle

y,sr=librosa.load("../samples/sample8.wav", sr=None)
print(f"{cycle_length(y=y,sr=sr,window=12000)}")



