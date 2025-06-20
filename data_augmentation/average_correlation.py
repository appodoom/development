import numpy as np
import librosa
import soundfile as sf


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
    return cycle_indices, beat_frames

def enhance(y, sr):
    cycle_indices, beat_frames = find_cycle_beat_indices(y, sr)
    cycle_indices_adjusted = [beat_frames[idx] for idx in cycle_indices]
    new_y = np.array([], dtype=y.dtype)
    permuted = np.random.permutation(cycle_indices_adjusted)
    samples = librosa.frames_to_samples(permuted)
    beat_samples = librosa.frames_to_samples(beat_frames)
    beat_dic = {str(bs): i for i, bs in enumerate(beat_samples)}
    cycle_count = cycle_indices[1] - cycle_indices[0]

    for j in range(len(samples) - 1):
        start_cycle = samples[j]
        k = beat_dic[str(start_cycle)]
        if k + cycle_count - 1 < len(beat_samples) - 1:
            num_samps = beat_samples[k + cycle_count ] - beat_samples[k]
        else:
            num_samps = beat_samples[-1] - beat_samples[k]

        temp = y[start_cycle:start_cycle + num_samps]

        if new_y.size == 0:
            new_y = temp
        else:
            new_y = np.concatenate([new_y, temp])

    return new_y

def enhance_sample(in_path, out_path, factor):
    y, sr = librosa.load(in_path, sr=None)
    out = y
    for _ in range(factor):
        new_y = enhance(y, sr)
        out = np.concatenate((out, new_y))
    
    sf.write(out_path, out, samplerate=sr)
    print("Done")

y, sr= librosa.load("../../sample8.wav", sr=None)
print(find_cycle_beat_indices(y=y, sr=sr))