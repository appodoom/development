import numpy as np
import librosa
import soundfile as sf

def find_cycle_beat_indices(
    y,
    sr,
    min_beats: int = 3,
    max_beats: int = 16,
    n_mels: int = 128,
    hop_length: int = 512
) -> np.ndarray:
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times=librosa.frame_to_samples(beat_frames)
    y2 = np.concatenate((y, y))
    mel1 = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
    mel2 = librosa.feature.melspectrogram(y=y2, sr=sr, n_mels=n_mels, hop_length=hop_length)
    mel1 = librosa.util.normalize(mel1, axis=1)
    mel2 = librosa.util.normalize(mel2, axis=1)
    beat_duration=60/tempo
    beat_durations=[]
    for i in range(3,17):
        beat_durations.append(i*beat_duration)
    # shift=0
    corrs = []
    # test=[sum(
    #         np.correlate(mel1[i], mel2[i, shift:shift + mel1.shape[1]]))]
    # shift+=1
    # while (sum(
    #         np.correlate(mel1[i], mel2[i, shift:shift + mel1.shape[1]]))<test[-1]):
    #     shift+=1
    #     test.append(sum(
    #         np.correlate(mel1[i], mel2[i, shift:shift + mel1.shape[1]])))
    for b in range(beat_times[0], len(y)):
        if b + mel1.shape[1] > mel2.shape[1]:
            break
        corr_val = sum(
            np.correlate(mel1[i], mel2[i, b:b + mel1.shape[1]])
            for i in range(n_mels)
        )
    # print(corrs[:20])
        corrs.append(corr_val)
    corrs = np.array(corrs)
    best_period = corrs[np.argmax(corrs)]/sr
    best_beats=np.argmin([np.abs(best_period-beat_dur) for beat_dur in beat_durations])+3
    # cycle_indices = np.arange(0, len(beat_frames), best_beats)
    return best_beats

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

y, sr= librosa.load("../sample8.wav", sr=None)
print(find_cycle_beat_indices(y=y, sr=sr))