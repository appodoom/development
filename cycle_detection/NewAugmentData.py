import numpy as np
import librosa
import soundfile as sf # type: ignore[import]

def adjust_tempo(tempo,beat_frames):
    print("initial tempo: ",tempo)
    print("initial Beat frames: ",beat_frames)
    batta =tempo   
    batta2=beat_frames
    while batta>150:
        batta=batta//2
        batta2 = np.array([batta2[i] for i in range(0,len(batta2),2)])

    while batta<50:
        batta*=2
        batta2=batta2//2

    print("Tempo done")
    return batta,batta2

def find_cycle_beat_indices(
    y,
    sr,
    min_beats: int = 3,
    max_beats: int = 16,
    n_mels: int = 128,
    hop_length: int = 512
) -> np.ndarray:
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    tempo, beat_frames = adjust_tempo(tempo=tempo,beat_frames=beat_frames)
    print("Tempo: ",tempo)
    print("Beat frames: ",beat_frames)
    y2 = np.concatenate((y, y))
    mel1 = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
    mel2 = librosa.feature.melspectrogram(y=y2, sr=sr, n_mels=n_mels, hop_length=hop_length)
    mel1 = librosa.util.normalize(mel1, axis=1)
    mel2 = librosa.util.normalize(mel2, axis=1)
    frames_per_beat = int(sr * 60 / (tempo * hop_length))
    corrs = []
    for b in range(min_beats, max_beats + 1):
        shift = b * frames_per_beat
        if shift + mel1.shape[1] > mel2.shape[1]:
            break
        corr_val = sum(
            np.correlate(mel1[i], mel2[i, shift:shift + mel1.shape[1]])
            for i in range(n_mels)
        )
        corrs.append(corr_val)
    print(corrs)
    corrs = np.array(corrs)
    best_beats = np.argmax(corrs) + min_beats
    print("Cycle lenght: ",best_beats)
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
