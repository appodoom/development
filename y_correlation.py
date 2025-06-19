import numpy as np
import librosa
# from NewAugmentData import find_cycle_beat_indices

# def adjust_tempo(tempo,beat_frames):
#     batta =tempo   
#     batta2=beat_frames
#     while batta>150:
#         batta=batta//2
#         batta2 = np.array([batta2[i] for i in range(0,len(batta2),2)])

#     while batta<50:
#         batta*=2
#         batta2=batta2//2

#     print("Tempo done")
#     return batta,batta2

# def find_cycle_beat_indices_time_based_on_y(y, sr,
#     min_beats: int = 3,
#     max_beats: int = 16,
# ) -> np.ndarray:
#     tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
#     tempo, beat_frames = adjust_tempo(tempo=tempo,beat_frames=beat_frames)
#     print("Tempo: ",tempo)

#     samples_per_beat = int(sr * 60 / tempo)
    
#     y2 = np.concatenate((y, y))
    
#     corrs = []
#     for b in range(min_beats, max_beats + 1):
#         shift = b * samples_per_beat
#         if shift + len(y) > len(y2):
#             break
#         corr_val = np.dot(y, y2[shift:shift + len(y)])
#         corrs.append(corr_val)
    
#     corrs = np.array(corrs)
#     best_beats = min_beats + np.argmax(corrs)
    
#     cycle_indices = np.arange(0, len(beat_frames), best_beats)
#     print("Cycle lenght: ",best_beats)

#     return cycle_indices, beat_frames
import numpy as np
import librosa

def find_cycle_beat_indices(
    y,
    sr,
    min_beats: int = 3,
    max_beats: int = 16,
    n_mels: int = 128,
    hop_length: int = 512
) -> np.ndarray:
    # 1) Rough beat‐track on the full clip to locate the first downbeat
    _, initial_beats = librosa.beat.beat_track(y=y, sr=sr)
    if len(initial_beats) < min_beats:
        raise ValueError(f"Too few beats detected: {len(initial_beats)}")

    first_sample = librosa.frames_to_samples(initial_beats[0], hop_length=hop_length)
    y_shifted = y[first_sample:]

    tempo, beats = librosa.beat.beat_track(y=y_shifted, sr=sr)
    print("tempo: ",tempo)
    if len(beats) < min_beats:
        raise ValueError(f"Too few beats after shifting: {len(beats)}")

    y2 = np.concatenate((y_shifted, y_shifted))

    mel1 = librosa.feature.melspectrogram(y=y_shifted, sr=sr,
                                          n_mels=n_mels,
                                          hop_length=hop_length)
    mel2 = librosa.feature.melspectrogram(y=y2, sr=sr,
                                          n_mels=n_mels,
                                          hop_length=hop_length)
    mel1 = librosa.util.normalize(mel1, axis=0)
    mel2 = librosa.util.normalize(mel2, axis=0)

    intervals = np.diff(beats)
    median_int = int(np.median(intervals))

    corrs = []
    for b in range(min_beats, max_beats + 1):
        shift = b * median_int
        if shift + mel1.shape[1] > mel2.shape[1]:
            break
        corr_val = sum(
            np.dot(mel1[i], mel2[i, shift:shift + mel1.shape[1]])
            for i in range(n_mels)
        )
        corrs.append(corr_val)

    corrs = np.array(corrs)
    best_beats = np.argmax(corrs) + min_beats

    cycle_idxs = np.arange(0, len(beats), best_beats)

    orig_cycle_frames = initial_beats[0] + beats[cycle_idxs]
    print("cycle length: ",best_beats)
    return orig_cycle_frames, initial_beats

y,sr = librosa.load("../samples/sample8.wav",sr=None)
# cycle_indices, beat_frames = find_cycle_beat_indices_time_based_on_y(y,sr=sr)
# print("Cycle frames: ",beat_frames)

cycle_indices1, beat_frames1 = find_cycle_beat_indices(y,sr=sr)
print("Cycle frames before: ",beat_frames1)


