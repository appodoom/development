import librosa
import numpy as np
import os
import glob
import json
import numpy as np


MEL_JSON_PATH = "mel_48000.json"
data_folder = "../../samples/khaled_initial"
bin_values = np.array([0.125,0.15, 0.225, 0.25, 0.275, 0.3, 0.4,0.5, 0.6,1])
   

def sliding_cross_correlation(X, Y):
    if X.shape[1] > Y.shape[1]:
        X, Y = Y, X

    n_freq, n_time_X = X.shape
    _, n_time_Y = Y.shape

    best_score = -np.inf
    best_offset = None 
    norm_X = np.linalg.norm(X)
    for offset in range(n_time_Y - n_time_X + 1):
        
        Y_slice = Y[:, offset:offset + n_time_X]
        if (norm_X!=0 and np.linalg.norm(Y_slice)!=0) :
            score = np.tensordot(X, Y_slice, axes=2) / (norm_X * np.linalg.norm(Y_slice)) 
        else: continue

        if score > best_score:
            best_score = score
            best_offset = offset

    return best_score, best_offset

def load_mel_from_json(file_path):
    with open(file_path,'r') as f:
        loaded = json.load(f)
    return { name: np.array(mel) for name, mel in loaded.items() }

def load_file(path):
    y, sr = librosa.load(path, sr=None)
    y = librosa.util.normalize(y)
    return y, sr

def get_onsets(onset_frames,y):
    onset_samples=[onset_frames[0]//2]
    for i in range(len(onset_frames)-1):
        avg=(onset_frames[i]+onset_frames[i+1])//2
        onset_samples.append(avg)
    onset_samples = librosa.frames_to_samples(onset_samples)
    return np.concatenate((onset_samples, [len(y)]))

def get_intervals(onsets):
    out = []
    for i in range(len(onsets) - 1):
        start = onsets[i]
        end = onsets[i+1]
        out.append((start, end))
    return np.array(out)
def adjust_tempo(tempo):
    while tempo > 150:
        tempo /= 2
    while tempo < 50:
        tempo*=2
    return tempo

def classify_onset_durations(i,y, sr, tempo,onset_frames):
    # 1. Extract onsets (times in seconds)
    y2=librosa.samples_to_time(y)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    onset_times = list(onset_times)
    onset_times.append(y2[-1])
    onset_times = np.array(onset_times)
    orig_tempo = tempo
    print(f"Orig tempo of sample {i} is : {orig_tempo}")
    # 2. Calculate deltas between consecutive onsets (in seconds)
    deltas = np.diff(onset_times)
    tempo = adjust_tempo(tempo)
    print(f"Tempo of sample {i} after adjustment is {tempo}")
    # 3. Define note durations (seconds)
    quarter_duration = 60.0 / tempo
    note_durations = {
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
        1.33:quarter_duration*(3)
        # 28:quarter_duration*(1/7),
        # 56:quarter_duration*(1/14),
        # 14:quarter_duration*(2/7),
        # 9.33:quarter_duration*(3/7),
        # 7:quarter_duration*(4/7),
        # 5.6:quarter_duration*(5/7),
        # 4.67:quarter_duration*(6/7),
        # 20:quarter_duration*(1/5),
        # 40:quarter_duration*(1/10),
        # 10:quarter_duration*(2/5),
        # 6.67:quarter_duration*(3/5),
        # 5:quarter_duration*(4/5),
        # 36:quarter_duration*(1/9),
        # 72:quarter_duration*(1/18),
        # 18:quarter_duration*(2/9),
        # 9:quarter_duration*(4/9),
        # 7.2:quarter_duration*(5/9),
        # 6:quarter_duration*(6/9),
        # 5.1:quarter_duration*(7/9),
        # 4.5:quarter_duration*(8/9)
    }

    # 4. Classify each delta to closest note duration
    classified_notes = []
    for d in deltas:
        # Find closest note duration by minimum absolute difference
        closest_note = min(note_durations.keys(), key=lambda n: abs(d - note_durations[n]))
        classified_notes.append(closest_note)

    # Note: deltas has length len(onsets) - 1, add an assumed last note (e.g. 16th)
    # or keep length one less than onsets as per difference array
    return classified_notes, onset_times

def classify_onsets_by_amp(
    y,
    onset_frames,
    bin_values,
    hop_length=512
):
    onset_samples = librosa.frames_to_samples(onset_frames, hop_length=hop_length)
    onset_amps = np.abs(y[onset_samples-1])
    bin_indices = [int(np.argmin(np.abs(bin_values - amp))) for amp in onset_amps]
    return bin_indices

def get_corpus(mel_dict:dict , folder_path:str,bin_values):
    wav_files = sorted(glob.glob(os.path.join(folder_path, '*.wav')))
    # for fundamental_pulse in mel_dict:
    #     os.makedirs(f"./cluster/{fundamental_pulse}", exist_ok=True)
    k=0
    print(wav_files)
    for i, wav_path in enumerate(wav_files):
        print(f"Curent path : {wav_path}")
        with open(f"./old{i+1}.txt", "w") as kokliko:
            hits=[]
            y , sr= librosa.load(wav_path, sr=None)
            tempo , _ = librosa.beat.beat_track(y=y,sr=sr)
            onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=512)
            onsets = get_onsets(onset_frames=onset_frames,y=y)
            intervals = get_intervals(onsets)
            # duration,_= classify_onset_durations(i,y,sr,tempo,onset_frames)
            # amplitudes = classify_onsets_by_amp(y=y,onset_frames=onset_frames,bin_values=bin_values)
            for interval in intervals:
                k+=1
                segment=y[interval[0]:interval[1]]
                M=librosa.feature.melspectrogram(y=segment, sr=sr)
                max_score=-np.inf
                best_pulse=''
                for fundamental_pulse in mel_dict:
                    score , _ =sliding_cross_correlation(M, mel_dict[fundamental_pulse])
                    if score>max_score:
                        max_score=score
                        best_pulse=fundamental_pulse
                print(f"Choosen pulse is : {best_pulse}")
                hits.append(best_pulse)
                # sf.write(os.path.join(f"./cluster/{best_pulse}", f"onset_{k}.wav"), y[interval[0]:interval[1]], sr)
            for p in range(len(hits)):
                kokliko.write(f"{hits[p]}\n")
            print(f"sample_{i+1} done!")
    print("Done :)")
mel_dict=load_mel_from_json(MEL_JSON_PATH)
get_corpus(mel_dict, data_folder,bin_values)
