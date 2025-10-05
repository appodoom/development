from sliding_cross_correlation import sliding_cross_correlation
from load_files import load_wav_file, load_json_file
from get_intervals import get_intervals, get_intervals_for_duration
from get_node_duration import get_note_duration
import librosa
import numpy as np


def adjust_tempo(y, sr):
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    # while tempo > 150:
    #     tempo = tempo // 2
    # while tempo < 50:
    #     tempo *= 2
    print(f"tempo : {tempo}")
    return tempo


def get_corpus(fundamentals_json_mel_path, wav_file_path):
    y, sr = load_wav_file(wav_file_path)
    y = librosa.resample(y, orig_sr=sr, target_sr=48000)
    sr = 48000
    y = np.concatenate([np.zeros(48000), y])
    fundamentals = load_json_file(fundamentals_json_mel_path)[0]
    intervals_duration = get_intervals_for_duration(y, sr)
    intervals = get_intervals(y, sr)
    tempo = adjust_tempo(y, sr)
    quarter_duration = 60.0 / tempo  # in seconds
    note_durations = get_note_duration(quarter_duration)  # in seconds
    classified_hits = []
    for interval, interval_dur in zip(intervals, intervals_duration):
        segment = y[interval[0] : interval[1]]
        mel = librosa.feature.melspectrogram(y=segment, sr=sr)  # TODO revisit
        best_score = -np.inf
        best_hit = ""
        for fundamental_hit in fundamentals:
            score, _ = sliding_cross_correlation(mel, fundamentals[fundamental_hit])
            # print(
            #     f"Current fundemental is: {fundamental_hit} with corr score = {score}"
            # )
            if score > best_score:
                best_score = score
                best_hit = fundamental_hit
        # print(f"Choosen fundemental is: {best_hit} with max corr score = {best_score}")
        # print("")
        hit_duration = (interval_dur[1] - interval_dur[0]) / sr  # in seconds
        min_diff = np.inf
        best_note = ""
        for note in note_durations:
            diff = abs(note_durations[note] - hit_duration)
            if diff < min_diff:
                min_diff = diff
                best_note = note
        # print(f"best_note : {best_note}")
        classified_hits.append(best_hit + "_" + str(best_note))
        # if best_note == 1:
        # print(f"Current_hit is : {best_hit}")
        # print(f"hit_duration= {hit_duration}")
        # print(f"note_durations[best_note] = {note_durations[best_note]}")
        # print(f"rem =  {hit_duration - note_durations[best_note]}")
        # print("")
        if best_note == 1:
            additional_min_diff = np.inf
            additional_note = ""
            whole_note_dur = note_durations[1]
            remaining_dur = abs(whole_note_dur - hit_duration)
            # print(remaining_dur)
            if remaining_dur != 0:
                nb_of_complete_dur = int((hit_duration // whole_note_dur)[0])
                # print(f"nb_of_complete_dur = {nb_of_complete_dur}")
                remaining_dur -= whole_note_dur * nb_of_complete_dur
                for note in note_durations:
                    diff = abs(note_durations[note] - remaining_dur)
                    if diff < additional_min_diff:
                        additional_min_diff = diff
                        additional_note = str(note)
                for _ in range(nb_of_complete_dur):
                    classified_hits.append("S_1")
                classified_hits.append(f"S_{additional_note}")
    return classified_hits, tempo


# classified_hits, _ = get_corpus(
#     fundamentals_json_mel_path="../mel.json",
#     wav_file_path="../data/skeleton_and_silence_cycle.wav",
# )
# print(classified_hits)
