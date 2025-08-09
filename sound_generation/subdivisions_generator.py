import librosa
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import random
from config import get_audio_data
from squeleton_generator import (
    squeleton_generator,
    in_bpm,
    in_maxsubd,
    in_num_cycles,
    in_skeleton,
)

# this code is part of track 3
# this will call the generate squeleton function
# and will put random hits based on the user input for each of the following parameters
# even subdivision percentage and the ones he already have for the squeleton part

even_subdivisions_percentage = 0.5
hit_probabilities = {
    "D": 0.3,
    "OTA": 0.2,
    "OTI": 0.1,
    "PA2": 0.15,
    "RA": 0.1,
    "T1": 0.1,
    "T2": 0.05,
    "S": 0.7,
}


def subdivisions_generator(
    y,
    maxsubd,
    squeleton_samples_indices,
    beat_length_in_samples,
    hit_probabilities,
    type,
    even_subdivisions_percentage,
):
    subdivisions_y = np.zeros(len(y))
    index_of_current_slot_samples = 0
    duration_in_sample_by_maxsub = beat_length_in_samples // maxsubd
    hits = list(hit_probabilities.keys())
    weights = list(hit_probabilities.values())
    added_hits_indicies_in_samples = []
    while index_of_current_slot_samples < len(subdivisions_y):
        if random.random() >= even_subdivisions_percentage:
            index_of_current_slot_samples += duration_in_sample_by_maxsub
            continue

        remaining = len(subdivisions_y) - index_of_current_slot_samples
        hit_choosen = random.choices(hits, weights=weights, k=1)[0]

        hit_y = get_audio_data(hit_choosen)
        add_len = min(len(hit_y), remaining)

        if hit_choosen == "S":
            index_of_current_slot_samples += duration_in_sample_by_maxsub
        else:
            for sk_start, sk_end in squeleton_samples_indices:
                if (
                    index_of_current_slot_samples >= sk_start
                    and (index_of_current_slot_samples + add_len)
                    <= sk_start + duration_in_sample_by_maxsub
                ):
                    index_of_current_slot_samples += duration_in_sample_by_maxsub
                    break
            else:
                subdivisions_y[
                    index_of_current_slot_samples : index_of_current_slot_samples
                    + add_len
                ] += hit_y[:add_len]
                added_hits_indicies_in_samples.append(
                    (
                        index_of_current_slot_samples,
                        index_of_current_slot_samples + add_len,
                    )
                )
                index_of_current_slot_samples += duration_in_sample_by_maxsub
    y += subdivisions_y
    sf.write(
        f"test_with_{type}_silence_{even_subdivisions_percentage}%.wav",
        y,
        samplerate=48000,
    )
    return y, added_hits_indicies_in_samples


def plot(wav_path, intervals):
    y, sr = librosa.load(wav_path, sr=None)
    times = np.arange(len(y)) / sr
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(times, y, linewidth=0.8)
    for start, end in intervals:
        t_start = start / sr
        t_end = end / sr
        ax.axvline(
            t_start,
            color="r",
            linestyle="--",
            label="start" if start == intervals[0][0] else "",
        )
        ax.axvline(
            t_end,
            color="g",
            linestyle="--",
            label="end" if end == intervals[0][1] else "",
        )

    ax.legend(loc="upper right")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Audio Waveform with Interval Markers")
    plt.tight_layout()
    plt.show()


squeleton_y, beat_length_in_samples, skeleton_length, squeleton_samples_indices = (
    squeleton_generator(in_bpm, in_skeleton, in_maxsubd, in_num_cycles, "test.wav")
)
y, added_hits_indicies_in_samples = subdivisions_generator(
    hit_probabilities=hit_probabilities,
    y=squeleton_y,
    squeleton_samples_indices=squeleton_samples_indices,
    beat_length_in_samples=beat_length_in_samples,
    maxsubd=in_maxsubd,
    type="even",
    even_subdivisions_percentage=even_subdivisions_percentage,
)

final_y, odd_indicies = subdivisions_generator(
    hit_probabilities=hit_probabilities,
    y=squeleton_y,
    squeleton_samples_indices=added_hits_indicies_in_samples
    + squeleton_samples_indices,
    beat_length_in_samples=beat_length_in_samples,
    maxsubd=3,
    type="odd",
    even_subdivisions_percentage=even_subdivisions_percentage,
)

# plot(wav_path="./test_with_odd_silence_70%.wav", intervals=squeleton_samples_indices)
