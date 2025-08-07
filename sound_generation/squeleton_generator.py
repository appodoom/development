import numpy as np
import soundfile as sf
from config import get_audio_data
# This code is part of track #3
# It is responsible to take user preference and generate derbakke music accordingly
# The generator will take

"""
    tempo in bpm
    skeleton (format discussed below)
    maximum subdivisions between two consecutive beats
    number of cycles
"""

in_bpm = 120  # bpm

in_skeleton = [(2, "D"), (1, "D"), (1, "PA2")]
# This means, first beat is doom, after 1 beat place tek, after 0.5 beats place open tek, after 1 beat doom, then reiterate

in_maxsubd = 1000  # smallest note is an 8th of a beat
in_num_cycles = 16  # cycles


def squeleton_generator(bpm, skeleton, maxsubd, num_cycles, out_path, sr=48000):
    beat_length_in_samples = int((60 / bpm) * sr)
    skeleton_length = len(skeleton)
    num_of_beats_in_audio = num_cycles * skeleton_length

    length_in_samples = int(
        sum([x[0] * beat_length_in_samples for x in skeleton]) * num_cycles
    )
    squeleton_samples_indices = []
    y = np.zeros(length_in_samples + beat_length_in_samples)

    accumulator = i = 0
    while accumulator <= num_of_beats_in_audio:
        accumulator += skeleton[i % skeleton_length][0]
        curr_beat = skeleton[i % skeleton_length][1]
        y_hit = get_audio_data(curr_beat, sr)
        hit_timestamp = int(accumulator * beat_length_in_samples)
        end_index = hit_timestamp + len(y_hit)

        # place curr_beat on accumulator
        if end_index <= len(y):
            y[hit_timestamp:end_index] += y_hit
            squeleton_samples_indices.append((hit_timestamp, end_index))
        i += 1
    return y, beat_length_in_samples, skeleton_length, squeleton_samples_indices
    # sf.write(out_path, data=y, samplerate=sr)


squeleton_generator(in_bpm, in_skeleton, in_maxsubd, in_num_cycles, "test.wav")
