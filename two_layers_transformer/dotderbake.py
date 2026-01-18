import numpy as np
from config import get_audio_data
from algorithm import apply_cross_fade
import soundfile as sf


def play_from_dotderbake(file_path):
    with open(file_path, "r") as f:
        data = f.read()

    lines = data.split("\n")

    if len(lines) != 4:
        print("Wrong format")
        return

    initial_tempo = float(lines[0])
    tempos = [float(i) for i in lines[1].split(" ")]
    skeleton_tokens = lines[2].split(" ")
    var_tokens = lines[3].split(" ")

    regenerate(initial_tempo, tempos, skeleton_tokens, var_tokens)


def subdivisions_regenerator(
    tokens,
    tempos,
    y,
    added_hits_intervals,
    sr=48000,
):
    # Initialize tempo tracking
    current_tempo = tempos[0]
    if len(tokens) < 3:
        return y

    added_hits_intervals = sorted(added_hits_intervals, key=lambda x: x[0])
    subdivisions_y = np.zeros(len(y))

    curr_sample = 0
    beat_index = 0
    curr_token = 0
    chosen_div = None
    while curr_token < len(tokens) and chosen_div is None:
        if tokens[curr_token].startswith("SUBD_"):
            try:
                chosen_div = int(tokens[curr_token].split("_")[1])
            except ValueError:
                chosen_div = None
        curr_token += 1
    if not chosen_div:
        return y
    # Calculate beat length for current tempo
    beat_length_in_samples = int(60 * sr / current_tempo)
    maxsubd_length_arr = [
        int(beat_length_in_samples / chosen_div) for _ in range(chosen_div - 1)
    ]
    maxsubd_length_arr.append(beat_length_in_samples - sum(maxsubd_length_arr))

    new_added_hits_intervals = []

    index_of_curr_subd_in_beat = 0
    while curr_sample < len(subdivisions_y) and curr_token < len(tokens):
        # Check if we need to update tempo (new beat)
        if index_of_curr_subd_in_beat >= chosen_div:
            beat_index += 1
            index_of_curr_subd_in_beat = 0

            # Update tempo if available
            if beat_index < len(tempos):
                new_tempo = tempos[beat_index]
                if new_tempo != current_tempo:
                    current_tempo = new_tempo

                beat_length_in_samples = int(60 * sr / current_tempo)

            # Update subdivision only if an explicit SUBD token is provided
            if curr_token < len(tokens) and tokens[curr_token].startswith("SUBD_"):
                try:
                    chosen_div = int(tokens[curr_token].split("_")[1])
                except ValueError:
                    chosen_div = chosen_div
                curr_token += 1

            if chosen_div <= 0:
                break
            maxsubd_length_arr = [
                int(beat_length_in_samples / chosen_div) for _ in range(chosen_div - 1)
            ]
            maxsubd_length_arr.append(beat_length_in_samples - sum(maxsubd_length_arr))

        if curr_token >= len(tokens):
            break
        if tokens[curr_token].startswith("SUBD_"):
            if index_of_curr_subd_in_beat != 0:
                beat_index += 1
                index_of_curr_subd_in_beat = 0
                if beat_index < len(tempos):
                    current_tempo = tempos[beat_index]
                    beat_length_in_samples = int(60 * sr / current_tempo)
            try:
                chosen_div = int(tokens[curr_token].split("_")[1])
            except ValueError:
                chosen_div = chosen_div
            curr_token += 1
            if chosen_div <= 0:
                continue
            maxsubd_length_arr = [
                int(beat_length_in_samples / chosen_div) for _ in range(chosen_div - 1)
            ]
            maxsubd_length_arr.append(beat_length_in_samples - sum(maxsubd_length_arr))
            continue

        remaining = len(subdivisions_y) - curr_sample
        if not tokens[curr_token].startswith("HIT_"):
            curr_token += 1
            continue
        chosen_hit = tokens[curr_token].split("_")[1]
        curr_token += 1
        if curr_token >= len(tokens) or not tokens[curr_token].startswith("AMP_"):
            continue
        try:
            chosen_amplitude = float(tokens[curr_token].split("_")[1])
        except ValueError:
            curr_token += 1
            continue
        curr_token += 1
        if chosen_hit == "S":
            curr_sample += maxsubd_length_arr[index_of_curr_subd_in_beat]
        else:
            hit_y_raw = np.asarray(get_audio_data(chosen_hit, sr), dtype=np.float32)
            add_len = min(len(hit_y_raw), remaining)
            hit_y = apply_cross_fade(hit_y_raw[:add_len])

            subdivisions_y[curr_sample : curr_sample + add_len] += (
                chosen_amplitude * hit_y[:add_len]
            )
            new_added_hits_intervals.append((curr_sample, curr_sample + add_len))
            curr_sample += maxsubd_length_arr[index_of_curr_subd_in_beat]

        index_of_curr_subd_in_beat += 1

    y += subdivisions_y
    return y


def skeleton_regenerator(amplitude, tempos, tokens, sr=48000):
    # Initialize with first tempo
    current_tempo = tempos[0]
    beat_length_in_samples = int((60 / current_tempo) * sr)
    num_of_beats_in_audio = len(tempos)
    skeleton_hits_intervals = []
    y = np.zeros(0, dtype=np.float32)

    expected_hit_timestamp = 0
    curr_beat = 0
    tempo_index = 0  # Track which tempo we're using

    curr_token = 0
    while curr_beat < num_of_beats_in_audio and curr_token < len(tokens):
        beat_duration = float(tokens[curr_token].split("_")[1])  # delay in beats
        curr_beat += beat_duration

        # Update tempo if we've moved to a new beat index
        if int(curr_beat) > tempo_index and int(curr_beat) < len(tempos):
            tempo_index = int(curr_beat)
            new_tempo = tempos[tempo_index]
            current_tempo = new_tempo
            beat_length_in_samples = int((60 / current_tempo) * sr)

        curr_token += 1
        curr_hit = tokens[curr_token].split("_")[1]

        y_hit_raw = np.asarray(get_audio_data(curr_hit, sr), dtype=np.float32)
        y_hit = apply_cross_fade(y_hit_raw)

        expected_hit_timestamp += int(beat_duration * beat_length_in_samples)

        curr_token += 1
        adjusted_hit_timestamp = expected_hit_timestamp + int(
            tokens[curr_token].split("_")[1]
        )
        end_of_hit_timestamp = adjusted_hit_timestamp + len(y_hit)

        # Padding and adding the hit
        if end_of_hit_timestamp > len(y):
            pad_len = end_of_hit_timestamp - len(y)
            y = np.pad(y, (0, pad_len), mode="constant")

        y[adjusted_hit_timestamp:end_of_hit_timestamp] += amplitude * y_hit
        skeleton_hits_intervals.append((adjusted_hit_timestamp, end_of_hit_timestamp))
        curr_token += 1

    # Return from first hit timestamp
    start_time = skeleton_hits_intervals[0][0] if skeleton_hits_intervals else 0
    return (
        y[start_time:],
        beat_length_in_samples,
        skeleton_hits_intervals,
    )


def regenerate(initial_tempo, tempos, skeleton_tokens, var_tokens, sr=48000):
    VOLUME = 3

    y_sk, _, skeleton_hits_intervals = skeleton_regenerator(
        amplitude=VOLUME, tokens=skeleton_tokens, tempos=tempos
    )
    y = subdivisions_regenerator(var_tokens, tempos, y_sk, skeleton_hits_intervals)
    sf.write("input.wav", data=y, samplerate=48000)


play_from_dotderbake("./input.derbake")
