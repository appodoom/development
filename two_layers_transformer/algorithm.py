import numpy as np

# import numpy.typing as npt
from config import get_audio_data
import soundfile as sf
import random


def apply_cross_fade(hit_audio, fade_samples=500, attack_preserve=0):
    if len(hit_audio) <= fade_samples * 2:
        fade_samples = max(8, len(hit_audio) // 4)

    hit_audio = hit_audio.copy()

    # Cosine fade-in (very smooth attack)
    fade_in = 0.5 * (1 - np.cos(np.linspace(0, np.pi, fade_samples)))

    # Cosine fade-out (very smooth release)
    fade_out = 0.5 * (1 + np.cos(np.linspace(0, np.pi, fade_samples)))

    hit_audio[:fade_samples] *= fade_in
    hit_audio[-fade_samples:] *= fade_out

    return hit_audio


def get_random_proba_list(weights):
    output = []
    for weight in weights:
        choice = random.uniform(0, weight)
        output.append(choice)
    return output


def subdivisions_generator(
    y,
    maxsubd,
    added_hits_intervals,
    beat_length_in_samples,
    hit_probabilities,
    subdiv_proba,
    amplitudes,
    amplitudes_proba_list,
    tempos,
    sr=48000,
):
    subdiv_array = []
    tokens = []

    # Initialize tempo tracking
    current_tempo = tempos[0]

    for i in range(len(subdiv_proba)):
        subdiv_array.append(i)

    if sum(subdiv_proba) == 0:
        return y, [], ""

    maxsubdi = random.choices(population=subdiv_array, weights=subdiv_proba, k=1)[0]
    added_hits_intervals = sorted(added_hits_intervals, key=lambda x: x[0])
    subdivisions_y = np.zeros(len(y))

    curr_sample = 0
    # tempo_index = 0
    beat_index = 0

    chosen_div = maxsubd - maxsubdi
    tokens.append(f"SUBD_{chosen_div}")

    # Calculate beat length for current tempo
    beat_length_in_samples = int(60 * sr / current_tempo)
    maxsubd_length_arr = [
        int(beat_length_in_samples / chosen_div) for _ in range(chosen_div - 1)
    ]
    maxsubd_length_arr.append(beat_length_in_samples - sum(maxsubd_length_arr))

    hits = list(hit_probabilities[maxsubdi].keys())
    weights = list(hit_probabilities[maxsubdi].values())
    new_added_hits_intervals = []

    index_of_curr_subd_in_beat = 0
    sample_of_last_beat = 0

    while curr_sample < len(subdivisions_y):
        # Check if we need to update tempo (new beat)
        if curr_sample >= sample_of_last_beat + beat_length_in_samples:
            beat_index += 1
            index_of_curr_subd_in_beat = 0
            sample_of_last_beat = curr_sample

            # Update tempo if available
            if beat_index < len(tempos):
                new_tempo = tempos[beat_index]
                if new_tempo != current_tempo:
                    # tempo_diff = new_tempo - current_tempo
                    current_tempo = new_tempo

                beat_length_in_samples = int(60 * sr / current_tempo)

            # Get new random subdivision for the new beat
            maxsubdi = random.choices(
                population=subdiv_array, weights=subdiv_proba, k=1
            )[0]
            chosen_div = maxsubd - maxsubdi
            tokens.append(f"SUBD_{chosen_div}")
            maxsubd_length_arr = [
                int(beat_length_in_samples / chosen_div) for _ in range(chosen_div - 1)
            ]
            maxsubd_length_arr.append(beat_length_in_samples - sum(maxsubd_length_arr))

            hits = list(hit_probabilities[maxsubdi].keys())
            weights = list(hit_probabilities[maxsubdi].values())

        remaining = len(subdivisions_y) - curr_sample
        random_proba_list = get_random_proba_list(weights)
        chosen_hit = random.choices(hits, weights=random_proba_list, k=1)[0]
        chosen_amplitude = random.choices(
            population=amplitudes, weights=amplitudes_proba_list, k=1
        )[0]

        if chosen_hit == "S":
            tokens.append(f"HIT_{chosen_hit}")
            tokens.append(f"AMP_{chosen_amplitude}")
            curr_sample += maxsubd_length_arr[index_of_curr_subd_in_beat]
        else:
            hit_y_raw = np.asarray(get_audio_data(chosen_hit, sr), dtype=np.float32)
            add_len = min(len(hit_y_raw), remaining)
            hit_y = apply_cross_fade(hit_y_raw[:add_len])

            no_overlap = True
            for start, end in added_hits_intervals:
                if start <= curr_sample < end:
                    curr_sample += maxsubd_length_arr[index_of_curr_subd_in_beat]
                    no_overlap = False
                    break

            if no_overlap:
                subdivisions_y[curr_sample : curr_sample + add_len] += (
                    chosen_amplitude * hit_y[:add_len]
                )
                new_added_hits_intervals.append((curr_sample, curr_sample + add_len))
                curr_sample += maxsubd_length_arr[index_of_curr_subd_in_beat]
                tokens.append(f"HIT_{chosen_hit}")
                tokens.append(f"AMP_{chosen_amplitude}")
            else:
                tokens.append("HIT_S")
                tokens.append(f"AMP_{chosen_amplitude}")

        index_of_curr_subd_in_beat += 1

    y += subdivisions_y
    new_added_hits_intervals.extend(added_hits_intervals)
    return y, new_added_hits_intervals, " ".join(tokens)


# TODO: REDO (WRONG UNDERSTANDING)
def get_deviated_sample(
    start_of_window: int,
    end_of_window: int,
    expected_hit_timestamp: int,
    shift_proba: float,
):
    if random.random() >= shift_proba:
        return expected_hit_timestamp
    return int(random.uniform(start_of_window, end_of_window))


"""
    get_window_by_beat returns a tuple containing the window start and window end
    where window is the allowed interval in which the hit can fall (mimick human error)
"""


def get_window_by_beat(expected_hit_timestamp: int, beat_len: int) -> tuple[int, int]:
    half = int(0.05 * beat_len)
    start_of_window = max(0, expected_hit_timestamp - half)
    end_of_window = expected_hit_timestamp + half
    return (start_of_window, end_of_window)


"""
    skeleton_generator generates an audio of desired length just playing the skeleton provided
"""


def skeleton_generator(
    amplitude: float,
    skeleton: list[tuple[float, str]],
    num_cycles: int,
    tempos: list[float],
    shift_proba: float,
    sr=48000,
):
    # Initialize with first tempo
    current_tempo = tempos[0]
    beat_length_in_samples = int((60 / current_tempo) * sr)
    skeleton_length = len(skeleton)
    num_of_beats_in_audio = num_cycles * sum(x[0] for x in skeleton)
    tokens = []

    # Track tempo tokens - start with initial tempo difference (0.0)

    skeleton_hits_intervals = []
    y = np.zeros(0, dtype=np.float32)

    expected_hit_timestamp = 0
    curr_beat = 0.0
    i = 0
    tempo_index = 0  # Track which tempo we're using

    while curr_beat < num_of_beats_in_audio:
        beat_duration = skeleton[i % skeleton_length][0]  # delay in beats
        curr_beat += beat_duration

        # Update tempo if we've moved to a new beat index
        if int(curr_beat) > tempo_index and int(curr_beat) < len(tempos):
            tempo_index = int(curr_beat)
            new_tempo = tempos[tempo_index]
            # tempo_diff = new_tempo - current_tempo
            current_tempo = new_tempo
            beat_length_in_samples = int((60 / current_tempo) * sr)

        tokens.append(f"DELAY_{beat_duration}")
        curr_hit = skeleton[i % skeleton_length][1]

        y_hit_raw = np.asarray(get_audio_data(curr_hit, sr), dtype=np.float32)
        y_hit = apply_cross_fade(y_hit_raw)
        tokens.append(f"HIT_{curr_hit}")

        expected_hit_timestamp += int(beat_duration * beat_length_in_samples)

        start_of_window, end_of_window = get_window_by_beat(
            expected_hit_timestamp, beat_length_in_samples
        )
        adjusted_hit_timestamp = get_deviated_sample(
            start_of_window, end_of_window, expected_hit_timestamp, shift_proba
        )

        deviation_samples = adjusted_hit_timestamp - expected_hit_timestamp
        tokens.append(f"DEV_{deviation_samples}")

        end_of_hit_timestamp = adjusted_hit_timestamp + len(y_hit)

        # Padding and adding the hit
        if end_of_hit_timestamp > len(y):
            pad_len = end_of_hit_timestamp - len(y)
            y = np.pad(y, (0, pad_len), mode="constant")

        y[adjusted_hit_timestamp:end_of_hit_timestamp] += amplitude * y_hit
        skeleton_hits_intervals.append((adjusted_hit_timestamp, end_of_hit_timestamp))
        i += 1

    # Return from first hit timestamp
    start_time = skeleton_hits_intervals[0][0] if skeleton_hits_intervals else 0
    return (
        y[start_time:],
        beat_length_in_samples,
        skeleton_hits_intervals,
        " ".join(tokens),
    )


"""
    get_subdivision_hit_probabilities is a utility function that transforms the matrix into a list of dictionaries where
    each entry in the list corresponds to a subdivision
"""


def get_subdivision_hit_probabilities(
    maxsubd: int,
    number_of_hits: int,
    hits_list: list[str],
    probabilities_dict: dict[str, list],
) -> list[dict[str, float]]:
    out = []

    for col_index in range(maxsubd):
        current_process = {}
        sum_of_probabilities = 0
        for j in range(number_of_hits):
            current_hit = hits_list[j]
            current_process[current_hit] = probabilities_dict[current_hit][col_index]
            sum_of_probabilities += probabilities_dict[current_hit][col_index]
        if sum_of_probabilities > 100:
            raise ValueError(
                f"Column {col_index} probabilities sum to {sum_of_probabilities} (>100). "
                "Reduce one or more values so that the sum ≤ 100."
            )
        # adding silence with other hits
        current_process["S"] = 100 - sum_of_probabilities
        out.append(current_process)

    return out


"""
    get_available_choices gives you a list of possible actions you can do to the tempo:
    1: keep the same
    2: increase
    3: decrease
"""


def get_available_choices(
    current_tempo: float, initial_tempo: float, allowed_tempo_deviation: float
) -> list[int]:
    lower = initial_tempo - allowed_tempo_deviation
    upper = initial_tempo + allowed_tempo_deviation
    choices = [1]  # keep
    if current_tempo <= lower:
        choices.append(2)  # increase
    elif current_tempo >= upper:
        choices.append(3)  # decrease
    else:
        choices.extend([2, 3])
    return choices


"""
    get_tempos returns a list of length number_of_beats. 
    tempos[i]: the tempo between beat i and beat i+1
"""


def get_tempos(
    number_of_beats: int, initial_tempo: float, allowed_tempo_deviation: float
):
    tempos = []
    current_tempo = initial_tempo
    i = 0

    # for each beat, decide whether to increase, decrease or keep the same tempo as the beat before
    while i <= number_of_beats:
        choices = get_available_choices(
            current_tempo, initial_tempo, allowed_tempo_deviation
        )
        choice = random.choice(choices)
        if choice == 2:  # Increase
            deviation = random.uniform(
                0, initial_tempo + allowed_tempo_deviation - current_tempo
            )
            tempos.append(current_tempo + deviation)
        elif choice == 3:  # Decrease
            deviation = random.uniform(
                0, initial_tempo + allowed_tempo_deviation - current_tempo
            )
            tempos.append(max(1, current_tempo - deviation))
        else:  # Keep
            tempos.append(current_tempo)
        i += 1

    return tempos, " ".join([str(i) for i in tempos])


"""
    merge_skeleton_with_variations generates both skeleton music and then variations
"""


def merge_skeleton_with_variations(
    maxsubd: int,
    probabilities_dict: dict[str, list],
    bpm: float,
    skeleton: list[tuple[float, str]],
    num_cycles: int,
    subdiv_proba: list[float],
    amplitudes: list[float],
    amplitudes_proba_list: list[float],
    cycle_length: float,
    shift_proba: float,
    allowed_tempo_deviation: float,
    sr: int = 48000,
):
    # calculating the total number of beats in the audio
    num_of_beats = int(num_cycles * sum(float(x[0]) for x in skeleton))

    # get the list of tempos for every beat
    tempos, tempo_tokens = get_tempos(
        number_of_beats=num_of_beats,
        initial_tempo=bpm,
        allowed_tempo_deviation=allowed_tempo_deviation,
    )

    # getting the notes
    hits_list = list(probabilities_dict.keys())
    number_of_hits = len(hits_list)

    subdivision_hit_probabilities = get_subdivision_hit_probabilities(
        maxsubd=maxsubd,
        number_of_hits=number_of_hits,
        hits_list=hits_list,
        probabilities_dict=probabilities_dict,
    )

    y, beat_length_in_samples, added_hits_intervals, skeleton_tokens = (
        skeleton_generator(
            shift_proba=shift_proba,
            amplitude=amplitudes[-1],  # always play at highest amplitude
            skeleton=skeleton,
            num_cycles=num_cycles,
            sr=sr,
            tempos=tempos,
        )
    )
    y, added_hits_intervals, var_tokens = subdivisions_generator(
        y=y,
        maxsubd=maxsubd,
        amplitudes=amplitudes,
        amplitudes_proba_list=amplitudes_proba_list,
        added_hits_intervals=added_hits_intervals,
        beat_length_in_samples=beat_length_in_samples,
        hit_probabilities=subdivision_hit_probabilities,
        subdiv_proba=subdiv_proba,
        tempos=tempos,
    )

    return y, tempo_tokens + "\n" + skeleton_tokens + "\n" + var_tokens


"""
    get_probability_dict transforms our matrix into a dictionary of note: probability vector
    params:
        - list matrix: the variation matrix
        - list[str] notes: naming of the matrix rows
"""


def get_probability_dict(matrix: list, notes: list[str]) -> dict[str, list]:
    return dict(zip(notes, matrix))


"""
    generate_derbouka is the main entry point of the algorithm, it takes the various
    settings given by the user, and generates the derbouka music desired.

    params:
        - str uuid: unique id of music
        - int num_cycles: number of cycles
        - float cycle_length: length of cycle in beats
        - float bpm: tempo in bpm
        - int maxsubd: the maximum subdivision of a beat
        - float shift_proba: the probability of wrong placement
        - float allowed_tempo_deviation: +- bpm by which the music is allowed to move
        - list[tuple[float, str]] skeleton: the skeleton of the cycle
        - list matrix: the variation matrix
        - float amplitude_variation: probability for a hit to fall in the middle bin
"""


def generate_derbouka(
    uuid: str,
    num_cycles: int,
    cycle_length: float,
    bpm: float,
    maxsubd: int,
    shift_proba: float,
    allowed_tempo_deviation: float,
    skeleton: list[tuple[float, str]],
    matrix: list,
    amplitude_variation: float,
) -> None:
    # supported notes
    notes = ["D", "OTA", "OTI", "PA2"]
    VOLUME = 3

    # amplitude bins
    amplitudes = [0.1015 * VOLUME, 0.5 * VOLUME, 1 * VOLUME]

    # amplitude bins probabilities
    amplitudes_proba_list = [
        (1 - amplitude_variation) / 2,
        amplitude_variation,
        (1 - amplitude_variation) / 2,
    ]
    # subdivisions probability vector
    subdiv_proba = matrix[0]
    # the rest of variation percentages
    matrix = matrix[1:]
    probabilities_dict = get_probability_dict(matrix=matrix, notes=notes)

    y_generated, tokens = merge_skeleton_with_variations(
        amplitudes=amplitudes,
        amplitudes_proba_list=amplitudes_proba_list,
        shift_proba=shift_proba,
        maxsubd=maxsubd,
        bpm=bpm,
        probabilities_dict=probabilities_dict,
        skeleton=skeleton,
        num_cycles=num_cycles,
        subdiv_proba=subdiv_proba,
        cycle_length=cycle_length,
        allowed_tempo_deviation=allowed_tempo_deviation,
    )

    with open(f"./data/{uuid}.derbake", "w") as f:
        f.write(str(bpm) + "\n" + tokens)
    # writing the result
    sf.write(f"./data/{uuid}.wav", data=y_generated, samplerate=48000)
