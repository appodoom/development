import librosa
import numpy as np
import soundfile as sf  # type: ignore[import]
from augment_data import find_cycle_beat_indices

def overlay_clicks(
    y: np.ndarray,
    sr: int,
    beat_frames: np.ndarray,
    cycle_start_frames: np.ndarray,
    click_freq_beat: float = 1000,
    click_freq_cycle: float = 2000,
    gain_beat: float = 0.3,
    gain_cycle: float = 0.3,
) -> np.ndarray:
    """
    Generate two click-trains (one at every beat, one at every cycle start)
    and mix them into y.
    """
    # total length in samples
    length = len(y)

    # “regular” clicks at every beat
    click_beats = librosa.clicks(
        frames=beat_frames,
        sr=sr,
        click_freq=click_freq_beat,
        length=length,
    )

    # “special” clicks at each cycle start
    click_cycles = librosa.clicks(
        frames=cycle_start_frames,
        sr=sr,
        click_freq=click_freq_cycle,
        length=length,
    )

    return y + gain_beat * click_beats + gain_cycle * click_cycles


def enhance_sample_with_clicks(in_path, out_path, factor=1):
    # 1) load
    y, sr = librosa.load(in_path, sr=None)

    # 2) detect beats & cycles
    cycle_idxs, beat_frames = find_cycle_beat_indices(y, sr)
    # convert cycle‐idxs (indices into beat_frames) to actual frame positions
    cycle_start_frames = beat_frames[cycle_idxs]

    # 3) “enhance” (your existing shuffling/concatenation routine)
    out = y.copy()
    # for _ in range(factor):
    #     new_y = enhance(y, sr)   # your existing enhance()
    #     out = np.concatenate((out, new_y))

    # 4) because out is multiple concatenations, we need to tile our frame‐lists
    #    into sample‐offsets for each repetition
    base_len = len(y)
    beat_samples = librosa.frames_to_samples(beat_frames)
    cycle_samples = librosa.frames_to_samples(cycle_start_frames)

    # build full lists of sample‐positions in the “out” signal
    all_beat_samples = np.hstack([
        beat_samples + i * base_len for i in range(factor + 1)
    ])
    all_cycle_samples = np.hstack([
        cycle_samples + i * base_len for i in range(factor + 1)
    ])

    # convert those back into frames (for librosa.clicks)
    all_beat_frames = librosa.samples_to_frames(all_beat_samples, hop_length=512)
    all_cycle_frames = librosa.samples_to_frames(all_cycle_samples, hop_length=512)

    # 5) overlay clicks on the entire out
    out_with_clicks = overlay_clicks(
        out,
        sr,
        beat_frames=all_beat_frames,
        cycle_start_frames=all_cycle_frames,
        click_freq_beat=1000,
        click_freq_cycle=2000,
        gain_beat=0.3,
        gain_cycle=0.3,
    )

    # 6) write
    sf.write(out_path, out_with_clicks, sr)
    print(f"Wrote {out_path}")

enhance_sample_with_clicks("../sample8.wav","./click_sample8.wav")
#sample 8 is 5

# implement tempo code and chek for the window size because there is two beats so fast wara ba3ed!
# there is a problem triplet
