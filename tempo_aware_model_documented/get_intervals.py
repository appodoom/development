"""
Detect onset frames in an audio signal and return their sample indices.
And build contiguous (start_sample, end_sample) intervals between successive onsets.
"""

from typing import List, Tuple
import numpy as np
import librosa


def get_onsets(
    y: np.ndarray,
    sr: int,
    hop_length: int = 512,
) -> np.ndarray:
    """
    Parameters
    ----------
    y : np.ndarray, shape (n_samples,) or (n_channels, n_samples)
        Audio time-series (mono by default; stereo if you loaded with mono=False).
    sr : int
        Sampling rate in Hz.
    hop_length : int, optional (default=512)
        Number of samples between successive frames used by the onset detector.

    Returns
    -------
    onset_samples : np.ndarray, dtype=int, shape (n_onsets,)
        1D array of onset **sample indices** (not frames). These are obtained by
        converting onset frame indices via `librosa.frames_to_samples(...)`.
    """
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop_length)
    return librosa.frames_to_samples(onset_frames, hop_length=hop_length)


def get_intervals(
    y: np.ndarray,
    sr: int,
    hop_length: int = 512,
) -> List[Tuple[int, int]]:
    """

    Parameters
    ----------
    y : np.ndarray, shape (n_samples,) or (n_channels, n_samples)
        Audio time-series.
    sr : int
        Sampling rate in Hz.
    hop_length : int, optional (default=512)
        Hop length used when detecting onsets (must match `get_onsets`).

    Returns
    -------
    intervals : list[tuple[int, int]]
        Each tuple is an interval in **sample indices**:
        (onset_samples[i], onset_samples[i+1]) for i in [0 .. n_onsets-2].
    """
    boundaries = get_onsets(y, sr, hop_length=hop_length)  # in samples
    intervals = [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]
    return intervals


def get_intervals_for_duration(y, sr, hop_length=512):
    boundaries = librosa.onset.onset_detect(
        y=y, sr=sr, hop_length=hop_length, units="samples"
    )
    intervals = [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]
    intervals.append((boundaries[-1], len(y)))
    return intervals
