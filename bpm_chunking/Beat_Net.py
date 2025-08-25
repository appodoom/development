# pip install librosa madmom BeatNet
# (Per BeatNet README, install librosa & madmom first.)
import collections, collections.abc
collections.MutableSequence = collections.abc.MutableSequence  # shim for py3.10+
from typing import Sequence
import numpy as np
if not hasattr(np, 'float'):   np.float = float
if not hasattr(np, 'int'):     np.int = int
if not hasattr(np, 'complex'): np.complex = np.complex128
if not hasattr(np, 'bool'):    np.bool = bool

from BeatNet.BeatNet import BeatNet

def beatnet_offline_beats(wav_path: str) -> np.ndarray:
    """
    Run BeatNet in offline mode (DBN decoding) and return beat timestamps (seconds).

    Parameters
    ----------
    wav_path : str
        Path to an audio file (e.g., .wav). BeatNet will resample to 22050 Hz internally.

    Returns
    -------
    np.ndarray, shape (N,)
        Beat times in seconds.
    """
    # model=1 (one of the pretrained CRNNs), offline + DBN uses non-causal Viterbi decoding
    est = BeatNet(model=1, mode='offline', inference_model='DBN', plot=[], thread=False)
    out = est.process(wav_path)          # BeatNet returns a (num_beats, 2) array in offline mode

    # BeatNet README: first column contains beat times; second column is downbeat-related info
    # (mirrors madmom’s convention). We just return the beat-time column.
    out = np.asarray(out)
    if out.ndim == 2 and out.shape[1] >= 1:
        return out[:, 0]
    elif out.ndim == 1:
        # Fallback if a 1D array is ever returned
        return out
    else:
        raise RuntimeError(f"Unexpected BeatNet output shape: {out.shape}")

# Example:
# beats = beatnet_offline_beats("path/to/song.wav")
# print(beats)
print(beatnet_offline_beats("../../samples/sample1.wav"))