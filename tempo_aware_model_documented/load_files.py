import json
from typing import Dict, List, Tuple

import librosa
import numpy as np


def load_wav_file(wav_file_path: str) -> Tuple[np.ndarray, float]:
    """
    Parameters
    ----------
    wav_file_path : str
        Path to a .wav audio file.

    Returns
    -------
    y : np.ndarray, shape (n_samples,)
        Audio time-series samples (mono). Values are L2-normalized.
    sr : int
        Sample rate in Hz (native rate, since sr=None).
    """
    y, sr = librosa.load(wav_file_path, sr=None)
    y = librosa.util.normalize(y)
    return y, sr


def load_json_file(json_file_path: str) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """
    Parameters
    ----------
    json_file_path : str
        Path to a .json file containing a dict like {name: array_like}.

    Returns
    -------
    json_data : dict[str, np.ndarray]
        Same mapping with values converted to NumPy arrays (float32 where possible).
    list_of_hits : list[str]
        List of keys (names) found in the JSON, in file order.
    """
    with open(json_file_path, "r") as f:
        mels = json.load(f)

    json_data = {
        name: np.asarray(mel, dtype=np.float32)
        if np.array(mel).dtype.kind in "fi"
        else np.asarray(mel)
        for name, mel in mels.items()
    }
    list_of_hits = list(mels.keys())
    return json_data, list_of_hits
