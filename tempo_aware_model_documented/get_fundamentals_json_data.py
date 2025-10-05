"""
Load multiple JSON files (each describing a fundamental and its variations) and
   aggregate them into a dictionary keyed by a derived fundamental name.

"""

from load_files import load_json_file
from typing import Dict, List, Tuple
import numpy as np


def get_fundamentals_data(
    json_mel_fundamentals_paths: List[str],
) -> Dict[str, Tuple[Dict[str, np.ndarray], List[str]]]:
    """
    Parameters
    ----------
    json_mel_fundamentals_paths : list[str]
        File paths to JSON files. Each JSON is expected to map variation names to
        array-like data (e.g., mel-spectrograms). This function relies on
        `load_json(json_file_path)` to parse the file into:
          - current_fundamental_mel_data: dict[str, np.ndarray]
          - current_list_of_variations : list[str]

    Returns
    -------
    fundamentals : dict[str, (dict[str, np.ndarray], list[str])]
        A mapping:
            fundamental_name -> (mel_data_by_variation, variation_names)
        where:
          - mel_data_by_variation: dict[str, np.ndarray]
              Keys are variation names; values are NumPy arrays (e.g., mel features).
          - variation_names: list[str]
              The list of variation names present in the JSON.
    """
    fundamentals: Dict[str, Tuple[Dict[str, np.ndarray], List[str]]] = {}
    for fund_path in json_mel_fundamentals_paths:
        current_fundamental_mel_data, current_list_of_variations = load_json_file(
            json_file_path=fund_path
        )
        current_fundamental_name = fund_path.split(".")[1][1:]
        fundamentals[current_fundamental_name] = (
            current_fundamental_mel_data,
            current_list_of_variations,
        )
    return fundamentals
