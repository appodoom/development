"""
Compute the maximum cosine-similarity (normalized cross-correlation) between a
   2D matrix X (n_freq x n_time_X) and all time-aligned slices of Y
   (n_freq x n_time_Y), sliding along the time axis.
"""

from typing import Optional, Tuple
import numpy as np


def sliding_cross_correlation(
    X: np.ndarray, Y: np.ndarray
) -> Tuple[float, Optional[int]]:
    """
    Parameters
    ----------
    X : np.ndarray, shape (n_freq, n_time_X)
        The query matrix (must share the same n_freq as Y).
    Y : np.ndarray, shape (n_freq, n_time_Y)
        The reference matrix to search in.

    Returns
    -------
    best_score : float
        The maximum cosine similarity in [-1, 1] across all valid offsets,
        or -np.inf if no valid comparison was possible.
    best_offset : Optional[int]
        The start column in Y where the best match occurs, or None if none exists.
    """
    if X.shape[1] > Y.shape[1]:
        X, Y = Y, X

    n_freq, n_time_X = X.shape
    _, n_time_Y = Y.shape

    best_score = -np.inf
    best_offset = None
    norm_X = np.linalg.norm(X)
    for offset in range(n_time_Y - n_time_X + 1):
        Y_slice = Y[:, offset : offset + n_time_X]
        if norm_X != 0 and np.linalg.norm(Y_slice) != 0:
            score = np.tensordot(X, Y_slice, axes=2) / (
                norm_X * np.linalg.norm(Y_slice)
            )
        else:
            continue

        if score > best_score:
            best_score = score
            best_offset = offset

    return best_score, best_offset
