import numpy as np
import librosa

def detect_cycle_length(
    wav_path: str,
    min_beats: int = 3,
    max_beats: int = 16,
    hop_length: int = 512,
    harmonic_threshold: float = 0.90
) -> int:
    """
    Estimate the darbukka/iqa‘ cycle length in beats.
    
    Args:
      wav_path: path to the input WAV file
      min_beats: smallest cycle to test (inclusive)
      max_beats: largest cycle to test (inclusive)
      hop_length: STFT / onset hop length
      harmonic_threshold: fraction of the max-corr cutoff for sub-multiples
    
    Returns:
      An integer tau between min_beats and max_beats.
    """
    # 1. Load & HPSS to isolate percussive
    y, sr = librosa.load(wav_path, mono=True)
    _, y_perc = librosa.effects.hpss(y)

    # 2. Onset-strength envelope
    onset_env = librosa.onset.onset_strength(
        y=y_perc, sr=sr, hop_length=hop_length
    )

    # 3. Beat-tracking
    tempo, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_env, sr=sr, hop_length=hop_length
    )

    # 4. Build beat-aligned vector a[i]
    a = onset_env[beat_frames]
    N = len(a)
    if N < min_beats * 2:
        raise RuntimeError(f"Too few beats ({N}) for cycle detection ≥{min_beats}.")

    # 5. Zero-mean
    a = a - np.mean(a)

    # 6. Compute normalized linear autocorrelation for each tau
    max_lag = min(max_beats, N // 2)
    if max_lag < min_beats:
        raise RuntimeError("No valid cycle-length candidates under these settings.")

    def normcorr(tau):
        L = N - tau
        x = a[:L]
        y2 = a[tau:tau+L]
        denom = np.linalg.norm(x) * np.linalg.norm(y2)
        return np.dot(x, y2) / denom if denom > 0 else 0.0

    scores = {tau: normcorr(tau) for tau in range(min_beats, max_lag + 1)}

    # 7. Pick best tau with sub-multiple check
    tau_max = max(scores, key=scores.get)
    C_max = scores[tau_max]

    # collect all tau <= tau_max whose score ≥ threshold * C_max
    candidates = [τ for τ, c in scores.items() if τ <= tau_max and c >= harmonic_threshold * C_max]
    best_tau = min(candidates) if candidates else tau_max

    return best_tau

if __name__ == "__main__":
    wav_file = "../../sample8.wav"
    cycle = detect_cycle_length(
        wav_file,
        min_beats=3,
        max_beats=16,
        hop_length=512,
        harmonic_threshold=0.90
    )
    print("Estimated cycle length:", cycle)
