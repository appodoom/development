## Experiment 1: Beat-Based Auto-Correlation Approaches

In our first attempt to estimate the underlying rhythmic cycle in darbuka recordings, we explored a family of “beat-based auto-correlation” methods. The goal was to find the optimal shift (in beats) that, when applied to the audio, maximizes self-similarity.

### 1. Rationale

- **Cycle hypothesis**  
  A repeating rhythmic pattern (the “cycle”) will cause the audio signal to line up with itself when shifted by exactly one cycle’s worth of beats.
- **Auto-correlation**  
  By computing the normalized auto-correlation between the original and shifted signal, the correct cycle length should correspond to the shift that maximizes the correlation.

### 2. Beat Shift Strategy

We constrained the candidate shifts to integer multiples of the estimated beat duration, between 3 and 16 beats. This range covers typical darbuka cycle lengths while keeping the search tractable.

1. **Estimate beat duration**  
   - **Tempo-based**: use `librosa.beat_track` to detect tempo (BPM) and convert to samples per beat \( \: \delta = sample\_rate\, \times {60 \over BPM} \)  

2. **Compute candidate shifts**  
   For each multiplier \( k \in \{3, 4, \dots, 16\} \), compute  
   \[
     \Delta_k = k \times \delta
   \]
3. **Auto-correlate**  
   For each shift \(\Delta_k\), compute the Pearson correlation between  
   \[
     x[n] \quad\text{and}\quad x[n + \Delta_k]
   \]
   where \(x[n]\) is the audio waveform (or spectrogram feature).

### 3. Raw Waveform vs. Mel Spectrogram

We applied the above procedure to two different signal representations:

- **Raw audio waveform**  
  Captures all frequency content, but may be dominated by transient onsets and noise.
- **Mel-spectrogram**  
  Compresses spectral information into perceptually-weighted bins, potentially smoothing out irrelevant variance.

### 4. Results and Observations



Despite two signal representations, none of the variants produced a reliable correlation peak at a consistent shift. In particular:

- **Tempo fluctuations** across a track meant the “samples per beat” estimate was only approximate.
- **Transient onsets** (especially rapid rolls and ornamentation) broke the stationarity assumption required for clean auto-correlation peaks.
- **Spectrogram smoothing** reduced noise but also blurred the precise timing information needed to detect the cycle.

> **Conclusion:**  
> The simple beat-shift auto-correlation approach—whether applied to raw audio or mel-spectrograms and using tempo-derived versus sample-average beat durations—fails to yield a robust cycle estimate on darbuka recordings. Next, we will explore time-domain envelope alignment and adaptive windowing methods.

