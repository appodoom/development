import librosa
import numpy as np
import soundfile as sf

y, sr = librosa.load("../../samples/sample8.wav", sr=None)
tempos = librosa.beat.tempo(y=y,
                            sr=sr,
                            hop_length=512,
                            aggregate=None)

intervals = []
i = 0
N = len(tempos)
THRESH = 10  # maximum allowed BPM variation

while i < N:
    start = i
    # initialize window min/max
    min_t = max_t = tempos[i]
    i += 1

    # grow the window until variation > THRESH
    while i < N:
        t = tempos[i]
        min_t = min(min_t, t)
        max_t = max(max_t, t)
        if max_t - min_t > THRESH:
            break
        i += 1

    # end of this “steady‐tempo” segment is i‐1
    end = i - 1
    intervals.append((start, end))


# intervals now holds non‐overlapping segments
print(intervals)
difference=np.diff(np.array(intervals))
print(difference)


def getBeatPerChunk(intervals,tempos,y,sr):
    beat_timeStamps = {}
    for i in range(len(intervals)):
        interval = intervals[i]
        tempo=tempos[interval[0]]
        start_sample = librosa.frames_to_samples(interval[0])
        end_sample = librosa.frames_to_samples(interval[1])
        current_y = y[start_sample:end_sample+1]
        current_beats = librosa.beat.beat_track(y=current_y,start_bpm=tempo)
        beat_timeStamps[i]=current_beats
    return beat_timeStamps

def getAudiosByChunk(intervals,tempos,y,sr):
    for i in range(len(intervals)):
        interval = intervals[i]
        start_sample = librosa.frames_to_samples(interval[0])
        end_sample = librosa.frames_to_samples(interval[1])
        current_y = y[start_sample:end_sample+1]
        sf.write(file=f"Chunk{i}_tempo_{tempos[interval[0]]}.wav",data=current_y,samplerate=sr)

beat_timeStamps = getBeatPerChunk(intervals,tempos,y,sr)
getAudiosByChunk(intervals,tempos,y,sr)