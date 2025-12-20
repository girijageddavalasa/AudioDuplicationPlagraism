import numpy as np

def segment_audio(audio, sr, segment_length=3.0, overlap=0.5):
    segments = []
    step = segment_length * (1 - overlap)
    duration = len(audio) / sr
    idx = 0

    for start in np.arange(0, duration - segment_length, step):
        start_sample = int(start * sr)
        end_sample = int((start + segment_length) * sr)

        segments.append({
            "segment_id": f"seg_{idx}",
            "start": start,
            "end": start + segment_length,
            "audio": audio[start_sample:end_sample]
        })
        idx += 1

    return segments
