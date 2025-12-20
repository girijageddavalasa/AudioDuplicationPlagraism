import librosa
import numpy as np

def extract_features(segments, sr, prefix):
    features = []

    for i, seg in enumerate(segments):
        y = seg["audio"]

        # Melody
        pitches, mags = librosa.piptrack(y=y, sr=sr)
        pitch = pitches[mags.argmax(axis=0), range(mags.shape[1])]
        pitch = pitch[pitch > 0]
        melody = (
            np.diff(librosa.hz_to_midi(pitch)).tolist()
            if len(pitch) > 1 else []
        )

        # Rhythm
        onsets = librosa.onset.onset_detect(y=y, sr=sr, units="time")
        rhythm = (
            (np.diff(onsets) / np.mean(np.diff(onsets))).tolist()
            if len(onsets) > 1 else []
        )

        # Harmony
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        harmony = chroma.mean(axis=1).tolist()

        features.append({
            "segment_id": f"{prefix}_{i}",
            "start": seg["start"],
            "end": seg["end"],
            "melody": melody,
            "rhythm": rhythm,
            "harmony": harmony
        })

    return features
