import librosa

def load_audio(path, sr=22050):
    audio, sr = librosa.load(path, sr=sr, mono=True)
    duration = librosa.get_duration(y=audio, sr=sr)
    return audio, sr, duration
