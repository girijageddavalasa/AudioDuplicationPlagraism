import os
import numpy as np
import librosa
import soundfile as sf

# =========================
# CONFIG
# =========================
QUERY_ROOT = "data/query"
CANDIDATES_ROOT = "data/candidates"
NOISE_LEVEL = 0.01  # 0.002 = light, 0.005 = medium, 0.01 = heavy

# =========================
# FUNCTIONS
# =========================
def add_gaussian_noise(audio, noise_level):
    noise = np.random.randn(len(audio))
    return audio + noise_level * noise

# =========================
# PROCESS
# =========================
for genre in os.listdir(QUERY_ROOT):
    genre_query_path = os.path.join(QUERY_ROOT, genre)

    if not os.path.isdir(genre_query_path):
        continue

    # Create same genre folder inside candidates
    genre_candidate_path = os.path.join(CANDIDATES_ROOT, genre)
    os.makedirs(genre_candidate_path, exist_ok=True)

    print(f"\nProcessing genre: {genre}")

    for file in os.listdir(genre_query_path):
        if not file.endswith(".wav"):
            continue

        input_path = os.path.join(genre_query_path, file)

        # Load audio
        audio, sr = librosa.load(input_path, sr=None)

        # Add noise
        noisy_audio = add_gaussian_noise(audio, NOISE_LEVEL)

        # Normalize to avoid clipping
        noisy_audio = noisy_audio / np.max(np.abs(noisy_audio))

        # Save into candidates folder
        noisy_filename = file.replace(".wav", "_noisy.wav")
        output_path = os.path.join(genre_candidate_path, noisy_filename)

        sf.write(output_path, noisy_audio, sr)

        print(f"  ✔ Saved to candidates/{genre}/{noisy_filename}")
