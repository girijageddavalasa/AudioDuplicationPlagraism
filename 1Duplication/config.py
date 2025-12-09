# Audio Processing Configuration

# Sample rate for audio processing
SAMPLE_RATE = 22050

# FFT parameters
N_FFT = 2048
HOP_LENGTH = 512

# Mel spectrogram parameters
N_MELS = 128

# Chroma parameters
N_CHROMA = 12

# Folder paths (use relative paths)
QUERY_FOLDER = 'data/query_audio'
REFERENCE_FOLDER = 'data/reference_audio'
TEST_FOLDER = 'data/test_audio'
OUTPUT_FOLDER = 'output'

# Database configuration
DB_PATH = 'data/audio_fingerprints.db'

# Logging
LOG_LEVEL = 'INFO'
LOG_FILE = 'logs/audio_processing.log'
