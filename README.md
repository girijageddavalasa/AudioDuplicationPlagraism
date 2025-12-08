# AUDIO DUPLICATION AND PLAGIARISM DETECTION

---

## Project Architecture

![System Architecture](architecure.jpg)

---

## Dataset

The primary dataset used:

- **GTZAN Music Genre Classification Dataset**  
  [https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)

Additional self-created datasets for chord analysis:

- [Samsung 5th Nov Dataset](https://www.kaggle.com/datasets/girijageddavalasa/samsung5thnov)
- [Sample 4 Dataset](https://www.kaggle.com/datasets/girijageddavalasa/sample4)

---

## 0Corpus — Audio Corpus Management

A comprehensive audio corpus pipeline standardizes **data ingestion, metadata extraction, and lifecycle tracking** for all audio files within the system.

**Key Features:**

- Automated technical analysis: sample rate, channels, encoding subtype, duration, bitrate, compression ratio.
- File integrity: SHA256 fingerprinting for duplication checks.
- Metadata capture: dataset name, source type, reference URLs.
- Full provenance and version tracking with an SQLite-backed schema.
- Structured logging and audit trails for all corpus modifications.

This foundation ensures **reliability, scalability, and traceability** for large-scale audio analysis workflows, supporting future integration into AI-driven similarity search and plagiarism detection modules.

---

## 1Duplication — Fingerprinting & Chroma Analysis

Implemented dual pipelines for detecting both **exact and transformed copies**:

### Dejavu-Based Fingerprinting

- Generates spectrograms and identifies strong spectral peaks.
- Converts peaks into **compact hash fingerprints**.
- Stores fingerprints in a database.
- Matches query audio against reference tracks to find near-exact duplicates.

### Chroma & Feature Analysis

- Extracts **deep chroma-based features**:
  - Chroma STFT, CQT, CENS
  - MFCCs, Mel Spectrograms
  - Spectral contrast, roll-off, tonnetz, tempo, harmonic/percussive ratios
- Saves both JSON feature files and visual plots (waveform, spectrogram).
- Enables detection of **melodic-specific or speed-modified plagiarism**.

Together, these modules provide **audio fingerprinting precision** alongside **semantic similarity detection** for robust plagiarism analysis.

---

## 2Plagiarism_Similarity — MIDI & Chord Matching

MIDI conversion enables the extraction of **chords and harmonic progression** from tracks for higher-level similarity analysis.

- Tested chord-level clustering using self-curated datasets.
- Experiments performed in:
  - `tryingcompatibilityforall` notebook → cluster prediction
  - `finalreport` notebook → similarity testing
- Current similarity methods:
  - **Containment similarity**
  - **n-gram based similarity** (variable _n_)
- Planned upgrade: **WER (Word Error Rate)** for accuracy-based comparison.

MIDI and chord embeddings enhance cross-song similarity detection even across genre or arrangement differences.
