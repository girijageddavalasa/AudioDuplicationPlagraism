import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ChromaAudioFeatureExtractor:
    """
    Extract comprehensive audio features using Chroma and librosa.
    Handles music, speech, beats, and general audio recordings.
    """
    
    def _init_(self, config_path='config.py'):
        """Initialize with configuration"""
        self.load_config(config_path)
        self.setup_output_dirs()
        
    def load_config(self, config_path):
        """Load configuration from config.py"""
        config = {}
        with open(config_path, 'r') as f:
            exec(f.read(), config)
        
        # Audio processing parameters
        self.sample_rate = config.get('SAMPLE_RATE', 22050)
        self.hop_length = config.get('HOP_LENGTH', 512)
        self.n_fft = config.get('N_FFT', 2048)
        self.n_mels = config.get('N_MELS', 128)
        self.n_chroma = config.get('N_CHROMA', 12)
        
        # Folder paths
        self.query_folder = config.get('QUERY_FOLDER', 'query_audio')
        self.reference_folder = config.get('REFERENCE_FOLDER', 'reference_audio')
        self.test_folder = config.get('TEST_FOLDER', 'test_audio')
        self.output_folder = config.get('OUTPUT_FOLDER', 'output')
        
    def setup_output_dirs(self):
        """Create output subdirectories"""
        self.features_dir = os.path.join(self.output_folder, 'chroma_features')
        self.waveforms_dir = os.path.join(self.output_folder, 'chroma_waveforms')
        self.spectrograms_dir = os.path.join(self.output_folder, 'chroma_spectrograms')
        
        for directory in [self.features_dir, self.waveforms_dir, self.spectrograms_dir]:
            os.makedirs(directory, exist_ok=True)
    
    def extract_comprehensive_features(self, audio_path):
        """
        Extract all audio features from a file.
        Works with music, speech, beats, and recordings.
        """
        print(f"\nProcessing: {os.path.basename(audio_path)}")
        
        # Load audio file (full duration)
        y, sr = librosa.load(audio_path, sr=self.sample_rate, duration=None)
        duration = librosa.get_duration(y=y, sr=sr)
        
        print(f"  Duration: {duration:.2f}s, Sample Rate: {sr}Hz")
        
        features = {
            'filename': os.path.basename(audio_path),
            'duration': float(duration),
            'sample_rate': int(sr),
            'timestamp': datetime.now().isoformat()
        }
        
        # 1. Chroma Features (pitch class profiles)
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=self.n_fft, 
                                                    hop_length=self.hop_length, 
                                                    n_chroma=self.n_chroma)
        chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=self.hop_length,
                                                 n_chroma=self.n_chroma)
        chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr, hop_length=self.hop_length,
                                                   n_chroma=self.n_chroma)
        
        features['chroma_stft_mean'] = chroma_stft.mean(axis=1).tolist()
        features['chroma_stft_std'] = chroma_stft.std(axis=1).tolist()
        features['chroma_cqt_mean'] = chroma_cqt.mean(axis=1).tolist()
        features['chroma_cens_mean'] = chroma_cens.mean(axis=1).tolist()
        
        # 2. Spectral Features
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, 
                                                               hop_length=self.hop_length)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, 
                                                             hop_length=self.hop_length)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, 
                                                                 hop_length=self.hop_length)[0]
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y, hop_length=self.hop_length)[0]
        
        features['spectral_centroid_mean'] = float(np.mean(spectral_centroid))
        features['spectral_centroid_std'] = float(np.std(spectral_centroid))
        features['spectral_rolloff_mean'] = float(np.mean(spectral_rolloff))
        features['spectral_bandwidth_mean'] = float(np.mean(spectral_bandwidth))
        features['zero_crossing_rate_mean'] = float(np.mean(zero_crossing_rate))
        
        # 3. MFCC (Mel-frequency cepstral coefficients) - good for speech and music
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=self.hop_length)
        features['mfcc_mean'] = mfcc.mean(axis=1).tolist()
        features['mfcc_std'] = mfcc.std(axis=1).tolist()
        
        # 4. Mel Spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=self.n_mels,
                                                          hop_length=self.hop_length)
        features['mel_spectrogram_mean'] = np.mean(librosa.power_to_db(mel_spectrogram), axis=1).tolist()
        
        # 5. Tempo and Beat tracking
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=self.hop_length)
        features['tempo'] = float(tempo)
        features['num_beats'] = int(len(beats))
        
        # 6. Tonnetz (tonal centroid features)
        tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
        features['tonnetz_mean'] = tonnetz.mean(axis=1).tolist()
        
        # 7. RMS Energy
        rms = librosa.feature.rms(y=y, hop_length=self.hop_length)[0]
        features['rms_mean'] = float(np.mean(rms))
        features['rms_std'] = float(np.std(rms))
        
        # 8. Harmonic and Percussive components (good for music vs speech)
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        features['harmonic_ratio'] = float(np.sum(np.abs(y_harmonic)) / np.sum(np.abs(y)))
        features['percussive_ratio'] = float(np.sum(np.abs(y_percussive)) / np.sum(np.abs(y)))
        
        # Store time-series data for visualization
        features['_time_series'] = {
            'chroma_stft': chroma_stft.tolist(),
            'chroma_cqt': chroma_cqt.tolist(),
            'mfcc': mfcc.tolist(),
            'spectral_centroid': spectral_centroid.tolist(),
            'beats': beats.tolist(),
            'waveform': y[::100].tolist()  # Downsampled for storage
        }
        
        return features, y, sr
    
    def generate_waveform_plots(self, audio_path, y, sr, features):
        """Generate comprehensive waveform visualizations"""
        filename = os.path.splitext(os.path.basename(audio_path))[0]
        
        fig, axes = plt.subplots(4, 1, figsize=(15, 12))
        fig.suptitle(f'Audio Analysis: {filename}', fontsize=16, fontweight='bold')
        
        # 1. Waveform
        librosa.display.waveshow(y, sr=sr, ax=axes[0], color='blue', alpha=0.7)
        axes[0].set_title('Waveform')
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Amplitude')
        axes[0].grid(True, alpha=0.3)
        
        # 2. Chroma STFT
        chroma_data = np.array(features['_time_series']['chroma_stft'])
        img1 = librosa.display.specshow(chroma_data, y_axis='chroma', x_axis='time',
                                        sr=sr, hop_length=self.hop_length, ax=axes[1],
                                        cmap='coolwarm')
        axes[1].set_title('Chroma STFT (Pitch Classes)')
        axes[1].set_ylabel('Pitch Class')
        fig.colorbar(img1, ax=axes[1], format='%+2.0f')
        
        # 3. MFCC
        mfcc_data = np.array(features['_time_series']['mfcc'])
        img2 = librosa.display.specshow(mfcc_data, x_axis='time', sr=sr,
                                        hop_length=self.hop_length, ax=axes[2],
                                        cmap='viridis')
        axes[2].set_title('MFCC (Mel-frequency Cepstral Coefficients)')
        axes[2].set_ylabel('MFCC Coefficients')
        fig.colorbar(img2, ax=axes[2], format='%+2.0f')
        
        # 4. Spectral Features
        time_frames = librosa.frames_to_time(range(len(features['_time_series']['spectral_centroid'])),
                                              sr=sr, hop_length=self.hop_length)
        axes[3].plot(time_frames, features['_time_series']['spectral_centroid'], 
                     label='Spectral Centroid', color='red', linewidth=1.5)
        axes[3].set_title('Spectral Centroid Over Time')
        axes[3].set_xlabel('Time (s)')
        axes[3].set_ylabel('Frequency (Hz)')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_path = os.path.join(self.waveforms_dir, f'{filename}_waveform.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Waveform saved: {output_path}")
        
    def generate_spectrogram_plots(self, audio_path, y, sr, features):
        """Generate detailed spectrogram visualizations"""
        filename = os.path.splitext(os.path.basename(audio_path))[0]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Spectrograms: {filename}', fontsize=16, fontweight='bold')
        
        # 1. Mel Spectrogram
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=self.n_mels,
                                                   hop_length=self.hop_length)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        img1 = librosa.display.specshow(mel_spec_db, y_axis='mel', x_axis='time',
                                        sr=sr, hop_length=self.hop_length,
                                        ax=axes[0, 0], cmap='magma')
        axes[0, 0].set_title('Mel Spectrogram')
        fig.colorbar(img1, ax=axes[0, 0], format='%+2.0f dB')
        
        # 2. Chroma CQT
        chroma_cqt = np.array(features['_time_series']['chroma_cqt'])
        img2 = librosa.display.specshow(chroma_cqt, y_axis='chroma', x_axis='time',
                                        sr=sr, hop_length=self.hop_length,
                                        ax=axes[0, 1], cmap='coolwarm')
        axes[0, 1].set_title('Chroma CQT')
        fig.colorbar(img2, ax=axes[0, 1])
        
        # 3. STFT Magnitude
        D = librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length)
        D_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        img3 = librosa.display.specshow(D_db, y_axis='log', x_axis='time',
                                        sr=sr, hop_length=self.hop_length,
                                        ax=axes[1, 0], cmap='jet')
        axes[1, 0].set_title('STFT Magnitude Spectrogram')
        fig.colorbar(img3, ax=axes[1, 0], format='%+2.0f dB')
        
        # 4. Harmonic-Percussive Separation
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        axes[1, 1].plot(librosa.times_like(y_harmonic, sr=sr)[::1000], 
                       y_harmonic[::1000], alpha=0.7, label='Harmonic', linewidth=0.8)
        axes[1, 1].plot(librosa.times_like(y_percussive, sr=sr)[::1000], 
                       y_percussive[::1000], alpha=0.7, label='Percussive', linewidth=0.8)
        axes[1, 1].set_title('Harmonic-Percussive Separation')
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Amplitude')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_path = os.path.join(self.spectrograms_dir, f'{filename}_spectrogram.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Spectrogram saved: {output_path}")
    
    def save_features_to_file(self, features, audio_path):
        """Save extracted features to JSON file"""
        filename = os.path.splitext(os.path.basename(audio_path))[0]
        
        # Create a clean copy without time series data for the main JSON
        clean_features = {k: v for k, v in features.items() if k != '_time_series'}
        
        output_path = os.path.join(self.features_dir, f'{filename}_features.json')
        
        with open(output_path, 'w') as f:
            json.dump(clean_features, f, indent=2)
        
        print(f"  Features saved: {output_path}")
        
        return output_path
    
    def process_audio_file(self, audio_path, generate_plots=True):
        """Process a single audio file"""
        try:
            # Extract features
            features, y, sr = self.extract_comprehensive_features(audio_path)
            
            # Save features to JSON
            self.save_features_to_file(features, audio_path)
            
            # Generate visualizations
            if generate_plots:
                self.generate_waveform_plots(audio_path, y, sr, features)
                self.generate_spectrogram_plots(audio_path, y, sr, features)
            
            return features
            
        except Exception as e:
            print(f"  ERROR processing {audio_path}: {str(e)}")
            return None
    
    def process_folder(self, folder_path, generate_plots=True):
        """Process all audio files in a folder"""
        audio_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac']
        audio_files = []
        
        for ext in audio_extensions:
            audio_files.extend(Path(folder_path).glob(f'*{ext}'))
            audio_files.extend(Path(folder_path).glob(f'*{ext.upper()}'))
        
        if not audio_files:
            print(f"No audio files found in {folder_path}")
            return []
        
        print(f"\nFound {len(audio_files)} audio files in {folder_path}")
        print("=" * 60)
        
        all_features = []
        for audio_file in audio_files:
            features = self.process_audio_file(str(audio_file), generate_plots)
            if features:
                all_features.append(features)
        
        return all_features
    
    def create_summary_report(self, all_features):
        """Create a summary report of all processed files"""
        if not all_features:
            return
        
        summary_path = os.path.join(self.output_folder, 'chroma_summary_report.txt')
        
        with open(summary_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("CHROMA AUDIO FEATURE EXTRACTION - SUMMARY REPORT\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total files processed: {len(all_features)}\n\n")
            
            for idx, features in enumerate(all_features, 1):
                f.write(f"\n{idx}. {features['filename']}\n")
                f.write("-" * 80 + "\n")
                f.write(f"Duration: {features['duration']:.2f}s\n")
                f.write(f"Sample Rate: {features['sample_rate']}Hz\n")
                f.write(f"Tempo: {features['tempo']:.2f} BPM\n")
                f.write(f"Number of Beats: {features['num_beats']}\n")
                f.write(f"Harmonic Ratio: {features['harmonic_ratio']:.3f}\n")
                f.write(f"Percussive Ratio: {features['percussive_ratio']:.3f}\n")
                f.write(f"Spectral Centroid (mean): {features['spectral_centroid_mean']:.2f}Hz\n")
                f.write(f"RMS Energy (mean): {features['rms_mean']:.4f}\n")
                f.write(f"\nChroma STFT Mean: {[f'{x:.3f}' for x in features['chroma_stft_mean'][:6]]}...\n")
                f.write(f"MFCC Mean (first 5): {[f'{x:.3f}' for x in features['mfcc_mean'][:5]]}...\n")
        
        print(f"\nSummary report saved: {summary_path}")


def main():
    """Main execution function"""
    print("\n" + "=" * 80)
    print("CHROMA-BASED AUDIO FEATURE EXTRACTION")
    print("=" * 80)
    
    # Initialize extractor
    extractor = ChromaAudioFeatureExtractor(config_path='config.py')
    
    # Process folders
    folders_to_process = [
        ('query_audio', extractor.query_folder),
        ('reference_audio', extractor.reference_folder),
        ('test_audio', extractor.test_folder)
    ]
    
    all_results = []
    
    for folder_name, folder_path in folders_to_process:
        if os.path.exists(folder_path):
            print(f"\n{'='*80}")
            print(f"Processing folder: {folder_name}")
            print(f"{'='*80}")
            features = extractor.process_folder(folder_path, generate_plots=True)
            all_results.extend(features)
        else:
            print(f"\nFolder not found: {folder_path}")
    
    # Create summary report
    if all_results:
        extractor.create_summary_report(all_results)
        print(f"\n{'='*80}")
        print(f"PROCESSING COMPLETE!")
        print(f"Total files processed: {len(all_results)}")
        print(f"Output saved in: {extractor.output_folder}/")
        print(f"  - Features: {extractor.features_dir}/")
        print(f"  - Waveforms: {extractor.waveforms_dir}/")
        print(f"  - Spectrograms: {extractor.spectrograms_dir}/")
        print(f"{'='*80}\n")


if _name_ == "_main_":
    main()