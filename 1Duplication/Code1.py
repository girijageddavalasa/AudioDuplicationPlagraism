import os
import numpy as np
import librosa
import hashlib
from pathlib import Path
from datetime import datetime
import json
import matplotlib.pyplot as plt
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


class DejavuAudioFingerprinter:
   
    def _init_(self, config_path='config.py'):
        self.load_config(config_path)
        self.setup_output_dirs()
        self.fingerprint_database = {}
        self.audio_metadata = {}
        
    def load_config(self, config_path):
        import importlib.util
        
        try:
            spec = importlib.util.spec_from_file_location("config", config_path)
            if spec is None:
                print(f"Config file not found: {config_path}. Using defaults.")
                self._set_default_config()
                return
                
            config_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(config_module)
            
        except Exception as e:
            print(f"Error loading config: {e}. Using defaults.")
            self._set_default_config()
            return
        
        # Audio processing parameters
        self.sample_rate = getattr(config_module, 'SAMPLE_RATE', 22050)
        self.hop_length = getattr(config_module, 'HOP_LENGTH', 512)
        self.n_fft = getattr(config_module, 'N_FFT', 2048)
        
        # Dejavu-specific parameters
        self.peak_neighborhood_size = 10  # Increased for better peak isolation
        self.min_amplitude_threshold = 0.1
        self.fan_value = 15  # Number of fingerprints per peak
        self.target_zone_size = 5
        
        # Folder paths
        self.query_folder = getattr(config_module, 'QUERY_FOLDER', 'query_audio')
        self.reference_folder = getattr(config_module, 'REFERENCE_FOLDER', 'reference_audio')
        self.test_folder = getattr(config_module, 'TEST_FOLDER', 'test_audio')
        self.output_folder = getattr(config_module, 'OUTPUT_FOLDER', 'output')
        
    def _set_default_config(self):
        """Set default configuration"""
        self.sample_rate = 22050
        self.hop_length = 512
        self.n_fft = 2048
        self.peak_neighborhood_size = 20
        self.min_amplitude_threshold = 0.1
        self.fan_value = 15
        self.target_zone_size = 5
        self.query_folder = 'query_audio'
        self.reference_folder = 'reference_audio'
        self.test_folder = 'test_audio'
        self.output_folder = 'output'
        
    def setup_output_dirs(self):
        self.dejavu_dir = os.path.join(self.output_folder, 'dejavu_fingerprints')
        self.dejavu_plots_dir = os.path.join(self.output_folder, 'dejavu_visualizations')
        self.dejavu_matches_dir = os.path.join(self.output_folder, 'dejavu_matches')
        
        for directory in [self.dejavu_dir, self.dejavu_plots_dir, self.dejavu_matches_dir]:
            os.makedirs(directory, exist_ok=True)
    
    def generate_spectrogram(self, audio_path):
        """Generate spectrogram from audio file"""
        print(f"\nGenerating spectrogram for: {os.path.basename(audio_path)}")
        
        # Load full audio
        y, sr = librosa.load(audio_path, sr=self.sample_rate, duration=None, mono=True)
        duration = librosa.get_duration(y=y, sr=sr)
        
        print(f"  Duration: {duration:.2f}s, Sample Rate: {sr}Hz")
        print(f"  Audio amplitude range: [{np.min(y):.3f}, {np.max(np.abs(y)):.3f}]")
        
        # Compute STFT
        D = librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length)
        magnitude = np.abs(D)
        
        # Convert to dB scale
        magnitude_db = librosa.amplitude_to_db(magnitude, ref=np.max)
        print(f'  Spectrogram shape: {magnitude_db.shape}')
        print(f'  dB range: [{np.min(magnitude_db):.2f}, {np.max(magnitude_db):.2f}]')
        
        return magnitude_db, sr, duration, y
    
    def find_peaks(self, spectrogram):
        print("  Finding spectral peaks...")
        
        from scipy.ndimage import maximum_filter
        
        # Normalize spectrogram to 0-1 range
        spec_min = np.min(spectrogram)
        spec_max = np.max(spectrogram)
        spec_norm = (spectrogram - spec_min) / (spec_max - spec_min + 1e-6)
        
        # Use adaptive threshold based on percentile
        # 85th percentile = only strongest 15% of energy becomes candidate peaks
        threshold = np.percentile(spec_norm, 85)
        print(f"    Adaptive threshold (percentile 85): {threshold:.3f} (normalized)")
        
        # Find local maxima in normalized spectrogram
        struct = np.ones((self.peak_neighborhood_size, self.peak_neighborhood_size))
        local_max = maximum_filter(spec_norm, footprint=struct) == spec_norm
        
        # Apply threshold
        background = (spec_norm > threshold)
        peaks = local_max & background
        
        # Get peak coordinates
        peak_coords = np.argwhere(peaks)
        
        print(f"  Found {len(peak_coords)} peaks ({len(peak_coords)/spectrogram.size*100:.2f}% density)")
        
        # If no peaks found, relax threshold progressively
        if len(peak_coords) == 0:
            print("    Warning: No peaks at 85th percentile. Attempting lower thresholds...")
            for percentile in [75, 60, 50]:
                threshold = np.percentile(spec_norm, percentile)
                background = (spec_norm > threshold)
                peaks = local_max & background
                peak_coords = np.argwhere(peaks)
                print(f"    Percentile {percentile}: Found {len(peak_coords)} peaks")
                
                if len(peak_coords) > 10:  # Accept if reasonable number of peaks found
                    print(f"    ✓ Using threshold at {percentile}th percentile")
                    break
        
        return peak_coords
    
    def generate_fingerprints(self, peaks):
        """
        Generate fingerprints from peaks using combinatorial hashing.
        Each fingerprint is a hash of (freq1, freq2, time_delta).
        """
        print("  Generating fingerprints...")
        
        fingerprints = []
        
        if len(peaks) == 0:
            print(f"  No peaks to generate fingerprints from")
            return fingerprints
        
        # Sort peaks by time
        peaks = peaks[peaks[:, 1].argsort()]
        
        for i in range(len(peaks)):
            for j in range(1, min(self.fan_value, len(peaks) - i)):
                freq1 = peaks[i][0]
                freq2 = peaks[i + j][0]
                time1 = peaks[i][1]
                time2 = peaks[i + j][1]
                time_delta = time2 - time1
                
                # Create hash from frequency and time information
                hash_input = f"{freq1}|{freq2}|{time_delta}"
                fingerprint_hash = hashlib.sha1(hash_input.encode('utf-8')).hexdigest()[:20]
                
                fingerprints.append({
                    'hash': fingerprint_hash,
                    'time_offset': int(time1),
                    'freq1': int(freq1),
                    'freq2': int(freq2),
                    'time_delta': int(time_delta)
                })
        
        print(f"  Generated {len(fingerprints)} fingerprints from {len(peaks)} peaks")
        
        return fingerprints
    
    def visualize_fingerprints(self, audio_path, spectrogram, peaks, fingerprints):
        """Visualize the fingerprinting process"""
        filename = os.path.splitext(os.path.basename(audio_path))[0]
        
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        fig.suptitle(f'Dejavu Fingerprinting: {filename}', fontsize=16, fontweight='bold')
        
        # 1. Spectrogram with peaks
        img1 = librosa.display.specshow(spectrogram, y_axis='log', x_axis='time',
                                        sr=self.sample_rate, hop_length=self.hop_length,
                                        ax=axes[0], cmap='viridis')
        
        # Plot peaks
        if len(peaks) > 0:
            time_frames = librosa.frames_to_time(peaks[:, 1], sr=self.sample_rate, 
                                                  hop_length=self.hop_length)
            freq_bins = librosa.fft_frequencies(sr=self.sample_rate, n_fft=self.n_fft)[peaks[:, 0]]
            axes[0].scatter(time_frames, freq_bins, c='red', s=5, alpha=0.6, label='Detected Peaks')
            print(f"    Visualizing {len(peaks)} peaks on spectrogram")
        else:
            print(f"    No peaks to visualize")
        
        axes[0].set_title(f'Spectrogram with Detected Peaks ({len(peaks)} peaks)')
        axes[0].legend()
        fig.colorbar(img1, ax=axes[0], format='%+2.0f dB')
        
        # 2. Fingerprint distribution
        if fingerprints:
            time_offsets = [fp['time_offset'] for fp in fingerprints[:2000]]
            freq_deltas = [abs(fp['freq2'] - fp['freq1']) for fp in fingerprints[:2000]]
            
            axes[1].scatter(time_offsets, freq_deltas, alpha=0.4, s=2, c='blue')
            axes[1].set_title(f'Fingerprint Distribution (showing first {min(2000, len(fingerprints))} of {len(fingerprints)})')
            axes[1].set_xlabel('Time Offset (frames)')
            axes[1].set_ylabel('Frequency Delta (bins)')
            axes[1].grid(True, alpha=0.3)
        else:
            axes[1].text(0.5, 0.5, 'No fingerprints generated', ha='center', va='center',
                        transform=axes[1].transAxes, fontsize=14, color='red')
        
        plt.tight_layout()
        
        output_path = os.path.join(self.dejavu_plots_dir, f'{filename}_fingerprints.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Visualization saved: {output_path}")
    

    
    def save_fingerprints(self, audio_path, fingerprints, metadata):
        """Save fingerprints to JSON file"""
        filename = os.path.splitext(os.path.basename(audio_path))[0]
        
        output_data = {
            'filename': filename,
            'metadata': metadata,
            'num_fingerprints': len(fingerprints),
            'fingerprints': fingerprints[:100],  # Save first 100 for inspection
            'timestamp': datetime.now().isoformat()
        }
        
        output_path = os.path.join(self.dejavu_dir, f'{filename}_fingerprints.json')
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"  Fingerprints saved: {output_path}")
        
        # Store in database
        song_id = filename
        self.fingerprint_database[song_id] = fingerprints
        self.audio_metadata[song_id] = metadata
        
        return output_path
    
    def fingerprint_audio(self, audio_path, visualize=True):
        """Complete fingerprinting process for an audio file"""
        try:
            # Generate spectrogram
            spectrogram, sr, duration, waveform = self.generate_spectrogram(audio_path)
            
            # Find peaks
            peaks = self.find_peaks(spectrogram)
            
            # Generate fingerprints
            fingerprints = self.generate_fingerprints(peaks)
            
            # Metadata
            metadata = {
                'duration': float(duration),
                'sample_rate': int(sr),
                'num_peaks': int(len(peaks)),
                'num_fingerprints': len(fingerprints)
            }
            
            # Save fingerprints
            self.save_fingerprints(audio_path, fingerprints, metadata)
            
            # Visualize
            if visualize:
                self.visualize_fingerprints(audio_path, spectrogram, peaks, fingerprints)
            
            return fingerprints, metadata
            
        except Exception as e:
            print(f"  ERROR processing {audio_path}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None

    
    
    def match_fingerprints(self, query_fingerprints, query_filename):
        """
        Match query fingerprints against database.
        Returns best matches with confidence scores.
        """
        print(f"\nMatching {query_filename} against database...")
        
        if not self.fingerprint_database:
            print("  Database is empty. Please fingerprint reference audio first.")
            return []
        
        matches = defaultdict(int)
        
        # Create hash lookup for query
        query_hashes = {fp['hash']: fp['time_offset'] for fp in query_fingerprints}
        print(f"  Query fingerprints: {len(query_hashes)} unique hashes")
        
        # Match against each song in database
        for song_id, song_fingerprints in self.fingerprint_database.items():
            time_differences = []
            hash_matches = 0
            
            for fp in song_fingerprints:
                if fp['hash'] in query_hashes:
                    hash_matches += 1
                    time_diff = fp['time_offset'] - query_hashes[fp['hash']]
                    time_differences.append(time_diff)
            
            # Find most common time difference (indicates alignment)
            if time_differences:
                from collections import Counter
                counter = Counter(time_differences)
                most_common_diff, count = counter.most_common(1)[0]
                
                confidence = count / len(query_fingerprints) if len(query_fingerprints) > 0 else 0
                matches[song_id] = {
                    'match_count': count,
                    'time_offset': most_common_diff,
                    'confidence': confidence,
                    'metadata': self.audio_metadata.get(song_id, {})
                }
        
        # Sort by match count
        sorted_matches = sorted(matches.items(), key=lambda x: x[1]['match_count'], reverse=True)
        
        print(f"  Found {len(sorted_matches)} potential matches")
        
        return sorted_matches
    
    def save_match_results(self, query_filename, matches):
        """Save matching results to file"""
        output_data = {
            'query_file': query_filename,
            'timestamp': datetime.now().isoformat(),
            'num_matches': len(matches),
            'matches': [
                {
                    'reference_file': song_id,
                    'match_count': data['match_count'],
                    'confidence': f"{data['confidence']:.2%}",
                    'time_offset': data['time_offset'],
                    'metadata': data['metadata']
                }
                for song_id, data in matches[:10]  # Top 10 matches
            ]
        }
        
        filename = os.path.splitext(query_filename)[0]
        output_path = os.path.join(self.dejavu_matches_dir, f'{filename}_matches.json')
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"  Match results saved: {output_path}")


    def process_folder(self, folder_path, mode='fingerprint'):
        """
        Process all audio files in a folder.
        mode: 'fingerprint' to build database, 'query' to match against database
        """
        audio_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac']
        audio_files = []
        
        for ext in audio_extensions:
            audio_files.extend(Path(folder_path).glob(f'*{ext}'))
            audio_files.extend(Path(folder_path).glob(f'*{ext.upper()}'))
        
        if not audio_files:
            print(f"No audio files found in {folder_path}")
            return
        
        print(f"\nFound {len(audio_files)} audio files in {folder_path}")
        print("=" * 60)
        
        for audio_file in sorted(audio_files):
            if mode == 'fingerprint':
                self.fingerprint_audio(str(audio_file), visualize=True)
            elif mode == 'query':
                fingerprints, metadata = self.fingerprint_audio(str(audio_file), visualize=False)
                if fingerprints and metadata['num_fingerprints'] > 0:
                    matches = self.match_fingerprints(fingerprints, os.path.basename(str(audio_file)))
                    if matches:
                        self.save_match_results(os.path.basename(str(audio_file)), matches)
                        
                        # Print top matches
                        print(f"\n  Top 3 matches for {os.path.basename(str(audio_file))}:")
                        for i, (song_id, data) in enumerate(matches[:3], 1):
                            print(f"    {i}. {song_id}: {data['match_count']} matches ({data['confidence']:.2%} confidence)")


def main():
    """Main execution function"""
    print("\n" + "=" * 80)
    print("DEJAVU-STYLE AUDIO FINGERPRINTING (FIXED VERSION)")
    print("=" * 80)
    
    # Initialize fingerprinter
    fingerprinter = DejavuAudioFingerprinter(config_path='config.py')
    
    # Step 1: Build fingerprint database from reference audio
    print(f"\n{'='*80}")
    print("STEP 1: Building fingerprint database from reference audio")
    print(f"{'='*80}")
    
    if os.path.exists(fingerprinter.reference_folder):
        fingerprinter.process_folder(fingerprinter.reference_folder, mode='fingerprint')
    else:
        print(f"Reference folder not found: {fingerprinter.reference_folder}")
    
    # Also fingerprint test audio for database
    if os.path.exists(fingerprinter.test_folder):
        print(f"\n{'='*80}")
        print("STEP 1b: Adding test audio to database")
        print(f"{'='*80}")
        fingerprinter.process_folder(fingerprinter.test_folder, mode='fingerprint')
    
    # Step 2: Query matching
    print(f"\n{'='*80}")
    print("STEP 2: Matching query audio against database")
    print(f"{'='*80}")
    
    if os.path.exists(fingerprinter.query_folder):
        fingerprinter.process_folder(fingerprinter.query_folder, mode='query')
    else:
        print(f"Query folder not found: {fingerprinter.query_folder}")
    
    # Summary
    print(f"\n{'='*80}")
    print("PROCESSING COMPLETE!")
    print(f"Database size: {len(fingerprinter.fingerprint_database)} audio files")
    print(f"Output saved in: {fingerprinter.output_folder}/")
    print(f"  - Fingerprints: {fingerprinter.dejavu_dir}/")
    print(f"  - Visualizations: {fingerprinter.dejavu_plots_dir}/")
    print(f"  - Match results: {fingerprinter.dejavu_matches_dir}/")
    print(f"{'='*80}\n")


if _name_ == "_main_":
    main()