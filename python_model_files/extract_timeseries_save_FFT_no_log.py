import numpy as np
import pandas as pd
import soundfile as sf
from pathlib import Path
import pickle
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

class TimeSeriesExtractor:
    """
    Extracts Log-Filter-Bank Energy (LFBE) features, uses a single global
    PCA model for dimensionality reduction, and then standardizes the data
    with a single global scaler.
    """

    def __init__(self,
                 t_pts: int = 1000,  # Number of time points to extract
                 sr: int = 16000,
                 n_freq_ave: int = 1):  # Number of frequency bins to average
        self.t_pts = t_pts
        self.sr = sr
        self.n_freq_ave = n_freq_ave

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def _load_audio(self, audio_path: Path):
        """Load audio with soundfile """
        y, sr = sf.read(audio_path, always_2d=False)
        return y.astype(np.float32), sr

    def extract_features_from_file(self, audio_path: Path):
        try:
            # Load audio
            y, sr = self._load_audio(audio_path)
            
            # Truncate if t_pts specified
            if self.t_pts is not None:
                y = y[:self.t_pts]
            
            # Compute FFT
            yf = np.fft.fft(y)
            xf = np.fft.fftfreq(len(y), 1/sr)
            
            # Get positive frequencies only
            pos_mask = xf >= 0
            yf_abs = np.abs(yf[pos_mask])
            
            # Calculate magnitude spectrum
            spectrum = yf_abs.astype(np.float32)
            
            # Average frequency bins if n_freq_ave > 1
            if self.n_freq_ave > 1:
                n_bins = len(spectrum)
                n_averaged_bins = n_bins // self.n_freq_ave
                reshaped = spectrum[:n_averaged_bins * self.n_freq_ave]
                reshaped = reshaped.reshape(-1, self.n_freq_ave)
                spectrum = np.mean(reshaped, axis=1)
                
            return spectrum
            
        except Exception as e:
            print(f"[Error] {audio_path}: {e}")
            return np.zeros(self.t_pts//2 + 1 if self.t_pts else 0, dtype=np.float32)

    # ------------------------- dataset helpers -----------------------
    def parse_filename(self, filepath: Path):
        name = filepath.name
        parts = name.replace('.wav', '').split('_')
        if len(parts) >= 4:
            return {'label': parts[0], 'car_id': int(parts[2]), 'filepath': str(filepath)}
        return {'label': 'unknown', 'car_id': -1, 'filepath': str(filepath)}

    def process_dataset_folder(self, folder_path: str):
        folder = Path(folder_path)
        files = list(folder.glob('*.wav'))
        print(f"Processing {len(files)} wav files in {folder_path}")

        data = {}
        for wav in tqdm(files, desc="Extracting FFT"):
            meta = self.parse_filename(wav)
            if meta['car_id'] == -1: continue
            
            features = self.extract_features_from_file(wav)
            # Separate normal and anomaly data
            data_type = 'normal' if meta['label'] == 'normal' else 'anomaly'
            bucket = data.setdefault(meta['car_id'], {})
            type_bucket = bucket.setdefault(data_type, {'features': [], 'metadata': []})
            type_bucket['features'].append(features)
            type_bucket['metadata'].append(meta)

        # Convert features to numpy arrays
        for car_id, car_data in data.items():
            for data_type in car_data:
                car_data[data_type]['features'] = np.array(car_data[data_type]['features'], dtype=np.float32)
        return data

    def save_processed_data(self, data_dict, dataset_type, out_dir):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        
        # Save individual car data
        for cid, bucket in data_dict.items():
            # Handle training data
            if dataset_type == 'train':
                if cid <= 4:  # Cars 1-4 are normal
                    if 'normal' in bucket:
                        features = bucket['normal']['features']
                        np.save(out/f"{dataset_type}_car_{cid:02d}_fft.npy", features)
                        
                        # Save average FFT for this car
                        avg_fft = np.mean(features, axis=0)
                        np.save(out/f"{dataset_type}_car_{cid:02d}_avg_fft.npy", avg_fft)
                        
                        # Save metadata
                        with open(out/f"{dataset_type}_car_{cid:02d}_metadata.pkl", 'wb') as f:
                            pickle.dump(bucket['normal']['metadata'], f)
                else:  # Cars 5-7 are labeled as anomaly
                    if 'normal' in bucket:  # Data is still in 'normal' bucket due to file naming
                        features = bucket['normal']['features']
                        np.save(out/f"{dataset_type}_car_{cid:02d}_anomaly_fft.npy", features)
                        with open(out/f"{dataset_type}_car_{cid:02d}_anomaly_metadata.pkl", 'wb') as f:
                            pickle.dump(bucket['normal']['metadata'], f)
            
            # Handle test data
            else:
                # Save normal test data
                if 'normal' in bucket:
                    np.save(out/f"{dataset_type}_car_{cid:02d}_fft.npy", 
                           bucket['normal']['features'])
                    with open(out/f"{dataset_type}_car_{cid:02d}_metadata.pkl", 'wb') as f:
                        pickle.dump(bucket['normal']['metadata'], f)
                
                # Save anomaly test data
                if 'anomaly' in bucket:
                    np.save(out/f"{dataset_type}_car_{cid:02d}_anomaly_fft.npy", 
                           bucket['anomaly']['features'])
                    with open(out/f"{dataset_type}_car_{cid:02d}_anomaly_metadata.pkl", 'wb') as f:
                        pickle.dump(bucket['anomaly']['metadata'], f)
        
        # For training data, compute overall average across cars 1-4 only
        if dataset_type == 'train':
            all_features = []
            for cid in range(1, 5):  # Only use cars 1-4 for overall average
                if cid in data_dict and 'normal' in data_dict[cid]:
                    all_features.extend(data_dict[cid]['normal']['features'])
            
            if all_features:
                overall_avg = np.mean(all_features, axis=0)
                np.save(out/f"{dataset_type}_all_cars_avg_fft.npy", overall_avg)
        
        print(f"Saved FFT data to {out}")

# ========================================================================
# MAIN SCRIPT
# ========================================================================
def main():
    # Define frequency averaging first
    n_freq_ave = 900  # Average every n_freq_ave frequency bins
    t_pts = None  # Number of time points to extract, None is all
    
    ###
    ### ResCon laptop
    ###
    train_dir = r"C:\Users\DanielGauthier\Documents\MLPerf\ToyCar\train"
    test_dir = r"C:\Users\DanielGauthier\Documents\MLPerf\ToyCar\test"
    output_dir = fr"C:\Users\DanielGauthier\Documents\MLPerf\FFT_ave_{n_freq_ave}_no_log"
    ###
    ### Home desktop
    #train_dir = r"C:\Users\danie\Documents\ToyCar\train"
    #test_dir = r"C:\Users\danie\Documents\ToyCar\test"
    #output_dir = fr"C:\Users\danie\Documents\ToyCar\FFT_ave_{n_freq_ave}"
    ###

    print("="*60)
    print("TIME SERIES EXTRACTION")
    print(f"Time points: {t_pts}")
    print(f"Frequency averaging: {n_freq_ave} bins")
    print("="*60)

    extractor = TimeSeriesExtractor(t_pts=t_pts, n_freq_ave=n_freq_ave)
    
    # Process first file to get dimensions
    first_file = list(Path(train_dir).glob('*.wav'))[0]
    first_features = extractor.extract_features_from_file(first_file)
    print(f"Number of frequency bins after averaging: {len(first_features)}")
    
    # Process and save training data
    train_data = extractor.process_dataset_folder(train_dir)
    extractor.save_processed_data(train_data, 'train', output_dir)
    
    # Process and save test data
    test_data = extractor.process_dataset_folder(test_dir)
    extractor.save_processed_data(test_data, 'test', output_dir)
    
    print("\n" + "="*60)
    print("TIME SERIES EXTRACTION COMPLETE!")
    print(f"Raw time series saved to: {output_dir}")
    print("="*60)

if __name__ == '__main__':
    main()
