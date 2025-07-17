import numpy as np
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

class AnomalyDetector:
    def __init__(self):
        # Training data should have cars 1-4 (normal) and 5-7 (anomaly)
        self.train_normal = {}    # For normal training data
        self.train_anomaly = {}   # For anomaly training data
        
        # Testing data should only have cars 1-4 (both normal and anomaly)
        self.test_normal = {}     # For normal test data
        self.test_anomaly = {}    # For anomaly test data

class FFT_AnomalyDetector(AnomalyDetector):
    """FFT-based anomaly detector that processes frequency domain data."""
    
    def __init__(self, scale: float = 1.0, alpha: float = 1.0, f_min: float = 0, f_max: float = 8000):
        super().__init__()
        self.avg_fft = None          # Overall average FFT
        self.sampling_rate = 16000   # Hz
        self.threshold = np.float32(0.1)         # Initialize threshold value as float32
        self.scale = np.float32(scale)          # Scaling factor for data as float32
        self.alpha = np.float32(alpha)          # Regularization strength for RidgeClassifier as float32
        self.f_min = np.float32(f_min)          # Minimum frequency to consider as float32
        self.f_max = np.float32(f_max)          # Maximum frequency to consider as float32
    
    def load_data(self, data_dir: str, car_ids: List[int]):
        """
        Loads FFT data from files, applying frequency filtering.
        """
        data_dir_path = Path(data_dir)
        print(f"\nChecking directory: {data_dir_path}")

        # First load overall average to determine number of frequency bins
        avg_fft_path = data_dir_path / 'train_all_cars_avg_fft.npy'
        if avg_fft_path.exists():
            self.avg_fft = np.load(avg_fft_path).astype(np.float32)
            n_freq_points = len(self.avg_fft)
            
            # Calculate frequency values and create mask
            freqs = np.linspace(0, self.sampling_rate/2, n_freq_points, dtype=np.float32)
            self.freq_mask = (freqs >= self.f_min) & (freqs <= self.f_max)
            print(f"Original number of frequency bins: {n_freq_points}")
            print(f"Using frequencies between {self.f_min} Hz and {self.f_max} Hz")
            print(f"Number of frequency bins after mask: {np.sum(self.freq_mask)}")
            
            # Apply frequency mask to average FFT
            self.avg_fft = self.avg_fft[self.freq_mask]
        else:
            print(f"Error: Overall average FFT file not found at {avg_fft_path}")
            return

        # Load normal training data (cars 1-4)
        for car_id in range(1, 5):
            fft_path = data_dir_path / f'train_car_{car_id:02d}_fft.npy'
            if fft_path.exists():
                fft_data = np.load(fft_path).astype(np.float32)
                # Apply frequency mask and subtract average FFT
                fft_data = fft_data[:, self.freq_mask]
                #fft_data = (fft_data - self.avg_fft) * self.scale
                fft_data = fft_data * self.scale
                self.train_normal[car_id] = fft_data
                
                # Calculate and print statistics for car 1
                if car_id == 1:
                    mean_vals = np.mean(fft_data, axis=0).astype(np.float32)
                    std_vals = np.std(fft_data, axis=0).astype(np.float32)
                    print(f"\nCar 1 Training Statistics (after scaling):")
                    print(f"Scale factor: {self.scale}")
                    print(f"Mean across samples: {mean_vals.mean():.6f}")
                    print(f"Standard deviation across samples: {std_vals.mean():.6f}")
                    print(f"Max absolute mean value: {np.abs(mean_vals).max():.6f}")
                    print(f"Max standard deviation: {std_vals.max():.6f}")
                
                print(f"Loaded normal training data for Car {car_id}, shape: {fft_data.shape}")
        
        # Load anomaly training data (cars 5-7)
        for car_id in range(5, 8):
            fft_path = data_dir_path / f'train_car_{car_id:02d}_anomaly_fft.npy'
            if fft_path.exists():
                fft_data = np.load(fft_path).astype(np.float32)
                # Apply frequency mask and subtract average FFT
                fft_data = fft_data[:, self.freq_mask]
                #fft_data = (fft_data - self.avg_fft) * self.scale
                fft_data = fft_data * self.scale
                self.train_anomaly[car_id] = fft_data
                print(f"Loaded anomaly training data for Car {car_id}, shape: {fft_data.shape}")
        
        # Load test data for evaluation cars
        for car_id in car_ids:
            # Load normal test data
            normal_path = data_dir_path / f'test_car_{car_id:02d}_fft.npy'
            if normal_path.exists():
                test_data = np.load(normal_path).astype(np.float32)
                # Apply frequency mask and subtract average FFT
                test_data = test_data[:, self.freq_mask]
                #test_data = (test_data - self.avg_fft) * self.scale
                test_data = test_data * self.scale
                self.test_normal[car_id] = test_data
                print(f"Loaded normal test data for Car {car_id}, shape: {test_data.shape}")
            
            # Load anomaly test data
            anomaly_path = data_dir_path / f'test_car_{car_id:02d}_anomaly_fft.npy'
            if anomaly_path.exists():
                test_data = np.load(anomaly_path).astype(np.float32)
                # Apply frequency mask and subtract average FFT
                test_data = test_data[:, self.freq_mask]
                #test_data = (test_data - self.avg_fft) * self.scale
                test_data = test_data * self.scale
                self.test_anomaly[car_id] = test_data
                print(f"Loaded anomaly test data for Car {car_id}, shape: {test_data.shape}")


    def train_classifier(self):
        """Train Ridge Classifier on normal (cars 1-4) vs anomaly (cars 5-7) data"""
        # Combine normal training data
        normal_data = []
        normal_labels = []
        for car_id in range(1, 5):
            if car_id in self.train_normal:
                normal_data.append(self.train_normal[car_id])
                normal_labels.extend([0] * len(self.train_normal[car_id]))
        
        # Combine anomaly training data
        anomaly_data = []
        anomaly_labels = []
        for car_id in range(5, 8):
            if car_id in self.train_anomaly:
                anomaly_data.append(self.train_anomaly[car_id])
                anomaly_labels.extend([1] * len(self.train_anomaly[car_id]))
        
        # Combine all training data and ensure float32
        X_train = np.vstack(normal_data + anomaly_data).astype(np.float32)
        y_train = np.array(normal_labels + anomaly_labels)
        print(f"Training data shape: {X_train.shape}, Labels shape: {y_train.shape}")
        
        # Train classifier with specified alpha
        self.classifier = RidgeClassifier(alpha=float(self.alpha), class_weight='balanced')
        self.classifier.fit(X_train, y_train)
        
        # Save the trained weight matrix
        weight_matrix = self.classifier.coef_.astype(np.float32)
        np.save('trained_weight_matrix.npy', weight_matrix)
        print(f"Saved trained weight matrix with shape: {weight_matrix.shape}")
        
        # Also save bias if available
        if hasattr(self.classifier, 'intercept_'):
            bias = self.classifier.intercept_.astype(np.float32)
            np.save('trained_bias.npy', bias)
            print(f"Saved bias with shape: {bias.shape}")
        
        # Get training predictions and plot confusion matrix
        y_pred = self.classifier.predict(X_train)
        cm = confusion_matrix(y_train, y_pred)
        
        plt.figure(figsize=(8, 8))
        ConfusionMatrixDisplay(confusion_matrix=cm, 
                          display_labels=['Normal', 'Anomaly']).plot()
        plt.title('Training Data Confusion Matrix')
        plt.show()
        
        return y_train, y_pred
    
    def find_optimal_threshold(self, test_cars):
        """Find optimal threshold across all test cars."""
        all_true = []
        all_scores = []
        
        # Collect all test data
        for car_id in test_cars:
            if car_id in self.test_normal and car_id in self.test_anomaly:
                X_test = np.vstack([self.test_normal[car_id], self.test_anomaly[car_id]]).astype(np.float32)
                y_true = np.hstack([np.zeros(len(self.test_normal[car_id])), 
                                  np.ones(len(self.test_anomaly[car_id]))])
                scores = self.classifier.decision_function(X_test).astype(np.float32)
                
                all_true.extend(y_true)
                all_scores.extend(scores)
        
        # Find optimal threshold using ROC curve
        fpr, tpr, thresholds = roc_curve(all_true, all_scores)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = np.float32(thresholds[optimal_idx])
        
        return optimal_threshold

    def evaluate_test_car(self, car_id: int, threshold: float):
        """Evaluate test data for a single car using trained classifier"""
        if car_id not in self.test_normal or car_id not in self.test_anomaly:
            print(f"Error: Missing test data for Car {car_id}")
            return None, None, None
        
        # Combine test data and create labels
        X_test = np.vstack([self.test_normal[car_id], self.test_anomaly[car_id]]).astype(np.float32)
        y_true = np.hstack([np.zeros(len(self.test_normal[car_id])), 
                       np.ones(len(self.test_anomaly[car_id]))])
        
        # Get decision function scores and ensure float32
        scores = self.classifier.decision_function(X_test).astype(np.float32)
        
        # Make predictions using the provided threshold
        y_pred = (scores >= np.float32(threshold)).astype(int)
        
        # Calculate ROC AUC
        fpr, tpr, _ = roc_curve(y_true, scores)
        roc_auc = auc(fpr, tpr)
        
        # Plot confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 8))
        ConfusionMatrixDisplay(confusion_matrix=cm,
                          display_labels=['Normal', 'Anomaly']).plot()
        plt.title(f'Test Data Confusion Matrix - Car {car_id}\nThreshold: {threshold:.3f}')
        plt.show()
        
        return y_true, scores, y_pred, np.float32(roc_auc)

def main():
    # Configuration - ensure all are float32
    N_FREQ_AVE = 800
    ## with mean subtracted
    # SCALE = 3.0
    # ALPHA = 2.0  # Regularization strength for RidgeClassifier
    SCALE = np.float32(1.0)
    ALPHA = np.float32(1.e-3)  # Regularization strength for RidgeClassifier
    F_MIN = np.float32(1200.)  # Minimum frequency to consider (Hz)
    F_MAX = np.float32(7900.) # Maximum frequency to consider (Hz)
    DATA_DIR = "/Users/jaymain/MLPerf Tiny C++ MBED/FFT_ave_800_no_log"
    TEST_CARS = [1, 2, 3, 4]
    
    # Initialize detector with all parameters
    detector = FFT_AnomalyDetector(scale=SCALE, alpha=ALPHA, f_min=F_MIN, f_max=F_MAX) # type: ignore
    
    # Initialize and load data
    print("\n=== Loading FFT Data ===")
    detector.load_data(DATA_DIR, TEST_CARS)
    
    # Train classifier and show training results
    print("\n=== Training Classifier ===")
    y_train, y_pred_train = detector.train_classifier()
    
    # Find optimal threshold across all test cars
    print("\n=== Finding Optimal Threshold ===")
    optimal_threshold = detector.find_optimal_threshold(TEST_CARS)
    print(f"Optimal threshold across all cars: {optimal_threshold:.3f}")
    
    # Initialize list to store AUC scores
    auc_scores = []
    
    # Evaluate each test car
    for car_id in TEST_CARS:
        print(f"\n=== Analyzing Car {car_id} ===")
        
        # Get predictions and scores using optimal threshold - FIXED METHOD CALL AND UNPACKING
        y_true, y_scores, y_pred, roc_auc = detector.evaluate_test_car(car_id, optimal_threshold) # type: ignore
        
        if y_true is not None:
            # Compute and plot ROC curve
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            car_auc = auc(fpr, tpr)
            auc_scores.append(np.float32(car_auc))
            
            plt.figure(figsize=(8, 8))
            plt.plot(fpr, tpr, color='darkorange', lw=2,
                    label=f'ROC curve (AUC = {car_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - Car {car_id}')
            plt.legend(loc="lower right")
            plt.grid(True)
            plt.show()
    
    # Print composite results
    composite_auc = np.mean(auc_scores).astype(np.float32)
    print("\n=== Composite Results ===")
    print(f"Individual AUC scores: {[f'{score:.3f}' for score in auc_scores]}")
    print(f"Composite AUC score: {composite_auc:.3f}")
    
    # Convert weights to C++ format
    print("\n=== Converting to C++ Format ===")
    try:
        import subprocess
        result = subprocess.run(['python', 'convert_weights_to_cpp.py'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("Successfully converted weights to C++ format")
        else:
            print(f"Error converting weights: {result.stderr}")
    except Exception as e:
        print(f"Could not run conversion script: {e}")
        print("Please run 'python convert_weights_to_cpp.py' manually")

if __name__ == "__main__":
    main()