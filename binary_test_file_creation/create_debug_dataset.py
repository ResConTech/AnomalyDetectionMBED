#!/usr/bin/env python3

import os
import shutil
from pathlib import Path

def create_debug_dataset():
    """Create debug dataset with 52 samples total (13 per car: 6 normal, 7 anomaly)"""
    
    source_dir = Path("/Users/jaymain/MLPerf Tiny C++ MBED/test_samples_corrected")
    debug_dir = Path("/Users/jaymain/MLPerf Tiny C++ MBED/test_samples_debug")
    
    # Create debug directory
    debug_dir.mkdir(exist_ok=True)
    
    # Clear existing files
    for file in debug_dir.glob("*"):
        file.unlink()
    
    copied_files = []
    
    # For each car (1-4)
    for car_id in range(1, 5):
        car_files = []
        
        # Get 6 normal samples
        normal_pattern = f"car_{car_id:02d}_sample_*_normal.bin"
        normal_files = sorted(list(source_dir.glob(normal_pattern)))[:6]
        
        # Get 7 anomaly samples  
        anomaly_pattern = f"car_{car_id:02d}_sample_*_anomaly.bin"
        anomaly_files = sorted(list(source_dir.glob(anomaly_pattern)))[:7]
        
        # Copy files to debug directory
        for file in normal_files + anomaly_files:
            dest_file = debug_dir / file.name
            shutil.copy2(file, dest_file)
            copied_files.append(file.name)
            car_files.append(file.name)
        
        print(f"Car {car_id}: {len(normal_files)} normal + {len(anomaly_files)} anomaly = {len(car_files)} total")
    
    print(f"\nTotal debug samples: {len(copied_files)}")
    
    # Create debug CSV
    csv_content = []
    
    # Sort files for consistent ordering
    copied_files.sort()
    
    for filename in copied_files:
        # Determine class label
        if "normal" in filename:
            predicted_class = 0
        elif "anomaly" in filename:
            predicted_class = 1
        else:
            predicted_class = 0  # default
        
        # CSV format: input file name, total classes, predicted class, window width, stride
        csv_line = f"{filename},2,{predicted_class},412,412"
        csv_content.append(csv_line)
    
    # Write debug CSV
    debug_csv_path = debug_dir / "y_labels_debug.csv"
    with open(debug_csv_path, 'w') as f:
        f.write('\n'.join(csv_content))
    
    print(f"Debug CSV written to: {debug_csv_path}")
    print(f"Debug samples directory: {debug_dir}")
    
    # Create summary
    summary_path = debug_dir / "README.txt"
    with open(summary_path, 'w') as f:
        f.write("Debug Dataset for MLPerf Tiny Anomaly Detection\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total samples: {len(copied_files)}\n")
        f.write("Per car: 13 samples (6 normal + 7 anomaly)\n")
        f.write("Cars: 1-4\n")
        f.write("File size: 412 bytes each (103 float32 values)\n")
        f.write("Format: Little-endian float32\n")
        f.write("Window width: 412 bytes\n")
        f.write("Stride: 412 bytes (single window per sample)\n\n")
        f.write("Files breakdown:\n")
        
        for car_id in range(1, 5):
            normal_count = len([f for f in copied_files if f"car_{car_id:02d}" in f and "normal" in f])
            anomaly_count = len([f for f in copied_files if f"car_{car_id:02d}" in f and "anomaly" in f])
            f.write(f"  Car {car_id}: {normal_count} normal + {anomaly_count} anomaly = {normal_count + anomaly_count} total\n")
    
    return len(copied_files)

if __name__ == "__main__":
    total_samples = create_debug_dataset()
    print(f"\n✅ Debug dataset created with {total_samples} samples")
    print("✅ Debug CSV file created: y_labels_debug.csv")
    print("✅ Ready for debugging and initial testing")