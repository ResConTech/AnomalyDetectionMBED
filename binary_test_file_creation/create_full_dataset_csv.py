#!/usr/bin/env python3

import os
from pathlib import Path

def create_full_dataset_csv():
    """Create CSV file for the full dataset with all 2,459 samples"""
    
    source_dir = Path("/Users/jaymain/MLPerf Tiny C++ MBED/test_samples_corrected")
    
    # Get all .bin files
    all_files = list(source_dir.glob("*.bin"))
    
    print(f"Found {len(all_files)} binary files")
    
    # Sort files for consistent ordering
    all_files.sort()
    
    csv_content = []
    normal_count = 0
    anomaly_count = 0
    
    for file_path in all_files:
        filename = file_path.name
        
        # Determine class label
        if "normal" in filename:
            predicted_class = 0
            normal_count += 1
        elif "anomaly" in filename:
            predicted_class = 1
            anomaly_count += 1
        else:
            predicted_class = 0  # default
            normal_count += 1
        
        # CSV format: input file name, total classes, predicted class, window width, stride
        csv_line = f"{filename},2,{predicted_class},412,412"
        csv_content.append(csv_line)
    
    # Write full CSV
    full_csv_path = source_dir / "y_labels_full.csv"
    with open(full_csv_path, 'w') as f:
        f.write('\n'.join(csv_content))
    
    print(f"Full CSV written to: {full_csv_path}")
    print(f"Total samples: {len(all_files)}")
    print(f"Normal samples: {normal_count}")
    print(f"Anomaly samples: {anomaly_count}")
    
    # Create summary by car
    car_summary = {}
    for filename in [f.name for f in all_files]:
        car_id = filename.split('_')[1]  # Extract car ID from filename
        if car_id not in car_summary:
            car_summary[car_id] = {'normal': 0, 'anomaly': 0}
        
        if "normal" in filename:
            car_summary[car_id]['normal'] += 1
        elif "anomaly" in filename:
            car_summary[car_id]['anomaly'] += 1
    
    print(f"\nBreakdown by car:")
    for car_id in sorted(car_summary.keys()):
        normal = car_summary[car_id]['normal']
        anomaly = car_summary[car_id]['anomaly']
        total = normal + anomaly
        print(f"  Car {car_id}: {normal} normal + {anomaly} anomaly = {total} total")
    
    # Create summary file
    summary_path = source_dir / "README_full.txt"
    with open(summary_path, 'w') as f:
        f.write("Full Dataset for MLPerf Tiny Anomaly Detection\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total samples: {len(all_files)}\n")
        f.write(f"Normal samples: {normal_count}\n")
        f.write(f"Anomaly samples: {anomaly_count}\n")
        f.write("Cars: 1-4\n")
        f.write("File size: 412 bytes each (103 float32 values)\n")
        f.write("Format: Little-endian float32\n")
        f.write("Window width: 412 bytes\n")
        f.write("Stride: 412 bytes (single window per sample)\n\n")
        f.write("Files breakdown:\n")
        
        for car_id in sorted(car_summary.keys()):
            normal = car_summary[car_id]['normal']
            anomaly = car_summary[car_id]['anomaly']
            total = normal + anomaly
            f.write(f"  Car {car_id}: {normal} normal + {anomaly} anomaly = {total} total\n")
    
    return len(all_files)

if __name__ == "__main__":
    total_samples = create_full_dataset_csv()
    print(f"\n✅ Full dataset CSV created with {total_samples} samples")
    print("✅ Ready for complete MLPerf Tiny benchmarking")