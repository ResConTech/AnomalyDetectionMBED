#!/usr/bin/env python3

import numpy as np
import os

def convert_weights_to_cpp():
    """Convert trained Ridge classifier weights to C++ format"""
    
    # Load the trained weights and bias
    weight_matrix_path = 'trained_weight_matrix.npy'
    bias_path = 'trained_bias.npy'
    
    if not os.path.exists(weight_matrix_path):
        print(f"Error: {weight_matrix_path} not found")
        return
    
    if not os.path.exists(bias_path):
        print(f"Error: {bias_path} not found")
        return
    
    # Load the weights
    weight_matrix = np.load(weight_matrix_path)
    bias = np.load(bias_path)
    
    print(f"Weight matrix shape: {weight_matrix.shape}")
    print(f"Bias shape: {bias.shape}")
    
    # Ridge classifier weights should be a 1D array
    if len(weight_matrix.shape) == 2:
        weights = weight_matrix.flatten()
    else:
        weights = weight_matrix
    
    # Combine weights and bias
    all_params = np.concatenate([weights, bias])
    
    print(f"Total parameters: {len(all_params)} (expected: {len(weights)} weights + {len(bias)} bias)")
    
    # Generate C++ code
    cpp_code = f"""// Auto-generated Ridge Classifier weights
// Generated from trained_weight_matrix.npy and trained_bias.npy
// Features: {len(weights)}, Bias: {len(bias)}, Total: {len(all_params)}

#include "ngrc_weights.h"

const float ridge_weights[RIDGE_TOTAL_PARAMS] = {{
"""
    
    # Add weights in groups of 8 for readability
    for i in range(0, len(all_params), 8):
        group = all_params[i:i+8]
        line = "    " + ", ".join([f"{w:.8f}f" for w in group])
        if i + 8 < len(all_params):
            line += ","
        cpp_code += line + "\n"
    
    cpp_code += "};\n"
    
    # Write to file
    output_path = 'ridge_weights_generated.cpp'
    with open(output_path, 'w') as f:
        f.write(cpp_code)
    
    print(f"Generated C++ weights file: {output_path}")
    
    # Also generate a header update
    header_update = f"""
// Update for model_config.h:
#define NUM_FFT_BINS {len(weights)}
#define NUM_FFT_FEATURES {len(weights)}
#define RIDGE_WEIGHTS_SIZE {len(weights)}
#define RIDGE_TOTAL_PARAMS {len(all_params)}
"""
    
    with open('model_config_update.txt', 'w') as f:
        f.write(header_update)
    
    print(f"Generated config update: model_config_update.txt")
    
    # Show summary
    print(f"\nSummary:")
    print(f"  Features: {len(weights)}")
    print(f"  Bias terms: {len(bias)}")
    print(f"  Total parameters: {len(all_params)}")
    print(f"  Weight range: {weights.min():.6f} to {weights.max():.6f}")
    print(f"  Bias value: {bias[0]:.6f}")
    
    return len(all_params)

if __name__ == "__main__":
    convert_weights_to_cpp()