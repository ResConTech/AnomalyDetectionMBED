#include "polynomial_expansion.h"

// Simple feature preparation for Ridge Classifier
// No polynomial expansion needed - just copy FFT features and add bias term
void ngrc_expand_features(const float* fft_features,
                         int timestep,
                         float* features_with_bias) {
    // Copy FFT features directly (no expansion needed)
    for (int i = 0; i < NUM_FFT_FEATURES; i++) {
        features_with_bias[i] = fft_features[i];
    }
    
    // Add bias term as the last element
    features_with_bias[NUM_FFT_FEATURES] = 1.0f;
}