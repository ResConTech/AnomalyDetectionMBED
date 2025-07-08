// NGRC_Model/Src/polynomial_expansion.cpp
#include "polynomial_expansion.h"

void extract_delayed_features(const float* original_features,
                             int timestep,
                             float* delayed_features) {
    // Extract features at timestep + MAX_ABS_DELAY + delay
    // This ensures we're looking at the correct historical data
    
    // First delay (-1): previous timestep
    int t1 = timestep + MAX_ABS_DELAY + DELAY_1;  // t + 6 - 1 = t + 5
    for (int feat = 0; feat < NUM_PCA_FEATURES; feat++) {
        delayed_features[feat] = original_features[t1 * NUM_PCA_FEATURES + feat];
    }
    
    // Second delay (-6): 6 timesteps ago  
    int t2 = timestep + MAX_ABS_DELAY + DELAY_2;  // t + 6 - 6 = t
    for (int feat = 0; feat < NUM_PCA_FEATURES; feat++) {
        delayed_features[NUM_PCA_FEATURES + feat] = original_features[t2 * NUM_PCA_FEATURES + feat];
    }
}

void generate_polynomial_expansion(const float* delayed_features,
                                  float* expanded_vector) {
    int idx = 0;
    
    // Step 1: All linear terms (40 terms)
    for (int i = 0; i < NUM_LINEAR_TERMS; i++) {
        expanded_vector[idx++] = delayed_features[i];
    }
    
    // Step 2: All quadratic terms (820 terms)
    // This matches Python's combinations_with_replacement for degree 2
    
    // First, all unique pairs (i,j) where i < j
    for (int i = 0; i < NUM_LINEAR_TERMS; i++) {
        for (int j = i; j < NUM_LINEAR_TERMS; j++) {
            expanded_vector[idx++] = delayed_features[i] * delayed_features[j];
        }
    }
    
    // Step 3: Add bias term (1.0) as the last element
    expanded_vector[idx] = 1.0f;
    
    // Verify we filled exactly NGRC_EFFECTIVE_TERMS
    // idx should be 860 after quadratic terms, 861 after bias
}

void ngrc_expand_features(const float* original_features,
                         int timestep,
                         float* expanded_vector) {
    // Temporary buffer for linear terms
    float delayed_features[NUM_LINEAR_TERMS];
    
    // Extract features with delays
    extract_delayed_features(original_features, timestep, delayed_features);
    
    // Generate polynomial expansion
    generate_polynomial_expansion(delayed_features, expanded_vector);
}