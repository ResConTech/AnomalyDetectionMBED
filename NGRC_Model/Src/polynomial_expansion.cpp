// NGRC_Model/Src/polynomial_expansion.cpp
#include "polynomial_expansion.h"

// Eigen library for optimized vector operations
#include "Dense.h"

using Eigen::Map;
using Eigen::Matrix;
using Eigen::RowMajor;

// Eigen optimization flags for embedded systems
#define EIGEN_NO_MALLOC              // Prevent dynamic allocation
#define EIGEN_STACK_ALLOCATION_LIMIT 0x8000  // 32KB stack limit
#define EIGEN_FAST_MATH              // Enable aggressive optimizations

void extract_delayed_features(const float* original_features,
                             int timestep,
                             float* delayed_features) {
    // Optimized feature extraction using Eigen vector operations
    // Map original features as a matrix for efficient row access
    const Map<const Matrix<float, WINDOW_SIZE, NUM_PCA_FEATURES, RowMajor>> 
        features_matrix(original_features);
    Map<Matrix<float, NUM_LINEAR_TERMS, 1>> output(delayed_features);
    
    // Calculate timestep indices for delays
    int t1 = timestep + MAX_ABS_DELAY + DELAY_1;  // t + 6 - 1 = t + 5
    int t2 = timestep + MAX_ABS_DELAY + DELAY_2;  // t + 6 - 6 = t
    
    // Vectorized memory copy - much faster than loops
    output.head<NUM_PCA_FEATURES>() = features_matrix.row(t1);
    output.tail<NUM_PCA_FEATURES>() = features_matrix.row(t2);
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
    
    // All unique pairs (i,j) where i <= j (includes squares)
    for (int i = 0; i < NUM_LINEAR_TERMS; i++) {
        for (int j = i; j < NUM_LINEAR_TERMS; j++) {
            expanded_vector[idx++] = delayed_features[i] * delayed_features[j];
        }
    }
    
    // Step 3: Add bias term as the last element
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