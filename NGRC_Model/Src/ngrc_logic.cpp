#include "ngrc_logic.h"
#include "ngrc_weights.h"

// Eigen library for optimized vector operations
#include "Dense.h"

using Eigen::Map;
using Eigen::Matrix;
using Eigen::RowMajor;

// Eigen optimization flags for embedded systems
#define EIGEN_NO_MALLOC              // Prevent dynamic allocation
#define EIGEN_STACK_ALLOCATION_LIMIT 0x8000  // 32KB stack limit
#define EIGEN_FAST_MATH              // Enable aggressive optimizations

void ngrc_predict(const float* fft_features_with_bias, 
                  float* out_classification_score) {
    // Ridge Classifier: score = dot(weights, features) + bias
    // Input: FFT features + bias term (length = NUM_FFT_FEATURES + 1)
    // Output: Single classification score
    
    // Map existing arrays to Eigen types (zero-copy)
    const Map<const Matrix<float, 1, RIDGE_TOTAL_PARAMS, RowMajor>> 
        weights_vector(reinterpret_cast<const float*>(ridge_weights));
    const Map<const Matrix<float, RIDGE_TOTAL_PARAMS, 1>> 
        input_vector(fft_features_with_bias);
    
    // Single dot product operation
    float score = weights_vector.dot(input_vector);
    
    // Store result (single value)
    out_classification_score[0] = score;
}

float ridge_classify(const float* fft_features_with_bias) {
    // Direct Ridge classification without intermediate buffer
    float score;
    ngrc_predict(fft_features_with_bias, &score);
    return score;
}