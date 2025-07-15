// NGRC_Model/Src/ngrc_logic.cpp
#include "ngrc_logic.h"
#include "ngrc_weights.h"

// Eigen library for optimized matrix operations
#include "Dense.h"

using Eigen::Map;
using Eigen::Matrix;
using Eigen::RowMajor;

// Eigen optimization flags for embedded systems
#define EIGEN_NO_MALLOC              // Prevent dynamic allocation
#define EIGEN_STACK_ALLOCATION_LIMIT 0x8000  // 32KB stack limit
#define EIGEN_FAST_MATH              // Enable aggressive optimizations

void ngrc_predict(const float* current_expanded_x_vector, 
                  float* out_predicted_features) {
    // Optimized matrix-vector multiplication using Eigen
    // weights: [20][861], x_vector: [861], out: [20]
    
    // Map existing arrays to Eigen types (zero-copy)
    const Map<const Matrix<float, NUM_PCA_FEATURES, NGRC_EFFECTIVE_TERMS, RowMajor>> 
        weights_matrix(reinterpret_cast<const float*>(ngrc_trained_weights));
    const Map<const Matrix<float, NGRC_EFFECTIVE_TERMS, 1>> 
        input_vector(current_expanded_x_vector);
    Map<Matrix<float, NUM_PCA_FEATURES, 1>> 
        output_vector(out_predicted_features);
    
    // Single vectorized operation - replaces nested loops
    output_vector = weights_matrix * input_vector;
}

float calculate_mse(const float* predicted, const float* actual) {
    // Optimized MSE calculation using Eigen vector operations
    const Map<const Matrix<float, NUM_PCA_FEATURES, 1>> pred_vec(predicted);
    const Map<const Matrix<float, NUM_PCA_FEATURES, 1>> actual_vec(actual);
    
    // Vectorized difference, square, and sum operations
    Matrix<float, NUM_PCA_FEATURES, 1> diff = pred_vec - actual_vec;
    return diff.squaredNorm() / NUM_PCA_FEATURES;
}