// NGRC_Model/Src/ngrc_logic.cpp
#include "ngrc_logic.h"
#include "ngrc_weights.h"

void ngrc_predict(const float* current_expanded_x_vector, 
                  float* out_predicted_features) {
    // Perform matrix-vector multiplication
    // out = weights * x_vector
    // weights: [20][861], x_vector: [861], out: [20]
    
    for (int feature_idx = 0; feature_idx < NUM_PCA_FEATURES; feature_idx++) {
        float sum = 0.0f;
        
        // Dot product of weight row with input vector
        for (int term_idx = 0; term_idx < NGRC_EFFECTIVE_TERMS; term_idx++) {
            sum += ngrc_trained_weights[feature_idx][term_idx] * 
                   current_expanded_x_vector[term_idx];
        }
        
        out_predicted_features[feature_idx] = sum;
    }
}

float calculate_mse(const float* predicted, const float* actual) {
    float sum_squared_error = 0.0f;
    
    for (int i = 0; i < NUM_PCA_FEATURES; i++) {
        float error = predicted[i] - actual[i];
        sum_squared_error += error * error;
    }
    
    // Return mean squared error
    return sum_squared_error / NUM_PCA_FEATURES;
}