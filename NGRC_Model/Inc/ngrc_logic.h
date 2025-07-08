// NGRC_Model/Inc/ngrc_logic.h
#ifndef NGRC_LOGIC_H
#define NGRC_LOGIC_H

#include "model_config.h"

/**
 * @brief Perform NGRC prediction for a single timestep
 * 
 * This function computes the matrix-vector multiplication:
 * out_predicted_features = ngrc_trained_weights * current_expanded_x_vector
 * 
 * @param current_expanded_x_vector Pointer to the pre-expanded NGRC input vector
 *                                  Size: NGRC_EFFECTIVE_TERMS (861) floats
 * @param out_predicted_features    Pointer to output buffer for predicted features
 *                                  Size: NUM_PCA_FEATURES (20) floats
 */
void ngrc_predict(const float* current_expanded_x_vector, 
                  float* out_predicted_features);

/**
 * @brief Calculate Mean Squared Error between two feature vectors
 * 
 * @param predicted Pointer to predicted features (20 floats)
 * @param actual    Pointer to actual features (20 floats)
 * @return float    Mean squared error
 */
float calculate_mse(const float* predicted, const float* actual);

#endif // NGRC_LOGIC_H