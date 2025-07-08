// NGRC_Model/Inc/polynomial_expansion.h
#ifndef POLYNOMIAL_EXPANSION_H
#define POLYNOMIAL_EXPANSION_H

#include "model_config.h"

/**
 * @brief Extract features with time delays from the original feature buffer
 * 
 * @param original_features Pointer to full feature sequence [NUM_FRAMES][NUM_PCA_FEATURES]
 * @param timestep Current timestep (0 to NUM_VALID_TIMESTEPS_FOR_PREDICTION-1)
 * @param delayed_features Output buffer for linear terms [NUM_LINEAR_TERMS]
 *                         Layout: [feature_0_delay_1, ..., feature_19_delay_1, 
 *                                  feature_0_delay_2, ..., feature_19_delay_2]
 */
void extract_delayed_features(const float* original_features,
                             int timestep,
                             float* delayed_features);

/**
 * @brief Generate polynomial expansion from linear features
 * 
 * Creates all monomials up to degree 2, matching Python's all_terms() output
 * 
 * @param delayed_features Input linear terms [NUM_LINEAR_TERMS]
 * @param expanded_vector Output buffer [NGRC_EFFECTIVE_TERMS] including bias
 */
void generate_polynomial_expansion(const float* delayed_features,
                                  float* expanded_vector);

/**
 * @brief Complete NGRC feature expansion for one timestep
 * 
 * Convenience function that combines extraction and expansion
 * 
 * @param original_features Full feature sequence [NUM_FRAMES][NUM_PCA_FEATURES]
 * @param timestep Current timestep (0 to NUM_VALID_TIMESTEPS_FOR_PREDICTION-1)
 * @param expanded_vector Output buffer [NGRC_EFFECTIVE_TERMS]
 */
void ngrc_expand_features(const float* original_features,
                         int timestep,
                         float* expanded_vector);

#endif // POLYNOMIAL_EXPANSION_H