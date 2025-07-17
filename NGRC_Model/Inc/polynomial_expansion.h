// polynomial_expansion.h
#ifndef POLYNOMIAL_EXPANSION_H
#define POLYNOMIAL_EXPANSION_H

#include "model_config.h"

/**
 * @brief Prepare FFT features for Ridge Classifier
 * 
 * Copies FFT features directly and adds bias term (no expansion needed)
 * 
 * @param fft_features Single FFT feature vector [NUM_FFT_FEATURES]
 * @param timestep Ignored (kept for compatibility)
 * @param features_with_bias Output buffer [RIDGE_TOTAL_PARAMS] = FFT features + bias
 */
void ngrc_expand_features(const float* fft_features,
                         int timestep,
                         float* features_with_bias);

#endif // POLYNOMIAL_EXPANSION_H