// ngrc_logic.h
#ifndef NGRC_LOGIC_H
#define NGRC_LOGIC_H

#include "model_config.h"

/**
 * @brief Perform Ridge Classifier prediction for FFT features
 * 
 * This function computes the Ridge classification score:
 * score = dot(ridge_weights, fft_features_with_bias)
 * 
 * @param fft_features_with_bias Pointer to FFT features with bias term appended
 *                               Size: RIDGE_TOTAL_PARAMS (NUM_FFT_FEATURES + 1) floats
 * @param out_classification_score Pointer to output buffer for classification score
 *                                 Size: 1 float
 */
void ngrc_predict(const float* fft_features_with_bias, 
                  float* out_classification_score);

/**
 * @brief Direct Ridge classification function
 * 
 * @param fft_features_with_bias Pointer to FFT features with bias term
 * @return float Classification score (continuous value for AUC calculation)
 */
float ridge_classify(const float* fft_features_with_bias);

#endif // NGRC_LOGIC_H