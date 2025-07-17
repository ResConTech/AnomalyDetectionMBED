// ngrc_weights.h
#ifndef NGRC_WEIGHTS_H
#define NGRC_WEIGHTS_H

#include "model_config.h"

// Ridge Classifier weights vector
// Dimensions: [RIDGE_TOTAL_PARAMS] = [NUM_FFT_FEATURES + 1]
// The last element is the bias term
extern const float ridge_weights[RIDGE_TOTAL_PARAMS];

#endif // NGRC_WEIGHTS_H