// NGRC_Model/Inc/ngrc_weights.h
#ifndef NGRC_WEIGHTS_H
#define NGRC_WEIGHTS_H

#include "model_config.h"

// NGRC trained weights matrix
// Dimensions: [NUM_PCA_FEATURES][NGRC_EFFECTIVE_TERMS] = [20][861]
// This matrix includes the bias term in the last column
extern const float ngrc_trained_weights[NUM_PCA_FEATURES][NGRC_EFFECTIVE_TERMS];

#endif // NGRC_WEIGHTS_H