// NGRC_Model/Inc/model_config.h
#ifndef NGRC_MODEL_CONFIG_H
#define NGRC_MODEL_CONFIG_H

// Core NGRC dimensions
#define NUM_FRAMES 128          // Total frames in input sequence
#define NUM_PCA_FEATURES 20     // Number of PCA components

// NGRC polynomial expansion parameters
#define NUM_DELAYS 2            // Number of delay terms
#define DELAY_1 -1              // First delay value
#define DELAY_2 -6              // Second delay value  
#define MAX_ABS_DELAY 6         // Maximum absolute delay value
#define POLYNOMIAL_DEGREE 2     // Maximum polynomial degree

// NGRC expansion dimensions
#define NUM_LINEAR_TERMS 40     // 20 features × 2 delays
#define NUM_QUADRATIC_TERMS 820 // (40 choose 2) + 40 = 780 + 40 = 820
#define NUM_NGRC_INPUT_TERMS 860  // Linear + quadratic terms
#define NGRC_EFFECTIVE_TERMS 861  // Including bias term

// Output dimensions
#define NUM_NGRC_OUTPUTS NUM_PCA_FEATURES  // 20 outputs (predict all features)

// Valid timesteps for prediction (considering max delay)
#define NUM_VALID_TIMESTEPS_FOR_PREDICTION 121  // 128 - 6

// Windowed inference configuration
#define WINDOW_SIZE 7                     // Minimum timesteps for 1 prediction
#define WINDOW_OVERLAP 6                  // Overlap for temporal dependencies  
#define PREDICTIONS_PER_WINDOW 1          // Single prediction per inference (mathematically equivalent)

// Windowed data payload size
#define WINDOWED_SAMPLE_FLOAT_COUNT (WINDOW_SIZE * NUM_PCA_FEATURES)  // 140 floats
#define WINDOWED_PAYLOAD_SIZE_BYTES (WINDOWED_SAMPLE_FLOAT_COUNT * sizeof(float))  // 560 bytes

// Legacy full sample configuration (kept for reference)
#define ORIGINAL_SAMPLE_FLOAT_COUNT (NUM_FRAMES * NUM_PCA_FEATURES)  // 2,560 floats
#define PAYLOAD_SIZE_FLOATS WINDOWED_SAMPLE_FLOAT_COUNT              // Use windowed size
#define NGRC_PAYLOAD_SIZE_BYTES WINDOWED_PAYLOAD_SIZE_BYTES          // Use windowed size

// Buffer size for a single expanded vector (used during inference)
#define EXPANDED_VECTOR_SIZE NGRC_EFFECTIVE_TERMS  // 861 floats

// For MLPerf Tiny API
#define MAX_DB_INPUT_SIZE WINDOWED_PAYLOAD_SIZE_BYTES

// Model version string for MLPerf
#define TH_MODEL_VERSION EE_MODEL_VERSION_AD01

#endif // NGRC_MODEL_CONFIG_H