// model_config.h
#ifndef MODEL_CONFIG_H
#define MODEL_CONFIG_H

// FFT-based Ridge Classifier for Anomaly Detection
// MLPerf Tiny benchmark implementation

// FFT parameters
#define SAMPLING_RATE 16000           // Hz
#define F_MIN 300.0f                  // Minimum frequency to consider (Hz)
#define F_MAX 7900.0f                 // Maximum frequency to consider (Hz)

// FFT dimensions after frequency masking (300-7900 Hz)
#define NUM_FFT_FEATURES 103          // Input features to Ridge classifier
#define RIDGE_WEIGHTS_SIZE 103        // Weight vector size (103 features)
#define RIDGE_TOTAL_PARAMS 104        // Weights + bias (103 + 1)

// Single sample per inference
#define WINDOWED_SAMPLE_FLOAT_COUNT NUM_FFT_FEATURES
#define WINDOWED_PAYLOAD_SIZE_BYTES (WINDOWED_SAMPLE_FLOAT_COUNT * sizeof(float))

// MLPerf Tiny API compatibility
#define MAX_DB_INPUT_SIZE WINDOWED_PAYLOAD_SIZE_BYTES
#define PAYLOAD_SIZE_FLOATS WINDOWED_SAMPLE_FLOAT_COUNT
#define TH_MODEL_VERSION EE_MODEL_VERSION_AD01

#endif // MODEL_CONFIG_H