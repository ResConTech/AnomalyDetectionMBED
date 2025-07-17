// submitter_implemented.cpp
#include "PinNames.h"
#include "mbed.h"
#include "api/submitter_implemented.h"
#include "api/internally_implemented.h"
#include "model_config.h"
#include "ngrc_logic.h"
#include "polynomial_expansion.h"

// Mbed OS objects
static mbed::Timer timer;
static mbed::DigitalOut timestampPin(D7);

// Conditional serial port based on energy mode
#if EE_CFG_ENERGY_MODE == 1
// Energy mode: Use D0/D1 for external serial connection
static mbed::UnbufferedSerial pc(D1, D0);
#else
// Performance mode: Use USB serial
static mbed::UnbufferedSerial pc(USBTX, USBRX);
#endif

// The console redirect works for both
FileHandle *mbed::mbed_override_console(int fd)
{
    return &pc;
}

// FFT-based Ridge Classifier data management
static float data_payload_buffer[WINDOWED_SAMPLE_FLOAT_COUNT]; // Single FFT sample
static const float *original_features_ptr = nullptr;
static const float *expanded_vectors_ptr = nullptr; // Deprecated - not used
static float current_anomaly_score = 0.0f;

// Temporary buffer for FFT features with bias term
static float expanded_vector_buffer[RIDGE_TOTAL_PARAMS];

// Energy mode configuration
#if EE_CFG_ENERGY_MODE == 1
#define BAUD_RATE 9600
#else
#define BAUD_RATE 115200 // More reliable than 921600
#endif

void th_serialport_initialize(void)
{
    pc.baud(BAUD_RATE);
}

void th_timestamp_initialize(void)
{
    timer.start();
    timestampPin = 0; // Start with pin low

    // Print timestamp mode
    th_printf(EE_MSG_TIMESTAMP_MODE);
}

void th_timestamp(void)
{
    unsigned long microseconds = timer.elapsed_time().count();
    th_printf(EE_MSG_TIMESTAMP, microseconds);

    // Toggle timestamp pin for energy measurement
    timestampPin = !timestampPin;
}

void th_printf(const char *fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    char buffer[256];
    vsnprintf(buffer, sizeof(buffer), fmt, args);
    pc.write(buffer, strlen(buffer)); // Write directly to serial
    va_end(args);
}

char th_getchar()
{
    return getchar();
}

void th_command_ready(char volatile *p_command)
{
    p_command = p_command; // Unused
    ee_serial_command_parser_callback((char *)p_command);
}

void th_final_initialize(void)
{
    th_serialport_initialize();
    th_timestamp_initialize();

    // Now we should be ready for clean communication
}

void th_load_tensor(void)
{
    // Load single FFT sample (NUM_FFT_FEATURES floats)
    size_t expected_bytes = WINDOWED_SAMPLE_FLOAT_COUNT * sizeof(float);
    size_t bytes = ee_get_buffer(reinterpret_cast<uint8_t *>(data_payload_buffer),
                                 expected_bytes);

    if (bytes != expected_bytes)
    {
        th_printf("e-[Invalid payload size: expected %d, got %d]\r\n",
                  expected_bytes, bytes);
        original_features_ptr = nullptr;
        return;
    }

    original_features_ptr = data_payload_buffer;
}

void th_infer(void)
{
    // FFT-based Ridge Classifier inference
    // Single FFT sample per window (Option A)
    
    if (original_features_ptr == nullptr) {
        current_anomaly_score = 0.0f;
        return;
    }

    // Prepare FFT features with bias term
    // original_features_ptr points to a single FFT vector (NUM_FFT_FEATURES floats)
    ngrc_expand_features(original_features_ptr, 0, expanded_vector_buffer);
    
    // Perform Ridge classification
    float classification_score_buffer[1];
    ngrc_predict(expanded_vector_buffer, classification_score_buffer);
    
    // Store classification score as anomaly score
    current_anomaly_score = classification_score_buffer[0];
}

void th_results(void)
{

    th_printf("m-results-[%0.3f]\r\n", current_anomaly_score);
}

// Optional functions (empty implementations)
void th_pre() {}
void th_post() {}

// Stub implementations for unused libc functions
int th_strncmp(const char *str1, const char *str2, size_t n)
{
    return strncmp(str1, str2, n);
}

char *th_strncpy(char *dest, const char *src, size_t n)
{
    return strncpy(dest, src, n);
}

size_t th_strnlen(const char *str, size_t maxlen)
{
    size_t len = 0;
    while (len < maxlen && str[len] != '\0')
    {
        len++;
    }
    return len;
}

char *th_strcat(char *dest, const char *src)
{
    return strcat(dest, src);
}

char *th_strtok(char *str, const char *sep)
{
    return strtok(str, sep);
}

int th_atoi(const char *str)
{
    return atoi(str);
}

void *th_memset(void *b, int c, size_t len)
{
    return memset(b, c, len);
}

void *th_memcpy(void *dst, const void *src, size_t n)
{
    return memcpy(dst, src, n);
}

int th_vprintf(const char *format, va_list ap)
{
    return vprintf(format, ap);
}
