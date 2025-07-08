// main.cpp
#include "mbed.h"
#include "api/internally_implemented.h"

// External declaration for the serial callback
extern void ee_serial_callback(char c);

// External function from submitter_implemented
extern char th_getchar();

int main() {
    // Initialize the benchmark
    ee_benchmark_initialize();
    
    // Main loop - poll for serial input and feed to the framework
    while (1) {
        // Get character (blocking)
        char c = th_getchar();
        
        // Feed it to the framework's serial callback
        ee_serial_callback(c);
    }
    
    return 0;
}