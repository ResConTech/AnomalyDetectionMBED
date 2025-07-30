# MLPerf Tiny NGRC Anomaly Detection Benchmark
Implementation of Next Generation Reservoir Computing (NGRC) model for MLPerf Tiny benchmarking suite on MBED-compatible microcontrollers.
Built for Nucleo L4R5ZI MCU. 

## Project Structure
```
MLPerf Tiny C++ MBED/
├── main.cpp
├── submitter_implemented.cpp
├── mbed_app.json
├── README.md
├── api/
│   ├── internally_implemented.cpp
│   ├── internally_implemented.h
│   └── submitter_implemented.h
└── NGRC_Model/
    ├── Inc/
    │   ├── model_config.h
    │   ├── ngrc_logic.h
    │   ├── ngrc_weights.h
    │   └── polynomial_expansion.h
    └── Src/
        ├── ngrc_logic.cpp
        ├── ngrc_weights.cpp
        └── polynomial_expansion.cpp
```

## main.cpp
Entry point that initializes the benchmark framework and enters infinite polling loop for serial communication.
Calls `ee_benchmark_initialize()` to setup MLPerf framework, then continuously polls for characters via `th_getchar()`.
Each received character is fed to `ee_serial_callback()` for command assembly and processing.
Implements blocking I/O model where main thread waits for serial input and dispatches to framework.
Simple architecture that bridges hardware serial interface with MLPerf command processing system.
No local state management - all command parsing and execution handled by framework layers.
Critical for establishing host-device communication channel required for benchmark execution.
Serves as minimal hardware abstraction layer between MBED serial drivers and MLPerf protocol.

## submitter_implemented.cpp
Core implementation of MLPerf API functions and NGRC inference logic for anomaly detection.
Manages dual-mode operation: energy measurement mode (9600 baud, D0/D1 pins) and performance mode (115200 baud, USB).
Implements `th_load_tensor()` to receive 2560 floats (128 frames × 20 PCA features) from host via buffer system.
Executes NGRC inference in `th_infer()` by processing 121 valid timesteps with polynomial expansion and prediction.
Calculates anomaly score as mean squared error between predicted and actual features across all timesteps.
Provides hardware timing via `th_timestamp()` with microsecond precision using MBED Timer and GPIO toggle.
Handles all MLPerf-required I/O functions including printf, getchar, and string manipulation wrappers.
Contains static buffers for input data (2560 floats) and single expanded vector (861 floats) to minimize memory usage.

## api/internally_implemented.cpp
MLPerf Tiny benchmark framework implementing standardized communication protocol and execution flow.
Provides command parser that handles `db load`, `db HEXDATA`, `infer N W`, `results`, and utility commands.
Manages global input buffer (10KB) for receiving hex-encoded test data from host system.
Implements `ee_infer()` benchmark orchestration with warmup cycles, timing measurement, and result collection.
Contains `ee_serial_callback()` ISR handler that assembles commands from character stream with '%' termination.
Provides `ee_buffer_parse()` for data loading operations including hex-to-binary conversion and buffer management.
Implements standardized response format with `m-` (message) and `e-` (error) prefixes for host parsing.
Serves as abstraction layer between hardware-specific submitter code and MLPerf benchmark requirements.

## api/internally_implemented.h
Header file defining MLPerf Tiny API constants, message formats, and function declarations.
Establishes communication protocol with command definitions (`EE_CMD_NAME`, `EE_CMD_TIMESTAMP`) and delimiters.
Defines standardized message formats for benchmark responses including timing, initialization, and error messages.
Declares core framework functions for command parsing, buffer management, and benchmark execution.
Contains version information and device identification constants required by MLPerf specification.
Provides enum definitions for argument parsing states and status codes used throughout framework.
Establishes buffer size limits and communication parameters for reliable host-device interaction.
Critical interface file that ensures compatibility between framework implementation and submitter code.

## api/submitter_implemented.h
Header defining submitter-implemented functions required by MLPerf Tiny API specification.
Declares core inference functions: `th_load_tensor()`, `th_infer()`, `th_results()` that must be implemented.
Defines platform abstraction functions for timing (`th_timestamp()`), I/O (`th_printf()`, `th_getchar()`), and initialization.
Contains energy mode configuration macros and message format definitions for benchmark communication.
Includes libc wrapper function declarations for string manipulation and memory operations.
Establishes model version definition (`TH_MODEL_VERSION`) and vendor identification strings.
Provides conditional compilation support for energy vs performance measurement modes.
Serves as contract between MLPerf framework and hardware-specific implementation requirements.

## NGRC_Model/Src/ngrc_logic.cpp
Core NGRC prediction algorithm implementing matrix-vector multiplication for feature forecasting.
Function `ngrc_predict()` performs dot product between pre-trained weights (20×861) and expanded input vector (861 terms).
Implements optimized nested loop structure: outer loop iterates over 20 output features, inner loop over 861 input terms.
Accumulates sum of products for each feature prediction using floating-point arithmetic.
Contains `calculate_mse()` function that computes mean squared error between predicted and actual feature vectors.
MSE calculation loops through 20 features, squares differences, and returns normalized error metric.
Critical performance component that executes 121 times per inference (once per valid timestep).
Represents the mathematical core of NGRC anomaly detection where prediction accuracy determines anomaly scores.

## NGRC_Model/Src/polynomial_expansion.cpp
Implements polynomial feature expansion from time-delayed input features to NGRC input space.
Function `extract_delayed_features()` extracts features from timesteps t-1 and t-6 based on current timestep.
Handles temporal indexing with `MAX_ABS_DELAY` offset to ensure valid historical data access.
Function `generate_polynomial_expansion()` creates 861-term vector: 40 linear + 820 quadratic + 1 bias term.
Quadratic terms generated using combinations with replacement algorithm matching Python's sklearn implementation.
Function `ngrc_expand_features()` combines delay extraction and polynomial expansion for single timestep.
Uses temporary buffer for delayed features (40 floats) to minimize memory allocation overhead.
Critical preprocessing step that transforms raw PCA features into high-dimensional NGRC input space.

## NGRC_Model/Src/ngrc_weights.cpp
Contains pre-trained NGRC weight matrix as constant data array for inference operations.
Stores 20×861 floating-point weight matrix representing learned relationships between expanded features and outputs.
Matrix includes bias terms in final column, eliminating need for separate bias vector storage.
Weights trained offline using historical data to predict future PCA features from polynomial-expanded inputs.
Large data file (approximately 69KB) containing model parameters essential for NGRC functionality.
Memory-efficient storage using const qualifier and 2D array structure for direct matrix access.
Critical model component that determines prediction accuracy and anomaly detection performance.
Represents the learned knowledge of the NGRC system encoded as numerical coefficients.

## NGRC_Model/Inc/model_config.h
Defines all model dimensions, parameters, and configuration constants for NGRC implementation.
Establishes core dimensions: 128 input frames, 20 PCA features, 2 time delays (-1, -6 timesteps).
Calculates derived dimensions: 40 linear terms, 820 quadratic terms, 861 total expanded terms.
Defines buffer sizes for input data (10KB) and model processing requirements.
Contains MLPerf integration constants including model version identifier and payload specifications.
Establishes valid timestep range (121 steps) accounting for maximum delay constraints.
Provides single source of truth for all dimensional parameters used across model implementation.
Critical configuration file that ensures consistency between data structures and algorithm implementations.

## mbed_app.json
MBED OS configuration file specifying platform settings, memory allocation, and communication parameters.
Configures main stack size to 64KB to accommodate model processing and buffer requirements.
Sets serial communication baud rate to 9600 for consistent energy measurement across different hardware.
Disables newline conversion to maintain binary data integrity during hex data transmission.
Enables minimal floating-point printf support required for result formatting and debug output.
Specifies bare-metal profile for minimal OS overhead and deterministic performance characteristics.
Configures platform-specific serial drivers and memory management for embedded execution.
Essential configuration file that ensures proper hardware initialization and resource allocation for benchmark execution.
For H7A3ZI-Q benchmarking, remove the stack size portion of the json file. Also remember to press the reset button on the MCU before benchmarking in performance mode.

## python_model_files
Python development environment containing Ridge classifier training pipeline and data preprocessing utilities for NGRC anomaly detection.
`NGRCAnom-FFT-32bit.py` implements FFT-based anomaly detection using Ridge classifier with frequency domain filtering (1200-7900 Hz).
Trains on cars 1-4 (normal) vs cars 5-7 (anomaly + normal) and cars 1-4 (anomaly) using 32-bit float precision for embedded compatibility.
`extract_timeseries_save_FFT_no_log.py` processes raw audio files into frequency domain representations with configurable bin averaging.
Extracts FFT magnitude spectra from 16kHz audio, applies frequency averaging, and saves preprocessed data as NumPy arrays.
`convert_weights_to_cpp.py` converts trained Ridge classifier weights and bias terms into C++ constant arrays.
Generates `ridge_weights_generated.cpp` with properly formatted float arrays and configuration updates for embedded deployment.
Essential development toolkit for training, validating, and deploying the anomaly detection model to embedded hardware.

## binary_test_file_creation
Binary test data generation utilities for creating MLPerf Tiny benchmark datasets with proper frequency filtering and formatting.
`generate_binary_test_files_corrected.py` processes FFT data into individual .bin files using 300-7900 Hz frequency range (103 features).
`generate_binary_test_files_1200hz.py` creates alternative dataset with 1200-7900 Hz range (91 features) for model comparison.
Converts NumPy arrays to little-endian float32 binary format compatible with STM32 microcontroller memory layout.
`create_full_dataset_csv.py` generates complete CSV manifest files for 2,459 samples with class labels and windowing parameters.
`create_debug_dataset.py` creates reduced test sets (52 samples total) for rapid development and debugging workflows.
Both generation scripts create structured filenames (car_XX_sample_YYY_label.bin) and comprehensive README documentation.
Critical preprocessing pipeline that bridges Python model development with embedded C++ inference testing framework.

## Energy vs Peformance Mode 
To switch between energy (1) and peformance mode (0) setup for the benchmark, you need to change line 44 in submitter_implemented.h, and change the baud rate in mbed_app.json to 9600 (lines 9 & 10). Changing the variable to 1 and building will enable energy mode measurements and changing to 0 will enable performance mode measurements. Every change in benchmarking mode will require you to 
rebuild and reflash.

## mbed setup 
To use and flash this build to a MCU, you must follow these steps. 
1. Download Mbed Studio 
2. Create a template project
3. Select your MCU in Mbed Studio 
4. Select Mbed OS version 6.7.0
5. Copy these folders and their contents to the mbed studio project
6. Select build profile 'Release'
7. Click clean build 
8. Once build is successful, and MCU is connected to your PC, click the play button and flash to the board. 
