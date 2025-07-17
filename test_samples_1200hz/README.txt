Binary Test Files for MLPerf Tiny Anomaly Detection (1200Hz)
============================================================

Generated from: /Users/jaymain/MLPerf Tiny C++ MBED/FFT_ave_800_no_log
Total files: 2459
Features per sample: 91
File size: 364 bytes
Format: Little-endian float32
Frequency range: 1200.0-7900.0 Hz (after masking)
Alpha parameter: 10^-3 (more regularized model)

File naming convention:
  car_XX_sample_YYY_normal.bin   - Normal samples
  car_XX_sample_YYY_anomaly.bin  - Anomaly samples
  Where XX = car ID (01-04), YYY = sample index (000-349)

Cars 1-4 are the test vehicles
Normal samples: 350 per car
Anomaly samples: 264-265 per car
