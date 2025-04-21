# rust-gemm
A GEMM like set of single (f32) and double (f64) precision functions that use CPU vector instructions including AVX2 AVX512 and NEON.

The nightly build of rustc must be used to get support for AVX512.

To run the code:  
rustup override set nightly  
RUSTFLAGS="-C target-cpu=native" cargo run --release  

Output from a 4 core ARM N1 NEON CPU:  
```
f64 scalar: 1.30s
f64 parallel: 332.13ms
Speedup: 3.92
f64 vector: 870.29ms
Speedup: 1.50
f64 vector parallel: 123.81ms
Speedup: 10.52
 
f32 scalar: 1.29s
f32 parallel: 321.12ms
Speedup: 4.01
f32 vector: 439.63ms
Speedup: 2.93
f32 vector parallel: 52.15ms
Speedup: 24.67
Threads: 4
Running with NEON
```
Output from a 16 thread AMD 5700x AVX2 CPU:
```
f64 scalar: 698.57ms
f64 parallel: 86.11ms
Speedup: 8.11
f64 vector: 210.74ms
Speedup: 3.31
f64 vector parallel: 27.28ms
Speedup: 25.61
 
f32 scalar: 702.37ms
f32 parallel: 81.28ms
Speedup: 8.64
f32 vector: 93.65ms
Speedup: 7.50
f32 vector parallel: 13.53ms
Speedup: 51.90
Threads: 16
Running with AVX2
```
Output from a 52 thread Xeon Platinum 8170 AVX512 CPU:
```
f64 scalar: 1.19s
f64 parallel: 83.54ms
Speedup: 14.19
f64 vector: 350.67ms
Speedup: 3.38
f64 vector parallel: 17.12ms
Speedup: 69.24
 
f32 scalar: 1.17s
f32 parallel: 65.47ms
Speedup: 17.92
f32 vector: 169.73ms
Speedup: 6.91
f32 vector parallel: 7.21ms
Speedup: 162.65
Threads: 52
Runnung with AVX512
```
