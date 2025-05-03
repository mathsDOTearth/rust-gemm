# rust-gemm
A GEMM like set of single (f32) and double (f64) precision functions that use CPU vector instructions including AVX2 AVX512 and NEON.  There is additional RVV 1.0 support via a C function.

The nightly build of rustc must be used to get support for AVX512.

Sadly I was struggling for Rust native RVV support so have added this support via C

To run the code:
```  
rustup override set nightly  
RUSTFLAGS="-C target-cpu=native" cargo run --release  
```

Output from a 4 core ARM N1 NEON CPU:  
```
f64 scalar: 1.30s
f64 parallel: 343.25ms
Speedup: 3.79
✔ f64 SIMD‑parallel matches reference (tol=0.00000001)
f64 vector: 862.37ms
Speedup: 1.51
✔ f64 SIMD‑parallel matches reference (tol=0.00000001)
f64 vector parallel: 125.90ms
Speedup: 10.33
✔ f64 SIMD‑parallel matches reference (tol=0.00000001)
 
f32 scalar: 1.27s
f32 parallel: 319.94ms
Speedup: 3.98
✔ f32 SIMD‑parallel matches reference (tol=0.001)
f32 vector: 436.38ms
Speedup: 2.92
✔ f32 SIMD‑parallel matches reference (tol=0.001)
f32 vector parallel: 55.48ms
Speedup: 22.95
✔ f32 SIMD‑parallel matches reference (tol=0.001)
 
Threads: 4
Running with NEON
```
Output from a 8 core RISC-V Spacemit K1 RVV 1.0
```
f64 scalar: 8.64s
f64 parallel: 1.18s
Speedup: 7.30
✔ f64 SIMD‑parallel matches reference (tol=0.00000001)
f64 vector: 3.80s
Speedup: 2.27
✔ f64 SIMD‑parallel matches reference (tol=0.00000001)
f64 vector parallel: 1.02s
Speedup: 8.46
✔ f64 SIMD‑parallel matches reference (tol=0.00000001)
 
f32 scalar: 7.09s
f32 parallel: 924.44ms
Speedup: 7.67
✔ f32 SIMD‑parallel matches reference (tol=0.001)
f32 vector: 2.67s
Speedup: 2.66
✔ f32 SIMD‑parallel matches reference (tol=0.001)
f32 vector parallel: 453.77ms
Speedup: 15.63
✔ f32 SIMD‑parallel matches reference (tol=0.001)
 
Threads: 8
Running with RVV 1.0
```
Output from a 16 thread AMD 5700x AVX2 CPU:
```
f64 scalar: 698.75ms
f64 parallel: 78.39ms
Speedup: 8.91
✔ f64 SIMD‑parallel matches reference (tol=0.00000001)
f64 vector: 217.75ms
Speedup: 3.21
✔ f64 SIMD‑parallel matches reference (tol=0.00000001)
f64 vector parallel: 23.60ms
Speedup: 29.60
✔ f64 SIMD‑parallel matches reference (tol=0.00000001)
 
f32 scalar: 698.18ms
f32 parallel: 82.54ms
Speedup: 8.46
✔ f32 SIMD‑parallel matches reference (tol=0.001)
f32 vector: 95.68ms
Speedup: 7.30
✔ f32 SIMD‑parallel matches reference (tol=0.001)
f32 vector parallel: 12.30ms
Speedup: 56.77
✔ f32 SIMD‑parallel matches reference (tol=0.001)
 
Threads: 16
Running with AVX2
```
Output from a 52 thread Xeon Platinum 8170 AVX512 CPU:
```
f64 scalar: 1.19s
f64 parallel: 70.87ms
Speedup: 16.73
✔ f64 SIMD‑parallel matches reference (tol=0.00000001)
f64 vector: 339.51ms
Speedup: 3.49
✔ f64 SIMD‑parallel matches reference (tol=0.00000001)
f64 vector parallel: 17.47ms
Speedup: 67.86
✔ f64 SIMD‑parallel matches reference (tol=0.00000001)
 
f32 scalar: 1.20s
f32 parallel: 74.75ms
Speedup: 15.99
✔ f32 SIMD‑parallel matches reference (tol=0.001)
f32 vector: 168.33ms
Speedup: 7.10
✔ f32 SIMD‑parallel matches reference (tol=0.001)
f32 vector parallel: 7.31ms
Speedup: 163.37
✔ f32 SIMD‑parallel matches reference (tol=0.001)
 
Threads: 52
Runnung with AVX512
```
