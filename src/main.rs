//! GEMM with scalar, AVX2/NEON, and AVX512 support for f64/f32
//! Written By Rich from mathsDOTearth 2025
#![cfg_attr(target_arch = "x86_64", feature(avx512_target_feature))]
#![cfg_attr(target_arch = "x86_64", feature(stdarch_x86_avx512))]

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
use rayon::prelude::*;
use rayon::current_num_threads;
use unirand::MarsagliaUniRng;
use std::time::Instant;

// SIMD abstraction
pub unsafe trait SimdElem: Copy + Sized {
    type Scalar: Copy + Send + Sync + std::ops::Add<Output = Self::Scalar> + std::ops::Mul<Output = Self::Scalar>;
    type Reg;
    const LANES: usize;
    unsafe fn zero() -> Self::Reg;
    unsafe fn load(ptr: *const Self::Scalar) -> Self::Reg;
    unsafe fn store(ptr: *mut Self::Scalar, v: Self::Reg);
    unsafe fn fmadd(acc: Self::Reg, a: Self::Reg, b: Self::Reg) -> Self::Reg;
    unsafe fn reduce(v: Self::Reg) -> Self::Scalar;
}

// x86_64 AVX512 f64
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
unsafe impl SimdElem for f64 {
    type Scalar = f64;
    type Reg    = __m512d;
    const LANES : usize = 8;
    #[inline(always)] unsafe fn zero() -> __m512d { _mm512_setzero_pd() }
    #[inline(always)] unsafe fn load(ptr: *const f64) -> __m512d { _mm512_loadu_pd(ptr) }
    #[inline(always)] unsafe fn store(ptr: *mut f64, v: __m512d) { _mm512_storeu_pd(ptr, v) }
    #[inline(always)] unsafe fn fmadd(acc: __m512d, a: __m512d, b: __m512d) -> __m512d {
        _mm512_fmadd_pd(a, b, acc)
    }
    #[inline(always)] unsafe fn reduce(v: __m512d) -> f64 {
        let mut buf = [0f64; 8];
        _mm512_storeu_pd(buf.as_mut_ptr(), v);
        buf.iter().copied().sum()
    }
}
// x86_64 AVX512 f32
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
unsafe impl SimdElem for f32 {
    type Scalar = f32;
    type Reg    = __m512;
    const LANES : usize = 16;
    #[inline(always)] unsafe fn zero() -> __m512 { _mm512_setzero_ps() }
    #[inline(always)] unsafe fn load(ptr: *const f32) -> __m512 { _mm512_loadu_ps(ptr) }
    #[inline(always)] unsafe fn store(ptr: *mut f32, v: __m512) { _mm512_storeu_ps(ptr, v) }
    #[inline(always)] unsafe fn fmadd(acc: __m512, a: __m512, b: __m512) -> __m512 {
        _mm512_fmadd_ps(a, b, acc)
    }
    #[inline(always)] unsafe fn reduce(v: __m512) -> f32 {
        let mut buf = [0f32; 16];
        _mm512_storeu_ps(buf.as_mut_ptr(), v);
        buf.iter().copied().sum()
    }
}

// x86_64 AVX2 f64
#[cfg(all(target_arch = "x86_64", not(target_feature = "avx512f")))]
unsafe impl SimdElem for f64 {
    type Scalar = f64;
    type Reg    = __m256d;
    const LANES : usize = 4;
    #[inline(always)] unsafe fn zero() -> __m256d { _mm256_setzero_pd() }
    #[inline(always)] unsafe fn load(ptr: *const f64) -> __m256d { _mm256_loadu_pd(ptr) }
    #[inline(always)] unsafe fn store(ptr: *mut f64, v: __m256d) { _mm256_storeu_pd(ptr, v) }
    #[inline(always)] unsafe fn fmadd(acc: __m256d, a: __m256d, b: __m256d) -> __m256d {
        _mm256_fmadd_pd(a, b, acc)
    }
    #[inline(always)] unsafe fn reduce(v: __m256d) -> f64 {
        let tmp = _mm256_hadd_pd(v, v);
        let arr: [f64; 4] = std::mem::transmute(tmp);
        arr[0] + arr[2]
    }
}
// x86_64 AVX2 f32
#[cfg(all(target_arch = "x86_64", not(target_feature = "avx512f")))]
unsafe impl SimdElem for f32 {
    type Scalar = f32;
    type Reg    = __m256;
    const LANES : usize = 8;
    #[inline(always)] unsafe fn zero() -> __m256 { _mm256_setzero_ps() }
    #[inline(always)] unsafe fn load(ptr: *const f32) -> __m256 { _mm256_loadu_ps(ptr) }
    #[inline(always)] unsafe fn store(ptr: *mut f32, v: __m256) { _mm256_storeu_ps(ptr, v) }
    #[inline(always)] unsafe fn fmadd(acc: __m256, a: __m256, b: __m256) -> __m256 {
        _mm256_fmadd_ps(a, b, acc)
    }
    #[inline(always)] unsafe fn reduce(v: __m256) -> f32 {
        let tmp1 = _mm256_hadd_ps(v, v);
        let tmp2 = _mm256_hadd_ps(tmp1, tmp1);
        _mm_cvtss_f32(_mm256_castps256_ps128(tmp2))
    }
}

// aarch64 NEON f64
#[cfg(target_arch = "aarch64")]
unsafe impl SimdElem for f64 {
    type Scalar = f64;
    type Reg    = float64x2_t;
    const LANES : usize = 2;
    #[inline(always)] unsafe fn zero() -> float64x2_t { vdupq_n_f64(0.0) }
    #[inline(always)] unsafe fn load(ptr: *const f64) -> float64x2_t { vld1q_f64(ptr) }
    #[inline(always)] unsafe fn store(ptr: *mut f64, v: float64x2_t) { vst1q_f64(ptr, v) }
    #[inline(always)] unsafe fn fmadd(acc: float64x2_t, a: float64x2_t, b: float64x2_t) -> float64x2_t {
        vfmaq_f64(acc, a, b)
    }
    #[inline(always)] unsafe fn reduce(v: float64x2_t) -> f64 {
        let arr: [f64;2] = std::mem::transmute(v);
        arr[0] + arr[1]
    }
}

// aarch64 NEON f32
#[cfg(target_arch = "aarch64")]
unsafe impl SimdElem for f32 {
    type Scalar = f32;
    type Reg    = float32x4_t;
    const LANES : usize = 4;
    #[inline(always)] unsafe fn zero() -> float32x4_t { vdupq_n_f32(0.0) }
    #[inline(always)] unsafe fn load(ptr: *const f32) -> float32x4_t { vld1q_f32(ptr) }
    #[inline(always)] unsafe fn store(ptr: *mut f32, v: float32x4_t) { vst1q_f32(ptr, v) }
    #[inline(always)] unsafe fn fmadd(acc: float32x4_t, a: float32x4_t, b: float32x4_t) -> float32x4_t {
        vfmaq_f32(acc, a, b)
    }
    #[inline(always)] unsafe fn reduce(v: float32x4_t) -> f32 {
        let arr: [f32;4] = std::mem::transmute(v);
        arr.iter().copied().sum()
    }
}

/// SIMD dot-product
pub unsafe fn dot_generic<E: SimdElem>(a: &[E::Scalar], b: &[E::Scalar]) -> E::Scalar {
    let mut i = 0; let len = a.len(); let mut acc = E::zero();
    while i + E::LANES <= len {
        let va = E::load(a.as_ptr().add(i));
        let vb = E::load(b.as_ptr().add(i));
        acc = E::fmadd(acc, va, vb);
        i += E::LANES;
    }
    let mut total = E::reduce(acc);
    while i < len { total = total + (a[i] * b[i]); i += 1; }
    total
}

/// Transpose
pub fn transpose<T: Copy + Default>(m: &[Vec<T>]) -> Vec<Vec<T>> {
    let r = m.len(); let c = m[0].len(); let mut t = vec![vec![T::default(); r]; c];
    for i in 0..r { for j in 0..c { t[j][i] = m[i][j]; }}
    t
}

// Scalar GEMM f64/f32
pub fn gemm_scalar_f64(alpha: f64, a: &[Vec<f64>], b_t: &[Vec<f64>], beta: f64, c: &mut [Vec<f64>], parallel: bool) {
    let rows = a.len(); let cols = b_t.len();
    if parallel {
        c.par_iter_mut().enumerate().for_each(|(i, row_c)| {
            for j in 0..cols { let mut sum=0f64; for k in 0..a[0].len() { sum+=a[i][k]*b_t[j][k]; } row_c[j]=alpha*sum+beta*row_c[j]; }
        });
    } else {
        for i in 0..rows { for j in 0..cols { let mut sum=0f64; for k in 0..a[0].len() { sum+=a[i][k]*b_t[j][k]; } c[i][j]=alpha*sum+beta*c[i][j]; }}
    }
}

pub fn gemm_scalar_f32(alpha: f32, a: &[Vec<f32>], b_t: &[Vec<f32>], beta: f32, c: &mut [Vec<f32>], parallel: bool) {
    let rows = a.len();
    let cols = b_t.len();
    if parallel {
        c.par_iter_mut().enumerate().for_each(|(i, row_c)| {
            for j in 0..cols {
                let mut sum = 0f32;
                for k in 0..a[0].len() {
                    sum += a[i][k] * b_t[j][k];
                }
                row_c[j] = alpha * sum + beta * row_c[j];
            }
        });
    } else {
        for i in 0..rows {
            for j in 0..cols {
                let mut sum = 0f32;
                for k in 0..a[0].len() {
                    sum += a[i][k] * b_t[j][k];
                }
                c[i][j] = alpha * sum + beta * c[i][j];
            }
        }
    }
}

/// Convenience wrappers for SIMD
pub fn gemm_f64(alpha: f64, a: &[Vec<f64>], b_t: &[Vec<f64>], beta: f64, c: &mut [Vec< f64>], parallel: bool) {
    if parallel {
        c.par_iter_mut().enumerate().for_each(|(i, row_c)| {
            for j in 0..b_t.len() {
                let sum = unsafe { dot_generic::<f64>(&a[i], &b_t[j]) };
                row_c[j] = alpha * sum + beta * row_c[j];
            }
        });
    } else {
        for i in 0..a.len() {
            for j in 0..b_t.len() {
                let sum = unsafe { dot_generic::<f64>(&a[i], &b_t[j]) };
                c[i][j] = alpha * sum + beta * c[i][j];
            }
        }
    }
}

pub fn gemm_f32(alpha: f32, a: &[Vec<f32>], b_t: &[Vec<f32>], beta: f32, c: &mut [Vec<f32>], parallel: bool) {
    if parallel {
        c.par_iter_mut().enumerate().for_each(|(i, row_c)| {
            for j in 0..b_t.len() {
                let sum = unsafe { dot_generic::<f32>(&a[i], &b_t[j]) };
                row_c[j] = alpha * sum + beta * row_c[j];
            }
        });
    } else {
        for i in 0..a.len() {
            for j in 0..b_t.len() {
                let sum = unsafe { dot_generic::<f32>(&a[i], &b_t[j]) };
                c[i][j] = alpha * sum + beta * c[i][j];
            }
        }
    }
}

// ----------------------
// main with full benchmarking
// ----------------------
fn main() {
    const N: usize = 1024;
    let mut rng = MarsagliaUniRng::new(); rng.rinit(170);

    // f64 data
    let a64: Vec<Vec<f64>> = (0..N).map(|_| (0..N).map(|_| rng.uni() as f64).collect()).collect();
    let bt64 = transpose(&a64);
    let mut c64 = vec![vec![0f64; N]; N];

    let t0 = Instant::now(); gemm_scalar_f64(1.0, &a64, &bt64, 0.0, &mut c64, false); let f64_scalar = t0.elapsed();
    println!("f64 scalar: {:.2?}", f64_scalar);

    let t1 = Instant::now(); gemm_scalar_f64(1.0, &a64, &bt64, 0.0, &mut c64, true); let f64_par = t1.elapsed();
    println!("f64 parallel: {:.2?}", f64_par);
    println!("Speedup: {:.2}", f64_scalar.as_secs_f64() / f64_par.as_secs_f64());

    let t2 = Instant::now(); gemm_f64(1.0, &a64, &bt64, 0.0, &mut c64, false); let f64_vec = t2.elapsed();
    println!("f64 vector: {:.2?}", f64_vec);
    println!("Speedup: {:.2}", f64_scalar.as_secs_f64() / f64_vec.as_secs_f64());

    let t3 = Instant::now(); gemm_f64(1.0, &a64, &bt64, 0.0, &mut c64, true); let f64_vec_par = t3.elapsed();
    println!("f64 vector parallel: {:.2?}", f64_vec_par);
    println!("Speedup: {:.2}", f64_scalar.as_secs_f64() / f64_vec_par.as_secs_f64());

    println!(" ");
  
    // f32 data
    let a32: Vec<Vec<f32>> = (0..N).map(|_| (0..N).map(|_| rng.uni() as f32).collect()).collect();
    let bt32 = transpose(&a32);
    let mut c32 = vec![vec![0f32; N]; N];

    let t4 = Instant::now(); gemm_scalar_f32(1.0, &a32, &bt32, 0.0, &mut c32, false); let f32_scalar = t4.elapsed();
    println!("f32 scalar: {:.2?}", f32_scalar);

    let t5 = Instant::now(); gemm_scalar_f32(1.0, &a32, &bt32, 0.0, &mut c32, true); let f32_par = t5.elapsed();
    println!("f32 parallel: {:.2?}", f32_par);
    println!("Speedup: {:.2}", f32_scalar.as_secs_f64() / f32_par.as_secs_f64());

    let t6 = Instant::now(); gemm_f32(1.0, &a32, &bt32, 0.0, &mut c32, false); let f32_vec = t6.elapsed();
    println!("f32 vector: {:.2?}", f32_vec);
    println!("Speedup: {:.2}", f32_scalar.as_secs_f64() / f32_vec.as_secs_f64());

    let t7 = Instant::now(); gemm_f32(1.0, &a32, &bt32, 0.0, &mut c32, true); let f32_vec_par = t7.elapsed();
    println!("f32 vector parallel: {:.2?}", f32_vec_par);
    println!("Speedup: {:.2}", f32_scalar.as_secs_f64() / f32_vec_par.as_secs_f64());

    println!("Threads: {}", current_num_threads());
    
    #[cfg(target_arch = "aarch64")]
    println!("Running with NEON");
    #[cfg(all(target_arch = "x86_64", not(target_feature = "avx512f")))]
    println!("Running with AVX2");
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    println!("Runnung with AVX512");
    
}
