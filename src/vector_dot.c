// src/vector_dot.c
#include <riscv_vector.h>
#include <stddef.h>
#include <stdint.h>

// ---------------------------------------------------------------------------
// Single‐precision float dot‐product via RVV
// ---------------------------------------------------------------------------
float vector_dot_f32(const float *a, const float *b, size_t n) {
    size_t i = 0;
    float acc = 0.0f;
    while (i < n) {
        // ask hardware for up to vlen lanes of e32,m1
        size_t vl = __riscv_vsetvlmax_e32m1();
        if (vl > n - i) vl = n - i;

        // load vectors
        vfloat32m1_t va = __riscv_vle32_v_f32m1(a + i, vl);
        vfloat32m1_t vb = __riscv_vle32_v_f32m1(b + i, vl);

        // multiply‐accumulate across lanes
        vfloat32m1_t vp = __riscv_vfmacc_vv_f32m1(va, vb, (__riscv_vundefined_f32m1()), vl);

        // horizontal sum via reduction intrinsic (lane 0 ← sum)
        vfloat32m1_t vs = __riscv_vfredsum_vs_f32m1(vp, vp, vl);

        // extract lane 0 into a scalar and add to accumulator
        float tmp;
        __riscv_vse32_v_f32m1(&tmp, vs, vl);
        acc += tmp;

        i += vl;
    }
    return acc;
}

// ---------------------------------------------------------------------------
// Double‐precision float dot‐product via RVV
// ---------------------------------------------------------------------------
double vector_dot_f64(const double *a, const double *b, size_t n) {
    size_t i = 0;
    double acc = 0.0;
    while (i < n) {
        size_t vl = __riscv_vsetvlmax_e64m1();
        if (vl > n - i) vl = n - i;

        vfloat64m1_t va = __riscv_vle64_v_f64m1(a + i, vl);
        vfloat64m1_t vb = __riscv_vle64_v_f64m1(b + i, vl);

        vfloat64m1_t vp = __riscv_vfmacc_vv_f64m1(va, vb, (__riscv_vundefined_f64m1()), vl);
        vfloat64m1_t vs = __riscv_vfredsum_vs_f64m1(vp, vp, vl);

        double tmp;
        __riscv_vse64_v_f64m1(&tmp, vs, vl);
        acc += tmp;

        i += vl;
    }
    return acc;
}
