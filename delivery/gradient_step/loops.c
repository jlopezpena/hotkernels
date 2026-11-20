// Loops for the gradient-step study (docs/gradient-step-findings.md), called
// from Python through ctypes. They are measurement code, not kernels: the
// study builds them with `clang -O3 -ffp-contract=off` into a library of its
// own, and nothing in the package uses them.
//
// Every loop computes what NumPy computes, bit for bit. `w -= a * g` in NumPy
// is two ufunc calls, so the product is rounded to float64 before it is
// subtracted; the fused loops do the same, and -ffp-contract=off keeps the
// compiler from turning the pair into a fused multiply-subtract, which
// rounds once and would give different bits.
//
// `ahead` is the prefetch distance in bytes. With `ahead` > 0 each 128-byte
// line of `w` is prefetched for a store (`prfm pstl1keep`) that many bytes
// before it is loaded; with 0 there is no prefetch. 128 bytes is the line of
// the Apple M1's caches. Prefetches stop `ahead` bytes before the end of `w`,
// so none names an address outside it.
#include <arm_neon.h>
#include <stddef.h>

// Doubles in one 128-byte cache line, the block each loop step handles.
#define LINE 16

// Prefetch the line at `p` for a store into the first-level cache.
static inline void prefetch_for_store(const double *p) {
    __builtin_prefetch(p, 1, 3);
}

// w[i] = w[i] - b[i] for i < n.
void study_subtract(double *w, const double *b, size_t n, size_t ahead) {
    size_t i = 0, stop = 0, step = ahead / sizeof(double);
    if (ahead > 0 && n > step) stop = n - step;
    for (; i + LINE <= n; i += LINE) {
        if (i < stop) prefetch_for_store(w + i + step);
        for (size_t j = 0; j < LINE; j += 2)
            vst1q_f64(w + i + j, vsubq_f64(vld1q_f64(w + i + j), vld1q_f64(b + i + j)));
    }
    for (; i < n; i++) w[i] = w[i] - b[i];
}

// w[i] = w[i] - (a * g[i]) for i < n, the product rounded first.
void study_fused(double *w, const double *g, double a, size_t n, size_t ahead) {
    size_t i = 0, stop = 0, step = ahead / sizeof(double);
    if (ahead > 0 && n > step) stop = n - step;
    const float64x2_t va = vdupq_n_f64(a);
    for (; i + LINE <= n; i += LINE) {
        if (i < stop) prefetch_for_store(w + i + step);
        for (size_t j = 0; j < LINE; j += 2) {
            float64x2_t p = vmulq_f64(va, vld1q_f64(g + i + j));
            vst1q_f64(w + i + j, vsubq_f64(vld1q_f64(w + i + j), p));
        }
    }
    for (; i < n; i++) {
        double p = a * g[i];
        w[i] = w[i] - p;
    }
}

// m[i] = m[i] + (d[i] * d[i]) for i < n, the product rounded first: Adagrad's
// `mem += dparam * dparam` in one pass.
void study_fused_square_add(double *m, const double *d, size_t n) {
    for (size_t i = 0; i < n; i++) {
        double p = d[i] * d[i];
        m[i] = m[i] + p;
    }
}
