// bottleneck's nanstd(a, axis=0, ddof=ddof) over the matrix whose element
// (i, j) is a[i * step + j], for j < cols and as many rows as fit in n
// elements. Writes one standard deviation per column to out and returns cols,
// or 0 with no rows.
//
// Two passes over each column in order, as bottleneck makes them: the sum
// from +0.0 and the count of the elements that are not NaN; then, if the
// count exceeds ddof, the sum from +0.0 of the squared differences from the
// mean, and sqrt(sum / (count - ddof)). Here a block of columns is carried at
// once, a row at a time, so the additions of a row are independent and the
// compiler makes vector code of them. A NaN adds +0.0 instead of being
// skipped, which is the same: each sum starts at +0.0, never becomes -0.0,
// and x + 0.0 is x.
#define BLOCK 32
// bottleneck's NaN for a column with no result: NPY_NAN, the positive quiet
// NaN. Spelled as the compiler's constant, which is that NaN everywhere.
#define POSITIVE_NAN __builtin_nan("")

// The compiler must not fuse a multiply and an add on its own: where
// bottleneck's build does not, the two roundings are part of the result.
#pragma STDC FP_CONTRACT OFF

// bottleneck's `asum += ai * ai` as its build for this architecture computes
// it. On ARM64 clang contracts it into one fused multiply-add, rounded once
// (the wheel's nanstd_one_float64 has `fmadd d3, d3, d3, d0`). On x86-64 a
// build for the baseline instruction set has no fused multiply-add, so the
// square is rounded and then the sum.
static inline double add_square(double d, double sum) {
#if defined(__aarch64__)
    return __builtin_fma(d, d, sum);
#else
    double square = d * d;
    return sum + square;
#endif
}

size_t nanstd_cols_f64(const double *a, double *out, size_t n, size_t cols, size_t step, int64_t ddof) {
    if (n < cols) return 0;
    size_t rows = (n - cols) / step + 1;
    for (size_t j0 = 0; j0 < cols; j0 += BLOCK) {
        size_t w = cols - j0 < BLOCK ? cols - j0 : BLOCK;
        double sum[BLOCK], mean[BLOCK];
        int64_t count[BLOCK];
        for (size_t k = 0; k < w; k++) { sum[k] = 0.0; count[k] = 0; }
        for (size_t i = 0; i < rows; i++) {
            const double *row = a + i * step + j0;
            for (size_t k = 0; k < w; k++) {
                double v = row[k];
                int present = v == v;
                sum[k] += present ? v : 0.0;
                count[k] += present;
            }
        }
        // The mean only where bottleneck forms it, so that no division raises
        // a floating-point flag bottleneck's does not. A column whose count
        // does not exceed ddof gets NaN below, whatever the next pass leaves.
        for (size_t k = 0; k < w; k++) {
            mean[k] = count[k] > ddof ? sum[k] / (double)count[k] : 0.0;
            sum[k] = 0.0;
        }
        for (size_t i = 0; i < rows; i++) {
            const double *row = a + i * step + j0;
            for (size_t k = 0; k < w; k++) {
                double v = row[k];
                double d = v == v ? v - mean[k] : 0.0;
                sum[k] = add_square(d, sum[k]);
            }
        }
        for (size_t k = 0; k < w; k++) {
            out[j0 + k] = count[k] > ddof ? __builtin_sqrt(sum[k] / (double)(count[k] - ddof)) : POSITIVE_NAN;
        }
    }
    return cols;
}
