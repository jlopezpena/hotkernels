// bottleneck's nanmean(a, axis=0) over the matrix whose element (i, j) is
// a[i * step + j], for j < cols and as many rows as fit in n elements.
// Writes one mean per column to out and returns cols, or 0 with no rows.
//
// Each column is summed in order from +0.0, NaN skipped, as bottleneck sums
// it, so the result is bottleneck's bit for bit. bottleneck finishes one
// column before starting the next and each addition waits for the one
// before. Here a block of columns is carried at once, one accumulator each,
// a row at a time: the additions of a row are independent, and the compiler
// makes vector code of them. A NaN adds +0.0 instead of being skipped, which
// is the same: the sum starts at +0.0, never becomes -0.0, and x + 0.0 is x.
#define BLOCK 32
// bottleneck's NaN for a column with no number: NPY_NAN, the positive quiet
// NaN. Spelled as the compiler's constant, which is that NaN everywhere; 0.0 /
// 0.0 would give a negative one on x86-64.
#define POSITIVE_NAN __builtin_nan("")

size_t nanmean_cols_f64(const double *a, double *out, size_t n, size_t cols, size_t step) {
    if (n < cols) return 0;
    size_t rows = (n - cols) / step + 1;
    for (size_t j0 = 0; j0 < cols; j0 += BLOCK) {
        size_t w = cols - j0 < BLOCK ? cols - j0 : BLOCK;
        double sum[BLOCK];
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
        for (size_t k = 0; k < w; k++) out[j0 + k] = count[k] > 0 ? sum[k] / (double)count[k] : POSITIVE_NAN;
    }
    return cols;
}
