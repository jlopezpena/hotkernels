// bottleneck's nanmin(a, axis=0) over the matrix whose element (i, j) is
// a[i * step + j], for j < cols and as many rows as fit in n elements.
// Writes the smallest element of each column to out and returns cols, or 0 with
// no rows.
//
// bottleneck starts each column at +inf and replaces it with each element
// `v` in order for which `v <= extreme`: a NaN never compares, and among
// equal elements the last wins, which decides the sign of a zero result. A
// column where no comparison held gives NaN. The same test in the same order
// gives the same bits. Here a block of columns is carried at once, a row at a
// time, so the compiler makes vector code of the comparisons.
#define BLOCK 32
// bottleneck's NaN for a column with no number: NPY_NAN, the positive quiet
// NaN. Spelled as the compiler's constant, which is that NaN everywhere.
#define POSITIVE_NAN __builtin_nan("")

size_t nanmin_cols_f64(const double *a, double *out, size_t n, size_t cols, size_t step) {
    if (n < cols) return 0;
    size_t rows = (n - cols) / step + 1;
    for (size_t j0 = 0; j0 < cols; j0 += BLOCK) {
        size_t w = cols - j0 < BLOCK ? cols - j0 : BLOCK;
        double extreme[BLOCK];
        uint64_t found[BLOCK];
        for (size_t k = 0; k < w; k++) { extreme[k] = __builtin_inf(); found[k] = 0; }
        for (size_t i = 0; i < rows; i++) {
            const double *row = a + i * step + j0;
            for (size_t k = 0; k < w; k++) {
                double v = row[k];
                int take = v <= extreme[k];
                extreme[k] = take ? v : extreme[k];
                found[k] |= take;
            }
        }
        for (size_t k = 0; k < w; k++) out[j0 + k] = found[k] ? extreme[k] : POSITIVE_NAN;
    }
    return cols;
}
