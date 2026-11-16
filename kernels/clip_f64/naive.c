// Clamps every element to [lo, hi]. A NaN element stays NaN, because both
// comparisons are false for it.
void clip_f64(const double *x, double *out, size_t n, double lo, double hi) {
    for (size_t i = 0; i < n; i++) {
        double v = x[i];
        out[i] = v < lo ? lo : (v > hi ? hi : v);
    }
}
