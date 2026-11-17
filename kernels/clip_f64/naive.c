// Clamps every element to [lo, hi]. A NaN element stays NaN, because both
// comparisons are false for it.
//
// Valid for lo <= hi with neither bound NaN. For those bounds the result is
// NumPy's bit for bit, signed zeros included. For lo > hi NumPy returns hi for
// every element that is not NaN, and for a NaN bound it returns NaN; this loop does
// neither, so a caller must not pass such bounds.
void clip_f64(const double *x, double *out, size_t n, double lo, double hi) {
    for (size_t i = 0; i < n; i++) {
        double v = x[i];
        out[i] = v < lo ? lo : (v > hi ? hi : v);
    }
}
