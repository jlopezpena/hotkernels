// Running sum, accumulated left to right. The first element is copied, not
// added to zero, so a leading negative zero stays negative, as in NumPy.
void cumsum_f64(const double *x, double *out, size_t n) {
    if (n == 0) return;
    double s = x[0];
    out[0] = s;
    for (size_t i = 1; i < n; i++) {
        s += x[i];
        out[i] = s;
    }
}
