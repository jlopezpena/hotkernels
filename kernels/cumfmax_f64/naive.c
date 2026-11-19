// Running maximum that ignores NaN: out[i] is the largest of x[0..i] that is not NaN,
// with NumPy's choice between equal values and between NaNs. The comparisons are
// written out because the C library's fmax() is not declared here, and the compiler's
// own would keep the earlier of two zeros where NumPy keeps the later.
void cumfmax_f64(const double *x, double *out, size_t n) {
    if (n == 0) {
        return;
    }
    double m = x[0];
    out[0] = m;
    if (n == 2) {
        // NumPy computes a single pair with its operands the other way round: of two
        // equal values the first is kept, and of two NaNs the second.
        double v = x[1];
        out[1] = m != m ? v : (v > m ? v : m);
        return;
    }
    for (size_t i = 1; i < n; i++) {
        double v = x[i];
        // A NaN element changes nothing. Otherwise the element replaces a NaN, a
        // smaller value, and an equal one: the later of two zeros is kept.
        if (v == v && (m != m || v >= m)) {
            m = v;
        }
        out[i] = m;
    }
}
