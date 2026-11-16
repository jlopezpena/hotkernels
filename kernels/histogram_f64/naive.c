// The edge arithmetic below has to round exactly as NumPy's does, one
// operation at a time, so the compiler must not fuse a multiply and an add.
#pragma STDC FP_CONTRACT OFF

// Left edge of bin i, as np.linspace(lo, hi, nbins + 1) computes it.
static inline double edge(size_t i, size_t nbins, double step, double lo, double hi) {
    return i == nbins ? hi : (double)i * step + lo;
}

// Counts how many elements of x fall into each of nbins equal-width bins
// spanning [lo, hi]. Follows np.histogram step by step:
//   - elements outside [lo, hi], and NaN, are skipped;
//   - the bin is first estimated by arithmetic, which can be off by one near an
//     edge, then corrected by comparing with the edges themselves;
//   - the last bin includes hi.
void histogram_f64(const double *x, int64_t *counts, size_t n, size_t nbins, double lo, double hi) {
    for (size_t b = 0; b < nbins; b++) counts[b] = 0;
    double width = hi - lo;
    double step = width / (double)nbins;
    for (size_t i = 0; i < n; i++) {
        double v = x[i];
        if (!(v >= lo && v <= hi)) continue;
        size_t b = (size_t)(((v - lo) / width) * (double)nbins);
        if (b == nbins) b--;
        if (v < edge(b, nbins, step, lo, hi)) b--;
        if (v >= edge(b + 1, nbins, step, lo, hi) && b != nbins - 1) b++;
        counts[b]++;
    }
}
