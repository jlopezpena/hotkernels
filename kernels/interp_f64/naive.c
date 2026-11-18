// np.interp(x, xp, fp) for `queries` query points in x and n sample points in
// xp and fp, with the default left and right values (fp[0] and fp[n-1]). xp is
// ascending and has no NaN.
//
// This follows NumPy's arr_interp case by case, so the result is bit-identical:
//   - a NaN query gives that NaN (except with a single sample point);
//   - j is the largest index with xp[j] <= x;
//   - the interpolated value is slope * (x - xp[j]) + fp[j], rounded as
//     NumPy's build for the platform rounds it (see mul_add), with NumPy's two
//     fallbacks when it comes out as NaN.

// The compiler must not fuse a multiply and an add on its own: where NumPy
// does not, the two roundings are part of the result.
#pragma STDC FP_CONTRACT OFF

// a * b + c as NumPy's build for this architecture computes it in arr_interp.
// On ARM64 the compiler contracts the expression into one fused multiply-add,
// rounded once. On x86-64 it does not: NumPy's baseline build has no FMA, so
// the product is rounded and then the sum. Measured against np.interp on each.
static inline double mul_add(double a, double b, double c) {
#if defined(__aarch64__)
    return __builtin_fma(a, b, c);
#else
    double product = a * b;
    return product + c;
#endif
}

void interp_f64(const double *x, const double *xp, const double *fp, double *out, size_t n,
                size_t queries) {
    double left = fp[0], right = fp[n - 1];
    if (n == 1) {
        for (size_t i = 0; i < queries; i++) {
            double v = x[i];
            out[i] = v < xp[0] ? left : (v > xp[0] ? right : fp[0]);
        }
        return;
    }
    for (size_t i = 0; i < queries; i++) {
        double v = x[i];
        if (v != v) { out[i] = v; continue; }
        if (v > xp[n - 1]) { out[i] = right; continue; }
        if (v < xp[0]) { out[i] = left; continue; }
        // Count the sample points that are <= v; j is the last of them.
        size_t lo = 0, hi = n;
        while (lo < hi) {
            size_t mid = lo + ((hi - lo) >> 1);
            if (v >= xp[mid]) lo = mid + 1;
            else hi = mid;
        }
        size_t j = lo - 1;
        if (j == n - 1 || xp[j] == v) { out[i] = fp[j]; continue; }
        double slope = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j]);
        double r = mul_add(slope, v - xp[j], fp[j]);
        if (r != r) {
            r = mul_add(slope, v - xp[j + 1], fp[j + 1]);
            if (r != r && fp[j] == fp[j + 1]) r = fp[j];
        }
        out[i] = r;
    }
}
