// Index of the minimum, with NumPy's rules: the first minimum wins, and the
// first NaN wins over everything.
int64_t argmin_f32(const float *x, size_t n) {
    float best = x[0];
    if (best != best) return 0;
    size_t best_i = 0;
    for (size_t i = 1; i < n; i++) {
        float v = x[i];
        if (v != v) return (int64_t)i;
        if (v < best) {
            best = v;
            best_i = i;
        }
    }
    return (int64_t)best_i;
}
