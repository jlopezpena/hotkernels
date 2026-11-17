// Running sum, accumulated left to right.
void cumsum_i64(const int64_t *x, int64_t *out, size_t n) {
    uint64_t s = 0;
    for (size_t i = 0; i < n; i++) {
        s += (uint64_t)x[i];  // unsigned, so overflow wraps as NumPy's does
        out[i] = (int64_t)s;
    }
}
