// For each of the n elements of x, writes whether it equals any of the 8 test values.
void isin_i64(const int64_t *x, const int64_t *test, bool *out, size_t n) {
    for (size_t i = 0; i < n; i++) {
        bool hit = false;
        for (size_t j = 0; j < 8; j++) hit |= (x[i] == test[j]);
        out[i] = hit;
    }
}
