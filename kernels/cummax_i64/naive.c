// Running maximum: out[i] is the largest of x[0..i].
void cummax_i64(const int64_t *x, int64_t *out, size_t n) {
    if (n == 0) {
        return;
    }
    int64_t m = x[0];
    for (size_t i = 0; i < n; i++) {
        m = x[i] > m ? x[i] : m;
        out[i] = m;
    }
}
