// Running count of the true elements: out[i] is how many of mask[0..i] are true.
void cumsum_bool(const bool *mask, int64_t *out, size_t n) {
    int64_t s = 0;
    for (size_t i = 0; i < n; i++) {
        s += mask[i];
        out[i] = s;
    }
}
