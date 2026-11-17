// For each of the `needles` values in v, writes how many of the n elements of
// a are smaller than it.
void rank_u8(const uint8_t *a, const uint8_t *v, int64_t *out, size_t n, size_t needles) {
    for (size_t j = 0; j < needles; j++) {
        int64_t smaller = 0;
        for (size_t i = 0; i < n; i++) smaller += a[i] < v[j];
        out[j] = smaller;
    }
}
