// Counts how many elements of x equal each value from 0 to bins - 1. Elements outside that range are ignored.
void bincount_i64(const int64_t *x, int64_t *counts, size_t n, size_t bins) {
    for (size_t b = 0; b < bins; b++) counts[b] = 0;
    for (size_t i = 0; i < n; i++) {
        if ((uint64_t)x[i] < bins) counts[x[i]]++;
    }
}
