// Writes the distinct values of x in [lo, lo + span) to out, ascending, and returns how many. Other elements are ignored.
// out is the table: a one marks each value present, then the marked values are moved to its front.
size_t unique_i64(const int64_t *x, int64_t *out, size_t n, int64_t lo, size_t span) {
    for (size_t j = 0; j < span; j++) out[j] = 0;
    for (size_t i = 0; i < n; i++) {
        const uint64_t offset = (uint64_t)x[i] - (uint64_t)lo;
        if (offset < span) out[offset] = 1;
    }
    size_t count = 0;
    for (size_t j = 0; j < span; j++) {
        if (out[j]) out[count++] = (int64_t)((uint64_t)lo + j);
    }
    return count;
}
