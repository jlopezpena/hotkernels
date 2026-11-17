// Counts how many times each byte value occurs in x. counts has 256 elements.
void bincount_u8(const uint8_t *x, int64_t *counts, size_t n) {
    for (size_t b = 0; b < 256; b++) counts[b] = 0;
    for (size_t i = 0; i < n; i++) counts[x[i]]++;
}
