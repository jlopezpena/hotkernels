// Sum of the bytes of x.
uint64_t sum_u8(const uint8_t *x, size_t n) {
    uint64_t s = 0;
    for (size_t i = 0; i < n; i++) s += x[i];
    return s;
}
