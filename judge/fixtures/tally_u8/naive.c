// Counts, for each remainder b below nbuckets, the elements of x that leave
// that remainder when divided by nbuckets.
void tally_u8(const uint8_t *x, int64_t *counts, size_t n, size_t nbuckets) {
    for (size_t b = 0; b < nbuckets; b++) counts[b] = 0;
    for (size_t i = 0; i < n; i++) counts[x[i] % nbuckets]++;
}
