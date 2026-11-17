// Writes the index of every true element of mask to out, ascending. Returns how many.
//
// Branchless, like compress_f32: every index is stored, and the output position
// advances only when the element is true. `out` is a scratch buffer.
size_t nonzero_bool(const bool *mask, int64_t *out, size_t n) {
    size_t k = 0;
    for (size_t i = 0; i < n; i++) {
        out[k] = (int64_t)i;
        k += mask[i];
    }
    return k;
}
