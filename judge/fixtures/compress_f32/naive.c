// Copies x[i] to out wherever mask[i] is true. Returns how many were copied.
size_t compress_f32(const float *x, const bool *mask, float *out, size_t n) {
    size_t k = 0;
    for (size_t i = 0; i < n; i++) {
        if (mask[i]) out[k++] = x[i];
    }
    return k;
}
