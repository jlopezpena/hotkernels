// Converts every byte to a float.
void cast_u8_f32(const uint8_t *x, float *out, size_t n) {
    for (size_t i = 0; i < n; i++) out[i] = (float)x[i];
}
