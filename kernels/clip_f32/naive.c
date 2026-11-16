// Clamps every element to [lo, hi]. A NaN element stays NaN, because both
// comparisons are false for it.
void clip_f32(const float *x, float *out, size_t n, float lo, float hi) {
    for (size_t i = 0; i < n; i++) {
        float v = x[i];
        out[i] = v < lo ? lo : (v > hi ? hi : v);
    }
}
