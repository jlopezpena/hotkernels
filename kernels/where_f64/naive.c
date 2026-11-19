// Takes out[i] from a where c[i] is true and from b where it is not.
void where_f64(const bool *c, const double *a, const double *b, double *out, size_t n) {
    for (size_t i = 0; i < n; i++) {
        out[i] = c[i] ? a[i] : b[i];
    }
}
