// Number of elements that are not zero.
size_t count_nonzero_f64(const double *x, size_t n) {
    size_t c = 0;
    for (size_t i = 0; i < n; i++) c += (x[i] != 0);
    return c;
}
