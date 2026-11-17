// NumPy's ordering for floats: NaN is greater than everything, and equal to itself.
static inline bool less(double a, double b) { return a < b || (b != b && a == a); }

// For each of the `needles` values in v, writes the first index i such that
// the needle is less than a[i], or n if there is none (side='right'): the
// index after the last element equal to the needle. a holds n values sorted
// ascending, NaN last.
void searchsorted_right_f64(const double *a, const double *v, int64_t *out, size_t n, size_t needles) {
    for (size_t j = 0; j < needles; j++) {
        double key = v[j];
        size_t lo = 0, hi = n;
        while (lo < hi) {
            size_t mid = lo + ((hi - lo) >> 1);
            if (less(key, a[mid])) hi = mid;
            else lo = mid + 1;
        }
        out[j] = (int64_t)lo;
    }
}
