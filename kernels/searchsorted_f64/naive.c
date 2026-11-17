// NumPy's ordering for floats: NaN is greater than everything, and equal to itself.
static inline bool less(double a, double b) { return a < b || (b != b && a == a); }

// For each of the n needles in v, writes the first index i such that a[i] is
// not less than the needle (side='left'). a holds n values sorted ascending,
// NaN last.
void searchsorted_f64(const double *a, const double *v, int64_t *out, size_t n) {
    for (size_t j = 0; j < n; j++) {
        double key = v[j];
        size_t lo = 0, hi = n;
        while (lo < hi) {
            size_t mid = lo + ((hi - lo) >> 1);
            if (less(a[mid], key)) lo = mid + 1;
            else hi = mid;
        }
        out[j] = (int64_t)lo;
    }
}
