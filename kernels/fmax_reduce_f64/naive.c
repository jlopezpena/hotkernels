// Maximum that ignores NaN, np.fmax.reduce(x): the largest element that is not a
// NaN, and a NaN only if every element is one. n is at least 1.
//
// The sign of a zero maximum and the NaN returned for an array of NaNs are NumPy's,
// and those depend on the order in which NumPy's loop combines the elements. So this
// follows that loop as it runs on x86-64 with AVX2: LANES running maxima that all
// start at x[0], the elements from x[1] on taken LANES at a time, the lanes combined,
// and the leftover elements one at a time. The comparisons are written out because
// the C library's fmax() is not declared here.

// Elements in one vector of NumPy's loop.
#define LANES 4

// One step of a lane: a NaN element changes nothing; otherwise the element replaces a
// NaN, a smaller value and an equal one, so the later of two zeros is kept.
static double lane_step(double acc, double v) {
    if (v != v) {
        return acc;
    }
    return acc > v ? acc : v;
}

// Two lanes combined: a number over a NaN, and the second of two equal values. Of two
// NaNs either will do: a lane is only a NaN while it still holds x[0].
static double lane_pair(double a, double b) {
    if (a != a) {
        return b;
    }
    if (b != b) {
        return a;
    }
    return a > b ? a : b;
}

double fmax_reduce_f64(const double *x, size_t n) {
    double acc[LANES];
    for (size_t j = 0; j < LANES; j++) {
        acc[j] = x[0];
    }
    size_t i = 1;
    for (; i + LANES <= n; i += LANES) {
        for (size_t j = 0; j < LANES; j++) {
            acc[j] = lane_step(acc[j], x[i + j]);
        }
    }
    double r = lane_pair(lane_pair(acc[0], acc[2]), lane_pair(acc[1], acc[3]));
    for (; i < n; i++) {
        double v = x[i];
        // A leftover element replaces a NaN and a smaller value. An equal value does
        // not replace, so here the earlier of two zeros is kept.
        if (r != r || v > r) {
            r = v;
        }
    }
    return r;
}
