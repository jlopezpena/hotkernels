// Naive C baselines for the phase 2 survey.
//
// Each kernel is the simplest correct loop that matches NumPy's semantics.
// It is compiled by survey/measure.py with `clang -O3 -mcpu=native` and timed inside C
// by the bench_* functions, so no Python call overhead is included.
//
// These are survey baselines, not kernel-contract `naive.c` files. A candidate
// that is chosen for phase 3 gets its own reviewed naive.c under kernels/.

#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>

typedef float f32;
typedef double f64;
typedef int64_t i64;
typedef uint8_t u8;

#define NOINLINE __attribute__((noinline))

static inline uint64_t now_ns(void) { return clock_gettime_nsec_np(CLOCK_UPTIME_RAW); }

// Ask macOS for the highest QoS class so the timing thread stays on P-cores.
int survey_init(void) { return pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0); }

// The kernel is called through a volatile function pointer so the compiler can
// neither inline it into the timing loop nor hoist it out.
#define BENCH(name, params, call)                                              \
    uint64_t bench_##name params {                                             \
        __typeof__(&name) volatile kfn = &name;                                 \
        volatile i64 sink = 0;                                                 \
        uint64_t t0 = now_ns();                                                \
        for (i64 r = 0; r < reps; r++) sink += kfn call;                        \
        uint64_t t1 = now_ns();                                                \
        (void)sink;                                                            \
        return t1 - t0;                                                        \
    }

// As BENCH, for kernels that write an output buffer. If `fresh` is non-zero the
// output is malloc'd and freed on every call (`fresh` bytes), which is what
// NumPy does when no out= is given. Otherwise the preallocated `out0` is used.
#define BENCH_OUT(name, OUT_T, params, call)                                   \
    uint64_t bench_##name params {                                             \
        __typeof__(&name) volatile kfn = &name;                                 \
        volatile i64 sink = 0;                                                 \
        uint64_t t0 = now_ns();                                                \
        for (i64 r = 0; r < reps; r++) {                                       \
            OUT_T *out = fresh ? (OUT_T *)malloc((size_t)fresh) : out0;        \
            sink += kfn call;                                                   \
            if (fresh) free(out);                                              \
        }                                                                      \
        uint64_t t1 = now_ns();                                                \
        (void)sink;                                                            \
        return t1 - t0;                                                        \
    }

// ---------------------------------------------------------------- argmax/argmin
// NumPy: first occurrence wins; for floats the first NaN wins.

#define ARG_FLOAT(name, T, OP)                                                 \
    NOINLINE i64 name(const T *x, i64 n, i64 stride) {                         \
        T best = x[0];                                                         \
        i64 bi = 0;                                                            \
        if (best != best) return 0;                                            \
        for (i64 i = 1; i < n; i++) {                                          \
            T v = x[i * stride];                                               \
            if (v != v) return i;                                              \
            if (v OP best) { best = v; bi = i; }                               \
        }                                                                      \
        return bi;                                                             \
    }                                                                          \
    BENCH(name, (const T *x, i64 n, i64 stride, i64 reps), (x, n, stride))

#define ARG_INT(name, T, OP)                                                   \
    NOINLINE i64 name(const T *x, i64 n, i64 stride) {                         \
        T best = x[0];                                                         \
        i64 bi = 0;                                                            \
        for (i64 i = 1; i < n; i++) {                                          \
            T v = x[i * stride];                                               \
            if (v OP best) { best = v; bi = i; }                               \
        }                                                                      \
        return bi;                                                             \
    }                                                                          \
    BENCH(name, (const T *x, i64 n, i64 stride, i64 reps), (x, n, stride))

ARG_FLOAT(argmax_f32, f32, >)
ARG_FLOAT(argmax_f64, f64, >)
ARG_FLOAT(argmin_f64, f64, <)
ARG_INT(argmax_i64, i64, >)

// ---------------------------------------------------------------- max / sum (controls)
// NumPy is known to have SIMD for these; they check that the harness is sane.

NOINLINE i64 max_f32(const f32 *x, i64 n, f32 *res) {
    f32 m = x[0];
    for (i64 i = 0; i < n; i++) {
        f32 v = x[i];
        if (v != v) { *res = v; return 0; }
        if (v > m) m = v;
    }
    *res = m;
    return 0;
}
BENCH(max_f32, (const f32 *x, i64 n, f32 *res, i64 reps), (x, n, res))

#define SUM(name, T)                                                           \
    NOINLINE i64 name(const T *x, i64 n, i64 stride, T *res) {                 \
        T s = 0;                                                               \
        for (i64 i = 0; i < n; i++) s += x[i * stride];                        \
        *res = s;                                                              \
        return 0;                                                              \
    }                                                                          \
    BENCH(name, (const T *x, i64 n, i64 stride, T *res, i64 reps), (x, n, stride, res))

SUM(sum_f32, f32)
SUM(sum_f64, f64)

// ---------------------------------------------------------------- boolean mask selection

#define COMPRESS(name, T)                                                      \
    NOINLINE i64 name(const T *x, const u8 *mask, T *out, i64 n) {             \
        i64 k = 0;                                                             \
        for (i64 i = 0; i < n; i++)                                            \
            if (mask[i]) out[k++] = x[i];                                      \
        return k;                                                              \
    }                                                                          \
    BENCH_OUT(name, T, (const T *x, const u8 *mask, T *out0, i64 n, i64 fresh, i64 reps), (x, mask, out, n))

COMPRESS(compress_f32, f32)
COMPRESS(compress_f64, f64)

// ---------------------------------------------------------------- count_nonzero

#define COUNT_NONZERO(name, T)                                                 \
    NOINLINE i64 name(const T *x, i64 n) {                                     \
        i64 c = 0;                                                             \
        for (i64 i = 0; i < n; i++) c += (x[i] != 0);                          \
        return c;                                                              \
    }                                                                          \
    BENCH(name, (const T *x, i64 n, i64 reps), (x, n))

COUNT_NONZERO(count_nonzero_bool, u8)
COUNT_NONZERO(count_nonzero_i64, i64)
COUNT_NONZERO(count_nonzero_f64, f64)

// ---------------------------------------------------------------- nonzero

NOINLINE i64 nonzero_bool(const u8 *mask, i64 *out, i64 n) {
    i64 k = 0;
    for (i64 i = 0; i < n; i++)
        if (mask[i]) out[k++] = i;
    return k;
}
BENCH_OUT(nonzero_bool, i64, (const u8 *mask, i64 *out0, i64 n, i64 fresh, i64 reps), (mask, out, n))

// ---------------------------------------------------------------- cumsum and friends

#define CUMSUM(name, T)                                                        \
    NOINLINE i64 name(const T *x, T *out, i64 n) {                             \
        T s = 0;                                                               \
        for (i64 i = 0; i < n; i++) { s += x[i]; out[i] = s; }                 \
        return 0;                                                              \
    }                                                                          \
    BENCH_OUT(name, T, (const T *x, T *out0, i64 n, i64 fresh, i64 reps), (x, out, n))

CUMSUM(cumsum_f32, f32)
CUMSUM(cumsum_f64, f64)
CUMSUM(cumsum_i64, i64)

// np.minimum.accumulate: NaN propagates from the first NaN onwards.
NOINLINE i64 cummin_f64(const f64 *x, f64 *out, i64 n) {
    if (n == 0) return 0;
    f64 m = x[0];
    out[0] = m;
    for (i64 i = 1; i < n; i++) {
        f64 v = x[i];
        if (m == m && (v != v || v < m)) m = v;
        out[i] = m;
    }
    return 0;
}
BENCH_OUT(cummin_f64, f64, (const f64 *x, f64 *out0, i64 n, i64 fresh, i64 reps), (x, out, n))

NOINLINE i64 diff_f64(const f64 *x, f64 *out, i64 n) {
    for (i64 i = 0; i + 1 < n; i++) out[i] = x[i + 1] - x[i];
    return 0;
}
BENCH_OUT(diff_f64, f64, (const f64 *x, f64 *out0, i64 n, i64 fresh, i64 reps), (x, out, n))

// ---------------------------------------------------------------- searchsorted (side='left')
// `a` has n sorted elements, `v` has m needles. No NaN in the survey inputs.

#define SEARCHSORTED(name, T)                                                  \
    NOINLINE i64 name(const T *a, i64 n, const T *v, i64 m, i64 *out) {        \
        for (i64 j = 0; j < m; j++) {                                          \
            T key = v[j];                                                      \
            i64 lo = 0, hi = n;                                                \
            while (lo < hi) {                                                  \
                i64 mid = lo + ((hi - lo) >> 1);                               \
                if (a[mid] < key) lo = mid + 1; else hi = mid;                 \
            }                                                                  \
            out[j] = lo;                                                       \
        }                                                                      \
        return 0;                                                              \
    }                                                                          \
    BENCH_OUT(name, i64, (const T *a, i64 n, const T *v, i64 m, i64 *out0, i64 fresh, i64 reps), (a, n, v, m, out))

SEARCHSORTED(searchsorted_f64, f64)
SEARCHSORTED(searchsorted_i64, i64)

// ---------------------------------------------------------------- clip
// NaN in x propagates (both comparisons are false).

#define CLIP(name, T)                                                          \
    NOINLINE i64 name(const T *x, T *out, i64 n, T lo, T hi) {                 \
        for (i64 i = 0; i < n; i++) {                                          \
            T v = x[i];                                                        \
            out[i] = v < lo ? lo : (v > hi ? hi : v);                          \
        }                                                                      \
        return 0;                                                              \
    }                                                                          \
    BENCH_OUT(name, T, (const T *x, T *out0, i64 n, T lo, T hi, i64 fresh, i64 reps), (x, out, n, lo, hi))

CLIP(clip_f32, f32)
CLIP(clip_f64, f64)
CLIP(clip_i64, i64)

// ---------------------------------------------------------------- any / all

NOINLINE i64 any_bool(const u8 *x, i64 n) {
    for (i64 i = 0; i < n; i++)
        if (x[i]) return 1;
    return 0;
}
BENCH(any_bool, (const u8 *x, i64 n, i64 reps), (x, n))

NOINLINE i64 all_bool(const u8 *x, i64 n) {
    for (i64 i = 0; i < n; i++)
        if (!x[i]) return 0;
    return 1;
}
BENCH(all_bool, (const u8 *x, i64 n, i64 reps), (x, n))

// ---------------------------------------------------------------- take
// mode='raise': negative indices wrap once, anything else out of range fails.

NOINLINE i64 take_f64(const f64 *a, i64 n, const i64 *idx, i64 m, f64 *out) {
    for (i64 j = 0; j < m; j++) {
        i64 i = idx[j];
        if (i < 0) i += n;
        if (i < 0 || i >= n) return -1;
        out[j] = a[i];
    }
    return 0;
}
BENCH_OUT(take_f64, f64, (const f64 *a, i64 n, const i64 *idx, i64 m, f64 *out0, i64 fresh, i64 reps), (a, n, idx, m, out))

// ---------------------------------------------------------------- isin
// Simplest correct loop: compare against every test value.

NOINLINE i64 isin_i64(const i64 *x, i64 n, const i64 *test, i64 m, u8 *out) {
    for (i64 i = 0; i < n; i++) {
        u8 hit = 0;
        for (i64 j = 0; j < m; j++) hit |= (x[i] == test[j]);
        out[i] = hit;
    }
    return 0;
}
BENCH_OUT(isin_i64, u8, (const i64 *x, i64 n, const i64 *test, i64 m, u8 *out0, i64 fresh, i64 reps), (x, n, test, m, out))

// ---------------------------------------------------------------- where (three-argument)

NOINLINE i64 where_f64(const u8 *c, const f64 *a, const f64 *b, f64 *out, i64 n) {
    for (i64 i = 0; i < n; i++) out[i] = c[i] ? a[i] : b[i];
    return 0;
}
BENCH_OUT(where_f64, f64, (const u8 *c, const f64 *a, const f64 *b, f64 *out0, i64 n, i64 fresh, i64 reps), (c, a, b, out, n))

// ---------------------------------------------------------------- bincount
// Non-negative input only (NumPy raises otherwise); `out` has nbins slots.

NOINLINE i64 bincount_i64(const i64 *x, i64 n, i64 *out, i64 nbins) {
    for (i64 b = 0; b < nbins; b++) out[b] = 0;
    for (i64 i = 0; i < n; i++) out[x[i]]++;
    return 0;
}
BENCH_OUT(bincount_i64, i64, (const i64 *x, i64 n, i64 *out0, i64 nbins, i64 fresh, i64 reps), (x, n, out, nbins))

// ---------------------------------------------------------------- stepped slices

NOINLINE i64 copy_f64_strided(const f64 *x, f64 *out, i64 n, i64 stride) {
    for (i64 i = 0; i < n; i++) out[i] = x[i * stride];
    return 0;
}
BENCH_OUT(copy_f64_strided, f64, (const f64 *x, f64 *out0, i64 n, i64 stride, i64 fresh, i64 reps), (x, out, n, stride))

NOINLINE i64 add_f64_strided(const f64 *x, const f64 *y, f64 *out, i64 n, i64 stride) {
    for (i64 i = 0; i < n; i++) out[i] = x[i * stride] + y[i * stride];
    return 0;
}
BENCH_OUT(add_f64_strided, f64, (const f64 *x, const f64 *y, f64 *out0, i64 n, i64 stride, i64 fresh, i64 reps), (x, y, out, n, stride))

// ================================================================ wider net
// Candidates beyond the phase 2 list, added to get a shortlist of about ten.

ARG_INT(argmax_u8, u8, >)   // np.argmax on a bool array: "find first True"

NOINLINE i64 sum_u8(const u8 *x, i64 n) {
    uint64_t s = 0;
    for (i64 i = 0; i < n; i++) s += x[i];
    return (i64)s;
}
BENCH(sum_u8, (const u8 *x, i64 n, i64 reps), (x, n))

NOINLINE i64 dot_i64(const i64 *x, const i64 *y, i64 n) {
    uint64_t s = 0;
    for (i64 i = 0; i < n; i++) s += (uint64_t)x[i] * (uint64_t)y[i];
    return (i64)s;
}
BENCH(dot_i64, (const i64 *x, const i64 *y, i64 n, i64 reps), (x, y, n))

NOINLINE i64 array_equal_f64(const f64 *x, const f64 *y, i64 n) {
    for (i64 i = 0; i < n; i++)
        if (x[i] != y[i]) return 0;
    return 1;
}
BENCH(array_equal_f64, (const f64 *x, const f64 *y, i64 n, i64 reps), (x, y, n))

// np.packbits: most significant bit first, last byte padded with zeros.
NOINLINE i64 packbits(const u8 *x, u8 *out, i64 n) {
    i64 nbytes = (n + 7) / 8;
    for (i64 b = 0; b < nbytes; b++) {
        u8 v = 0;
        for (i64 j = 0; j < 8; j++) {
            i64 i = b * 8 + j;
            v = (u8)(v << 1) | (u8)(i < n && x[i] != 0);
        }
        out[b] = v;
    }
    return nbytes;
}
BENCH_OUT(packbits, u8, (const u8 *x, u8 *out0, i64 n, i64 fresh, i64 reps), (x, out, n))

#define CAST(name, S, D)                                                       \
    NOINLINE i64 name(const S *x, D *out, i64 n) {                             \
        for (i64 i = 0; i < n; i++) out[i] = (D)x[i];                          \
        return 0;                                                              \
    }                                                                          \
    BENCH_OUT(name, D, (const S *x, D *out0, i64 n, i64 fresh, i64 reps), (x, out, n))

CAST(cast_f64_f32, f64, f32)
CAST(cast_u8_f32, u8, f32)
CAST(cast_i64_f64, i64, f64)

// np.interp(v, xp, fp): xp increasing, no NaN in the survey inputs.
NOINLINE i64 interp_f64(const f64 *v, i64 m, const f64 *xp, const f64 *fp, i64 n, f64 *out) {
    for (i64 j = 0; j < m; j++) {
        f64 x = v[j];
        i64 lo = 0, hi = n;   // count of xp <= x
        while (lo < hi) {
            i64 mid = lo + ((hi - lo) >> 1);
            if (xp[mid] <= x) lo = mid + 1; else hi = mid;
        }
        if (lo == 0) out[j] = fp[0];
        else if (lo == n) out[j] = fp[n - 1];
        else {
            i64 k = lo - 1;
            f64 slope = (fp[k + 1] - fp[k]) / (xp[k + 1] - xp[k]);
            out[j] = slope * (x - xp[k]) + fp[k];
        }
    }
    return 0;
}
BENCH_OUT(interp_f64, f64, (const f64 *v, i64 m, const f64 *xp, const f64 *fp, i64 n, f64 *out0, i64 fresh, i64 reps), (v, m, xp, fp, n, out))

// np.add.at(acc, idx, v)
NOINLINE i64 add_at_f64(f64 *acc, const i64 *idx, const f64 *v, i64 n) {
    for (i64 i = 0; i < n; i++) acc[idx[i]] += v[i];
    return 0;
}
BENCH(add_at_f64, (f64 *acc, const i64 *idx, const f64 *v, i64 n, i64 reps), (acc, idx, v, n))

// a[idx] = v, with NumPy's index rules (see take_f64).
NOINLINE i64 put_f64(f64 *a, i64 n, const i64 *idx, const f64 *v, i64 m) {
    for (i64 j = 0; j < m; j++) {
        i64 i = idx[j];
        if (i < 0) i += n;
        if (i < 0 || i >= n) return -1;
        a[i] = v[j];
    }
    return 0;
}
BENCH(put_f64, (f64 *a, i64 n, const i64 *idx, const f64 *v, i64 m, i64 reps), (a, n, idx, v, m))

// x[mask] = value
NOINLINE i64 maskfill_f64(f64 *x, const u8 *mask, i64 n, f64 value) {
    for (i64 i = 0; i < n; i++)
        if (mask[i]) x[i] = value;
    return 0;
}
BENCH(maskfill_f64, (f64 *x, const u8 *mask, i64 n, f64 value, i64 reps), (x, mask, n, value))

// np.correlate(x, k, 'valid'): x has n + m - 1 elements, out has n.
NOINLINE i64 correlate_f64(const f64 *x, const f64 *k, f64 *out, i64 n, i64 m) {
    for (i64 i = 0; i < n; i++) {
        f64 s = 0;
        for (i64 j = 0; j < m; j++) s += x[i + j] * k[j];
        out[i] = s;
    }
    return 0;
}
BENCH_OUT(correlate_f64, f64, (const f64 *x, const f64 *k, f64 *out0, i64 n, i64 m, i64 fresh, i64 reps), (x, k, out, n, m))

// np.histogram(x, bins=nbins, range=(lo, hi)) counts, uniform bins.
NOINLINE i64 histogram_f64(const f64 *x, i64 n, i64 *out, i64 nbins, f64 lo, f64 hi) {
    for (i64 b = 0; b < nbins; b++) out[b] = 0;
    f64 norm = (f64)nbins / (hi - lo);
    for (i64 i = 0; i < n; i++) {
        f64 v = x[i];
        if (!(v >= lo && v <= hi)) continue;
        i64 b = (i64)((v - lo) * norm);
        if (b == nbins) b--;
        out[b]++;
    }
    return 0;
}
BENCH_OUT(histogram_f64, i64, (const f64 *x, i64 n, i64 *out0, i64 nbins, f64 lo, f64 hi, i64 fresh, i64 reps), (x, n, out, nbins, lo, hi))

// ================================================================ branchless variants
// The same operations as compress_f64 and nonzero_bool, written so that the
// loop has no data-dependent branch. They store one element past the used
// output (still inside the n-element capacity), which the judge's current
// output rule would reject. Measured to separate "NumPy mispredicts" from
// "SIMD is needed".

NOINLINE i64 compress_f64_branchless(const f64 *x, const u8 *mask, f64 *out, i64 n) {
    i64 k = 0;
    for (i64 i = 0; i < n; i++) { out[k] = x[i]; k += (mask[i] != 0); }
    return k;
}
BENCH_OUT(compress_f64_branchless, f64, (const f64 *x, const u8 *mask, f64 *out0, i64 n, i64 fresh, i64 reps), (x, mask, out, n))

NOINLINE i64 nonzero_bool_branchless(const u8 *mask, i64 *out, i64 n) {
    i64 k = 0;
    for (i64 i = 0; i < n; i++) { out[k] = i; k += (mask[i] != 0); }
    return k;
}
BENCH_OUT(nonzero_bool_branchless, i64, (const u8 *mask, i64 *out0, i64 n, i64 fresh, i64 reps), (mask, out, n))
