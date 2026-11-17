// void histogram_f64(const double *x, int64_t *counts, size_t n, size_t nbins, double lo, double hi);
// counts has exactly nbins slots; it arrives uninitialised.
//
// x0 = x, x1 = counts, x2 = n, x3 = nbins, d0 = lo, d1 = hi
//
// naive.c, for each element v with lo <= v <= hi:
//     b = (size_t)(((v - lo) / width) * nbins)        width = hi - lo
//     if (b == nbins) b--
//     if (v < edge(b)) b--                            edge(i) = (double)i * step + lo
//     if (v >= edge(b + 1) && b != nbins - 1) b++     step = width / nbins, edge(nbins) = hi
//     counts[b]++
// Every floating-point operation is rounded on its own. In particular an edge
// is a separate multiply and add, never fmadd.
//
// There are two vector paths. Both hand any lane they are not sure about to
// the scalar routine Lone, which is naive.c verbatim; the order of increments
// does not matter, so Lone can run for one lane in the middle of a block.
//
// ---- Compute path ----------------------------------------------------------
// Computes, two elements per vector, with exactly naive's operations:
//     t  = ((v - lo) / width) * nbins
//     b  = trunc(t) as a signed integer,  f = trunc(t) as a double
//     e0 = f * step + lo                  = edge(b)
//     e1 = (f + 1) * step + lo            = edge(b + 1) when b + 1 < nbins
//     q  = b + (v >= e0) + (v >= e1)
// and accepts the lane when 0 <= q - 1 < nbins - 1 (one unsigned compare),
// counting it in bin q - 1. Any other lane is redone by the scalar routine
// Lone, which is naive.c verbatim. Why an accepted lane is right (step is
// checked to be positive and finite before the vector path is used, so the
// edges are non-decreasing in b and edge(0) == lo):
//   - NaN: both compares are false and b = 0, so q = 0: not accepted.
//   - v < lo: v - lo is negative, so b <= 0. With b = 0, e0 = lo and both
//     compares are false; with b < 0, e1 <= lo so (v >= e1) is false. q <= 0.
//   - v > hi: v - lo >= width after rounding, so t >= nbins and q >= nbins.
//   - otherwise v is in range, b is naive's first estimate, and q - 1 <=
//     nbins - 2 means b <= nbins - 1, so naive's "b == nbins" step does
//     nothing. If v < e0, naive decrements and then tests v >= edge(b), which
//     is false: bin b - 1, and (v >= e1) is false too because e1 >= e0. If
//     v >= e0, naive keeps b and, b being at most nbins - 2 here, increments
//     exactly when v >= e1. Either way naive's bin is q - 1.
//   The last bin (q - 1 == nbins - 1) is never accepted, because there naive
//   uses hi as the edge and refuses to increment; Lone handles it.
//
// ---- Zone path -------------------------------------------------------------
// NOT a literal transcription of naive.c: it skips the two edge corrections
// for elements that are provably too far from a bin edge for either to fire,
// and sends every other element to Lone.
//
// It computes t exactly as naive does (fsub, fdiv, fmul), converts it to
// 32.32 fixed point z = trunc(t * 2^32) (exact scaling; saturating; NaN -> 0)
// and accepts a lane into bin b = z >> 32 when
//     DELTA <= fraction(z) < 1 - DELTA     and     b <= nbins - 2  (unsigned),
// with DELTA = 2^-12. So about one element in 2,048, plus everything in the
// last bin or out of range, takes the scalar route.
//
// Why an accepted lane is right. Write u = 2^-53, N = nbins, w = width,
// M = max(|lo|, |hi|), and tau = (v - lo) * N / w as a real number.
//   - NaN gives z = 0 (fraction 0); v < lo gives z <= 0 (fraction 0, or a
//     negative z whose b is huge as an unsigned number); v > hi gives
//     v - lo >= w after rounding, so t >= N and b >= N (or z saturates and
//     b = 2^31 - 1 >= N - 1). None is accepted, so an accepted v is in range.
//   - For v in range, t is tau after three rounded operations, so
//     |t - tau| <= 4 u N. Acceptance means b + DELTA <= t < b + 1 - DELTA.
//   - edge(k) = fl(fl(k * step) + lo) with step = fl(w / N) (a normal number,
//     checked). Expanding the three rounding errors, v >= edge(k) is certain
//     when tau >= k + D and v < edge(k) is certain when tau < k - D, where
//     D = 2.01 u N + 1.01 u M N / w.
//   - So if DELTA >= 4 u N + D, an accepted lane has edge(b) <= v < edge(b+1):
//     naive's first estimate is b, neither correction fires, the bin is b.
//   The entry check N < 2^31 and M * N <= w * 2^39 gives 4 u N + D < 2^-12.
//   The check is made in doubles, so it also requires fl(M * N) to be finite:
//   near the top of the range both products overflow to infinity, compare as
//   equal, and would pass for a range where the inequality is false. (With
//   fl(M * N) finite the rounded comparison is good enough: an infinite right
//   side means w * 2^39 really is the larger, and two finite products are each
//   within u of their real values, which the bound's slack absorbs.)
// The preconditions are checked once per call; if they fail the compute path
// is used.
//
// Register use
//   x4        scratch for zeroing
//   x5        counter being incremented
//   x6-x9     q for the four lanes of a block
//   x10       q - 1
//   x13, x14  scratch in Lone and in the zone path's checks
//   x15       return address for Lone
//   x16       nbins - 1
//   x17       counts - 8, so that counts[q - 1] is [x17, q, lsl #3]
//   v16 lo, v17 width, v18 (double)nbins, v19 step, v20 1.0 (each in both lanes)
//   d21 hi
//   d22, d23  v and scratch in Lone
//   v0-v7     block temporaries
//   d24, d25  scratch in the zone path's entry check

// Zone path: DELTA = 2^-12 as a 32-bit fraction is 2^20 = ZONE_DELTA << 12.
.set ZONE_DELTA, 0x100

// One vector of two elements at byte offset \off: leaves q in v\b.
// \v, \f, \g, \b are vector register numbers.
.macro BINS off, v, f, g, b
    ldr     q\v, [x0, #\off]
    fsub    v\f\().2d, v\v\().2d, v16.2d
    fdiv    v\f\().2d, v\f\().2d, v17.2d
    fmul    v\f\().2d, v\f\().2d, v18.2d        // t
    fcvtzs  v\b\().2d, v\f\().2d                // b
    frintz  v\f\().2d, v\f\().2d                // f
    fadd    v\g\().2d, v\f\().2d, v20.2d        // f + 1
    fmul    v\f\().2d, v\f\().2d, v19.2d
    fmul    v\g\().2d, v\g\().2d, v19.2d
    fadd    v\f\().2d, v\f\().2d, v16.2d        // e0
    fadd    v\g\().2d, v\g\().2d, v16.2d        // e1
    fcmge   v\f\().2d, v\v\().2d, v\f\().2d     // all ones where v >= e0
    fcmge   v\g\().2d, v\v\().2d, v\g\().2d     // all ones where v >= e1
    sub     v\b\().2d, v\b\().2d, v\f\().2d
    sub     v\b\().2d, v\b\().2d, v\g\().2d     // q
.endm

// Count one lane whose q is in \q, or hand it to the scalar routine.
.macro COUNT q, id
    sub     x10, \q, #1
    cmp     x10, x16
    b.hs    Lslow_\id
    ldr     x5, [x17, \q, lsl #3]
    add     x5, x5, #1
    str     x5, [x17, \q, lsl #3]
Lback_\id:
.endm

// Out-of-line entry to the scalar routine for the lane at byte offset \off.
.macro SLOW id, off
Lslow_\id:
    ldr     d22, [x0, #\off]
    adr     x15, Lback_\id
    b       Lone
.endm

// Zone path: count the lane whose fixed-point t is in x\n.
.macro ZCOUNT n, id
    add     w14, w\n, #ZONE_DELTA, lsl #12              // fraction + DELTA (wraps)
    cmp     w14, #(2 * ZONE_DELTA), lsl #12             // lower: within DELTA of an integer
    b.lo    Lslow_\id
    lsr     x13, x\n, #32                               // b
    cmp     x13, x16
    b.hs    Lslow_\id
    ldr     x5, [x1, x13, lsl #3]
    add     x5, x5, #1
    str     x5, [x1, x13, lsl #3]
Lback_\id:
.endm

// Zone path: one vector of two elements at byte offset \off, counted.
.macro ZONE_VEC off, id0, id1
    ldr     q0, [x0, #\off]
    fsub    v0.2d, v0.2d, v16.2d
    fdiv    v0.2d, v0.2d, v17.2d
    fmul    v0.2d, v0.2d, v18.2d                // t
    fcvtzs  v0.2d, v0.2d, #32                   // trunc(t * 2^32)
    fmov    x6, d0
    mov     x7, v0.d[1]
    ZCOUNT  6, \id0
    ZCOUNT  7, \id1
.endm

.text
.globl _histogram_f64
.p2align 2
_histogram_f64:
    cbz     x3, Ldone
    // Zero the counters, two per step.
    mov     x4, x1
    lsr     x5, x3, #1
    cbz     x5, Lzero_odd
Lzero_pair:
    stp     xzr, xzr, [x4], #16
    subs    x5, x5, #1
    b.ne    Lzero_pair
Lzero_odd:
    tbz     x3, #0, Lzeroed
    str     xzr, [x4]
Lzeroed:
    cbz     x2, Ldone

    fmov    d16, d0                 // lo
    fmov    d21, d1                 // hi
    fsub    d17, d1, d0             // width = hi - lo
    ucvtf   d18, x3                 // (double)nbins
    fdiv    d19, d17, d18           // step = width / nbins
    sub     x16, x3, #1
    sub     x17, x1, #8

    // The vector path needs step positive and finite (bit pattern in
    // [1, 0x7fefffffffffffff]) and nbins small enough to be exact as a double.
    fmov    x9, d19
    sub     x9, x9, #1
    mov     x10, #0x7ff0000000000000
    sub     x10, x10, #1
    cmp     x9, x10
    b.hs    Ltail
    lsr     x9, x3, #32
    cbnz    x9, Ltail

    dup     v16.2d, v16.d[0]
    dup     v17.2d, v17.d[0]
    dup     v18.2d, v18.d[0]
    dup     v19.2d, v19.d[0]
    fmov    d20, #1.0
    dup     v20.2d, v20.d[0]

    // Zone path entry check: nbins < 2^31, step normal (non-zero exponent),
    // and max(|lo|, |hi|) * nbins finite and <= width * 2^39.
    lsr     x9, x3, #31
    cbnz    x9, Lcompute
    fmov    x9, d19
    lsr     x9, x9, #52
    cbz     x9, Lcompute
    fabs    d24, d16
    fabs    d25, d21
    fmaxnm  d24, d24, d25
    fmul    d24, d24, d18
    fmov    x9, d24                 // not negative, so the bits order as the values do
    mov     x10, #0x7ff0000000000000
    cmp     x9, x10
    b.hs    Lcompute                // overflowed: infinity <= infinity below would pass
    mov     x9, #0x4260000000000000 // 2^39
    fmov    d25, x9
    fmul    d25, d17, d25
    fcmp    d24, d25
    b.hi    Lcompute                // also taken if the compare is unordered

    subs    x2, x2, #4
    b.lo    Lzone_done
Lzone_block:
    ZONE_VEC 0, e, f
    ZONE_VEC 16, g, h
    add     x0, x0, #32
    subs    x2, x2, #4
    b.hs    Lzone_block
Lzone_done:
    add     x2, x2, #4              // 0..3 elements left
    b       Ltail

Lcompute:
    subs    x2, x2, #4
    b.lo    Lblocks_done
Lblock:
    BINS    0, 0, 1, 2, 3
    BINS    16, 4, 5, 6, 7
    fmov    x6, d3
    mov     x7, v3.d[1]
    fmov    x8, d7
    mov     x9, v7.d[1]
    COUNT   x6, a
    COUNT   x7, b
    COUNT   x8, c
    COUNT   x9, d
    add     x0, x0, #32
    subs    x2, x2, #4
    b.hs    Lblock
Lblocks_done:
    add     x2, x2, #4              // 0..3 elements left

    // Whatever is left, one element at a time.
Ltail:
    cbz     x2, Ldone
Ltail_loop:
    ldr     d22, [x0], #8
    adr     x15, Ltail_next
    b       Lone
Ltail_next:
    subs    x2, x2, #1
    b.ne    Ltail_loop
Ldone:
    ret

    SLOW    a, 0
    SLOW    b, 8
    SLOW    c, 16
    SLOW    d, 24
    SLOW    e, 0
    SLOW    f, 8
    SLOW    g, 16
    SLOW    h, 24

    // naive.c for the single element v in d22. Returns to x15.
Lone:
    // Skip unless lo <= v <= hi. NaN fails the first compare.
    fcmp    d22, d16
    fccmp   d22, d21, #2, ge
    b.hi    Lone_done
    // b = (size_t)(((v - lo) / width) * nbins)
    fsub    d23, d22, d16
    fdiv    d23, d23, d17
    fmul    d23, d23, d18
    fcvtzu  x13, d23
    // if (b == nbins) b--
    cmp     x13, x3
    cset    w14, eq
    sub     x13, x13, x14
    // if (v < edge(b)) b--          (b < nbins here, so the edge is computed)
    ucvtf   d23, x13
    fmul    d23, d23, d19
    fadd    d23, d23, d16
    fcmp    d22, d23
    cset    w14, mi
    sub     x13, x13, x14
    // if (v >= edge(b + 1) && b != nbins - 1) b++
    add     x14, x13, #1
    ucvtf   d23, x14
    fmul    d23, d23, d19
    fadd    d23, d23, d16
    cmp     x14, x3
    fcsel   d23, d21, d23, eq       // edge(nbins) is exactly hi
    fcmp    d22, d23
    ccmp    x13, x16, #4, ge        // v >= edge: compare b with nbins - 1; else "equal"
    cinc    x13, x13, ne
    ldr     x14, [x1, x13, lsl #3]
    add     x14, x14, #1
    str     x14, [x1, x13, lsl #3]
Lone_done:
    br      x15
