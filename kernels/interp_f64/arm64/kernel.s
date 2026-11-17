// void interp_f64(const double *x, const double *xp, const double *fp, double *out, size_t n,
//                 size_t queries);
//
// np.interp(x, xp, fp) with `queries` values in x and n sample points in xp
// and fp; bit-identical to naive.c, which follows NumPy case by case.
//
// Queries are handled in groups of 64 (16 when n < 64), in two stages. A call
// with fewer queries than one group, or 3 to 15 sample points, takes the same
// two stages eight queries at a time in general registers (Lfew; one query,
// and two to four, have their own narrower copies). Two sample points go one
// query at a time through Lslow.
//
// Stage 1, the search. For each query v it finds
//     jc = number of i in [1, n-2] with xp[i] <= v
// which is NumPy's segment index j clamped to [0, n-2], so xp[jc], xp[jc+1],
// fp[jc], fp[jc+1] always exist. It is a branchless binary search over the
// m = n - 2 elements a = xp + 1, with p the largest power of two <= m:
//     first:  base = (a[p - 1] <= v) ? m - p : 0
//     then for h = p/2, p/4, ..., 1, and h = 1 once more:
//             if (a[base + h - 1] <= v) base += h
// The bases of four queries are the 32-bit lanes of one vector register, so
// 64 searches advance together and their loads overlap. One step for four
// queries moves the bases to integer registers, loads the four probes, and
// does the compare and the update as vector operations (16 instructions,
// repeated after Lg64_pass). jc is written to out as a 64-bit integer.
// When n < 64 the same count is made for sixteen queries at once by comparing
// every element of a with all of them (Lg16): with so few elements that is
// quicker than a search whose steps wait on each other.
//
// Stage 2, the arithmetic, two queries per vector, without branches:
//     t = v - xp[jc]
//     r = fma((fp[jc+1] - fp[jc]) / (xp[jc+1] - xp[jc]), t, fp[jc])
//     if !(t > 0)        r = fp[jc]      // exact hit, or below the range
//     if v >= xp[jc+1]   r = fp[jc+1]    // at or above the last sample point
// Inside the range and off a sample point, that is exactly naive.c's value.
// The rare cases are not decided here: whenever the fma result is NaN (a NaN
// query, infinities, the NaN fallbacks) the four queries of that iteration
// are recomputed by Lslow, which is naive.c line by line.
// From 1,048,576 points the fp loads are cache misses, so they are prefetched
// for the whole group as soon as stage 1 has produced the indices.
//
// Leftover queries (their number is not a multiple of the group size) are
// covered by one more group that ends exactly at the last query and
// recomputes a few results.
//
// Registers:
//   x1 xp, x2 fp, x4 n, x5 queries (on entry), x3 group size in bytes,
//   x9 group's queries,
//   x11 last group's queries, x12 out - x, x13 p/2, x14 a = xp + 1, x15 k,
//   x10 h, x17 &a[h - 1], x16 passes left, x0 and x5-x8 scratch
//   v0-v7, v16-v23 bases (stage 1), v24-v26 scratch,
//   v27 a[p - 1] (both lanes), v28 m - p (four lanes), v31 h (four lanes)
.text
.globl _interp_f64
.p2align 2
_interp_f64:
    cbz     x5, Ldone               // no queries
    cmp     x4, #1
    b.eq    Lone
    cmp     x4, #16
    b.lo    Lfew
    lsr     x6, x4, #32
    cbnz    x6, Lall_slow           // bases are 32-bit
    cmp     x5, #16
    b.lo    Lfew                    // too few queries for a group
    cmp     x4, #64
    b.lo    Lgroups
    cmp     x5, #64
    b.lo    Lfew                    // too few queries for a group of 64
Lgroups:
    mov     x11, x5                 // queries; x5 is scratch from here
    sub     x5, x4, #2              // m
    clz     x15, x5
    mov     x6, #63
    sub     x15, x6, x15            // k = floor(log2(m)), at least 3
    mov     x13, #1
    lsl     x6, x13, x15            // p
    sub     x7, x5, x6
    dup     v28.4s, w7              // m - p
    add     x14, x1, #8             // a
    add     x8, x14, x6, lsl #3
    sub     x8, x8, #8
    ld1r    {v27.2d}, [x8]          // a[p - 1]
    lsr     x13, x6, #1             // p/2
    sub     x12, x3, x0
    mov     x9, x0
    cmp     x4, #64
    b.lo    Lsetup16
    mov     x3, #512
    sub     x8, x11, #64
    add     x11, x0, x8, lsl #3     // first query of the last full group
    b       Lg64
Lsetup16:
    mov     x3, #128
    sub     x8, x11, #16
    add     x11, x0, x8, lsl #3     // first query of the last full group
    b       Lg16

Lg64:                               // stage 1 for 64 queries at x9
    ldp     q24, q25, [x9, #0]
    fcmge   v24.2d, v24.2d, v27.2d
    fcmge   v25.2d, v25.2d, v27.2d
    uzp1    v0.4s, v24.4s, v25.4s
    and     v0.16b, v0.16b, v28.16b   // (a[p - 1] <= v) ? m - p : 0
    ldp     q24, q25, [x9, #32]
    fcmge   v24.2d, v24.2d, v27.2d
    fcmge   v25.2d, v25.2d, v27.2d
    uzp1    v1.4s, v24.4s, v25.4s
    and     v1.16b, v1.16b, v28.16b   // (a[p - 1] <= v) ? m - p : 0
    ldp     q24, q25, [x9, #64]
    fcmge   v24.2d, v24.2d, v27.2d
    fcmge   v25.2d, v25.2d, v27.2d
    uzp1    v2.4s, v24.4s, v25.4s
    and     v2.16b, v2.16b, v28.16b   // (a[p - 1] <= v) ? m - p : 0
    ldp     q24, q25, [x9, #96]
    fcmge   v24.2d, v24.2d, v27.2d
    fcmge   v25.2d, v25.2d, v27.2d
    uzp1    v3.4s, v24.4s, v25.4s
    and     v3.16b, v3.16b, v28.16b   // (a[p - 1] <= v) ? m - p : 0
    ldp     q24, q25, [x9, #128]
    fcmge   v24.2d, v24.2d, v27.2d
    fcmge   v25.2d, v25.2d, v27.2d
    uzp1    v4.4s, v24.4s, v25.4s
    and     v4.16b, v4.16b, v28.16b   // (a[p - 1] <= v) ? m - p : 0
    ldp     q24, q25, [x9, #160]
    fcmge   v24.2d, v24.2d, v27.2d
    fcmge   v25.2d, v25.2d, v27.2d
    uzp1    v5.4s, v24.4s, v25.4s
    and     v5.16b, v5.16b, v28.16b   // (a[p - 1] <= v) ? m - p : 0
    ldp     q24, q25, [x9, #192]
    fcmge   v24.2d, v24.2d, v27.2d
    fcmge   v25.2d, v25.2d, v27.2d
    uzp1    v6.4s, v24.4s, v25.4s
    and     v6.16b, v6.16b, v28.16b   // (a[p - 1] <= v) ? m - p : 0
    ldp     q24, q25, [x9, #224]
    fcmge   v24.2d, v24.2d, v27.2d
    fcmge   v25.2d, v25.2d, v27.2d
    uzp1    v7.4s, v24.4s, v25.4s
    and     v7.16b, v7.16b, v28.16b   // (a[p - 1] <= v) ? m - p : 0
    ldp     q24, q25, [x9, #256]
    fcmge   v24.2d, v24.2d, v27.2d
    fcmge   v25.2d, v25.2d, v27.2d
    uzp1    v16.4s, v24.4s, v25.4s
    and     v16.16b, v16.16b, v28.16b   // (a[p - 1] <= v) ? m - p : 0
    ldp     q24, q25, [x9, #288]
    fcmge   v24.2d, v24.2d, v27.2d
    fcmge   v25.2d, v25.2d, v27.2d
    uzp1    v17.4s, v24.4s, v25.4s
    and     v17.16b, v17.16b, v28.16b   // (a[p - 1] <= v) ? m - p : 0
    ldp     q24, q25, [x9, #320]
    fcmge   v24.2d, v24.2d, v27.2d
    fcmge   v25.2d, v25.2d, v27.2d
    uzp1    v18.4s, v24.4s, v25.4s
    and     v18.16b, v18.16b, v28.16b   // (a[p - 1] <= v) ? m - p : 0
    ldp     q24, q25, [x9, #352]
    fcmge   v24.2d, v24.2d, v27.2d
    fcmge   v25.2d, v25.2d, v27.2d
    uzp1    v19.4s, v24.4s, v25.4s
    and     v19.16b, v19.16b, v28.16b   // (a[p - 1] <= v) ? m - p : 0
    ldp     q24, q25, [x9, #384]
    fcmge   v24.2d, v24.2d, v27.2d
    fcmge   v25.2d, v25.2d, v27.2d
    uzp1    v20.4s, v24.4s, v25.4s
    and     v20.16b, v20.16b, v28.16b   // (a[p - 1] <= v) ? m - p : 0
    ldp     q24, q25, [x9, #416]
    fcmge   v24.2d, v24.2d, v27.2d
    fcmge   v25.2d, v25.2d, v27.2d
    uzp1    v21.4s, v24.4s, v25.4s
    and     v21.16b, v21.16b, v28.16b   // (a[p - 1] <= v) ? m - p : 0
    ldp     q24, q25, [x9, #448]
    fcmge   v24.2d, v24.2d, v27.2d
    fcmge   v25.2d, v25.2d, v27.2d
    uzp1    v22.4s, v24.4s, v25.4s
    and     v22.16b, v22.16b, v28.16b   // (a[p - 1] <= v) ? m - p : 0
    ldp     q24, q25, [x9, #480]
    fcmge   v24.2d, v24.2d, v27.2d
    fcmge   v25.2d, v25.2d, v27.2d
    uzp1    v23.4s, v24.4s, v25.4s
    and     v23.16b, v23.16b, v28.16b   // (a[p - 1] <= v) ? m - p : 0
    mov     x10, x13
    add     x17, x14, x13, lsl #3
    sub     x17, x17, #8            // &a[p/2 - 1]
    dup     v31.4s, w13
    add     x16, x15, #1            // k steps and the last probe
Lg64_pass:
    fmov    x5, d0                // bases 0 and 1
    mov     x6, v0.d[1]              // bases 2 and 3
    lsr     x7, x5, #32
    lsr     x8, x6, #32
    ldr     d24, [x17, w5, uxtw #3]
    ldr     d25, [x17, x7, lsl #3]
    ins     v24.d[1], v25.d[0]
    ldr     d25, [x17, w6, uxtw #3]
    ldr     d26, [x17, x8, lsl #3]
    ins     v25.d[1], v26.d[0]
    ldr     q26, [x9, #0]
    fcmge   v24.2d, v26.2d, v24.2d
    ldr     q26, [x9, #16]
    fcmge   v25.2d, v26.2d, v25.2d
    uzp1    v24.4s, v24.4s, v25.4s  // four 32-bit masks
    mls     v0.4s, v24.4s, v31.4s    // base += h where the mask is -1
    fmov    x5, d1                // bases 0 and 1
    mov     x6, v1.d[1]              // bases 2 and 3
    lsr     x7, x5, #32
    lsr     x8, x6, #32
    ldr     d24, [x17, w5, uxtw #3]
    ldr     d25, [x17, x7, lsl #3]
    ins     v24.d[1], v25.d[0]
    ldr     d25, [x17, w6, uxtw #3]
    ldr     d26, [x17, x8, lsl #3]
    ins     v25.d[1], v26.d[0]
    ldr     q26, [x9, #32]
    fcmge   v24.2d, v26.2d, v24.2d
    ldr     q26, [x9, #48]
    fcmge   v25.2d, v26.2d, v25.2d
    uzp1    v24.4s, v24.4s, v25.4s  // four 32-bit masks
    mls     v1.4s, v24.4s, v31.4s    // base += h where the mask is -1
    fmov    x5, d2                // bases 0 and 1
    mov     x6, v2.d[1]              // bases 2 and 3
    lsr     x7, x5, #32
    lsr     x8, x6, #32
    ldr     d24, [x17, w5, uxtw #3]
    ldr     d25, [x17, x7, lsl #3]
    ins     v24.d[1], v25.d[0]
    ldr     d25, [x17, w6, uxtw #3]
    ldr     d26, [x17, x8, lsl #3]
    ins     v25.d[1], v26.d[0]
    ldr     q26, [x9, #64]
    fcmge   v24.2d, v26.2d, v24.2d
    ldr     q26, [x9, #80]
    fcmge   v25.2d, v26.2d, v25.2d
    uzp1    v24.4s, v24.4s, v25.4s  // four 32-bit masks
    mls     v2.4s, v24.4s, v31.4s    // base += h where the mask is -1
    fmov    x5, d3                // bases 0 and 1
    mov     x6, v3.d[1]              // bases 2 and 3
    lsr     x7, x5, #32
    lsr     x8, x6, #32
    ldr     d24, [x17, w5, uxtw #3]
    ldr     d25, [x17, x7, lsl #3]
    ins     v24.d[1], v25.d[0]
    ldr     d25, [x17, w6, uxtw #3]
    ldr     d26, [x17, x8, lsl #3]
    ins     v25.d[1], v26.d[0]
    ldr     q26, [x9, #96]
    fcmge   v24.2d, v26.2d, v24.2d
    ldr     q26, [x9, #112]
    fcmge   v25.2d, v26.2d, v25.2d
    uzp1    v24.4s, v24.4s, v25.4s  // four 32-bit masks
    mls     v3.4s, v24.4s, v31.4s    // base += h where the mask is -1
    fmov    x5, d4                // bases 0 and 1
    mov     x6, v4.d[1]              // bases 2 and 3
    lsr     x7, x5, #32
    lsr     x8, x6, #32
    ldr     d24, [x17, w5, uxtw #3]
    ldr     d25, [x17, x7, lsl #3]
    ins     v24.d[1], v25.d[0]
    ldr     d25, [x17, w6, uxtw #3]
    ldr     d26, [x17, x8, lsl #3]
    ins     v25.d[1], v26.d[0]
    ldr     q26, [x9, #128]
    fcmge   v24.2d, v26.2d, v24.2d
    ldr     q26, [x9, #144]
    fcmge   v25.2d, v26.2d, v25.2d
    uzp1    v24.4s, v24.4s, v25.4s  // four 32-bit masks
    mls     v4.4s, v24.4s, v31.4s    // base += h where the mask is -1
    fmov    x5, d5                // bases 0 and 1
    mov     x6, v5.d[1]              // bases 2 and 3
    lsr     x7, x5, #32
    lsr     x8, x6, #32
    ldr     d24, [x17, w5, uxtw #3]
    ldr     d25, [x17, x7, lsl #3]
    ins     v24.d[1], v25.d[0]
    ldr     d25, [x17, w6, uxtw #3]
    ldr     d26, [x17, x8, lsl #3]
    ins     v25.d[1], v26.d[0]
    ldr     q26, [x9, #160]
    fcmge   v24.2d, v26.2d, v24.2d
    ldr     q26, [x9, #176]
    fcmge   v25.2d, v26.2d, v25.2d
    uzp1    v24.4s, v24.4s, v25.4s  // four 32-bit masks
    mls     v5.4s, v24.4s, v31.4s    // base += h where the mask is -1
    fmov    x5, d6                // bases 0 and 1
    mov     x6, v6.d[1]              // bases 2 and 3
    lsr     x7, x5, #32
    lsr     x8, x6, #32
    ldr     d24, [x17, w5, uxtw #3]
    ldr     d25, [x17, x7, lsl #3]
    ins     v24.d[1], v25.d[0]
    ldr     d25, [x17, w6, uxtw #3]
    ldr     d26, [x17, x8, lsl #3]
    ins     v25.d[1], v26.d[0]
    ldr     q26, [x9, #192]
    fcmge   v24.2d, v26.2d, v24.2d
    ldr     q26, [x9, #208]
    fcmge   v25.2d, v26.2d, v25.2d
    uzp1    v24.4s, v24.4s, v25.4s  // four 32-bit masks
    mls     v6.4s, v24.4s, v31.4s    // base += h where the mask is -1
    fmov    x5, d7                // bases 0 and 1
    mov     x6, v7.d[1]              // bases 2 and 3
    lsr     x7, x5, #32
    lsr     x8, x6, #32
    ldr     d24, [x17, w5, uxtw #3]
    ldr     d25, [x17, x7, lsl #3]
    ins     v24.d[1], v25.d[0]
    ldr     d25, [x17, w6, uxtw #3]
    ldr     d26, [x17, x8, lsl #3]
    ins     v25.d[1], v26.d[0]
    ldr     q26, [x9, #224]
    fcmge   v24.2d, v26.2d, v24.2d
    ldr     q26, [x9, #240]
    fcmge   v25.2d, v26.2d, v25.2d
    uzp1    v24.4s, v24.4s, v25.4s  // four 32-bit masks
    mls     v7.4s, v24.4s, v31.4s    // base += h where the mask is -1
    fmov    x5, d16                // bases 0 and 1
    mov     x6, v16.d[1]              // bases 2 and 3
    lsr     x7, x5, #32
    lsr     x8, x6, #32
    ldr     d24, [x17, w5, uxtw #3]
    ldr     d25, [x17, x7, lsl #3]
    ins     v24.d[1], v25.d[0]
    ldr     d25, [x17, w6, uxtw #3]
    ldr     d26, [x17, x8, lsl #3]
    ins     v25.d[1], v26.d[0]
    ldr     q26, [x9, #256]
    fcmge   v24.2d, v26.2d, v24.2d
    ldr     q26, [x9, #272]
    fcmge   v25.2d, v26.2d, v25.2d
    uzp1    v24.4s, v24.4s, v25.4s  // four 32-bit masks
    mls     v16.4s, v24.4s, v31.4s    // base += h where the mask is -1
    fmov    x5, d17                // bases 0 and 1
    mov     x6, v17.d[1]              // bases 2 and 3
    lsr     x7, x5, #32
    lsr     x8, x6, #32
    ldr     d24, [x17, w5, uxtw #3]
    ldr     d25, [x17, x7, lsl #3]
    ins     v24.d[1], v25.d[0]
    ldr     d25, [x17, w6, uxtw #3]
    ldr     d26, [x17, x8, lsl #3]
    ins     v25.d[1], v26.d[0]
    ldr     q26, [x9, #288]
    fcmge   v24.2d, v26.2d, v24.2d
    ldr     q26, [x9, #304]
    fcmge   v25.2d, v26.2d, v25.2d
    uzp1    v24.4s, v24.4s, v25.4s  // four 32-bit masks
    mls     v17.4s, v24.4s, v31.4s    // base += h where the mask is -1
    fmov    x5, d18                // bases 0 and 1
    mov     x6, v18.d[1]              // bases 2 and 3
    lsr     x7, x5, #32
    lsr     x8, x6, #32
    ldr     d24, [x17, w5, uxtw #3]
    ldr     d25, [x17, x7, lsl #3]
    ins     v24.d[1], v25.d[0]
    ldr     d25, [x17, w6, uxtw #3]
    ldr     d26, [x17, x8, lsl #3]
    ins     v25.d[1], v26.d[0]
    ldr     q26, [x9, #320]
    fcmge   v24.2d, v26.2d, v24.2d
    ldr     q26, [x9, #336]
    fcmge   v25.2d, v26.2d, v25.2d
    uzp1    v24.4s, v24.4s, v25.4s  // four 32-bit masks
    mls     v18.4s, v24.4s, v31.4s    // base += h where the mask is -1
    fmov    x5, d19                // bases 0 and 1
    mov     x6, v19.d[1]              // bases 2 and 3
    lsr     x7, x5, #32
    lsr     x8, x6, #32
    ldr     d24, [x17, w5, uxtw #3]
    ldr     d25, [x17, x7, lsl #3]
    ins     v24.d[1], v25.d[0]
    ldr     d25, [x17, w6, uxtw #3]
    ldr     d26, [x17, x8, lsl #3]
    ins     v25.d[1], v26.d[0]
    ldr     q26, [x9, #352]
    fcmge   v24.2d, v26.2d, v24.2d
    ldr     q26, [x9, #368]
    fcmge   v25.2d, v26.2d, v25.2d
    uzp1    v24.4s, v24.4s, v25.4s  // four 32-bit masks
    mls     v19.4s, v24.4s, v31.4s    // base += h where the mask is -1
    fmov    x5, d20                // bases 0 and 1
    mov     x6, v20.d[1]              // bases 2 and 3
    lsr     x7, x5, #32
    lsr     x8, x6, #32
    ldr     d24, [x17, w5, uxtw #3]
    ldr     d25, [x17, x7, lsl #3]
    ins     v24.d[1], v25.d[0]
    ldr     d25, [x17, w6, uxtw #3]
    ldr     d26, [x17, x8, lsl #3]
    ins     v25.d[1], v26.d[0]
    ldr     q26, [x9, #384]
    fcmge   v24.2d, v26.2d, v24.2d
    ldr     q26, [x9, #400]
    fcmge   v25.2d, v26.2d, v25.2d
    uzp1    v24.4s, v24.4s, v25.4s  // four 32-bit masks
    mls     v20.4s, v24.4s, v31.4s    // base += h where the mask is -1
    fmov    x5, d21                // bases 0 and 1
    mov     x6, v21.d[1]              // bases 2 and 3
    lsr     x7, x5, #32
    lsr     x8, x6, #32
    ldr     d24, [x17, w5, uxtw #3]
    ldr     d25, [x17, x7, lsl #3]
    ins     v24.d[1], v25.d[0]
    ldr     d25, [x17, w6, uxtw #3]
    ldr     d26, [x17, x8, lsl #3]
    ins     v25.d[1], v26.d[0]
    ldr     q26, [x9, #416]
    fcmge   v24.2d, v26.2d, v24.2d
    ldr     q26, [x9, #432]
    fcmge   v25.2d, v26.2d, v25.2d
    uzp1    v24.4s, v24.4s, v25.4s  // four 32-bit masks
    mls     v21.4s, v24.4s, v31.4s    // base += h where the mask is -1
    fmov    x5, d22                // bases 0 and 1
    mov     x6, v22.d[1]              // bases 2 and 3
    lsr     x7, x5, #32
    lsr     x8, x6, #32
    ldr     d24, [x17, w5, uxtw #3]
    ldr     d25, [x17, x7, lsl #3]
    ins     v24.d[1], v25.d[0]
    ldr     d25, [x17, w6, uxtw #3]
    ldr     d26, [x17, x8, lsl #3]
    ins     v25.d[1], v26.d[0]
    ldr     q26, [x9, #448]
    fcmge   v24.2d, v26.2d, v24.2d
    ldr     q26, [x9, #464]
    fcmge   v25.2d, v26.2d, v25.2d
    uzp1    v24.4s, v24.4s, v25.4s  // four 32-bit masks
    mls     v22.4s, v24.4s, v31.4s    // base += h where the mask is -1
    fmov    x5, d23                // bases 0 and 1
    mov     x6, v23.d[1]              // bases 2 and 3
    lsr     x7, x5, #32
    lsr     x8, x6, #32
    ldr     d24, [x17, w5, uxtw #3]
    ldr     d25, [x17, x7, lsl #3]
    ins     v24.d[1], v25.d[0]
    ldr     d25, [x17, w6, uxtw #3]
    ldr     d26, [x17, x8, lsl #3]
    ins     v25.d[1], v26.d[0]
    ldr     q26, [x9, #480]
    fcmge   v24.2d, v26.2d, v24.2d
    ldr     q26, [x9, #496]
    fcmge   v25.2d, v26.2d, v25.2d
    uzp1    v24.4s, v24.4s, v25.4s  // four 32-bit masks
    mls     v23.4s, v24.4s, v31.4s    // base += h where the mask is -1
    lsr     x7, x10, #1
    cmp     x7, #0
    csinc   x10, x7, xzr, ne        // h = max(h / 2, 1): the last probe is h = 1 again
    add     x17, x14, x10, lsl #3
    sub     x17, x17, #8
    dup     v31.4s, w10
    subs    x16, x16, #1
    b.ne    Lg64_pass
    add     x6, x9, x12             // group's out
    ushll   v24.2d, v0.2s, #0
    ushll2  v25.2d, v0.4s, #0
    stp     q24, q25, [x6, #0]
    ushll   v24.2d, v1.2s, #0
    ushll2  v25.2d, v1.4s, #0
    stp     q24, q25, [x6, #32]
    ushll   v24.2d, v2.2s, #0
    ushll2  v25.2d, v2.4s, #0
    stp     q24, q25, [x6, #64]
    ushll   v24.2d, v3.2s, #0
    ushll2  v25.2d, v3.4s, #0
    stp     q24, q25, [x6, #96]
    ushll   v24.2d, v4.2s, #0
    ushll2  v25.2d, v4.4s, #0
    stp     q24, q25, [x6, #128]
    ushll   v24.2d, v5.2s, #0
    ushll2  v25.2d, v5.4s, #0
    stp     q24, q25, [x6, #160]
    ushll   v24.2d, v6.2s, #0
    ushll2  v25.2d, v6.4s, #0
    stp     q24, q25, [x6, #192]
    ushll   v24.2d, v7.2s, #0
    ushll2  v25.2d, v7.4s, #0
    stp     q24, q25, [x6, #224]
    ushll   v24.2d, v16.2s, #0
    ushll2  v25.2d, v16.4s, #0
    stp     q24, q25, [x6, #256]
    ushll   v24.2d, v17.2s, #0
    ushll2  v25.2d, v17.4s, #0
    stp     q24, q25, [x6, #288]
    ushll   v24.2d, v18.2s, #0
    ushll2  v25.2d, v18.4s, #0
    stp     q24, q25, [x6, #320]
    ushll   v24.2d, v19.2s, #0
    ushll2  v25.2d, v19.4s, #0
    stp     q24, q25, [x6, #352]
    ushll   v24.2d, v20.2s, #0
    ushll2  v25.2d, v20.4s, #0
    stp     q24, q25, [x6, #384]
    ushll   v24.2d, v21.2s, #0
    ushll2  v25.2d, v21.4s, #0
    stp     q24, q25, [x6, #416]
    ushll   v24.2d, v22.2s, #0
    ushll2  v25.2d, v22.4s, #0
    stp     q24, q25, [x6, #448]
    ushll   v24.2d, v23.2s, #0
    ushll2  v25.2d, v23.4s, #0
    stp     q24, q25, [x6, #480]
    mov     x5, x9
    mov     x7, #64
    // Far beyond the cache the fp loads of stage 2 are misses: start
    // them all now. (Below that the extra loads only cost time.)
    mov     x8, #1048576
    cmp     x4, x8
    b.lo    Lp2
    mov     x8, #0
Lg64_pf:
    add     x0, x6, x8
    ldp     x10, x16, [x0]
    ldp     x17, x0, [x0, #16]
    prfm    pldl1keep, [x2, x10, lsl #3]
    prfm    pldl1keep, [x2, x16, lsl #3]
    prfm    pldl1keep, [x2, x17, lsl #3]
    prfm    pldl1keep, [x2, x0, lsl #3]
    add     x8, x8, #32
    cmp     x8, #512
    b.lo    Lg64_pf
    b       Lp2

    // 16 <= n < 64: for sixteen queries at once, compare every element of a
    // with all of them and count. No step depends on the one before, so this
    // is quicker than a binary search while a is this short.
    // v16-v23 queries, v0-v7 counts, v24-v26 scratch
Lg16:
    ldp     q16, q17, [x9]
    ldp     q18, q19, [x9, #32]
    ldp     q20, q21, [x9, #64]
    ldp     q22, q23, [x9, #96]
    movi    v0.2d, #0
    movi    v1.2d, #0
    movi    v2.2d, #0
    movi    v3.2d, #0
    movi    v4.2d, #0
    movi    v5.2d, #0
    movi    v6.2d, #0
    movi    v7.2d, #0
    mov     x5, x14
    sub     x6, x4, #2              // m
Lg16_element:
    ld1r    {v24.2d}, [x5], #8
    fcmge   v25.2d, v16.2d, v24.2d
    sub     v0.2d, v0.2d, v25.2d     // count += 1 where a[i] <= v
    fcmge   v26.2d, v17.2d, v24.2d
    sub     v1.2d, v1.2d, v26.2d     // count += 1 where a[i] <= v
    fcmge   v25.2d, v18.2d, v24.2d
    sub     v2.2d, v2.2d, v25.2d     // count += 1 where a[i] <= v
    fcmge   v26.2d, v19.2d, v24.2d
    sub     v3.2d, v3.2d, v26.2d     // count += 1 where a[i] <= v
    fcmge   v25.2d, v20.2d, v24.2d
    sub     v4.2d, v4.2d, v25.2d     // count += 1 where a[i] <= v
    fcmge   v26.2d, v21.2d, v24.2d
    sub     v5.2d, v5.2d, v26.2d     // count += 1 where a[i] <= v
    fcmge   v25.2d, v22.2d, v24.2d
    sub     v6.2d, v6.2d, v25.2d     // count += 1 where a[i] <= v
    fcmge   v26.2d, v23.2d, v24.2d
    sub     v7.2d, v7.2d, v26.2d     // count += 1 where a[i] <= v
    subs    x6, x6, #1
    b.ne    Lg16_element
    add     x6, x9, x12             // group's out
    stp     q0, q1, [x6]
    stp     q2, q3, [x6, #32]
    stp     q4, q5, [x6, #64]
    stp     q6, q7, [x6, #96]
    mov     x5, x9
    mov     x7, #16
    b       Lp2

    // Stage 2 for x7 queries (a multiple of 4) at x5, with jc in out at x6.
    // Two queries per vector.
Lp2:
    movi    v24.2d, #0
    ldp     x8, x10, [x6, #0]       // jc of the two queries
    ldr     q0, [x5, #0]            // v
    add     x16, x1, x8, lsl #3
    add     x17, x2, x8, lsl #3
    ldr     q5, [x16]               // xp[jc], xp[jc+1] of the first query
    ldr     q7, [x17]               // fp[jc], fp[jc+1]
    add     x16, x1, x10, lsl #3
    add     x17, x2, x10, lsl #3
    ldr     q6, [x16]               // and of the second
    ldr     q16, [x17]
    zip1    v1.2d, v5.2d, v6.2d     // xp[jc] of both
    zip2    v2.2d, v5.2d, v6.2d     // xp[jc+1]
    zip1    v3.2d, v7.2d, v16.2d    // fp[jc]
    zip2    v4.2d, v7.2d, v16.2d    // fp[jc+1]
    fsub    v5.2d, v2.2d, v1.2d
    fsub    v6.2d, v4.2d, v3.2d
    fsub    v7.2d, v0.2d, v1.2d     // t = v - xp[jc]
    fdiv    v6.2d, v6.2d, v5.2d     // slope
    mov     v5.16b, v3.16b
    fmla    v5.2d, v6.2d, v7.2d     // slope * t + fp[jc], fused
    fmax    v24.2d, v24.2d, v5.2d   // stays NaN once any fma result is NaN
    fcmgt   v7.2d, v7.2d, #0.0
    fcmge   v0.2d, v0.2d, v2.2d
    bif     v5.16b, v3.16b, v7.16b  // !(t > 0): fp[jc]
    bit     v5.16b, v4.16b, v0.16b  // v >= xp[jc+1]: fp[jc+1]
    str     q5, [x6, #0]
    ldp     x8, x10, [x6, #16]       // jc of the two queries
    ldr     q0, [x5, #16]            // v
    add     x16, x1, x8, lsl #3
    add     x17, x2, x8, lsl #3
    ldr     q5, [x16]               // xp[jc], xp[jc+1] of the first query
    ldr     q7, [x17]               // fp[jc], fp[jc+1]
    add     x16, x1, x10, lsl #3
    add     x17, x2, x10, lsl #3
    ldr     q6, [x16]               // and of the second
    ldr     q16, [x17]
    zip1    v1.2d, v5.2d, v6.2d     // xp[jc] of both
    zip2    v2.2d, v5.2d, v6.2d     // xp[jc+1]
    zip1    v3.2d, v7.2d, v16.2d    // fp[jc]
    zip2    v4.2d, v7.2d, v16.2d    // fp[jc+1]
    fsub    v5.2d, v2.2d, v1.2d
    fsub    v6.2d, v4.2d, v3.2d
    fsub    v7.2d, v0.2d, v1.2d     // t = v - xp[jc]
    fdiv    v6.2d, v6.2d, v5.2d     // slope
    mov     v5.16b, v3.16b
    fmla    v5.2d, v6.2d, v7.2d     // slope * t + fp[jc], fused
    fmax    v24.2d, v24.2d, v5.2d   // stays NaN once any fma result is NaN
    fcmgt   v7.2d, v7.2d, #0.0
    fcmge   v0.2d, v0.2d, v2.2d
    bif     v5.16b, v3.16b, v7.16b  // !(t > 0): fp[jc]
    bit     v5.16b, v4.16b, v0.16b  // v >= xp[jc+1]: fp[jc+1]
    str     q5, [x6, #16]
    fmaxp   d24, v24.2d
    fcmp    d24, d24
    b.vs    Lp2_redo                // some fma result was NaN
Lp2_next:

    add     x5, x5, #32
    add     x6, x6, #32
    subs    x7, x7, #4
    b.ne    Lp2
    // Next group.
    add     x9, x9, x3
    cmp     x9, x11
    b.ls    Lagain
    add     x8, x11, x3             // end of the queries
    cmp     x9, x8
    b.eq    Ldone
    mov     x9, x11                 // leftovers: redo the last group's worth
Lagain:
    cmp     x3, #512
    b.eq    Lg64
    b       Lg16
Ldone:
    ret

    // ---- Fewer queries than a group: eight at once, in general registers.
    // Taken by every call with fewer queries than its group (fewer than 16,
    // or fewer than 64 from 64 sample points) and every call with 3 to 15
    // sample points. Stage 1 is the search at the top of the file, one fcmp +
    // csel per probe, with the eight bases in x5-x12 and the step size
    // shared. Stage 2 is the same arithmetic, one query at a time, and the
    // result replaces the query in its register. From eight queries, groups
    // of eight, leftovers covered by redoing the last eight. Below eight, the
    // missing lanes repeat the last query and only the real results are
    // stored, so no buffer is needed. Whenever an fma result is NaN, Lslow
    // takes over from this group to the end of the call. One query, and two
    // to four, have their own copies of the two stages (Lfew_one, Lfew4):
    // padding them to eight lanes costs up to twice their time.
    // x0 group's first query, x3 out - x, x14 the last full group's first
    // query (or the number of queries when there are fewer than eight),
    // x5-x12 bases, x13 h, x16 passes left, x17 &a[h - 1], x15 scratch;
    // d0-d7 queries then results, d16-d23 scratch, d24 NaN detector
Lfew:
    cmp     x4, #3
    b.lo    Lall_slow               // two points: nothing to search
    cmp     x5, #1
    b.eq    Lfew_one
    cmp     x5, #4
    b.ls    Lfew4
    mov     x14, x5                 // queries
    sub     x3, x3, x0              // out - x
    cmp     x14, #8
    b.hs    Lfew_full
    sub     x15, x14, #1            // last query; lanes past it repeat it
    ldp     d0, d1, [x0]            // five to seven queries: 0-4 exist
    ldp     d2, d3, [x0, #16]
    ldr     d4, [x0, #32]
    mov     x16, #5
    cmp     x16, x15
    csel    x16, x16, x15, ls
    ldr     d5, [x0, x16, lsl #3]
    mov     x16, #6
    cmp     x16, x15
    csel    x16, x16, x15, ls
    ldr     d6, [x0, x16, lsl #3]
    mov     x16, #7
    cmp     x16, x15
    csel    x16, x16, x15, ls
    ldr     d7, [x0, x16, lsl #3]
    b       Lfew_search
Lfew_one:                           // one query: the same search for it alone
    ldr     d0, [x0]
    sub     x15, x4, #2             // m
    clz     x16, x15
    mov     x17, #63
    sub     x17, x17, x16           // k = floor(log2(m))
    mov     x13, #1
    lsl     x13, x13, x17           // p
    sub     x15, x15, x13           // m - p
    ldr     d16, [x1, x13, lsl #3]  // a[p - 1] = xp[p]
    mov     x17, #64
    sub     x16, x17, x16           // k steps and the last probe
    fcmp    d16, d0
    csel    x5, x15, xzr, ls        // base = (a[p - 1] <= v) ? m - p : 0
    lsr     x13, x13, #1
    cmp     x13, #0
    csinc   x13, x13, xzr, ne       // h = max(p / 2, 1)
Lfew_one_pass:
    add     x17, x1, x13, lsl #3    // &a[h - 1] = xp + h
    ldr     d16, [x17, x5, lsl #3]
    fcmp    d16, d0
    add     x15, x5, x13
    csel    x5, x15, x5, ls         // base += h if a[base + h - 1] <= v
    lsr     x15, x13, #1
    cmp     x15, #0
    csinc   x13, x15, xzr, ne       // h = max(h / 2, 1): the last probe is h = 1 again
    subs    x16, x16, #1
    b.ne    Lfew_one_pass
    add     x15, x1, x5, lsl #3
    ldp     d16, d17, [x15]         // xp[jc], xp[jc+1]
    add     x15, x2, x5, lsl #3
    ldp     d18, d19, [x15]         // fp[jc], fp[jc+1]
    fsub    d20, d17, d16
    fsub    d21, d19, d18
    fsub    d22, d0, d16            // t = v - xp[jc]
    fdiv    d21, d21, d20           // slope
    fmadd   d23, d21, d22, d18      // slope * t + fp[jc], fused
    fcmp    d23, d23
    b.vs    Lfew_one_redo           // NaN: naive.c decides
    fcmp    d22, #0.0
    fcsel   d23, d23, d18, gt       // !(t > 0): fp[jc]
    fcmp    d0, d17
    fcsel   d0, d19, d23, ge        // v >= xp[jc+1]: fp[jc+1]
    str     d0, [x3]
    ret
Lfew_one_redo:
    mov     x5, #1
    b       Lall_slow
Lfew4:                              // 2 to 4 queries: four lanes, the last query repeated
    mov     x14, x5                 // queries
    sub     x15, x14, #1
    ldr     d0, [x0]
    ldr     d1, [x0, #8]
    mov     x16, #2
    cmp     x16, x15
    csel    x16, x16, x15, ls
    ldr     d2, [x0, x16, lsl #3]
    mov     x16, #3
    cmp     x16, x15
    csel    x16, x16, x15, ls
    ldr     d3, [x0, x16, lsl #3]
    sub     x15, x4, #2             // m
    clz     x16, x15
    mov     x17, #63
    sub     x17, x17, x16           // k = floor(log2(m))
    mov     x13, #1
    lsl     x13, x13, x17           // p
    sub     x15, x15, x13           // m - p
    ldr     d16, [x1, x13, lsl #3]  // a[p - 1] = xp[p]
    mov     x17, #64
    sub     x16, x17, x16           // k steps and the last probe
    fcmp    d16, d0
    csel    x5, x15, xzr, ls
    fcmp    d16, d1
    csel    x6, x15, xzr, ls
    fcmp    d16, d2
    csel    x7, x15, xzr, ls
    fcmp    d16, d3
    csel    x8, x15, xzr, ls
    lsr     x13, x13, #1
    cmp     x13, #0
    csinc   x13, x13, xzr, ne       // h = max(p / 2, 1)
Lfew4_pass:
    add     x17, x1, x13, lsl #3    // &a[h - 1] = xp + h
    ldr     d16, [x17, x5, lsl #3]
    fcmp    d16, d0
    add     x15, x5, x13
    csel    x5, x15, x5, ls
    ldr     d17, [x17, x6, lsl #3]
    fcmp    d17, d1
    add     x15, x6, x13
    csel    x6, x15, x6, ls
    ldr     d16, [x17, x7, lsl #3]
    fcmp    d16, d2
    add     x15, x7, x13
    csel    x7, x15, x7, ls
    ldr     d17, [x17, x8, lsl #3]
    fcmp    d17, d3
    add     x15, x8, x13
    csel    x8, x15, x8, ls
    lsr     x15, x13, #1
    cmp     x15, #0
    csinc   x13, x15, xzr, ne       // h = max(h / 2, 1): the last probe is h = 1 again
    subs    x16, x16, #1
    b.ne    Lfew4_pass
    movi    d24, #0
    add     x15, x1, x5, lsl #3
    ldp     d16, d17, [x15]
    add     x15, x2, x5, lsl #3
    ldp     d18, d19, [x15]
    fsub    d20, d17, d16
    fsub    d21, d19, d18
    fsub    d22, d0, d16
    fdiv    d21, d21, d20
    fmadd   d23, d21, d22, d18
    fmax    d24, d24, d23
    fcmp    d22, #0.0
    fcsel   d23, d23, d18, gt
    fcmp    d0, d17
    fcsel   d0, d19, d23, ge
    add     x15, x1, x6, lsl #3
    ldp     d16, d17, [x15]
    add     x15, x2, x6, lsl #3
    ldp     d18, d19, [x15]
    fsub    d20, d17, d16
    fsub    d21, d19, d18
    fsub    d22, d1, d16
    fdiv    d21, d21, d20
    fmadd   d23, d21, d22, d18
    fmax    d24, d24, d23
    fcmp    d22, #0.0
    fcsel   d23, d23, d18, gt
    fcmp    d1, d17
    fcsel   d1, d19, d23, ge
    add     x15, x1, x7, lsl #3
    ldp     d16, d17, [x15]
    add     x15, x2, x7, lsl #3
    ldp     d18, d19, [x15]
    fsub    d20, d17, d16
    fsub    d21, d19, d18
    fsub    d22, d2, d16
    fdiv    d21, d21, d20
    fmadd   d23, d21, d22, d18
    fmax    d24, d24, d23
    fcmp    d22, #0.0
    fcsel   d23, d23, d18, gt
    fcmp    d2, d17
    fcsel   d2, d19, d23, ge
    add     x15, x1, x8, lsl #3
    ldp     d16, d17, [x15]
    add     x15, x2, x8, lsl #3
    ldp     d18, d19, [x15]
    fsub    d20, d17, d16
    fsub    d21, d19, d18
    fsub    d22, d3, d16
    fdiv    d21, d21, d20
    fmadd   d23, d21, d22, d18
    fmax    d24, d24, d23
    fcmp    d22, #0.0
    fcsel   d23, d23, d18, gt
    fcmp    d3, d17
    fcsel   d3, d19, d23, ge
    fcmp    d24, d24
    b.vs    Lfew4_redo
    stp     d0, d1, [x3]
    cmp     x14, #3
    b.lo    Ldone
    str     d2, [x3, #16]
    b.eq    Ldone
    str     d3, [x3, #24]
    ret
Lfew4_redo:
    mov     x5, x14
    b       Lall_slow

Lfew_full:
    sub     x15, x14, #8
    add     x14, x0, x15, lsl #3    // first query of the last full group
Lfew_load:
    ldp     d0, d1, [x0]
    ldp     d2, d3, [x0, #16]
    ldp     d4, d5, [x0, #32]
    ldp     d6, d7, [x0, #48]
Lfew_search:
    sub     x15, x4, #2             // m
    clz     x16, x15
    mov     x17, #63
    sub     x17, x17, x16           // k = floor(log2(m))
    mov     x13, #1
    lsl     x13, x13, x17           // p
    sub     x15, x15, x13           // m - p
    ldr     d16, [x1, x13, lsl #3]  // a[p - 1] = xp[p]
    mov     x17, #64
    sub     x16, x17, x16           // k steps and the last probe: 64 - clz(m)
    fcmp    d16, d0
    csel    x5, x15, xzr, ls        // base = (a[p - 1] <= v) ? m - p : 0
    fcmp    d16, d1
    csel    x6, x15, xzr, ls
    fcmp    d16, d2
    csel    x7, x15, xzr, ls
    fcmp    d16, d3
    csel    x8, x15, xzr, ls
    fcmp    d16, d4
    csel    x9, x15, xzr, ls
    fcmp    d16, d5
    csel    x10, x15, xzr, ls
    fcmp    d16, d6
    csel    x11, x15, xzr, ls
    fcmp    d16, d7
    csel    x12, x15, xzr, ls
    lsr     x13, x13, #1
    cmp     x13, #0
    csinc   x13, x13, xzr, ne       // h = max(p / 2, 1)
Lfew_pass:
    add     x17, x1, x13, lsl #3    // &a[h - 1] = xp + h
    ldr     d16, [x17, x5, lsl #3]
    fcmp    d16, d0
    add     x15, x5, x13
    csel    x5, x15, x5, ls       // base += h if a[base + h - 1] <= v
    ldr     d17, [x17, x6, lsl #3]
    fcmp    d17, d1
    add     x15, x6, x13
    csel    x6, x15, x6, ls
    ldr     d16, [x17, x7, lsl #3]
    fcmp    d16, d2
    add     x15, x7, x13
    csel    x7, x15, x7, ls
    ldr     d17, [x17, x8, lsl #3]
    fcmp    d17, d3
    add     x15, x8, x13
    csel    x8, x15, x8, ls
    ldr     d16, [x17, x9, lsl #3]
    fcmp    d16, d4
    add     x15, x9, x13
    csel    x9, x15, x9, ls
    ldr     d17, [x17, x10, lsl #3]
    fcmp    d17, d5
    add     x15, x10, x13
    csel    x10, x15, x10, ls
    ldr     d16, [x17, x11, lsl #3]
    fcmp    d16, d6
    add     x15, x11, x13
    csel    x11, x15, x11, ls
    ldr     d17, [x17, x12, lsl #3]
    fcmp    d17, d7
    add     x15, x12, x13
    csel    x12, x15, x12, ls
    lsr     x15, x13, #1
    cmp     x15, #0
    csinc   x13, x15, xzr, ne       // h = max(h / 2, 1): the last probe is h = 1 again
    subs    x16, x16, #1
    b.ne    Lfew_pass
    movi    d24, #0
    add     x15, x1, x5, lsl #3
    ldp     d16, d17, [x15]  // xp[jc], xp[jc+1]
    add     x15, x2, x5, lsl #3
    ldp     d18, d19, [x15]  // fp[jc], fp[jc+1]
    fsub    d20, d17, d16
    fsub    d21, d19, d18
    fsub    d22, d0, d16  // t = v - xp[jc]
    fdiv    d21, d21, d20  // slope
    fmadd   d23, d21, d22, d18  // slope * t + fp[jc], fused
    fmax    d24, d24, d23  // stays NaN once any fma result is NaN
    fcmp    d22, #0.0
    fcsel   d23, d23, d18, gt  // !(t > 0): fp[jc]
    fcmp    d0, d17
    fcsel   d0, d19, d23, ge  // v >= xp[jc+1]: fp[jc+1]
    add     x15, x1, x6, lsl #3
    ldp     d16, d17, [x15]
    add     x15, x2, x6, lsl #3
    ldp     d18, d19, [x15]
    fsub    d20, d17, d16
    fsub    d21, d19, d18
    fsub    d22, d1, d16
    fdiv    d21, d21, d20
    fmadd   d23, d21, d22, d18
    fmax    d24, d24, d23
    fcmp    d22, #0.0
    fcsel   d23, d23, d18, gt
    fcmp    d1, d17
    fcsel   d1, d19, d23, ge
    add     x15, x1, x7, lsl #3
    ldp     d16, d17, [x15]
    add     x15, x2, x7, lsl #3
    ldp     d18, d19, [x15]
    fsub    d20, d17, d16
    fsub    d21, d19, d18
    fsub    d22, d2, d16
    fdiv    d21, d21, d20
    fmadd   d23, d21, d22, d18
    fmax    d24, d24, d23
    fcmp    d22, #0.0
    fcsel   d23, d23, d18, gt
    fcmp    d2, d17
    fcsel   d2, d19, d23, ge
    add     x15, x1, x8, lsl #3
    ldp     d16, d17, [x15]
    add     x15, x2, x8, lsl #3
    ldp     d18, d19, [x15]
    fsub    d20, d17, d16
    fsub    d21, d19, d18
    fsub    d22, d3, d16
    fdiv    d21, d21, d20
    fmadd   d23, d21, d22, d18
    fmax    d24, d24, d23
    fcmp    d22, #0.0
    fcsel   d23, d23, d18, gt
    fcmp    d3, d17
    fcsel   d3, d19, d23, ge
    add     x15, x1, x9, lsl #3
    ldp     d16, d17, [x15]
    add     x15, x2, x9, lsl #3
    ldp     d18, d19, [x15]
    fsub    d20, d17, d16
    fsub    d21, d19, d18
    fsub    d22, d4, d16
    fdiv    d21, d21, d20
    fmadd   d23, d21, d22, d18
    fmax    d24, d24, d23
    fcmp    d22, #0.0
    fcsel   d23, d23, d18, gt
    fcmp    d4, d17
    fcsel   d4, d19, d23, ge
    add     x15, x1, x10, lsl #3
    ldp     d16, d17, [x15]
    add     x15, x2, x10, lsl #3
    ldp     d18, d19, [x15]
    fsub    d20, d17, d16
    fsub    d21, d19, d18
    fsub    d22, d5, d16
    fdiv    d21, d21, d20
    fmadd   d23, d21, d22, d18
    fmax    d24, d24, d23
    fcmp    d22, #0.0
    fcsel   d23, d23, d18, gt
    fcmp    d5, d17
    fcsel   d5, d19, d23, ge
    add     x15, x1, x11, lsl #3
    ldp     d16, d17, [x15]
    add     x15, x2, x11, lsl #3
    ldp     d18, d19, [x15]
    fsub    d20, d17, d16
    fsub    d21, d19, d18
    fsub    d22, d6, d16
    fdiv    d21, d21, d20
    fmadd   d23, d21, d22, d18
    fmax    d24, d24, d23
    fcmp    d22, #0.0
    fcsel   d23, d23, d18, gt
    fcmp    d6, d17
    fcsel   d6, d19, d23, ge
    add     x15, x1, x12, lsl #3
    ldp     d16, d17, [x15]
    add     x15, x2, x12, lsl #3
    ldp     d18, d19, [x15]
    fsub    d20, d17, d16
    fsub    d21, d19, d18
    fsub    d22, d7, d16
    fdiv    d21, d21, d20
    fmadd   d23, d21, d22, d18
    fmax    d24, d24, d23
    fcmp    d22, #0.0
    fcsel   d23, d23, d18, gt
    fcmp    d7, d17
    fcsel   d7, d19, d23, ge
    fcmp    d24, d24
    b.vs    Lfew_redo               // some fma result was NaN
    add     x15, x0, x3             // this group's out
    cmp     x14, #8
    b.lo    Lfew_short
    stp     d0, d1, [x15]
    stp     d2, d3, [x15, #16]
    stp     d4, d5, [x15, #32]
    stp     d6, d7, [x15, #48]
    add     x0, x0, #64
    cmp     x0, x14
    b.ls    Lfew_load
    add     x15, x14, #64           // end of the queries
    cmp     x0, x15
    b.eq    Ldone
    mov     x0, x14                 // 1 to 7 left: redo the last eight
    b       Lfew_load
Lfew_short:                         // 5 to 7 queries: store only theirs
    stp     d0, d1, [x15]
    stp     d2, d3, [x15, #16]
    str     d4, [x15, #32]
    cmp     x14, #6
    b.lo    Ldone
    str     d5, [x15, #40]
    cmp     x14, #7
    b.lo    Ldone
    str     d6, [x15, #48]
    ret
Lfew_redo:                          // Lslow from this group to the end
    mov     x5, x0
    add     x6, x0, x3
    mov     x8, x14                 // fewer than eight: all of them
    cmp     x14, #8
    b.lo    Lfew_redo_go
    add     x8, x14, #64
    sub     x8, x8, x0
    lsr     x8, x8, #3              // queries from this group to the end
Lfew_redo_go:
    mov     x7, #0                  // return when done
    b       Lslow

Lone:                               // one sample point: every query gets fp[0]
    ldr     d0, [x2]
Lone_query:
    str     d0, [x3], #8
    subs    x5, x5, #1
    b.ne    Lone_query
    ret

Lall_slow:                          // x5 queries, one at a time
    mov     x8, x5
    mov     x5, x0
    mov     x6, x3
    mov     x7, #0                  // return when done
    b       Lslow
Lp2_redo:
    mov     x8, #4
    // naive.c for x8 queries at x5, results to x6. Rare path.
    // x10 i, x16 lo then j, x17 hi, x0 mid; d16 left, d17 right, d18 xp[0],
    // d19 xp[n-1]. Returns to stage 2 if x7 is not zero.
Lslow:
    ldr     d16, [x2]
    sub     x0, x4, #1
    ldr     d17, [x2, x0, lsl #3]
    ldr     d18, [x1]
    ldr     d19, [x1, x0, lsl #3]
    mov     x10, #0
Lslow_query:
    ldr     d0, [x5, x10, lsl #3]   // v
    fmov    d7, d0
    fcmp    d0, d0
    b.vs    Lslow_store             // NaN query: the result is that NaN
    fmov    d7, d17
    fcmp    d0, d19
    b.gt    Lslow_store             // above the range: right
    fmov    d7, d16
    fcmp    d0, d18
    b.mi    Lslow_store             // below the range: left
    mov     x16, #0                 // lo
    mov     x17, x4                 // hi
Lslow_search:                       // count the sample points <= v
    cmp     x16, x17
    b.hs    Lslow_found
    sub     x0, x17, x16
    add     x0, x16, x0, lsr #1     // mid
    ldr     d1, [x1, x0, lsl #3]
    fcmp    d0, d1
    b.ge    Lslow_right             // v >= xp[mid]
    mov     x17, x0
    b       Lslow_search
Lslow_right:
    add     x16, x0, #1
    b       Lslow_search
Lslow_found:
    sub     x16, x16, #1            // j
    ldr     d1, [x1, x16, lsl #3]   // xp[j]
    ldr     d2, [x2, x16, lsl #3]   // fp[j]
    fmov    d7, d2
    sub     x0, x4, #1
    cmp     x16, x0
    b.eq    Lslow_store             // last sample point
    fcmp    d1, d0
    b.eq    Lslow_store             // exact hit
    add     x0, x16, #1
    ldr     d3, [x1, x0, lsl #3]    // xp[j+1]
    ldr     d4, [x2, x0, lsl #3]    // fp[j+1]
    fsub    d5, d4, d2
    fsub    d6, d3, d1
    fdiv    d5, d5, d6              // slope
    fsub    d6, d0, d1
    fmadd   d7, d5, d6, d2          // slope * (v - xp[j]) + fp[j]
    fcmp    d7, d7
    b.vc    Lslow_store
    fsub    d6, d0, d3
    fmadd   d7, d5, d6, d4          // slope * (v - xp[j+1]) + fp[j+1]
    fcmp    d7, d7
    b.vc    Lslow_store
    fcmp    d2, d4
    b.ne    Lslow_store
    fmov    d7, d2
Lslow_store:
    str     d7, [x6, x10, lsl #3]
    add     x10, x10, #1
    cmp     x10, x8
    b.lo    Lslow_query
    cbnz    x7, Lp2_next
    ret
