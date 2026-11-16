// void interp_f64(const double *x, const double *xp, const double *fp, double *out, size_t n);
//
// np.interp(x, xp, fp) with n queries and n sample points; bit-identical to
// naive.c, which follows NumPy case by case.
//
// Queries are handled in groups of 64 (16 when n < 64), in two stages.
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
//
// Stage 2, the arithmetic, two queries per vector, without branches:
//     t = v - xp[jc]
//     r = fma((fp[jc+1] - fp[jc]) / (xp[jc+1] - xp[jc]), t, fp[jc])
//     if !(t > 0)        r = fp[jc]      // exact hit, or below the range
//     if v >= xp[jc+1]   r = fp[jc+1]    // at or above the last sample point
// Inside the range and off a sample point, that is exactly naive.c's value.
// The rare cases are not decided here: whenever the fma result is NaN (a NaN
// query, infinities, the NaN fallbacks) the four queries of that iteration
// are recomputed by Lslow, which is naive.c line by line. Lslow also handles
// everything when n < 16.
//
// Leftover queries (n not a multiple of the group size) are covered by one
// more group that ends exactly at the last query and recomputes a few results.
//
// Registers:
//   x1 xp, x2 fp, x4 n, x3 group size in bytes, x9 group's queries,
//   x11 last group's queries, x12 out - x, x13 p/2, x14 a = xp + 1, x15 k,
//   x10 h, x17 &a[h - 1], x16 passes left, x0 and x5-x8 scratch
//   v0-v7, v16-v23 bases (stage 1), v24-v26 scratch,
//   v27 a[p - 1] (both lanes), v28 m - p (four lanes), v31 h (four lanes)
.text
.globl _interp_f64
.p2align 2
_interp_f64:
    cmp     x4, #1
    b.eq    Lone
    cmp     x4, #16
    b.lo    Lall_slow
    lsr     x5, x4, #32
    cbnz    x5, Lall_slow           // bases are 32-bit
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
    sub     x8, x4, #64
    add     x11, x0, x8, lsl #3
    b       Lg64
Lsetup16:
    mov     x3, #128
    sub     x8, x4, #16
    add     x11, x0, x8, lsl #3
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
    b       Lp2

Lg16:                               // stage 1 for 16 queries at x9
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
    mov     x10, x13
    add     x17, x14, x13, lsl #3
    sub     x17, x17, #8            // &a[p/2 - 1]
    dup     v31.4s, w13
    add     x16, x15, #1            // k steps and the last probe
Lg16_pass:
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
    lsr     x7, x10, #1
    cmp     x7, #0
    csinc   x10, x7, xzr, ne        // h = max(h / 2, 1): the last probe is h = 1 again
    add     x17, x14, x10, lsl #3
    sub     x17, x17, #8
    dup     v31.4s, w10
    subs    x16, x16, #1
    b.ne    Lg16_pass
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
    mov     x5, x9
    mov     x7, #16
    b       Lp2

    // Stage 2 for x7 queries (a multiple of 4) at x5, with jc in out at x6.
    // Two queries per vector.
Lp2:
    movi    v24.2d, #0xffffffffffffffff
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
    fcmeq   v6.2d, v5.2d, v5.2d     // all ones unless that is NaN
    fcmgt   v7.2d, v7.2d, #0.0
    fcmge   v0.2d, v0.2d, v2.2d
    and     v24.16b, v24.16b, v6.16b
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
    fcmeq   v6.2d, v5.2d, v5.2d     // all ones unless that is NaN
    fcmgt   v7.2d, v7.2d, #0.0
    fcmge   v0.2d, v0.2d, v2.2d
    and     v24.16b, v24.16b, v6.16b
    bif     v5.16b, v3.16b, v7.16b  // !(t > 0): fp[jc]
    bit     v5.16b, v4.16b, v0.16b  // v >= xp[jc+1]: fp[jc+1]
    str     q5, [x6, #16]
    fmov    x8, d24
    mov     x10, v24.d[1]
    and     x8, x8, x10
    cmn     x8, #1
    b.ne    Lp2_redo                // some fma result was NaN
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

Lone:                               // one sample point: left = right = fp[0]
    ldr     d0, [x2]
    str     d0, [x3]
    ret

Lall_slow:
    mov     x5, x0
    mov     x6, x3
    mov     x8, x4
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
