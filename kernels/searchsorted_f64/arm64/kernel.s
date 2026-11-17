// void searchsorted_f64(const double *a, const double *v, int64_t *out, size_t n,
//                       size_t needles);
//
// a holds n sorted values; v holds `needles` values to search for. For each
// needle v[j], out[j] = first index i with !less(a[i], v[j]), where
// less(x, y) = x < y || (y is NaN && x is not NaN).
//
// Many needles are searched at once with a branchless lower bound. Let p be
// the largest power of two <= n. Each needle keeps an index `base`, with the
// answer always in [base, base + len] and base + len <= n:
//     first:  base = (a[p - 1] < key) ? n - p : 0          len = p
//     then for h = p/2, p/4, ..., 1:
//             if (a[base + h - 1] < key) base += h          len = h
//     last:   if (a[base] < key) base += 1
// The step size depends only on n, so all searches stay in step and their
// loads overlap. When n is a power of two the first step always gives 0.
// A NaN needle makes every compare false; its answer (the index of the first
// NaN in a, or n) is produced by a separate, rare path.
//
// There are four code paths. The length of a chooses one, and each needs at
// least one full group of needles:
//   n < 16            one needle at a time (Lscalar)
//   16 <= n < 64      sixteen needles at once, bases in x0-x15 (below)
//   64 <= n < 1,048,576   64 needles at once, bases in vector registers (Lv64)
//   n >= 1,048,576        128 needles at once, bases kept in out (Lbig)
// Leftover needles are covered by one more group that ends exactly at the last
// needle and recomputes a few results. That needs a full group to redo, so a
// call with fewer needles than its path's group takes a narrower path:
//   n >= 1,048,576, 64 to 127 needles   Lv64, whose 32-bit bases reach any
//                                       n below 2^32
//   otherwise                           Lscalar
//
// ---- The sixteen-needle path
//
// The search is as above, except that the last steps are replaced:
//     first:  base = (a[p - 1] < key) ? n - p : 0          len = p
//     then for h = p/2, p/4, ...:
//             if (a[base + h - 1] < key) base += h          len = h
//     last:   base += number of a[base .. base + 7] that are < key   (when len = 8)
//
// One step for one needle is four instructions:
//     ldr   d24, [x17, base, lsl #3]      // x17 = &a[h - 1]
//     fcmgt d24, key, d24                 // all ones if a[...] < key
//     fmov  x16, d24
//     sub   base, base, x16, lsl #log2(h) // base += h when the mask is -1
// The shift is a constant, so the code for each h is written out once
// (Lblock_<log2 h>, all the same size) and a group enters at the block for
// h = p/2. This form is used instead of fcmp + csel because fcmp runs at only
// one per cycle on this core.
//
// The last 4 probes of a search all fall in a[base .. base + 7], so they are
// replaced by loading those 8 elements and counting how many are below the
// key. That costs about the same number of instructions but is one step in
// the dependency chain instead of 4, and the chain is what limits the speed
// while the array fits in the cache.
//
// A group containing a NaN needle is handed to Lscalar.
//
// All sixteen bases live in x0-x15, so everything else is parked in vector
// registers and fetched when needed:
//   x0-x15   bases          x16 scratch        x17 &a[h - 1]
//   v0-v7, v16-v23  keys (both lanes)          v24, v29, v30 scratch
//   v25 = [next needle, next out]       v26 = [a, n]
//   v27 = [a[p - 1], entry block]       v28 = [&a[p/2 - 1], n - p]
//   v31 = [last full group's needle, out]
.text
.globl _searchsorted_f64
.p2align 2
_searchsorted_f64:
    mov     x14, x4                 // needles; x4 becomes k below
    cmp     x3, #16
    b.lo    Lsmall
    cmp     x14, #16
    b.lo    Lsmall
    clz     x4, x3
    mov     x5, #63
    sub     x4, x5, x4              // k = floor(log2(n)), p = 1 << k
    mov     x5, #1
    lsl     x5, x5, x4              // p
    sub     x6, x3, x5              // n - p
    add     x7, x0, x5, lsl #3
    ldur    d27, [x7, #-8]          // a[p - 1]
    cmp     x3, #64
    b.lo    Lsixteen
    cmp     x14, #64
    b.lo    Lsmall                  // too few needles for a group of 64
    lsr     x9, x3, #32
    cmp     x14, #128
    b.lo    Lfew
    mov     x10, #1048576
    cmp     x3, x10
    b.hs    Lbig
    b       Lv64
Lfew:                               // 64 to 127 needles: Lv64 if its bases can hold n
    cbnz    x9, Lsmall
    b       Lv64
Lsixteen:
    add     x7, x0, x5, lsl #2
    sub     x7, x7, #8              // &a[p/2 - 1]
    fmov    d28, x7
    mov     v28.d[1], x6
    fmov    d26, x0
    mov     v26.d[1], x3
    fmov    d25, x1
    mov     v25.d[1], x2
    sub     x9, x14, #16            // first needle of the last full group
    add     x10, x1, x9, lsl #3
    add     x11, x2, x9, lsl #3
    fmov    d31, x10
    mov     v31.d[1], x11
    adr     x9, Llast
    mov     x10, #264
    sub     x11, x4, #3
    msub    x9, x11, x10, x9        // k - 3 blocks before the last step
    mov     v27.d[1], x9
    mov     x16, x1
Lgroup:                             // x16 = this group's first needle
    ld1r    {v0.2d}, [x16], #8
    ld1r    {v1.2d}, [x16], #8
    ld1r    {v2.2d}, [x16], #8
    ld1r    {v3.2d}, [x16], #8
    ld1r    {v4.2d}, [x16], #8
    ld1r    {v5.2d}, [x16], #8
    ld1r    {v6.2d}, [x16], #8
    ld1r    {v7.2d}, [x16], #8
    ld1r    {v16.2d}, [x16], #8
    ld1r    {v17.2d}, [x16], #8
    ld1r    {v18.2d}, [x16], #8
    ld1r    {v19.2d}, [x16], #8
    ld1r    {v20.2d}, [x16], #8
    ld1r    {v21.2d}, [x16], #8
    ld1r    {v22.2d}, [x16], #8
    ld1r    {v23.2d}, [x16], #8
    // fmax returns NaN if either operand is NaN.
    fmax    d24, d0, d1
    fmax    d29, d2, d3
    fmax    d24, d24, d4
    fmax    d29, d29, d5
    fmax    d24, d24, d6
    fmax    d29, d29, d7
    fmax    d24, d24, d16
    fmax    d29, d29, d17
    fmax    d24, d24, d18
    fmax    d29, d29, d19
    fmax    d24, d24, d20
    fmax    d29, d29, d21
    fmax    d24, d24, d22
    fmax    d29, d29, d23
    fmax    d24, d24, d29
    fcmp    d24, d24
    b.vs    Lnan_group
    mov     x17, v28.d[1]           // n - p
    cbnz    x17, Lfirst
    mov     x0, #0
    mov     x1, #0
    mov     x2, #0
    mov     x3, #0
    mov     x4, #0
    mov     x5, #0
    mov     x6, #0
    mov     x7, #0
    mov     x8, #0
    mov     x9, #0
    mov     x10, #0
    mov     x11, #0
    mov     x12, #0
    mov     x13, #0
    mov     x14, #0
    mov     x15, #0
Lenter:
    mov     x16, v27.d[1]
    fmov    x17, d28
    br      x16
Lfirst:                             // base = (a[p - 1] < key) ? n - p : 0
    fcmgt   d24, d0, d27
    fmov    x0, d24
    and     x0, x0, x17
    fcmgt   d24, d1, d27
    fmov    x1, d24
    and     x1, x1, x17
    fcmgt   d24, d2, d27
    fmov    x2, d24
    and     x2, x2, x17
    fcmgt   d24, d3, d27
    fmov    x3, d24
    and     x3, x3, x17
    fcmgt   d24, d4, d27
    fmov    x4, d24
    and     x4, x4, x17
    fcmgt   d24, d5, d27
    fmov    x5, d24
    and     x5, x5, x17
    fcmgt   d24, d6, d27
    fmov    x6, d24
    and     x6, x6, x17
    fcmgt   d24, d7, d27
    fmov    x7, d24
    and     x7, x7, x17
    fcmgt   d24, d16, d27
    fmov    x8, d24
    and     x8, x8, x17
    fcmgt   d24, d17, d27
    fmov    x9, d24
    and     x9, x9, x17
    fcmgt   d24, d18, d27
    fmov    x10, d24
    and     x10, x10, x17
    fcmgt   d24, d19, d27
    fmov    x11, d24
    and     x11, x11, x17
    fcmgt   d24, d20, d27
    fmov    x12, d24
    and     x12, x12, x17
    fcmgt   d24, d21, d27
    fmov    x13, d24
    and     x13, x13, x17
    fcmgt   d24, d22, d27
    fmov    x14, d24
    and     x14, x14, x17
    fcmgt   d24, d23, d27
    fmov    x15, d24
    and     x15, x15, x17
    b       Lenter

    // One block per half-size h, largest first. Each is exactly
    // 66 instructions; the entry address is computed from that.
Lblock_4:                          // h = 2^4
    ldr     d24, [x17, x0, lsl #3]
    fcmgt   d24, d0, d24
    fmov    x16, d24
    sub     x0, x0, x16, lsl #4
    ldr     d24, [x17, x1, lsl #3]
    fcmgt   d24, d1, d24
    fmov    x16, d24
    sub     x1, x1, x16, lsl #4
    ldr     d24, [x17, x2, lsl #3]
    fcmgt   d24, d2, d24
    fmov    x16, d24
    sub     x2, x2, x16, lsl #4
    ldr     d24, [x17, x3, lsl #3]
    fcmgt   d24, d3, d24
    fmov    x16, d24
    sub     x3, x3, x16, lsl #4
    ldr     d24, [x17, x4, lsl #3]
    fcmgt   d24, d4, d24
    fmov    x16, d24
    sub     x4, x4, x16, lsl #4
    ldr     d24, [x17, x5, lsl #3]
    fcmgt   d24, d5, d24
    fmov    x16, d24
    sub     x5, x5, x16, lsl #4
    ldr     d24, [x17, x6, lsl #3]
    fcmgt   d24, d6, d24
    fmov    x16, d24
    sub     x6, x6, x16, lsl #4
    ldr     d24, [x17, x7, lsl #3]
    fcmgt   d24, d7, d24
    fmov    x16, d24
    sub     x7, x7, x16, lsl #4
    ldr     d24, [x17, x8, lsl #3]
    fcmgt   d24, d16, d24
    fmov    x16, d24
    sub     x8, x8, x16, lsl #4
    ldr     d24, [x17, x9, lsl #3]
    fcmgt   d24, d17, d24
    fmov    x16, d24
    sub     x9, x9, x16, lsl #4
    ldr     d24, [x17, x10, lsl #3]
    fcmgt   d24, d18, d24
    fmov    x16, d24
    sub     x10, x10, x16, lsl #4
    ldr     d24, [x17, x11, lsl #3]
    fcmgt   d24, d19, d24
    fmov    x16, d24
    sub     x11, x11, x16, lsl #4
    ldr     d24, [x17, x12, lsl #3]
    fcmgt   d24, d20, d24
    fmov    x16, d24
    sub     x12, x12, x16, lsl #4
    ldr     d24, [x17, x13, lsl #3]
    fcmgt   d24, d21, d24
    fmov    x16, d24
    sub     x13, x13, x16, lsl #4
    ldr     d24, [x17, x14, lsl #3]
    fcmgt   d24, d22, d24
    fmov    x16, d24
    sub     x14, x14, x16, lsl #4
    ldr     d24, [x17, x15, lsl #3]
    fcmgt   d24, d23, d24
    fmov    x16, d24
    sub     x15, x15, x16, lsl #4
    mov     x16, #64
    sub     x17, x17, x16           // &a[h/2 - 1]
Lblock_3:                          // h = 2^3
    ldr     d24, [x17, x0, lsl #3]
    fcmgt   d24, d0, d24
    fmov    x16, d24
    sub     x0, x0, x16, lsl #3
    ldr     d24, [x17, x1, lsl #3]
    fcmgt   d24, d1, d24
    fmov    x16, d24
    sub     x1, x1, x16, lsl #3
    ldr     d24, [x17, x2, lsl #3]
    fcmgt   d24, d2, d24
    fmov    x16, d24
    sub     x2, x2, x16, lsl #3
    ldr     d24, [x17, x3, lsl #3]
    fcmgt   d24, d3, d24
    fmov    x16, d24
    sub     x3, x3, x16, lsl #3
    ldr     d24, [x17, x4, lsl #3]
    fcmgt   d24, d4, d24
    fmov    x16, d24
    sub     x4, x4, x16, lsl #3
    ldr     d24, [x17, x5, lsl #3]
    fcmgt   d24, d5, d24
    fmov    x16, d24
    sub     x5, x5, x16, lsl #3
    ldr     d24, [x17, x6, lsl #3]
    fcmgt   d24, d6, d24
    fmov    x16, d24
    sub     x6, x6, x16, lsl #3
    ldr     d24, [x17, x7, lsl #3]
    fcmgt   d24, d7, d24
    fmov    x16, d24
    sub     x7, x7, x16, lsl #3
    ldr     d24, [x17, x8, lsl #3]
    fcmgt   d24, d16, d24
    fmov    x16, d24
    sub     x8, x8, x16, lsl #3
    ldr     d24, [x17, x9, lsl #3]
    fcmgt   d24, d17, d24
    fmov    x16, d24
    sub     x9, x9, x16, lsl #3
    ldr     d24, [x17, x10, lsl #3]
    fcmgt   d24, d18, d24
    fmov    x16, d24
    sub     x10, x10, x16, lsl #3
    ldr     d24, [x17, x11, lsl #3]
    fcmgt   d24, d19, d24
    fmov    x16, d24
    sub     x11, x11, x16, lsl #3
    ldr     d24, [x17, x12, lsl #3]
    fcmgt   d24, d20, d24
    fmov    x16, d24
    sub     x12, x12, x16, lsl #3
    ldr     d24, [x17, x13, lsl #3]
    fcmgt   d24, d21, d24
    fmov    x16, d24
    sub     x13, x13, x16, lsl #3
    ldr     d24, [x17, x14, lsl #3]
    fcmgt   d24, d22, d24
    fmov    x16, d24
    sub     x14, x14, x16, lsl #3
    ldr     d24, [x17, x15, lsl #3]
    fcmgt   d24, d23, d24
    fmov    x16, d24
    sub     x15, x15, x16, lsl #3
    mov     x16, #56
    sub     x17, x17, x16           // a
Llast:                              // x17 = a: count a[base .. base + 7] < key
    add     x16, x17, x0, lsl #3
    ldp     q24, q29, [x16]
    fcmgt   v24.2d, v0.2d, v24.2d
    fcmgt   v29.2d, v0.2d, v29.2d
    add     v24.2d, v24.2d, v29.2d
    ldp     q29, q30, [x16, #32]
    fcmgt   v29.2d, v0.2d, v29.2d
    fcmgt   v30.2d, v0.2d, v30.2d
    add     v29.2d, v29.2d, v30.2d
    add     v24.2d, v24.2d, v29.2d
    addp    d24, v24.2d             // minus the count
    fmov    x16, d24
    sub     x0, x0, x16
    add     x16, x17, x1, lsl #3
    ldp     q24, q29, [x16]
    fcmgt   v24.2d, v1.2d, v24.2d
    fcmgt   v29.2d, v1.2d, v29.2d
    add     v24.2d, v24.2d, v29.2d
    ldp     q29, q30, [x16, #32]
    fcmgt   v29.2d, v1.2d, v29.2d
    fcmgt   v30.2d, v1.2d, v30.2d
    add     v29.2d, v29.2d, v30.2d
    add     v24.2d, v24.2d, v29.2d
    addp    d24, v24.2d             // minus the count
    fmov    x16, d24
    sub     x1, x1, x16
    add     x16, x17, x2, lsl #3
    ldp     q24, q29, [x16]
    fcmgt   v24.2d, v2.2d, v24.2d
    fcmgt   v29.2d, v2.2d, v29.2d
    add     v24.2d, v24.2d, v29.2d
    ldp     q29, q30, [x16, #32]
    fcmgt   v29.2d, v2.2d, v29.2d
    fcmgt   v30.2d, v2.2d, v30.2d
    add     v29.2d, v29.2d, v30.2d
    add     v24.2d, v24.2d, v29.2d
    addp    d24, v24.2d             // minus the count
    fmov    x16, d24
    sub     x2, x2, x16
    add     x16, x17, x3, lsl #3
    ldp     q24, q29, [x16]
    fcmgt   v24.2d, v3.2d, v24.2d
    fcmgt   v29.2d, v3.2d, v29.2d
    add     v24.2d, v24.2d, v29.2d
    ldp     q29, q30, [x16, #32]
    fcmgt   v29.2d, v3.2d, v29.2d
    fcmgt   v30.2d, v3.2d, v30.2d
    add     v29.2d, v29.2d, v30.2d
    add     v24.2d, v24.2d, v29.2d
    addp    d24, v24.2d             // minus the count
    fmov    x16, d24
    sub     x3, x3, x16
    add     x16, x17, x4, lsl #3
    ldp     q24, q29, [x16]
    fcmgt   v24.2d, v4.2d, v24.2d
    fcmgt   v29.2d, v4.2d, v29.2d
    add     v24.2d, v24.2d, v29.2d
    ldp     q29, q30, [x16, #32]
    fcmgt   v29.2d, v4.2d, v29.2d
    fcmgt   v30.2d, v4.2d, v30.2d
    add     v29.2d, v29.2d, v30.2d
    add     v24.2d, v24.2d, v29.2d
    addp    d24, v24.2d             // minus the count
    fmov    x16, d24
    sub     x4, x4, x16
    add     x16, x17, x5, lsl #3
    ldp     q24, q29, [x16]
    fcmgt   v24.2d, v5.2d, v24.2d
    fcmgt   v29.2d, v5.2d, v29.2d
    add     v24.2d, v24.2d, v29.2d
    ldp     q29, q30, [x16, #32]
    fcmgt   v29.2d, v5.2d, v29.2d
    fcmgt   v30.2d, v5.2d, v30.2d
    add     v29.2d, v29.2d, v30.2d
    add     v24.2d, v24.2d, v29.2d
    addp    d24, v24.2d             // minus the count
    fmov    x16, d24
    sub     x5, x5, x16
    add     x16, x17, x6, lsl #3
    ldp     q24, q29, [x16]
    fcmgt   v24.2d, v6.2d, v24.2d
    fcmgt   v29.2d, v6.2d, v29.2d
    add     v24.2d, v24.2d, v29.2d
    ldp     q29, q30, [x16, #32]
    fcmgt   v29.2d, v6.2d, v29.2d
    fcmgt   v30.2d, v6.2d, v30.2d
    add     v29.2d, v29.2d, v30.2d
    add     v24.2d, v24.2d, v29.2d
    addp    d24, v24.2d             // minus the count
    fmov    x16, d24
    sub     x6, x6, x16
    add     x16, x17, x7, lsl #3
    ldp     q24, q29, [x16]
    fcmgt   v24.2d, v7.2d, v24.2d
    fcmgt   v29.2d, v7.2d, v29.2d
    add     v24.2d, v24.2d, v29.2d
    ldp     q29, q30, [x16, #32]
    fcmgt   v29.2d, v7.2d, v29.2d
    fcmgt   v30.2d, v7.2d, v30.2d
    add     v29.2d, v29.2d, v30.2d
    add     v24.2d, v24.2d, v29.2d
    addp    d24, v24.2d             // minus the count
    fmov    x16, d24
    sub     x7, x7, x16
    add     x16, x17, x8, lsl #3
    ldp     q24, q29, [x16]
    fcmgt   v24.2d, v16.2d, v24.2d
    fcmgt   v29.2d, v16.2d, v29.2d
    add     v24.2d, v24.2d, v29.2d
    ldp     q29, q30, [x16, #32]
    fcmgt   v29.2d, v16.2d, v29.2d
    fcmgt   v30.2d, v16.2d, v30.2d
    add     v29.2d, v29.2d, v30.2d
    add     v24.2d, v24.2d, v29.2d
    addp    d24, v24.2d             // minus the count
    fmov    x16, d24
    sub     x8, x8, x16
    add     x16, x17, x9, lsl #3
    ldp     q24, q29, [x16]
    fcmgt   v24.2d, v17.2d, v24.2d
    fcmgt   v29.2d, v17.2d, v29.2d
    add     v24.2d, v24.2d, v29.2d
    ldp     q29, q30, [x16, #32]
    fcmgt   v29.2d, v17.2d, v29.2d
    fcmgt   v30.2d, v17.2d, v30.2d
    add     v29.2d, v29.2d, v30.2d
    add     v24.2d, v24.2d, v29.2d
    addp    d24, v24.2d             // minus the count
    fmov    x16, d24
    sub     x9, x9, x16
    add     x16, x17, x10, lsl #3
    ldp     q24, q29, [x16]
    fcmgt   v24.2d, v18.2d, v24.2d
    fcmgt   v29.2d, v18.2d, v29.2d
    add     v24.2d, v24.2d, v29.2d
    ldp     q29, q30, [x16, #32]
    fcmgt   v29.2d, v18.2d, v29.2d
    fcmgt   v30.2d, v18.2d, v30.2d
    add     v29.2d, v29.2d, v30.2d
    add     v24.2d, v24.2d, v29.2d
    addp    d24, v24.2d             // minus the count
    fmov    x16, d24
    sub     x10, x10, x16
    add     x16, x17, x11, lsl #3
    ldp     q24, q29, [x16]
    fcmgt   v24.2d, v19.2d, v24.2d
    fcmgt   v29.2d, v19.2d, v29.2d
    add     v24.2d, v24.2d, v29.2d
    ldp     q29, q30, [x16, #32]
    fcmgt   v29.2d, v19.2d, v29.2d
    fcmgt   v30.2d, v19.2d, v30.2d
    add     v29.2d, v29.2d, v30.2d
    add     v24.2d, v24.2d, v29.2d
    addp    d24, v24.2d             // minus the count
    fmov    x16, d24
    sub     x11, x11, x16
    add     x16, x17, x12, lsl #3
    ldp     q24, q29, [x16]
    fcmgt   v24.2d, v20.2d, v24.2d
    fcmgt   v29.2d, v20.2d, v29.2d
    add     v24.2d, v24.2d, v29.2d
    ldp     q29, q30, [x16, #32]
    fcmgt   v29.2d, v20.2d, v29.2d
    fcmgt   v30.2d, v20.2d, v30.2d
    add     v29.2d, v29.2d, v30.2d
    add     v24.2d, v24.2d, v29.2d
    addp    d24, v24.2d             // minus the count
    fmov    x16, d24
    sub     x12, x12, x16
    add     x16, x17, x13, lsl #3
    ldp     q24, q29, [x16]
    fcmgt   v24.2d, v21.2d, v24.2d
    fcmgt   v29.2d, v21.2d, v29.2d
    add     v24.2d, v24.2d, v29.2d
    ldp     q29, q30, [x16, #32]
    fcmgt   v29.2d, v21.2d, v29.2d
    fcmgt   v30.2d, v21.2d, v30.2d
    add     v29.2d, v29.2d, v30.2d
    add     v24.2d, v24.2d, v29.2d
    addp    d24, v24.2d             // minus the count
    fmov    x16, d24
    sub     x13, x13, x16
    add     x16, x17, x14, lsl #3
    ldp     q24, q29, [x16]
    fcmgt   v24.2d, v22.2d, v24.2d
    fcmgt   v29.2d, v22.2d, v29.2d
    add     v24.2d, v24.2d, v29.2d
    ldp     q29, q30, [x16, #32]
    fcmgt   v29.2d, v22.2d, v29.2d
    fcmgt   v30.2d, v22.2d, v30.2d
    add     v29.2d, v29.2d, v30.2d
    add     v24.2d, v24.2d, v29.2d
    addp    d24, v24.2d             // minus the count
    fmov    x16, d24
    sub     x14, x14, x16
    add     x16, x17, x15, lsl #3
    ldp     q24, q29, [x16]
    fcmgt   v24.2d, v23.2d, v24.2d
    fcmgt   v29.2d, v23.2d, v29.2d
    add     v24.2d, v24.2d, v29.2d
    ldp     q29, q30, [x16, #32]
    fcmgt   v29.2d, v23.2d, v29.2d
    fcmgt   v30.2d, v23.2d, v30.2d
    add     v29.2d, v29.2d, v30.2d
    add     v24.2d, v24.2d, v29.2d
    addp    d24, v24.2d             // minus the count
    fmov    x16, d24
    sub     x15, x15, x16
    mov     x16, v25.d[1]
    stp     x0, x1, [x16, #0]
    stp     x2, x3, [x16, #16]
    stp     x4, x5, [x16, #32]
    stp     x6, x7, [x16, #48]
    stp     x8, x9, [x16, #64]
    stp     x10, x11, [x16, #80]
    stp     x12, x13, [x16, #96]
    stp     x14, x15, [x16, #112]
Lnext:
    movi    v24.4s, #128
    ushr    v24.2d, v24.2d, #32     // [128, 128]
    add     v25.2d, v25.2d, v24.2d
    fmov    x16, d25                // next needle
    fmov    x17, d31                // first needle of the last full group
    cmp     x16, x17
    b.ls    Lgroup
    add     x17, x17, #128          // end of the needles
    cmp     x16, x17
    b.eq    Ldone
    // 1 to 15 needles left: redo the last sixteen as one group.
    mov     v25.16b, v31.16b
    fmov    x16, d25
    b       Lgroup
Ldone:
    ret

    // ---- 64 needles at once, bases as 32-bit lanes of v0-v7 and v16-v23.
    // One step for four needles moves their bases to integer registers, loads
    // the four probes, and does the compare and the update as vector
    // operations: 16 instructions for four probes, and four times as many
    // searches in flight as the sixteen-needle path. Every step is the same
    // code, with h in a register, so there is one loop over the steps. The
    // needles are reloaded from v at each step; a group that held a NaN (found
    // with fmax during its first step) is fixed up afterwards at Lnanfix.
    // x0 a, x3 n, x4 k, x6 n - p, x14 needles (on entry), x9 group's needles,
    // x11 last group's needles,
    // x12 out - v, x13 p/2, x10 h, x17 &a[h - 1],
    // x16 passes left, x5-x8 scratch, v24-v26 scratch, v27 a[p - 1] (both
    // lanes), v28 n - p (four lanes), v29/v30 NaN check, v31 h (four lanes)
Lv64:
    dup     v27.2d, v27.d[0]
    dup     v28.4s, w6
    sub     x9, x14, #64
    add     x11, x1, x9, lsl #3
    sub     x12, x2, x1
    mov     x9, x1
    mov     x13, #1
    sub     x7, x4, #1
    lsl     x13, x13, x7
Lv64_group:
    cbnz    x6, Lv64_first
    movi    v0.4s, #0
    movi    v1.4s, #0
    movi    v2.4s, #0
    movi    v3.4s, #0
    movi    v4.4s, #0
    movi    v5.4s, #0
    movi    v6.4s, #0
    movi    v7.4s, #0
    movi    v16.4s, #0
    movi    v17.4s, #0
    movi    v18.4s, #0
    movi    v19.4s, #0
    movi    v20.4s, #0
    movi    v21.4s, #0
    movi    v22.4s, #0
    movi    v23.4s, #0
Lv64_start:
    add     x17, x0, x13, lsl #3
    sub     x17, x17, #8            // &a[p/2 - 1]
    mov     x10, x13
    dup     v31.4s, w13
    add     x16, x4, #1             // k steps and the last probe
    movi    v29.2d, #0
    movi    v30.2d, #0
    // First pass of the group; it also checks the needles for NaN.
    fmov    x5, d0
    mov     x6, v0.d[1]
    lsr     x7, x5, #32
    lsr     x8, x6, #32
    ldr     d24, [x17, w5, uxtw #3]
    ldr     d25, [x17, x7, lsl #3]
    ins     v24.d[1], v25.d[0]
    ldr     d25, [x17, w6, uxtw #3]
    ldr     d26, [x17, x8, lsl #3]
    ins     v25.d[1], v26.d[0]
    ldr     q26, [x9, #0]
    fmax    v29.2d, v29.2d, v26.2d
    fcmgt   v24.2d, v26.2d, v24.2d
    ldr     q26, [x9, #16]
    fmax    v30.2d, v30.2d, v26.2d
    fcmgt   v25.2d, v26.2d, v25.2d
    uzp1    v24.4s, v24.4s, v25.4s  // four 32-bit masks
    mls     v0.4s, v24.4s, v31.4s    // base += h where the mask is -1
    fmov    x5, d1
    mov     x6, v1.d[1]
    lsr     x7, x5, #32
    lsr     x8, x6, #32
    ldr     d24, [x17, w5, uxtw #3]
    ldr     d25, [x17, x7, lsl #3]
    ins     v24.d[1], v25.d[0]
    ldr     d25, [x17, w6, uxtw #3]
    ldr     d26, [x17, x8, lsl #3]
    ins     v25.d[1], v26.d[0]
    ldr     q26, [x9, #32]
    fmax    v29.2d, v29.2d, v26.2d
    fcmgt   v24.2d, v26.2d, v24.2d
    ldr     q26, [x9, #48]
    fmax    v30.2d, v30.2d, v26.2d
    fcmgt   v25.2d, v26.2d, v25.2d
    uzp1    v24.4s, v24.4s, v25.4s  // four 32-bit masks
    mls     v1.4s, v24.4s, v31.4s    // base += h where the mask is -1
    fmov    x5, d2
    mov     x6, v2.d[1]
    lsr     x7, x5, #32
    lsr     x8, x6, #32
    ldr     d24, [x17, w5, uxtw #3]
    ldr     d25, [x17, x7, lsl #3]
    ins     v24.d[1], v25.d[0]
    ldr     d25, [x17, w6, uxtw #3]
    ldr     d26, [x17, x8, lsl #3]
    ins     v25.d[1], v26.d[0]
    ldr     q26, [x9, #64]
    fmax    v29.2d, v29.2d, v26.2d
    fcmgt   v24.2d, v26.2d, v24.2d
    ldr     q26, [x9, #80]
    fmax    v30.2d, v30.2d, v26.2d
    fcmgt   v25.2d, v26.2d, v25.2d
    uzp1    v24.4s, v24.4s, v25.4s  // four 32-bit masks
    mls     v2.4s, v24.4s, v31.4s    // base += h where the mask is -1
    fmov    x5, d3
    mov     x6, v3.d[1]
    lsr     x7, x5, #32
    lsr     x8, x6, #32
    ldr     d24, [x17, w5, uxtw #3]
    ldr     d25, [x17, x7, lsl #3]
    ins     v24.d[1], v25.d[0]
    ldr     d25, [x17, w6, uxtw #3]
    ldr     d26, [x17, x8, lsl #3]
    ins     v25.d[1], v26.d[0]
    ldr     q26, [x9, #96]
    fmax    v29.2d, v29.2d, v26.2d
    fcmgt   v24.2d, v26.2d, v24.2d
    ldr     q26, [x9, #112]
    fmax    v30.2d, v30.2d, v26.2d
    fcmgt   v25.2d, v26.2d, v25.2d
    uzp1    v24.4s, v24.4s, v25.4s  // four 32-bit masks
    mls     v3.4s, v24.4s, v31.4s    // base += h where the mask is -1
    fmov    x5, d4
    mov     x6, v4.d[1]
    lsr     x7, x5, #32
    lsr     x8, x6, #32
    ldr     d24, [x17, w5, uxtw #3]
    ldr     d25, [x17, x7, lsl #3]
    ins     v24.d[1], v25.d[0]
    ldr     d25, [x17, w6, uxtw #3]
    ldr     d26, [x17, x8, lsl #3]
    ins     v25.d[1], v26.d[0]
    ldr     q26, [x9, #128]
    fmax    v29.2d, v29.2d, v26.2d
    fcmgt   v24.2d, v26.2d, v24.2d
    ldr     q26, [x9, #144]
    fmax    v30.2d, v30.2d, v26.2d
    fcmgt   v25.2d, v26.2d, v25.2d
    uzp1    v24.4s, v24.4s, v25.4s  // four 32-bit masks
    mls     v4.4s, v24.4s, v31.4s    // base += h where the mask is -1
    fmov    x5, d5
    mov     x6, v5.d[1]
    lsr     x7, x5, #32
    lsr     x8, x6, #32
    ldr     d24, [x17, w5, uxtw #3]
    ldr     d25, [x17, x7, lsl #3]
    ins     v24.d[1], v25.d[0]
    ldr     d25, [x17, w6, uxtw #3]
    ldr     d26, [x17, x8, lsl #3]
    ins     v25.d[1], v26.d[0]
    ldr     q26, [x9, #160]
    fmax    v29.2d, v29.2d, v26.2d
    fcmgt   v24.2d, v26.2d, v24.2d
    ldr     q26, [x9, #176]
    fmax    v30.2d, v30.2d, v26.2d
    fcmgt   v25.2d, v26.2d, v25.2d
    uzp1    v24.4s, v24.4s, v25.4s  // four 32-bit masks
    mls     v5.4s, v24.4s, v31.4s    // base += h where the mask is -1
    fmov    x5, d6
    mov     x6, v6.d[1]
    lsr     x7, x5, #32
    lsr     x8, x6, #32
    ldr     d24, [x17, w5, uxtw #3]
    ldr     d25, [x17, x7, lsl #3]
    ins     v24.d[1], v25.d[0]
    ldr     d25, [x17, w6, uxtw #3]
    ldr     d26, [x17, x8, lsl #3]
    ins     v25.d[1], v26.d[0]
    ldr     q26, [x9, #192]
    fmax    v29.2d, v29.2d, v26.2d
    fcmgt   v24.2d, v26.2d, v24.2d
    ldr     q26, [x9, #208]
    fmax    v30.2d, v30.2d, v26.2d
    fcmgt   v25.2d, v26.2d, v25.2d
    uzp1    v24.4s, v24.4s, v25.4s  // four 32-bit masks
    mls     v6.4s, v24.4s, v31.4s    // base += h where the mask is -1
    fmov    x5, d7
    mov     x6, v7.d[1]
    lsr     x7, x5, #32
    lsr     x8, x6, #32
    ldr     d24, [x17, w5, uxtw #3]
    ldr     d25, [x17, x7, lsl #3]
    ins     v24.d[1], v25.d[0]
    ldr     d25, [x17, w6, uxtw #3]
    ldr     d26, [x17, x8, lsl #3]
    ins     v25.d[1], v26.d[0]
    ldr     q26, [x9, #224]
    fmax    v29.2d, v29.2d, v26.2d
    fcmgt   v24.2d, v26.2d, v24.2d
    ldr     q26, [x9, #240]
    fmax    v30.2d, v30.2d, v26.2d
    fcmgt   v25.2d, v26.2d, v25.2d
    uzp1    v24.4s, v24.4s, v25.4s  // four 32-bit masks
    mls     v7.4s, v24.4s, v31.4s    // base += h where the mask is -1
    fmov    x5, d16
    mov     x6, v16.d[1]
    lsr     x7, x5, #32
    lsr     x8, x6, #32
    ldr     d24, [x17, w5, uxtw #3]
    ldr     d25, [x17, x7, lsl #3]
    ins     v24.d[1], v25.d[0]
    ldr     d25, [x17, w6, uxtw #3]
    ldr     d26, [x17, x8, lsl #3]
    ins     v25.d[1], v26.d[0]
    ldr     q26, [x9, #256]
    fmax    v29.2d, v29.2d, v26.2d
    fcmgt   v24.2d, v26.2d, v24.2d
    ldr     q26, [x9, #272]
    fmax    v30.2d, v30.2d, v26.2d
    fcmgt   v25.2d, v26.2d, v25.2d
    uzp1    v24.4s, v24.4s, v25.4s  // four 32-bit masks
    mls     v16.4s, v24.4s, v31.4s    // base += h where the mask is -1
    fmov    x5, d17
    mov     x6, v17.d[1]
    lsr     x7, x5, #32
    lsr     x8, x6, #32
    ldr     d24, [x17, w5, uxtw #3]
    ldr     d25, [x17, x7, lsl #3]
    ins     v24.d[1], v25.d[0]
    ldr     d25, [x17, w6, uxtw #3]
    ldr     d26, [x17, x8, lsl #3]
    ins     v25.d[1], v26.d[0]
    ldr     q26, [x9, #288]
    fmax    v29.2d, v29.2d, v26.2d
    fcmgt   v24.2d, v26.2d, v24.2d
    ldr     q26, [x9, #304]
    fmax    v30.2d, v30.2d, v26.2d
    fcmgt   v25.2d, v26.2d, v25.2d
    uzp1    v24.4s, v24.4s, v25.4s  // four 32-bit masks
    mls     v17.4s, v24.4s, v31.4s    // base += h where the mask is -1
    fmov    x5, d18
    mov     x6, v18.d[1]
    lsr     x7, x5, #32
    lsr     x8, x6, #32
    ldr     d24, [x17, w5, uxtw #3]
    ldr     d25, [x17, x7, lsl #3]
    ins     v24.d[1], v25.d[0]
    ldr     d25, [x17, w6, uxtw #3]
    ldr     d26, [x17, x8, lsl #3]
    ins     v25.d[1], v26.d[0]
    ldr     q26, [x9, #320]
    fmax    v29.2d, v29.2d, v26.2d
    fcmgt   v24.2d, v26.2d, v24.2d
    ldr     q26, [x9, #336]
    fmax    v30.2d, v30.2d, v26.2d
    fcmgt   v25.2d, v26.2d, v25.2d
    uzp1    v24.4s, v24.4s, v25.4s  // four 32-bit masks
    mls     v18.4s, v24.4s, v31.4s    // base += h where the mask is -1
    fmov    x5, d19
    mov     x6, v19.d[1]
    lsr     x7, x5, #32
    lsr     x8, x6, #32
    ldr     d24, [x17, w5, uxtw #3]
    ldr     d25, [x17, x7, lsl #3]
    ins     v24.d[1], v25.d[0]
    ldr     d25, [x17, w6, uxtw #3]
    ldr     d26, [x17, x8, lsl #3]
    ins     v25.d[1], v26.d[0]
    ldr     q26, [x9, #352]
    fmax    v29.2d, v29.2d, v26.2d
    fcmgt   v24.2d, v26.2d, v24.2d
    ldr     q26, [x9, #368]
    fmax    v30.2d, v30.2d, v26.2d
    fcmgt   v25.2d, v26.2d, v25.2d
    uzp1    v24.4s, v24.4s, v25.4s  // four 32-bit masks
    mls     v19.4s, v24.4s, v31.4s    // base += h where the mask is -1
    fmov    x5, d20
    mov     x6, v20.d[1]
    lsr     x7, x5, #32
    lsr     x8, x6, #32
    ldr     d24, [x17, w5, uxtw #3]
    ldr     d25, [x17, x7, lsl #3]
    ins     v24.d[1], v25.d[0]
    ldr     d25, [x17, w6, uxtw #3]
    ldr     d26, [x17, x8, lsl #3]
    ins     v25.d[1], v26.d[0]
    ldr     q26, [x9, #384]
    fmax    v29.2d, v29.2d, v26.2d
    fcmgt   v24.2d, v26.2d, v24.2d
    ldr     q26, [x9, #400]
    fmax    v30.2d, v30.2d, v26.2d
    fcmgt   v25.2d, v26.2d, v25.2d
    uzp1    v24.4s, v24.4s, v25.4s  // four 32-bit masks
    mls     v20.4s, v24.4s, v31.4s    // base += h where the mask is -1
    fmov    x5, d21
    mov     x6, v21.d[1]
    lsr     x7, x5, #32
    lsr     x8, x6, #32
    ldr     d24, [x17, w5, uxtw #3]
    ldr     d25, [x17, x7, lsl #3]
    ins     v24.d[1], v25.d[0]
    ldr     d25, [x17, w6, uxtw #3]
    ldr     d26, [x17, x8, lsl #3]
    ins     v25.d[1], v26.d[0]
    ldr     q26, [x9, #416]
    fmax    v29.2d, v29.2d, v26.2d
    fcmgt   v24.2d, v26.2d, v24.2d
    ldr     q26, [x9, #432]
    fmax    v30.2d, v30.2d, v26.2d
    fcmgt   v25.2d, v26.2d, v25.2d
    uzp1    v24.4s, v24.4s, v25.4s  // four 32-bit masks
    mls     v21.4s, v24.4s, v31.4s    // base += h where the mask is -1
    fmov    x5, d22
    mov     x6, v22.d[1]
    lsr     x7, x5, #32
    lsr     x8, x6, #32
    ldr     d24, [x17, w5, uxtw #3]
    ldr     d25, [x17, x7, lsl #3]
    ins     v24.d[1], v25.d[0]
    ldr     d25, [x17, w6, uxtw #3]
    ldr     d26, [x17, x8, lsl #3]
    ins     v25.d[1], v26.d[0]
    ldr     q26, [x9, #448]
    fmax    v29.2d, v29.2d, v26.2d
    fcmgt   v24.2d, v26.2d, v24.2d
    ldr     q26, [x9, #464]
    fmax    v30.2d, v30.2d, v26.2d
    fcmgt   v25.2d, v26.2d, v25.2d
    uzp1    v24.4s, v24.4s, v25.4s  // four 32-bit masks
    mls     v22.4s, v24.4s, v31.4s    // base += h where the mask is -1
    fmov    x5, d23
    mov     x6, v23.d[1]
    lsr     x7, x5, #32
    lsr     x8, x6, #32
    ldr     d24, [x17, w5, uxtw #3]
    ldr     d25, [x17, x7, lsl #3]
    ins     v24.d[1], v25.d[0]
    ldr     d25, [x17, w6, uxtw #3]
    ldr     d26, [x17, x8, lsl #3]
    ins     v25.d[1], v26.d[0]
    ldr     q26, [x9, #480]
    fmax    v29.2d, v29.2d, v26.2d
    fcmgt   v24.2d, v26.2d, v24.2d
    ldr     q26, [x9, #496]
    fmax    v30.2d, v30.2d, v26.2d
    fcmgt   v25.2d, v26.2d, v25.2d
    uzp1    v24.4s, v24.4s, v25.4s  // four 32-bit masks
    mls     v23.4s, v24.4s, v31.4s    // base += h where the mask is -1
    lsr     x7, x10, #1
    cmp     x7, #0
    csinc   x10, x7, xzr, ne        // h = max(h / 2, 1): the last probe is h = 1 again
    add     x17, x0, x10, lsl #3
    sub     x17, x17, #8
    dup     v31.4s, w10
    sub     x16, x16, #1
Lv64_pass:
    fmov    x5, d0
    mov     x6, v0.d[1]
    lsr     x7, x5, #32
    lsr     x8, x6, #32
    ldr     d24, [x17, w5, uxtw #3]
    ldr     d25, [x17, x7, lsl #3]
    ins     v24.d[1], v25.d[0]
    ldr     d25, [x17, w6, uxtw #3]
    ldr     d26, [x17, x8, lsl #3]
    ins     v25.d[1], v26.d[0]
    ldr     q26, [x9, #0]
    fcmgt   v24.2d, v26.2d, v24.2d
    ldr     q26, [x9, #16]
    fcmgt   v25.2d, v26.2d, v25.2d
    uzp1    v24.4s, v24.4s, v25.4s  // four 32-bit masks
    mls     v0.4s, v24.4s, v31.4s    // base += h where the mask is -1
    fmov    x5, d1
    mov     x6, v1.d[1]
    lsr     x7, x5, #32
    lsr     x8, x6, #32
    ldr     d24, [x17, w5, uxtw #3]
    ldr     d25, [x17, x7, lsl #3]
    ins     v24.d[1], v25.d[0]
    ldr     d25, [x17, w6, uxtw #3]
    ldr     d26, [x17, x8, lsl #3]
    ins     v25.d[1], v26.d[0]
    ldr     q26, [x9, #32]
    fcmgt   v24.2d, v26.2d, v24.2d
    ldr     q26, [x9, #48]
    fcmgt   v25.2d, v26.2d, v25.2d
    uzp1    v24.4s, v24.4s, v25.4s  // four 32-bit masks
    mls     v1.4s, v24.4s, v31.4s    // base += h where the mask is -1
    fmov    x5, d2
    mov     x6, v2.d[1]
    lsr     x7, x5, #32
    lsr     x8, x6, #32
    ldr     d24, [x17, w5, uxtw #3]
    ldr     d25, [x17, x7, lsl #3]
    ins     v24.d[1], v25.d[0]
    ldr     d25, [x17, w6, uxtw #3]
    ldr     d26, [x17, x8, lsl #3]
    ins     v25.d[1], v26.d[0]
    ldr     q26, [x9, #64]
    fcmgt   v24.2d, v26.2d, v24.2d
    ldr     q26, [x9, #80]
    fcmgt   v25.2d, v26.2d, v25.2d
    uzp1    v24.4s, v24.4s, v25.4s  // four 32-bit masks
    mls     v2.4s, v24.4s, v31.4s    // base += h where the mask is -1
    fmov    x5, d3
    mov     x6, v3.d[1]
    lsr     x7, x5, #32
    lsr     x8, x6, #32
    ldr     d24, [x17, w5, uxtw #3]
    ldr     d25, [x17, x7, lsl #3]
    ins     v24.d[1], v25.d[0]
    ldr     d25, [x17, w6, uxtw #3]
    ldr     d26, [x17, x8, lsl #3]
    ins     v25.d[1], v26.d[0]
    ldr     q26, [x9, #96]
    fcmgt   v24.2d, v26.2d, v24.2d
    ldr     q26, [x9, #112]
    fcmgt   v25.2d, v26.2d, v25.2d
    uzp1    v24.4s, v24.4s, v25.4s  // four 32-bit masks
    mls     v3.4s, v24.4s, v31.4s    // base += h where the mask is -1
    fmov    x5, d4
    mov     x6, v4.d[1]
    lsr     x7, x5, #32
    lsr     x8, x6, #32
    ldr     d24, [x17, w5, uxtw #3]
    ldr     d25, [x17, x7, lsl #3]
    ins     v24.d[1], v25.d[0]
    ldr     d25, [x17, w6, uxtw #3]
    ldr     d26, [x17, x8, lsl #3]
    ins     v25.d[1], v26.d[0]
    ldr     q26, [x9, #128]
    fcmgt   v24.2d, v26.2d, v24.2d
    ldr     q26, [x9, #144]
    fcmgt   v25.2d, v26.2d, v25.2d
    uzp1    v24.4s, v24.4s, v25.4s  // four 32-bit masks
    mls     v4.4s, v24.4s, v31.4s    // base += h where the mask is -1
    fmov    x5, d5
    mov     x6, v5.d[1]
    lsr     x7, x5, #32
    lsr     x8, x6, #32
    ldr     d24, [x17, w5, uxtw #3]
    ldr     d25, [x17, x7, lsl #3]
    ins     v24.d[1], v25.d[0]
    ldr     d25, [x17, w6, uxtw #3]
    ldr     d26, [x17, x8, lsl #3]
    ins     v25.d[1], v26.d[0]
    ldr     q26, [x9, #160]
    fcmgt   v24.2d, v26.2d, v24.2d
    ldr     q26, [x9, #176]
    fcmgt   v25.2d, v26.2d, v25.2d
    uzp1    v24.4s, v24.4s, v25.4s  // four 32-bit masks
    mls     v5.4s, v24.4s, v31.4s    // base += h where the mask is -1
    fmov    x5, d6
    mov     x6, v6.d[1]
    lsr     x7, x5, #32
    lsr     x8, x6, #32
    ldr     d24, [x17, w5, uxtw #3]
    ldr     d25, [x17, x7, lsl #3]
    ins     v24.d[1], v25.d[0]
    ldr     d25, [x17, w6, uxtw #3]
    ldr     d26, [x17, x8, lsl #3]
    ins     v25.d[1], v26.d[0]
    ldr     q26, [x9, #192]
    fcmgt   v24.2d, v26.2d, v24.2d
    ldr     q26, [x9, #208]
    fcmgt   v25.2d, v26.2d, v25.2d
    uzp1    v24.4s, v24.4s, v25.4s  // four 32-bit masks
    mls     v6.4s, v24.4s, v31.4s    // base += h where the mask is -1
    fmov    x5, d7
    mov     x6, v7.d[1]
    lsr     x7, x5, #32
    lsr     x8, x6, #32
    ldr     d24, [x17, w5, uxtw #3]
    ldr     d25, [x17, x7, lsl #3]
    ins     v24.d[1], v25.d[0]
    ldr     d25, [x17, w6, uxtw #3]
    ldr     d26, [x17, x8, lsl #3]
    ins     v25.d[1], v26.d[0]
    ldr     q26, [x9, #224]
    fcmgt   v24.2d, v26.2d, v24.2d
    ldr     q26, [x9, #240]
    fcmgt   v25.2d, v26.2d, v25.2d
    uzp1    v24.4s, v24.4s, v25.4s  // four 32-bit masks
    mls     v7.4s, v24.4s, v31.4s    // base += h where the mask is -1
    fmov    x5, d16
    mov     x6, v16.d[1]
    lsr     x7, x5, #32
    lsr     x8, x6, #32
    ldr     d24, [x17, w5, uxtw #3]
    ldr     d25, [x17, x7, lsl #3]
    ins     v24.d[1], v25.d[0]
    ldr     d25, [x17, w6, uxtw #3]
    ldr     d26, [x17, x8, lsl #3]
    ins     v25.d[1], v26.d[0]
    ldr     q26, [x9, #256]
    fcmgt   v24.2d, v26.2d, v24.2d
    ldr     q26, [x9, #272]
    fcmgt   v25.2d, v26.2d, v25.2d
    uzp1    v24.4s, v24.4s, v25.4s  // four 32-bit masks
    mls     v16.4s, v24.4s, v31.4s    // base += h where the mask is -1
    fmov    x5, d17
    mov     x6, v17.d[1]
    lsr     x7, x5, #32
    lsr     x8, x6, #32
    ldr     d24, [x17, w5, uxtw #3]
    ldr     d25, [x17, x7, lsl #3]
    ins     v24.d[1], v25.d[0]
    ldr     d25, [x17, w6, uxtw #3]
    ldr     d26, [x17, x8, lsl #3]
    ins     v25.d[1], v26.d[0]
    ldr     q26, [x9, #288]
    fcmgt   v24.2d, v26.2d, v24.2d
    ldr     q26, [x9, #304]
    fcmgt   v25.2d, v26.2d, v25.2d
    uzp1    v24.4s, v24.4s, v25.4s  // four 32-bit masks
    mls     v17.4s, v24.4s, v31.4s    // base += h where the mask is -1
    fmov    x5, d18
    mov     x6, v18.d[1]
    lsr     x7, x5, #32
    lsr     x8, x6, #32
    ldr     d24, [x17, w5, uxtw #3]
    ldr     d25, [x17, x7, lsl #3]
    ins     v24.d[1], v25.d[0]
    ldr     d25, [x17, w6, uxtw #3]
    ldr     d26, [x17, x8, lsl #3]
    ins     v25.d[1], v26.d[0]
    ldr     q26, [x9, #320]
    fcmgt   v24.2d, v26.2d, v24.2d
    ldr     q26, [x9, #336]
    fcmgt   v25.2d, v26.2d, v25.2d
    uzp1    v24.4s, v24.4s, v25.4s  // four 32-bit masks
    mls     v18.4s, v24.4s, v31.4s    // base += h where the mask is -1
    fmov    x5, d19
    mov     x6, v19.d[1]
    lsr     x7, x5, #32
    lsr     x8, x6, #32
    ldr     d24, [x17, w5, uxtw #3]
    ldr     d25, [x17, x7, lsl #3]
    ins     v24.d[1], v25.d[0]
    ldr     d25, [x17, w6, uxtw #3]
    ldr     d26, [x17, x8, lsl #3]
    ins     v25.d[1], v26.d[0]
    ldr     q26, [x9, #352]
    fcmgt   v24.2d, v26.2d, v24.2d
    ldr     q26, [x9, #368]
    fcmgt   v25.2d, v26.2d, v25.2d
    uzp1    v24.4s, v24.4s, v25.4s  // four 32-bit masks
    mls     v19.4s, v24.4s, v31.4s    // base += h where the mask is -1
    fmov    x5, d20
    mov     x6, v20.d[1]
    lsr     x7, x5, #32
    lsr     x8, x6, #32
    ldr     d24, [x17, w5, uxtw #3]
    ldr     d25, [x17, x7, lsl #3]
    ins     v24.d[1], v25.d[0]
    ldr     d25, [x17, w6, uxtw #3]
    ldr     d26, [x17, x8, lsl #3]
    ins     v25.d[1], v26.d[0]
    ldr     q26, [x9, #384]
    fcmgt   v24.2d, v26.2d, v24.2d
    ldr     q26, [x9, #400]
    fcmgt   v25.2d, v26.2d, v25.2d
    uzp1    v24.4s, v24.4s, v25.4s  // four 32-bit masks
    mls     v20.4s, v24.4s, v31.4s    // base += h where the mask is -1
    fmov    x5, d21
    mov     x6, v21.d[1]
    lsr     x7, x5, #32
    lsr     x8, x6, #32
    ldr     d24, [x17, w5, uxtw #3]
    ldr     d25, [x17, x7, lsl #3]
    ins     v24.d[1], v25.d[0]
    ldr     d25, [x17, w6, uxtw #3]
    ldr     d26, [x17, x8, lsl #3]
    ins     v25.d[1], v26.d[0]
    ldr     q26, [x9, #416]
    fcmgt   v24.2d, v26.2d, v24.2d
    ldr     q26, [x9, #432]
    fcmgt   v25.2d, v26.2d, v25.2d
    uzp1    v24.4s, v24.4s, v25.4s  // four 32-bit masks
    mls     v21.4s, v24.4s, v31.4s    // base += h where the mask is -1
    fmov    x5, d22
    mov     x6, v22.d[1]
    lsr     x7, x5, #32
    lsr     x8, x6, #32
    ldr     d24, [x17, w5, uxtw #3]
    ldr     d25, [x17, x7, lsl #3]
    ins     v24.d[1], v25.d[0]
    ldr     d25, [x17, w6, uxtw #3]
    ldr     d26, [x17, x8, lsl #3]
    ins     v25.d[1], v26.d[0]
    ldr     q26, [x9, #448]
    fcmgt   v24.2d, v26.2d, v24.2d
    ldr     q26, [x9, #464]
    fcmgt   v25.2d, v26.2d, v25.2d
    uzp1    v24.4s, v24.4s, v25.4s  // four 32-bit masks
    mls     v22.4s, v24.4s, v31.4s    // base += h where the mask is -1
    fmov    x5, d23
    mov     x6, v23.d[1]
    lsr     x7, x5, #32
    lsr     x8, x6, #32
    ldr     d24, [x17, w5, uxtw #3]
    ldr     d25, [x17, x7, lsl #3]
    ins     v24.d[1], v25.d[0]
    ldr     d25, [x17, w6, uxtw #3]
    ldr     d26, [x17, x8, lsl #3]
    ins     v25.d[1], v26.d[0]
    ldr     q26, [x9, #480]
    fcmgt   v24.2d, v26.2d, v24.2d
    ldr     q26, [x9, #496]
    fcmgt   v25.2d, v26.2d, v25.2d
    uzp1    v24.4s, v24.4s, v25.4s  // four 32-bit masks
    mls     v23.4s, v24.4s, v31.4s    // base += h where the mask is -1
    lsr     x7, x10, #1
    cmp     x7, #0
    csinc   x10, x7, xzr, ne        // h = max(h / 2, 1): the last probe is h = 1 again
    add     x17, x0, x10, lsl #3
    sub     x17, x17, #8
    dup     v31.4s, w10
    subs    x16, x16, #1
    b.ne    Lv64_pass
    add     x7, x9, x12             // group's out
    ushll   v24.2d, v0.2s, #0
    ushll2  v25.2d, v0.4s, #0
    stp     q24, q25, [x7, #0]
    ushll   v24.2d, v1.2s, #0
    ushll2  v25.2d, v1.4s, #0
    stp     q24, q25, [x7, #32]
    ushll   v24.2d, v2.2s, #0
    ushll2  v25.2d, v2.4s, #0
    stp     q24, q25, [x7, #64]
    ushll   v24.2d, v3.2s, #0
    ushll2  v25.2d, v3.4s, #0
    stp     q24, q25, [x7, #96]
    ushll   v24.2d, v4.2s, #0
    ushll2  v25.2d, v4.4s, #0
    stp     q24, q25, [x7, #128]
    ushll   v24.2d, v5.2s, #0
    ushll2  v25.2d, v5.4s, #0
    stp     q24, q25, [x7, #160]
    ushll   v24.2d, v6.2s, #0
    ushll2  v25.2d, v6.4s, #0
    stp     q24, q25, [x7, #192]
    ushll   v24.2d, v7.2s, #0
    ushll2  v25.2d, v7.4s, #0
    stp     q24, q25, [x7, #224]
    ushll   v24.2d, v16.2s, #0
    ushll2  v25.2d, v16.4s, #0
    stp     q24, q25, [x7, #256]
    ushll   v24.2d, v17.2s, #0
    ushll2  v25.2d, v17.4s, #0
    stp     q24, q25, [x7, #288]
    ushll   v24.2d, v18.2s, #0
    ushll2  v25.2d, v18.4s, #0
    stp     q24, q25, [x7, #320]
    ushll   v24.2d, v19.2s, #0
    ushll2  v25.2d, v19.4s, #0
    stp     q24, q25, [x7, #352]
    ushll   v24.2d, v20.2s, #0
    ushll2  v25.2d, v20.4s, #0
    stp     q24, q25, [x7, #384]
    ushll   v24.2d, v21.2s, #0
    ushll2  v25.2d, v21.4s, #0
    stp     q24, q25, [x7, #416]
    ushll   v24.2d, v22.2s, #0
    ushll2  v25.2d, v22.4s, #0
    stp     q24, q25, [x7, #448]
    ushll   v24.2d, v23.2s, #0
    ushll2  v25.2d, v23.4s, #0
    stp     q24, q25, [x7, #480]
    fmax    v29.2d, v29.2d, v30.2d
    fmaxp   d29, v29.2d
    fcmp    d29, d29
    b.vs    Lv64_nanfix
Lv64_next:
    mov     x6, v28.d[0]
    and     x6, x6, #0xffffffff     // n - p again (x6 was scratch)
    add     x9, x9, #512
    cmp     x9, x11
    b.ls    Lv64_group
    add     x7, x11, #512           // end of the needles
    cmp     x9, x7
    b.eq    Ldone
    mov     x9, x11                 // leftovers: redo the last 64 needles
    b       Lv64_group
Lv64_first:                         // base = (a[p - 1] < key) ? n - p : 0
    ldp     q24, q25, [x9, #0]
    fcmgt   v24.2d, v24.2d, v27.2d
    fcmgt   v25.2d, v25.2d, v27.2d
    uzp1    v0.4s, v24.4s, v25.4s
    and     v0.16b, v0.16b, v28.16b
    ldp     q24, q25, [x9, #32]
    fcmgt   v24.2d, v24.2d, v27.2d
    fcmgt   v25.2d, v25.2d, v27.2d
    uzp1    v1.4s, v24.4s, v25.4s
    and     v1.16b, v1.16b, v28.16b
    ldp     q24, q25, [x9, #64]
    fcmgt   v24.2d, v24.2d, v27.2d
    fcmgt   v25.2d, v25.2d, v27.2d
    uzp1    v2.4s, v24.4s, v25.4s
    and     v2.16b, v2.16b, v28.16b
    ldp     q24, q25, [x9, #96]
    fcmgt   v24.2d, v24.2d, v27.2d
    fcmgt   v25.2d, v25.2d, v27.2d
    uzp1    v3.4s, v24.4s, v25.4s
    and     v3.16b, v3.16b, v28.16b
    ldp     q24, q25, [x9, #128]
    fcmgt   v24.2d, v24.2d, v27.2d
    fcmgt   v25.2d, v25.2d, v27.2d
    uzp1    v4.4s, v24.4s, v25.4s
    and     v4.16b, v4.16b, v28.16b
    ldp     q24, q25, [x9, #160]
    fcmgt   v24.2d, v24.2d, v27.2d
    fcmgt   v25.2d, v25.2d, v27.2d
    uzp1    v5.4s, v24.4s, v25.4s
    and     v5.16b, v5.16b, v28.16b
    ldp     q24, q25, [x9, #192]
    fcmgt   v24.2d, v24.2d, v27.2d
    fcmgt   v25.2d, v25.2d, v27.2d
    uzp1    v6.4s, v24.4s, v25.4s
    and     v6.16b, v6.16b, v28.16b
    ldp     q24, q25, [x9, #224]
    fcmgt   v24.2d, v24.2d, v27.2d
    fcmgt   v25.2d, v25.2d, v27.2d
    uzp1    v7.4s, v24.4s, v25.4s
    and     v7.16b, v7.16b, v28.16b
    ldp     q24, q25, [x9, #256]
    fcmgt   v24.2d, v24.2d, v27.2d
    fcmgt   v25.2d, v25.2d, v27.2d
    uzp1    v16.4s, v24.4s, v25.4s
    and     v16.16b, v16.16b, v28.16b
    ldp     q24, q25, [x9, #288]
    fcmgt   v24.2d, v24.2d, v27.2d
    fcmgt   v25.2d, v25.2d, v27.2d
    uzp1    v17.4s, v24.4s, v25.4s
    and     v17.16b, v17.16b, v28.16b
    ldp     q24, q25, [x9, #320]
    fcmgt   v24.2d, v24.2d, v27.2d
    fcmgt   v25.2d, v25.2d, v27.2d
    uzp1    v18.4s, v24.4s, v25.4s
    and     v18.16b, v18.16b, v28.16b
    ldp     q24, q25, [x9, #352]
    fcmgt   v24.2d, v24.2d, v27.2d
    fcmgt   v25.2d, v25.2d, v27.2d
    uzp1    v19.4s, v24.4s, v25.4s
    and     v19.16b, v19.16b, v28.16b
    ldp     q24, q25, [x9, #384]
    fcmgt   v24.2d, v24.2d, v27.2d
    fcmgt   v25.2d, v25.2d, v27.2d
    uzp1    v20.4s, v24.4s, v25.4s
    and     v20.16b, v20.16b, v28.16b
    ldp     q24, q25, [x9, #416]
    fcmgt   v24.2d, v24.2d, v27.2d
    fcmgt   v25.2d, v25.2d, v27.2d
    uzp1    v21.4s, v24.4s, v25.4s
    and     v21.16b, v21.16b, v28.16b
    ldp     q24, q25, [x9, #448]
    fcmgt   v24.2d, v24.2d, v27.2d
    fcmgt   v25.2d, v25.2d, v27.2d
    uzp1    v22.4s, v24.4s, v25.4s
    and     v22.16b, v22.16b, v28.16b
    ldp     q24, q25, [x9, #480]
    fcmgt   v24.2d, v24.2d, v27.2d
    fcmgt   v25.2d, v25.2d, v27.2d
    uzp1    v23.4s, v24.4s, v25.4s
    and     v23.16b, v23.16b, v28.16b
    b       Lv64_start
Lv64_nanfix:
    mov     x10, x7
    mov     x8, #64
    mov     x16, #1
    b       Lnanfix

    // ---- Large n: the array is far bigger than the cache, so what matters is
    // how many probes are in flight at once. Needles are taken 128 at a time
    // and the block is swept once per step, with the bases kept in `out`; the
    // processor then has the loads of about a hundred needles outstanding
    // instead of sixteen.
    // x0 a, x3 n, x4 k, x9 block's needles, x10 block's out, x11 last block's
    // needles, x12 out - v, x13 h, x17 &a[h - 1], x16 passes left, x8 NaN seen
Lbig:
    dup     v27.2d, v27.d[0]
    dup     v28.2d, x6
    sub     x9, x14, #128
    add     x11, x1, x9, lsl #3
    sub     x12, x2, x1
    mov     x9, x1
Lbig_block:
    add     x10, x9, x12
    mov     x14, x9
    mov     x15, x10
    mov     x7, #32
    movi    v29.2d, #0
    movi    v30.2d, #0
Lbig_init:                          // base = (a[p - 1] < key) ? n - p : 0
    ldp     q0, q1, [x14], #32
    fmax    v29.2d, v29.2d, v0.2d
    fmax    v30.2d, v30.2d, v1.2d
    fcmgt   v2.2d, v0.2d, v27.2d
    fcmgt   v3.2d, v1.2d, v27.2d
    and     v2.16b, v2.16b, v28.16b
    and     v3.16b, v3.16b, v28.16b
    stp     q2, q3, [x15], #32
    subs    x7, x7, #1
    b.ne    Lbig_init
    fmax    v29.2d, v29.2d, v30.2d
    fmaxp   d29, v29.2d
    fcmp    d29, d29
    cset    x8, vs                  // some needle in the block is NaN
    mov     x13, #1
    sub     x6, x4, #1
    lsl     x13, x13, x6            // h = p/2
    add     x16, x4, #1             // k steps and the last probe
Lbig_pass:
    add     x17, x0, x13, lsl #3
    sub     x17, x17, #8            // &a[h - 1]
    mov     x14, x9
    mov     x15, x10
    mov     x7, #64
Lbig_probe:
    ldp     x1, x2, [x15]
    ldp     d0, d1, [x14], #16
    ldr     d24, [x17, x1, lsl #3]
    ldr     d25, [x17, x2, lsl #3]
    fcmgt   d24, d0, d24
    fcmgt   d25, d1, d25
    fmov    x6, d24
    and     x6, x6, x13
    add     x1, x1, x6
    fmov    x6, d25
    and     x6, x6, x13
    add     x2, x2, x6
    stp     x1, x2, [x15], #16
    subs    x7, x7, #1
    b.ne    Lbig_probe
    lsr     x6, x13, #1
    cmp     x6, #0
    csinc   x13, x6, xzr, ne        // h = max(h / 2, 1): the last probe is h = 1 again
    subs    x16, x16, #1
    b.ne    Lbig_pass
    cbnz    x8, Lbig_nanfix
Lbig_next:
    add     x9, x9, #1024
    cmp     x9, x11
    b.ls    Lbig_block
    add     x6, x11, #1024       // end of the needles
    cmp     x9, x6
    b.eq    Ldone
    mov     x9, x11                 // leftovers: redo the last 128 needles
    b       Lbig_block

    // A NaN needle compared false everywhere. Its answer is the index of the
    // first NaN in a (or n). Rare path.
Lbig_nanfix:
    mov     x8, #128
    mov     x16, #0
    // x9 needles, x10 their out, x8 how many, x16 which path to return to
Lnanfix:
    mov     x14, x9
    mov     x15, x10
Lbig_nf_loop:
    ldr     d0, [x14], #8
    fcmp    d0, d0
    b.vc    Lbig_nf_skip
    mov     x1, #0
    mov     x2, x3
Lbig_nf_search:
    cmp     x1, x2
    b.hs    Lbig_nf_store
    sub     x6, x2, x1
    add     x6, x1, x6, lsr #1
    ldr     d1, [x0, x6, lsl #3]
    fcmp    d1, d1
    csel    x2, x6, x2, vs          // a[mid] is NaN: hi = mid
    csinc   x1, x1, x6, vs          // otherwise lo = mid + 1
    b       Lbig_nf_search
Lbig_nf_store:
    str     x1, [x15]
Lbig_nf_skip:
    add     x15, x15, #8
    subs    x8, x8, #1
    b.ne    Lbig_nf_loop
    cbnz    x16, Lv64_next
    b       Lbig_next

Lnan_group:
    fmov    x0, d26
    mov     x3, v26.d[1]
    fmov    x1, d25
    mov     x2, v25.d[1]
    mov     x4, #16
    mov     x8, #1                  // come back to Lnext afterwards
    b       Lscalar
Lsmall:                             // x14 needles, one at a time
    cbz     x14, Ldone
    mov     x4, x14
    mov     x8, #0
    // Plain binary search for x4 needles of any value. Rare path.
    // x0 a, x1 needle, x2 out, x3 n; x5 lo, x6 hi, x7 mid.
Lscalar:
    ldr     d0, [x1], #8            // key
    mov     x5, #0
    mov     x6, x3
    fcmp    d0, d0
    b.vs    Lnan_search
Lsearch:
    cmp     x5, x6
    b.hs    Lstore
    sub     x7, x6, x5
    add     x7, x5, x7, lsr #1
    ldr     d1, [x0, x7, lsl #3]
    fcmp    d1, d0
    b.mi    Lright                  // a[mid] < key (false when unordered)
    mov     x6, x7
    b       Lsearch
Lright:
    add     x5, x7, #1
    b       Lsearch
Lnan_search:                        // NaN key: less(x, key) is "x is not NaN"
    cmp     x5, x6
    b.hs    Lstore
    sub     x7, x6, x5
    add     x7, x5, x7, lsr #1
    ldr     d1, [x0, x7, lsl #3]
    fcmp    d1, d1
    b.vc    Lnan_right
    mov     x6, x7
    b       Lnan_search
Lnan_right:
    add     x5, x7, #1
    b       Lnan_search
Lstore:
    str     x5, [x2], #8
    subs    x4, x4, #1
    b.ne    Lscalar
    cbnz    x8, Lnext
    ret
