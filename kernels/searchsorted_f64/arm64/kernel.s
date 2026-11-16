// void searchsorted_f64(const double *a, const double *v, int64_t *out, size_t n);
//
// For each needle v[j], out[j] = first index i with !less(a[i], v[j]), where
// less(x, y) = x < y || (y is NaN && x is not NaN).
//
// Sixteen needles are searched at once with a branchless lower bound. Let p be
// the largest power of two <= n. Each needle keeps an index `base`, with the
// answer always in [base, base + len]:
//     first:  base = (a[p - 1] < key) ? n - p : 0          len = p
//     then for h = p/2, p/4, ..., 1:
//             if (a[base + h - 1] < key) base += h          len = h
//     last:   if (a[base] < key) base += 1
// The probes of the sixteen searches are independent, so their loads overlap.
// When n is a power of two the first step always gives 0 and is skipped.
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
// A NaN needle makes every compare false, so a group containing one is handed
// to the plain one-at-a-time search at Lscalar, as are all needles when n < 16.
// Leftover needles (n not a multiple of 16) are covered by one more group that
// ends exactly at the last needle and recomputes a few results.
//
// From 1,048,576 elements up the array is far larger than the cache and the
// search is done a different way, at Lbig (described there).
//
// All sixteen bases live in x0-x15, so everything else is parked in vector
// registers and fetched when needed:
//   x0-x15   bases          x16 scratch        x17 &a[h - 1]
//   d0-d7, d16-d23  keys    d24, d29 scratch
//   v25 = [next needle, next out]       v26 = [a, n]
//   v27 = [a[p - 1], entry block]       v28 = [&a[p/2 - 1], n - p]
//   v30 = [128, 128]                    v31 = [last full group's needle, out]
.text
.globl _searchsorted_f64
.p2align 2
_searchsorted_f64:
    cmp     x3, #16
    b.lo    Lsmall
    clz     x4, x3
    mov     x5, #63
    sub     x4, x5, x4              // k = floor(log2(n)), p = 1 << k
    cmp     x4, #32
    b.hi    Lsmall                  // more levels than are written out
    mov     x5, #1
    lsl     x5, x5, x4              // p
    sub     x6, x3, x5              // n - p
    add     x7, x0, x5, lsl #3
    ldur    d27, [x7, #-8]          // a[p - 1]
    mov     x9, #1048576
    cmp     x3, x9
    b.hs    Lbig
    add     x7, x0, x5, lsl #2
    sub     x7, x7, #8              // &a[p/2 - 1]
    fmov    d28, x7
    mov     v28.d[1], x6
    fmov    d26, x0
    mov     v26.d[1], x3
    fmov    d25, x1
    mov     v25.d[1], x2
    sub     x9, x3, #16
    add     x10, x1, x9, lsl #3
    add     x11, x2, x9, lsl #3
    fmov    d31, x10
    mov     v31.d[1], x11
    mov     x9, #128
    dup     v30.2d, x9
    adr     x9, Lfinal
    mov     x10, #264
    msub    x9, x4, x10, x9         // k blocks before the final one
    mov     v27.d[1], x9
    mov     x16, x1
Lgroup:                             // x16 = this group's first needle
    ldp     d0, d1, [x16, #0]
    ldp     d2, d3, [x16, #16]
    ldp     d4, d5, [x16, #32]
    ldp     d6, d7, [x16, #48]
    ldp     d16, d17, [x16, #64]
    ldp     d18, d19, [x16, #80]
    ldp     d20, d21, [x16, #96]
    ldp     d22, d23, [x16, #112]
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
Lblock_31:                          // h = 2^31
    ldr     d24, [x17, x0, lsl #3]
    fcmgt   d24, d0, d24
    fmov    x16, d24
    sub     x0, x0, x16, lsl #31
    ldr     d24, [x17, x1, lsl #3]
    fcmgt   d24, d1, d24
    fmov    x16, d24
    sub     x1, x1, x16, lsl #31
    ldr     d24, [x17, x2, lsl #3]
    fcmgt   d24, d2, d24
    fmov    x16, d24
    sub     x2, x2, x16, lsl #31
    ldr     d24, [x17, x3, lsl #3]
    fcmgt   d24, d3, d24
    fmov    x16, d24
    sub     x3, x3, x16, lsl #31
    ldr     d24, [x17, x4, lsl #3]
    fcmgt   d24, d4, d24
    fmov    x16, d24
    sub     x4, x4, x16, lsl #31
    ldr     d24, [x17, x5, lsl #3]
    fcmgt   d24, d5, d24
    fmov    x16, d24
    sub     x5, x5, x16, lsl #31
    ldr     d24, [x17, x6, lsl #3]
    fcmgt   d24, d6, d24
    fmov    x16, d24
    sub     x6, x6, x16, lsl #31
    ldr     d24, [x17, x7, lsl #3]
    fcmgt   d24, d7, d24
    fmov    x16, d24
    sub     x7, x7, x16, lsl #31
    ldr     d24, [x17, x8, lsl #3]
    fcmgt   d24, d16, d24
    fmov    x16, d24
    sub     x8, x8, x16, lsl #31
    ldr     d24, [x17, x9, lsl #3]
    fcmgt   d24, d17, d24
    fmov    x16, d24
    sub     x9, x9, x16, lsl #31
    ldr     d24, [x17, x10, lsl #3]
    fcmgt   d24, d18, d24
    fmov    x16, d24
    sub     x10, x10, x16, lsl #31
    ldr     d24, [x17, x11, lsl #3]
    fcmgt   d24, d19, d24
    fmov    x16, d24
    sub     x11, x11, x16, lsl #31
    ldr     d24, [x17, x12, lsl #3]
    fcmgt   d24, d20, d24
    fmov    x16, d24
    sub     x12, x12, x16, lsl #31
    ldr     d24, [x17, x13, lsl #3]
    fcmgt   d24, d21, d24
    fmov    x16, d24
    sub     x13, x13, x16, lsl #31
    ldr     d24, [x17, x14, lsl #3]
    fcmgt   d24, d22, d24
    fmov    x16, d24
    sub     x14, x14, x16, lsl #31
    ldr     d24, [x17, x15, lsl #3]
    fcmgt   d24, d23, d24
    fmov    x16, d24
    sub     x15, x15, x16, lsl #31
    mov     x16, #8589934592
    sub     x17, x17, x16           // &a[h/2 - 1], or a after h = 1
Lblock_30:                          // h = 2^30
    ldr     d24, [x17, x0, lsl #3]
    fcmgt   d24, d0, d24
    fmov    x16, d24
    sub     x0, x0, x16, lsl #30
    ldr     d24, [x17, x1, lsl #3]
    fcmgt   d24, d1, d24
    fmov    x16, d24
    sub     x1, x1, x16, lsl #30
    ldr     d24, [x17, x2, lsl #3]
    fcmgt   d24, d2, d24
    fmov    x16, d24
    sub     x2, x2, x16, lsl #30
    ldr     d24, [x17, x3, lsl #3]
    fcmgt   d24, d3, d24
    fmov    x16, d24
    sub     x3, x3, x16, lsl #30
    ldr     d24, [x17, x4, lsl #3]
    fcmgt   d24, d4, d24
    fmov    x16, d24
    sub     x4, x4, x16, lsl #30
    ldr     d24, [x17, x5, lsl #3]
    fcmgt   d24, d5, d24
    fmov    x16, d24
    sub     x5, x5, x16, lsl #30
    ldr     d24, [x17, x6, lsl #3]
    fcmgt   d24, d6, d24
    fmov    x16, d24
    sub     x6, x6, x16, lsl #30
    ldr     d24, [x17, x7, lsl #3]
    fcmgt   d24, d7, d24
    fmov    x16, d24
    sub     x7, x7, x16, lsl #30
    ldr     d24, [x17, x8, lsl #3]
    fcmgt   d24, d16, d24
    fmov    x16, d24
    sub     x8, x8, x16, lsl #30
    ldr     d24, [x17, x9, lsl #3]
    fcmgt   d24, d17, d24
    fmov    x16, d24
    sub     x9, x9, x16, lsl #30
    ldr     d24, [x17, x10, lsl #3]
    fcmgt   d24, d18, d24
    fmov    x16, d24
    sub     x10, x10, x16, lsl #30
    ldr     d24, [x17, x11, lsl #3]
    fcmgt   d24, d19, d24
    fmov    x16, d24
    sub     x11, x11, x16, lsl #30
    ldr     d24, [x17, x12, lsl #3]
    fcmgt   d24, d20, d24
    fmov    x16, d24
    sub     x12, x12, x16, lsl #30
    ldr     d24, [x17, x13, lsl #3]
    fcmgt   d24, d21, d24
    fmov    x16, d24
    sub     x13, x13, x16, lsl #30
    ldr     d24, [x17, x14, lsl #3]
    fcmgt   d24, d22, d24
    fmov    x16, d24
    sub     x14, x14, x16, lsl #30
    ldr     d24, [x17, x15, lsl #3]
    fcmgt   d24, d23, d24
    fmov    x16, d24
    sub     x15, x15, x16, lsl #30
    mov     x16, #4294967296
    sub     x17, x17, x16           // &a[h/2 - 1], or a after h = 1
Lblock_29:                          // h = 2^29
    ldr     d24, [x17, x0, lsl #3]
    fcmgt   d24, d0, d24
    fmov    x16, d24
    sub     x0, x0, x16, lsl #29
    ldr     d24, [x17, x1, lsl #3]
    fcmgt   d24, d1, d24
    fmov    x16, d24
    sub     x1, x1, x16, lsl #29
    ldr     d24, [x17, x2, lsl #3]
    fcmgt   d24, d2, d24
    fmov    x16, d24
    sub     x2, x2, x16, lsl #29
    ldr     d24, [x17, x3, lsl #3]
    fcmgt   d24, d3, d24
    fmov    x16, d24
    sub     x3, x3, x16, lsl #29
    ldr     d24, [x17, x4, lsl #3]
    fcmgt   d24, d4, d24
    fmov    x16, d24
    sub     x4, x4, x16, lsl #29
    ldr     d24, [x17, x5, lsl #3]
    fcmgt   d24, d5, d24
    fmov    x16, d24
    sub     x5, x5, x16, lsl #29
    ldr     d24, [x17, x6, lsl #3]
    fcmgt   d24, d6, d24
    fmov    x16, d24
    sub     x6, x6, x16, lsl #29
    ldr     d24, [x17, x7, lsl #3]
    fcmgt   d24, d7, d24
    fmov    x16, d24
    sub     x7, x7, x16, lsl #29
    ldr     d24, [x17, x8, lsl #3]
    fcmgt   d24, d16, d24
    fmov    x16, d24
    sub     x8, x8, x16, lsl #29
    ldr     d24, [x17, x9, lsl #3]
    fcmgt   d24, d17, d24
    fmov    x16, d24
    sub     x9, x9, x16, lsl #29
    ldr     d24, [x17, x10, lsl #3]
    fcmgt   d24, d18, d24
    fmov    x16, d24
    sub     x10, x10, x16, lsl #29
    ldr     d24, [x17, x11, lsl #3]
    fcmgt   d24, d19, d24
    fmov    x16, d24
    sub     x11, x11, x16, lsl #29
    ldr     d24, [x17, x12, lsl #3]
    fcmgt   d24, d20, d24
    fmov    x16, d24
    sub     x12, x12, x16, lsl #29
    ldr     d24, [x17, x13, lsl #3]
    fcmgt   d24, d21, d24
    fmov    x16, d24
    sub     x13, x13, x16, lsl #29
    ldr     d24, [x17, x14, lsl #3]
    fcmgt   d24, d22, d24
    fmov    x16, d24
    sub     x14, x14, x16, lsl #29
    ldr     d24, [x17, x15, lsl #3]
    fcmgt   d24, d23, d24
    fmov    x16, d24
    sub     x15, x15, x16, lsl #29
    mov     x16, #2147483648
    sub     x17, x17, x16           // &a[h/2 - 1], or a after h = 1
Lblock_28:                          // h = 2^28
    ldr     d24, [x17, x0, lsl #3]
    fcmgt   d24, d0, d24
    fmov    x16, d24
    sub     x0, x0, x16, lsl #28
    ldr     d24, [x17, x1, lsl #3]
    fcmgt   d24, d1, d24
    fmov    x16, d24
    sub     x1, x1, x16, lsl #28
    ldr     d24, [x17, x2, lsl #3]
    fcmgt   d24, d2, d24
    fmov    x16, d24
    sub     x2, x2, x16, lsl #28
    ldr     d24, [x17, x3, lsl #3]
    fcmgt   d24, d3, d24
    fmov    x16, d24
    sub     x3, x3, x16, lsl #28
    ldr     d24, [x17, x4, lsl #3]
    fcmgt   d24, d4, d24
    fmov    x16, d24
    sub     x4, x4, x16, lsl #28
    ldr     d24, [x17, x5, lsl #3]
    fcmgt   d24, d5, d24
    fmov    x16, d24
    sub     x5, x5, x16, lsl #28
    ldr     d24, [x17, x6, lsl #3]
    fcmgt   d24, d6, d24
    fmov    x16, d24
    sub     x6, x6, x16, lsl #28
    ldr     d24, [x17, x7, lsl #3]
    fcmgt   d24, d7, d24
    fmov    x16, d24
    sub     x7, x7, x16, lsl #28
    ldr     d24, [x17, x8, lsl #3]
    fcmgt   d24, d16, d24
    fmov    x16, d24
    sub     x8, x8, x16, lsl #28
    ldr     d24, [x17, x9, lsl #3]
    fcmgt   d24, d17, d24
    fmov    x16, d24
    sub     x9, x9, x16, lsl #28
    ldr     d24, [x17, x10, lsl #3]
    fcmgt   d24, d18, d24
    fmov    x16, d24
    sub     x10, x10, x16, lsl #28
    ldr     d24, [x17, x11, lsl #3]
    fcmgt   d24, d19, d24
    fmov    x16, d24
    sub     x11, x11, x16, lsl #28
    ldr     d24, [x17, x12, lsl #3]
    fcmgt   d24, d20, d24
    fmov    x16, d24
    sub     x12, x12, x16, lsl #28
    ldr     d24, [x17, x13, lsl #3]
    fcmgt   d24, d21, d24
    fmov    x16, d24
    sub     x13, x13, x16, lsl #28
    ldr     d24, [x17, x14, lsl #3]
    fcmgt   d24, d22, d24
    fmov    x16, d24
    sub     x14, x14, x16, lsl #28
    ldr     d24, [x17, x15, lsl #3]
    fcmgt   d24, d23, d24
    fmov    x16, d24
    sub     x15, x15, x16, lsl #28
    mov     x16, #1073741824
    sub     x17, x17, x16           // &a[h/2 - 1], or a after h = 1
Lblock_27:                          // h = 2^27
    ldr     d24, [x17, x0, lsl #3]
    fcmgt   d24, d0, d24
    fmov    x16, d24
    sub     x0, x0, x16, lsl #27
    ldr     d24, [x17, x1, lsl #3]
    fcmgt   d24, d1, d24
    fmov    x16, d24
    sub     x1, x1, x16, lsl #27
    ldr     d24, [x17, x2, lsl #3]
    fcmgt   d24, d2, d24
    fmov    x16, d24
    sub     x2, x2, x16, lsl #27
    ldr     d24, [x17, x3, lsl #3]
    fcmgt   d24, d3, d24
    fmov    x16, d24
    sub     x3, x3, x16, lsl #27
    ldr     d24, [x17, x4, lsl #3]
    fcmgt   d24, d4, d24
    fmov    x16, d24
    sub     x4, x4, x16, lsl #27
    ldr     d24, [x17, x5, lsl #3]
    fcmgt   d24, d5, d24
    fmov    x16, d24
    sub     x5, x5, x16, lsl #27
    ldr     d24, [x17, x6, lsl #3]
    fcmgt   d24, d6, d24
    fmov    x16, d24
    sub     x6, x6, x16, lsl #27
    ldr     d24, [x17, x7, lsl #3]
    fcmgt   d24, d7, d24
    fmov    x16, d24
    sub     x7, x7, x16, lsl #27
    ldr     d24, [x17, x8, lsl #3]
    fcmgt   d24, d16, d24
    fmov    x16, d24
    sub     x8, x8, x16, lsl #27
    ldr     d24, [x17, x9, lsl #3]
    fcmgt   d24, d17, d24
    fmov    x16, d24
    sub     x9, x9, x16, lsl #27
    ldr     d24, [x17, x10, lsl #3]
    fcmgt   d24, d18, d24
    fmov    x16, d24
    sub     x10, x10, x16, lsl #27
    ldr     d24, [x17, x11, lsl #3]
    fcmgt   d24, d19, d24
    fmov    x16, d24
    sub     x11, x11, x16, lsl #27
    ldr     d24, [x17, x12, lsl #3]
    fcmgt   d24, d20, d24
    fmov    x16, d24
    sub     x12, x12, x16, lsl #27
    ldr     d24, [x17, x13, lsl #3]
    fcmgt   d24, d21, d24
    fmov    x16, d24
    sub     x13, x13, x16, lsl #27
    ldr     d24, [x17, x14, lsl #3]
    fcmgt   d24, d22, d24
    fmov    x16, d24
    sub     x14, x14, x16, lsl #27
    ldr     d24, [x17, x15, lsl #3]
    fcmgt   d24, d23, d24
    fmov    x16, d24
    sub     x15, x15, x16, lsl #27
    mov     x16, #536870912
    sub     x17, x17, x16           // &a[h/2 - 1], or a after h = 1
Lblock_26:                          // h = 2^26
    ldr     d24, [x17, x0, lsl #3]
    fcmgt   d24, d0, d24
    fmov    x16, d24
    sub     x0, x0, x16, lsl #26
    ldr     d24, [x17, x1, lsl #3]
    fcmgt   d24, d1, d24
    fmov    x16, d24
    sub     x1, x1, x16, lsl #26
    ldr     d24, [x17, x2, lsl #3]
    fcmgt   d24, d2, d24
    fmov    x16, d24
    sub     x2, x2, x16, lsl #26
    ldr     d24, [x17, x3, lsl #3]
    fcmgt   d24, d3, d24
    fmov    x16, d24
    sub     x3, x3, x16, lsl #26
    ldr     d24, [x17, x4, lsl #3]
    fcmgt   d24, d4, d24
    fmov    x16, d24
    sub     x4, x4, x16, lsl #26
    ldr     d24, [x17, x5, lsl #3]
    fcmgt   d24, d5, d24
    fmov    x16, d24
    sub     x5, x5, x16, lsl #26
    ldr     d24, [x17, x6, lsl #3]
    fcmgt   d24, d6, d24
    fmov    x16, d24
    sub     x6, x6, x16, lsl #26
    ldr     d24, [x17, x7, lsl #3]
    fcmgt   d24, d7, d24
    fmov    x16, d24
    sub     x7, x7, x16, lsl #26
    ldr     d24, [x17, x8, lsl #3]
    fcmgt   d24, d16, d24
    fmov    x16, d24
    sub     x8, x8, x16, lsl #26
    ldr     d24, [x17, x9, lsl #3]
    fcmgt   d24, d17, d24
    fmov    x16, d24
    sub     x9, x9, x16, lsl #26
    ldr     d24, [x17, x10, lsl #3]
    fcmgt   d24, d18, d24
    fmov    x16, d24
    sub     x10, x10, x16, lsl #26
    ldr     d24, [x17, x11, lsl #3]
    fcmgt   d24, d19, d24
    fmov    x16, d24
    sub     x11, x11, x16, lsl #26
    ldr     d24, [x17, x12, lsl #3]
    fcmgt   d24, d20, d24
    fmov    x16, d24
    sub     x12, x12, x16, lsl #26
    ldr     d24, [x17, x13, lsl #3]
    fcmgt   d24, d21, d24
    fmov    x16, d24
    sub     x13, x13, x16, lsl #26
    ldr     d24, [x17, x14, lsl #3]
    fcmgt   d24, d22, d24
    fmov    x16, d24
    sub     x14, x14, x16, lsl #26
    ldr     d24, [x17, x15, lsl #3]
    fcmgt   d24, d23, d24
    fmov    x16, d24
    sub     x15, x15, x16, lsl #26
    mov     x16, #268435456
    sub     x17, x17, x16           // &a[h/2 - 1], or a after h = 1
Lblock_25:                          // h = 2^25
    ldr     d24, [x17, x0, lsl #3]
    fcmgt   d24, d0, d24
    fmov    x16, d24
    sub     x0, x0, x16, lsl #25
    ldr     d24, [x17, x1, lsl #3]
    fcmgt   d24, d1, d24
    fmov    x16, d24
    sub     x1, x1, x16, lsl #25
    ldr     d24, [x17, x2, lsl #3]
    fcmgt   d24, d2, d24
    fmov    x16, d24
    sub     x2, x2, x16, lsl #25
    ldr     d24, [x17, x3, lsl #3]
    fcmgt   d24, d3, d24
    fmov    x16, d24
    sub     x3, x3, x16, lsl #25
    ldr     d24, [x17, x4, lsl #3]
    fcmgt   d24, d4, d24
    fmov    x16, d24
    sub     x4, x4, x16, lsl #25
    ldr     d24, [x17, x5, lsl #3]
    fcmgt   d24, d5, d24
    fmov    x16, d24
    sub     x5, x5, x16, lsl #25
    ldr     d24, [x17, x6, lsl #3]
    fcmgt   d24, d6, d24
    fmov    x16, d24
    sub     x6, x6, x16, lsl #25
    ldr     d24, [x17, x7, lsl #3]
    fcmgt   d24, d7, d24
    fmov    x16, d24
    sub     x7, x7, x16, lsl #25
    ldr     d24, [x17, x8, lsl #3]
    fcmgt   d24, d16, d24
    fmov    x16, d24
    sub     x8, x8, x16, lsl #25
    ldr     d24, [x17, x9, lsl #3]
    fcmgt   d24, d17, d24
    fmov    x16, d24
    sub     x9, x9, x16, lsl #25
    ldr     d24, [x17, x10, lsl #3]
    fcmgt   d24, d18, d24
    fmov    x16, d24
    sub     x10, x10, x16, lsl #25
    ldr     d24, [x17, x11, lsl #3]
    fcmgt   d24, d19, d24
    fmov    x16, d24
    sub     x11, x11, x16, lsl #25
    ldr     d24, [x17, x12, lsl #3]
    fcmgt   d24, d20, d24
    fmov    x16, d24
    sub     x12, x12, x16, lsl #25
    ldr     d24, [x17, x13, lsl #3]
    fcmgt   d24, d21, d24
    fmov    x16, d24
    sub     x13, x13, x16, lsl #25
    ldr     d24, [x17, x14, lsl #3]
    fcmgt   d24, d22, d24
    fmov    x16, d24
    sub     x14, x14, x16, lsl #25
    ldr     d24, [x17, x15, lsl #3]
    fcmgt   d24, d23, d24
    fmov    x16, d24
    sub     x15, x15, x16, lsl #25
    mov     x16, #134217728
    sub     x17, x17, x16           // &a[h/2 - 1], or a after h = 1
Lblock_24:                          // h = 2^24
    ldr     d24, [x17, x0, lsl #3]
    fcmgt   d24, d0, d24
    fmov    x16, d24
    sub     x0, x0, x16, lsl #24
    ldr     d24, [x17, x1, lsl #3]
    fcmgt   d24, d1, d24
    fmov    x16, d24
    sub     x1, x1, x16, lsl #24
    ldr     d24, [x17, x2, lsl #3]
    fcmgt   d24, d2, d24
    fmov    x16, d24
    sub     x2, x2, x16, lsl #24
    ldr     d24, [x17, x3, lsl #3]
    fcmgt   d24, d3, d24
    fmov    x16, d24
    sub     x3, x3, x16, lsl #24
    ldr     d24, [x17, x4, lsl #3]
    fcmgt   d24, d4, d24
    fmov    x16, d24
    sub     x4, x4, x16, lsl #24
    ldr     d24, [x17, x5, lsl #3]
    fcmgt   d24, d5, d24
    fmov    x16, d24
    sub     x5, x5, x16, lsl #24
    ldr     d24, [x17, x6, lsl #3]
    fcmgt   d24, d6, d24
    fmov    x16, d24
    sub     x6, x6, x16, lsl #24
    ldr     d24, [x17, x7, lsl #3]
    fcmgt   d24, d7, d24
    fmov    x16, d24
    sub     x7, x7, x16, lsl #24
    ldr     d24, [x17, x8, lsl #3]
    fcmgt   d24, d16, d24
    fmov    x16, d24
    sub     x8, x8, x16, lsl #24
    ldr     d24, [x17, x9, lsl #3]
    fcmgt   d24, d17, d24
    fmov    x16, d24
    sub     x9, x9, x16, lsl #24
    ldr     d24, [x17, x10, lsl #3]
    fcmgt   d24, d18, d24
    fmov    x16, d24
    sub     x10, x10, x16, lsl #24
    ldr     d24, [x17, x11, lsl #3]
    fcmgt   d24, d19, d24
    fmov    x16, d24
    sub     x11, x11, x16, lsl #24
    ldr     d24, [x17, x12, lsl #3]
    fcmgt   d24, d20, d24
    fmov    x16, d24
    sub     x12, x12, x16, lsl #24
    ldr     d24, [x17, x13, lsl #3]
    fcmgt   d24, d21, d24
    fmov    x16, d24
    sub     x13, x13, x16, lsl #24
    ldr     d24, [x17, x14, lsl #3]
    fcmgt   d24, d22, d24
    fmov    x16, d24
    sub     x14, x14, x16, lsl #24
    ldr     d24, [x17, x15, lsl #3]
    fcmgt   d24, d23, d24
    fmov    x16, d24
    sub     x15, x15, x16, lsl #24
    mov     x16, #67108864
    sub     x17, x17, x16           // &a[h/2 - 1], or a after h = 1
Lblock_23:                          // h = 2^23
    ldr     d24, [x17, x0, lsl #3]
    fcmgt   d24, d0, d24
    fmov    x16, d24
    sub     x0, x0, x16, lsl #23
    ldr     d24, [x17, x1, lsl #3]
    fcmgt   d24, d1, d24
    fmov    x16, d24
    sub     x1, x1, x16, lsl #23
    ldr     d24, [x17, x2, lsl #3]
    fcmgt   d24, d2, d24
    fmov    x16, d24
    sub     x2, x2, x16, lsl #23
    ldr     d24, [x17, x3, lsl #3]
    fcmgt   d24, d3, d24
    fmov    x16, d24
    sub     x3, x3, x16, lsl #23
    ldr     d24, [x17, x4, lsl #3]
    fcmgt   d24, d4, d24
    fmov    x16, d24
    sub     x4, x4, x16, lsl #23
    ldr     d24, [x17, x5, lsl #3]
    fcmgt   d24, d5, d24
    fmov    x16, d24
    sub     x5, x5, x16, lsl #23
    ldr     d24, [x17, x6, lsl #3]
    fcmgt   d24, d6, d24
    fmov    x16, d24
    sub     x6, x6, x16, lsl #23
    ldr     d24, [x17, x7, lsl #3]
    fcmgt   d24, d7, d24
    fmov    x16, d24
    sub     x7, x7, x16, lsl #23
    ldr     d24, [x17, x8, lsl #3]
    fcmgt   d24, d16, d24
    fmov    x16, d24
    sub     x8, x8, x16, lsl #23
    ldr     d24, [x17, x9, lsl #3]
    fcmgt   d24, d17, d24
    fmov    x16, d24
    sub     x9, x9, x16, lsl #23
    ldr     d24, [x17, x10, lsl #3]
    fcmgt   d24, d18, d24
    fmov    x16, d24
    sub     x10, x10, x16, lsl #23
    ldr     d24, [x17, x11, lsl #3]
    fcmgt   d24, d19, d24
    fmov    x16, d24
    sub     x11, x11, x16, lsl #23
    ldr     d24, [x17, x12, lsl #3]
    fcmgt   d24, d20, d24
    fmov    x16, d24
    sub     x12, x12, x16, lsl #23
    ldr     d24, [x17, x13, lsl #3]
    fcmgt   d24, d21, d24
    fmov    x16, d24
    sub     x13, x13, x16, lsl #23
    ldr     d24, [x17, x14, lsl #3]
    fcmgt   d24, d22, d24
    fmov    x16, d24
    sub     x14, x14, x16, lsl #23
    ldr     d24, [x17, x15, lsl #3]
    fcmgt   d24, d23, d24
    fmov    x16, d24
    sub     x15, x15, x16, lsl #23
    mov     x16, #33554432
    sub     x17, x17, x16           // &a[h/2 - 1], or a after h = 1
Lblock_22:                          // h = 2^22
    ldr     d24, [x17, x0, lsl #3]
    fcmgt   d24, d0, d24
    fmov    x16, d24
    sub     x0, x0, x16, lsl #22
    ldr     d24, [x17, x1, lsl #3]
    fcmgt   d24, d1, d24
    fmov    x16, d24
    sub     x1, x1, x16, lsl #22
    ldr     d24, [x17, x2, lsl #3]
    fcmgt   d24, d2, d24
    fmov    x16, d24
    sub     x2, x2, x16, lsl #22
    ldr     d24, [x17, x3, lsl #3]
    fcmgt   d24, d3, d24
    fmov    x16, d24
    sub     x3, x3, x16, lsl #22
    ldr     d24, [x17, x4, lsl #3]
    fcmgt   d24, d4, d24
    fmov    x16, d24
    sub     x4, x4, x16, lsl #22
    ldr     d24, [x17, x5, lsl #3]
    fcmgt   d24, d5, d24
    fmov    x16, d24
    sub     x5, x5, x16, lsl #22
    ldr     d24, [x17, x6, lsl #3]
    fcmgt   d24, d6, d24
    fmov    x16, d24
    sub     x6, x6, x16, lsl #22
    ldr     d24, [x17, x7, lsl #3]
    fcmgt   d24, d7, d24
    fmov    x16, d24
    sub     x7, x7, x16, lsl #22
    ldr     d24, [x17, x8, lsl #3]
    fcmgt   d24, d16, d24
    fmov    x16, d24
    sub     x8, x8, x16, lsl #22
    ldr     d24, [x17, x9, lsl #3]
    fcmgt   d24, d17, d24
    fmov    x16, d24
    sub     x9, x9, x16, lsl #22
    ldr     d24, [x17, x10, lsl #3]
    fcmgt   d24, d18, d24
    fmov    x16, d24
    sub     x10, x10, x16, lsl #22
    ldr     d24, [x17, x11, lsl #3]
    fcmgt   d24, d19, d24
    fmov    x16, d24
    sub     x11, x11, x16, lsl #22
    ldr     d24, [x17, x12, lsl #3]
    fcmgt   d24, d20, d24
    fmov    x16, d24
    sub     x12, x12, x16, lsl #22
    ldr     d24, [x17, x13, lsl #3]
    fcmgt   d24, d21, d24
    fmov    x16, d24
    sub     x13, x13, x16, lsl #22
    ldr     d24, [x17, x14, lsl #3]
    fcmgt   d24, d22, d24
    fmov    x16, d24
    sub     x14, x14, x16, lsl #22
    ldr     d24, [x17, x15, lsl #3]
    fcmgt   d24, d23, d24
    fmov    x16, d24
    sub     x15, x15, x16, lsl #22
    mov     x16, #16777216
    sub     x17, x17, x16           // &a[h/2 - 1], or a after h = 1
Lblock_21:                          // h = 2^21
    ldr     d24, [x17, x0, lsl #3]
    fcmgt   d24, d0, d24
    fmov    x16, d24
    sub     x0, x0, x16, lsl #21
    ldr     d24, [x17, x1, lsl #3]
    fcmgt   d24, d1, d24
    fmov    x16, d24
    sub     x1, x1, x16, lsl #21
    ldr     d24, [x17, x2, lsl #3]
    fcmgt   d24, d2, d24
    fmov    x16, d24
    sub     x2, x2, x16, lsl #21
    ldr     d24, [x17, x3, lsl #3]
    fcmgt   d24, d3, d24
    fmov    x16, d24
    sub     x3, x3, x16, lsl #21
    ldr     d24, [x17, x4, lsl #3]
    fcmgt   d24, d4, d24
    fmov    x16, d24
    sub     x4, x4, x16, lsl #21
    ldr     d24, [x17, x5, lsl #3]
    fcmgt   d24, d5, d24
    fmov    x16, d24
    sub     x5, x5, x16, lsl #21
    ldr     d24, [x17, x6, lsl #3]
    fcmgt   d24, d6, d24
    fmov    x16, d24
    sub     x6, x6, x16, lsl #21
    ldr     d24, [x17, x7, lsl #3]
    fcmgt   d24, d7, d24
    fmov    x16, d24
    sub     x7, x7, x16, lsl #21
    ldr     d24, [x17, x8, lsl #3]
    fcmgt   d24, d16, d24
    fmov    x16, d24
    sub     x8, x8, x16, lsl #21
    ldr     d24, [x17, x9, lsl #3]
    fcmgt   d24, d17, d24
    fmov    x16, d24
    sub     x9, x9, x16, lsl #21
    ldr     d24, [x17, x10, lsl #3]
    fcmgt   d24, d18, d24
    fmov    x16, d24
    sub     x10, x10, x16, lsl #21
    ldr     d24, [x17, x11, lsl #3]
    fcmgt   d24, d19, d24
    fmov    x16, d24
    sub     x11, x11, x16, lsl #21
    ldr     d24, [x17, x12, lsl #3]
    fcmgt   d24, d20, d24
    fmov    x16, d24
    sub     x12, x12, x16, lsl #21
    ldr     d24, [x17, x13, lsl #3]
    fcmgt   d24, d21, d24
    fmov    x16, d24
    sub     x13, x13, x16, lsl #21
    ldr     d24, [x17, x14, lsl #3]
    fcmgt   d24, d22, d24
    fmov    x16, d24
    sub     x14, x14, x16, lsl #21
    ldr     d24, [x17, x15, lsl #3]
    fcmgt   d24, d23, d24
    fmov    x16, d24
    sub     x15, x15, x16, lsl #21
    mov     x16, #8388608
    sub     x17, x17, x16           // &a[h/2 - 1], or a after h = 1
Lblock_20:                          // h = 2^20
    ldr     d24, [x17, x0, lsl #3]
    fcmgt   d24, d0, d24
    fmov    x16, d24
    sub     x0, x0, x16, lsl #20
    ldr     d24, [x17, x1, lsl #3]
    fcmgt   d24, d1, d24
    fmov    x16, d24
    sub     x1, x1, x16, lsl #20
    ldr     d24, [x17, x2, lsl #3]
    fcmgt   d24, d2, d24
    fmov    x16, d24
    sub     x2, x2, x16, lsl #20
    ldr     d24, [x17, x3, lsl #3]
    fcmgt   d24, d3, d24
    fmov    x16, d24
    sub     x3, x3, x16, lsl #20
    ldr     d24, [x17, x4, lsl #3]
    fcmgt   d24, d4, d24
    fmov    x16, d24
    sub     x4, x4, x16, lsl #20
    ldr     d24, [x17, x5, lsl #3]
    fcmgt   d24, d5, d24
    fmov    x16, d24
    sub     x5, x5, x16, lsl #20
    ldr     d24, [x17, x6, lsl #3]
    fcmgt   d24, d6, d24
    fmov    x16, d24
    sub     x6, x6, x16, lsl #20
    ldr     d24, [x17, x7, lsl #3]
    fcmgt   d24, d7, d24
    fmov    x16, d24
    sub     x7, x7, x16, lsl #20
    ldr     d24, [x17, x8, lsl #3]
    fcmgt   d24, d16, d24
    fmov    x16, d24
    sub     x8, x8, x16, lsl #20
    ldr     d24, [x17, x9, lsl #3]
    fcmgt   d24, d17, d24
    fmov    x16, d24
    sub     x9, x9, x16, lsl #20
    ldr     d24, [x17, x10, lsl #3]
    fcmgt   d24, d18, d24
    fmov    x16, d24
    sub     x10, x10, x16, lsl #20
    ldr     d24, [x17, x11, lsl #3]
    fcmgt   d24, d19, d24
    fmov    x16, d24
    sub     x11, x11, x16, lsl #20
    ldr     d24, [x17, x12, lsl #3]
    fcmgt   d24, d20, d24
    fmov    x16, d24
    sub     x12, x12, x16, lsl #20
    ldr     d24, [x17, x13, lsl #3]
    fcmgt   d24, d21, d24
    fmov    x16, d24
    sub     x13, x13, x16, lsl #20
    ldr     d24, [x17, x14, lsl #3]
    fcmgt   d24, d22, d24
    fmov    x16, d24
    sub     x14, x14, x16, lsl #20
    ldr     d24, [x17, x15, lsl #3]
    fcmgt   d24, d23, d24
    fmov    x16, d24
    sub     x15, x15, x16, lsl #20
    mov     x16, #4194304
    sub     x17, x17, x16           // &a[h/2 - 1], or a after h = 1
Lblock_19:                          // h = 2^19
    ldr     d24, [x17, x0, lsl #3]
    fcmgt   d24, d0, d24
    fmov    x16, d24
    sub     x0, x0, x16, lsl #19
    ldr     d24, [x17, x1, lsl #3]
    fcmgt   d24, d1, d24
    fmov    x16, d24
    sub     x1, x1, x16, lsl #19
    ldr     d24, [x17, x2, lsl #3]
    fcmgt   d24, d2, d24
    fmov    x16, d24
    sub     x2, x2, x16, lsl #19
    ldr     d24, [x17, x3, lsl #3]
    fcmgt   d24, d3, d24
    fmov    x16, d24
    sub     x3, x3, x16, lsl #19
    ldr     d24, [x17, x4, lsl #3]
    fcmgt   d24, d4, d24
    fmov    x16, d24
    sub     x4, x4, x16, lsl #19
    ldr     d24, [x17, x5, lsl #3]
    fcmgt   d24, d5, d24
    fmov    x16, d24
    sub     x5, x5, x16, lsl #19
    ldr     d24, [x17, x6, lsl #3]
    fcmgt   d24, d6, d24
    fmov    x16, d24
    sub     x6, x6, x16, lsl #19
    ldr     d24, [x17, x7, lsl #3]
    fcmgt   d24, d7, d24
    fmov    x16, d24
    sub     x7, x7, x16, lsl #19
    ldr     d24, [x17, x8, lsl #3]
    fcmgt   d24, d16, d24
    fmov    x16, d24
    sub     x8, x8, x16, lsl #19
    ldr     d24, [x17, x9, lsl #3]
    fcmgt   d24, d17, d24
    fmov    x16, d24
    sub     x9, x9, x16, lsl #19
    ldr     d24, [x17, x10, lsl #3]
    fcmgt   d24, d18, d24
    fmov    x16, d24
    sub     x10, x10, x16, lsl #19
    ldr     d24, [x17, x11, lsl #3]
    fcmgt   d24, d19, d24
    fmov    x16, d24
    sub     x11, x11, x16, lsl #19
    ldr     d24, [x17, x12, lsl #3]
    fcmgt   d24, d20, d24
    fmov    x16, d24
    sub     x12, x12, x16, lsl #19
    ldr     d24, [x17, x13, lsl #3]
    fcmgt   d24, d21, d24
    fmov    x16, d24
    sub     x13, x13, x16, lsl #19
    ldr     d24, [x17, x14, lsl #3]
    fcmgt   d24, d22, d24
    fmov    x16, d24
    sub     x14, x14, x16, lsl #19
    ldr     d24, [x17, x15, lsl #3]
    fcmgt   d24, d23, d24
    fmov    x16, d24
    sub     x15, x15, x16, lsl #19
    mov     x16, #2097152
    sub     x17, x17, x16           // &a[h/2 - 1], or a after h = 1
Lblock_18:                          // h = 2^18
    ldr     d24, [x17, x0, lsl #3]
    fcmgt   d24, d0, d24
    fmov    x16, d24
    sub     x0, x0, x16, lsl #18
    ldr     d24, [x17, x1, lsl #3]
    fcmgt   d24, d1, d24
    fmov    x16, d24
    sub     x1, x1, x16, lsl #18
    ldr     d24, [x17, x2, lsl #3]
    fcmgt   d24, d2, d24
    fmov    x16, d24
    sub     x2, x2, x16, lsl #18
    ldr     d24, [x17, x3, lsl #3]
    fcmgt   d24, d3, d24
    fmov    x16, d24
    sub     x3, x3, x16, lsl #18
    ldr     d24, [x17, x4, lsl #3]
    fcmgt   d24, d4, d24
    fmov    x16, d24
    sub     x4, x4, x16, lsl #18
    ldr     d24, [x17, x5, lsl #3]
    fcmgt   d24, d5, d24
    fmov    x16, d24
    sub     x5, x5, x16, lsl #18
    ldr     d24, [x17, x6, lsl #3]
    fcmgt   d24, d6, d24
    fmov    x16, d24
    sub     x6, x6, x16, lsl #18
    ldr     d24, [x17, x7, lsl #3]
    fcmgt   d24, d7, d24
    fmov    x16, d24
    sub     x7, x7, x16, lsl #18
    ldr     d24, [x17, x8, lsl #3]
    fcmgt   d24, d16, d24
    fmov    x16, d24
    sub     x8, x8, x16, lsl #18
    ldr     d24, [x17, x9, lsl #3]
    fcmgt   d24, d17, d24
    fmov    x16, d24
    sub     x9, x9, x16, lsl #18
    ldr     d24, [x17, x10, lsl #3]
    fcmgt   d24, d18, d24
    fmov    x16, d24
    sub     x10, x10, x16, lsl #18
    ldr     d24, [x17, x11, lsl #3]
    fcmgt   d24, d19, d24
    fmov    x16, d24
    sub     x11, x11, x16, lsl #18
    ldr     d24, [x17, x12, lsl #3]
    fcmgt   d24, d20, d24
    fmov    x16, d24
    sub     x12, x12, x16, lsl #18
    ldr     d24, [x17, x13, lsl #3]
    fcmgt   d24, d21, d24
    fmov    x16, d24
    sub     x13, x13, x16, lsl #18
    ldr     d24, [x17, x14, lsl #3]
    fcmgt   d24, d22, d24
    fmov    x16, d24
    sub     x14, x14, x16, lsl #18
    ldr     d24, [x17, x15, lsl #3]
    fcmgt   d24, d23, d24
    fmov    x16, d24
    sub     x15, x15, x16, lsl #18
    mov     x16, #1048576
    sub     x17, x17, x16           // &a[h/2 - 1], or a after h = 1
Lblock_17:                          // h = 2^17
    ldr     d24, [x17, x0, lsl #3]
    fcmgt   d24, d0, d24
    fmov    x16, d24
    sub     x0, x0, x16, lsl #17
    ldr     d24, [x17, x1, lsl #3]
    fcmgt   d24, d1, d24
    fmov    x16, d24
    sub     x1, x1, x16, lsl #17
    ldr     d24, [x17, x2, lsl #3]
    fcmgt   d24, d2, d24
    fmov    x16, d24
    sub     x2, x2, x16, lsl #17
    ldr     d24, [x17, x3, lsl #3]
    fcmgt   d24, d3, d24
    fmov    x16, d24
    sub     x3, x3, x16, lsl #17
    ldr     d24, [x17, x4, lsl #3]
    fcmgt   d24, d4, d24
    fmov    x16, d24
    sub     x4, x4, x16, lsl #17
    ldr     d24, [x17, x5, lsl #3]
    fcmgt   d24, d5, d24
    fmov    x16, d24
    sub     x5, x5, x16, lsl #17
    ldr     d24, [x17, x6, lsl #3]
    fcmgt   d24, d6, d24
    fmov    x16, d24
    sub     x6, x6, x16, lsl #17
    ldr     d24, [x17, x7, lsl #3]
    fcmgt   d24, d7, d24
    fmov    x16, d24
    sub     x7, x7, x16, lsl #17
    ldr     d24, [x17, x8, lsl #3]
    fcmgt   d24, d16, d24
    fmov    x16, d24
    sub     x8, x8, x16, lsl #17
    ldr     d24, [x17, x9, lsl #3]
    fcmgt   d24, d17, d24
    fmov    x16, d24
    sub     x9, x9, x16, lsl #17
    ldr     d24, [x17, x10, lsl #3]
    fcmgt   d24, d18, d24
    fmov    x16, d24
    sub     x10, x10, x16, lsl #17
    ldr     d24, [x17, x11, lsl #3]
    fcmgt   d24, d19, d24
    fmov    x16, d24
    sub     x11, x11, x16, lsl #17
    ldr     d24, [x17, x12, lsl #3]
    fcmgt   d24, d20, d24
    fmov    x16, d24
    sub     x12, x12, x16, lsl #17
    ldr     d24, [x17, x13, lsl #3]
    fcmgt   d24, d21, d24
    fmov    x16, d24
    sub     x13, x13, x16, lsl #17
    ldr     d24, [x17, x14, lsl #3]
    fcmgt   d24, d22, d24
    fmov    x16, d24
    sub     x14, x14, x16, lsl #17
    ldr     d24, [x17, x15, lsl #3]
    fcmgt   d24, d23, d24
    fmov    x16, d24
    sub     x15, x15, x16, lsl #17
    mov     x16, #524288
    sub     x17, x17, x16           // &a[h/2 - 1], or a after h = 1
Lblock_16:                          // h = 2^16
    ldr     d24, [x17, x0, lsl #3]
    fcmgt   d24, d0, d24
    fmov    x16, d24
    sub     x0, x0, x16, lsl #16
    ldr     d24, [x17, x1, lsl #3]
    fcmgt   d24, d1, d24
    fmov    x16, d24
    sub     x1, x1, x16, lsl #16
    ldr     d24, [x17, x2, lsl #3]
    fcmgt   d24, d2, d24
    fmov    x16, d24
    sub     x2, x2, x16, lsl #16
    ldr     d24, [x17, x3, lsl #3]
    fcmgt   d24, d3, d24
    fmov    x16, d24
    sub     x3, x3, x16, lsl #16
    ldr     d24, [x17, x4, lsl #3]
    fcmgt   d24, d4, d24
    fmov    x16, d24
    sub     x4, x4, x16, lsl #16
    ldr     d24, [x17, x5, lsl #3]
    fcmgt   d24, d5, d24
    fmov    x16, d24
    sub     x5, x5, x16, lsl #16
    ldr     d24, [x17, x6, lsl #3]
    fcmgt   d24, d6, d24
    fmov    x16, d24
    sub     x6, x6, x16, lsl #16
    ldr     d24, [x17, x7, lsl #3]
    fcmgt   d24, d7, d24
    fmov    x16, d24
    sub     x7, x7, x16, lsl #16
    ldr     d24, [x17, x8, lsl #3]
    fcmgt   d24, d16, d24
    fmov    x16, d24
    sub     x8, x8, x16, lsl #16
    ldr     d24, [x17, x9, lsl #3]
    fcmgt   d24, d17, d24
    fmov    x16, d24
    sub     x9, x9, x16, lsl #16
    ldr     d24, [x17, x10, lsl #3]
    fcmgt   d24, d18, d24
    fmov    x16, d24
    sub     x10, x10, x16, lsl #16
    ldr     d24, [x17, x11, lsl #3]
    fcmgt   d24, d19, d24
    fmov    x16, d24
    sub     x11, x11, x16, lsl #16
    ldr     d24, [x17, x12, lsl #3]
    fcmgt   d24, d20, d24
    fmov    x16, d24
    sub     x12, x12, x16, lsl #16
    ldr     d24, [x17, x13, lsl #3]
    fcmgt   d24, d21, d24
    fmov    x16, d24
    sub     x13, x13, x16, lsl #16
    ldr     d24, [x17, x14, lsl #3]
    fcmgt   d24, d22, d24
    fmov    x16, d24
    sub     x14, x14, x16, lsl #16
    ldr     d24, [x17, x15, lsl #3]
    fcmgt   d24, d23, d24
    fmov    x16, d24
    sub     x15, x15, x16, lsl #16
    mov     x16, #262144
    sub     x17, x17, x16           // &a[h/2 - 1], or a after h = 1
Lblock_15:                          // h = 2^15
    ldr     d24, [x17, x0, lsl #3]
    fcmgt   d24, d0, d24
    fmov    x16, d24
    sub     x0, x0, x16, lsl #15
    ldr     d24, [x17, x1, lsl #3]
    fcmgt   d24, d1, d24
    fmov    x16, d24
    sub     x1, x1, x16, lsl #15
    ldr     d24, [x17, x2, lsl #3]
    fcmgt   d24, d2, d24
    fmov    x16, d24
    sub     x2, x2, x16, lsl #15
    ldr     d24, [x17, x3, lsl #3]
    fcmgt   d24, d3, d24
    fmov    x16, d24
    sub     x3, x3, x16, lsl #15
    ldr     d24, [x17, x4, lsl #3]
    fcmgt   d24, d4, d24
    fmov    x16, d24
    sub     x4, x4, x16, lsl #15
    ldr     d24, [x17, x5, lsl #3]
    fcmgt   d24, d5, d24
    fmov    x16, d24
    sub     x5, x5, x16, lsl #15
    ldr     d24, [x17, x6, lsl #3]
    fcmgt   d24, d6, d24
    fmov    x16, d24
    sub     x6, x6, x16, lsl #15
    ldr     d24, [x17, x7, lsl #3]
    fcmgt   d24, d7, d24
    fmov    x16, d24
    sub     x7, x7, x16, lsl #15
    ldr     d24, [x17, x8, lsl #3]
    fcmgt   d24, d16, d24
    fmov    x16, d24
    sub     x8, x8, x16, lsl #15
    ldr     d24, [x17, x9, lsl #3]
    fcmgt   d24, d17, d24
    fmov    x16, d24
    sub     x9, x9, x16, lsl #15
    ldr     d24, [x17, x10, lsl #3]
    fcmgt   d24, d18, d24
    fmov    x16, d24
    sub     x10, x10, x16, lsl #15
    ldr     d24, [x17, x11, lsl #3]
    fcmgt   d24, d19, d24
    fmov    x16, d24
    sub     x11, x11, x16, lsl #15
    ldr     d24, [x17, x12, lsl #3]
    fcmgt   d24, d20, d24
    fmov    x16, d24
    sub     x12, x12, x16, lsl #15
    ldr     d24, [x17, x13, lsl #3]
    fcmgt   d24, d21, d24
    fmov    x16, d24
    sub     x13, x13, x16, lsl #15
    ldr     d24, [x17, x14, lsl #3]
    fcmgt   d24, d22, d24
    fmov    x16, d24
    sub     x14, x14, x16, lsl #15
    ldr     d24, [x17, x15, lsl #3]
    fcmgt   d24, d23, d24
    fmov    x16, d24
    sub     x15, x15, x16, lsl #15
    mov     x16, #131072
    sub     x17, x17, x16           // &a[h/2 - 1], or a after h = 1
Lblock_14:                          // h = 2^14
    ldr     d24, [x17, x0, lsl #3]
    fcmgt   d24, d0, d24
    fmov    x16, d24
    sub     x0, x0, x16, lsl #14
    ldr     d24, [x17, x1, lsl #3]
    fcmgt   d24, d1, d24
    fmov    x16, d24
    sub     x1, x1, x16, lsl #14
    ldr     d24, [x17, x2, lsl #3]
    fcmgt   d24, d2, d24
    fmov    x16, d24
    sub     x2, x2, x16, lsl #14
    ldr     d24, [x17, x3, lsl #3]
    fcmgt   d24, d3, d24
    fmov    x16, d24
    sub     x3, x3, x16, lsl #14
    ldr     d24, [x17, x4, lsl #3]
    fcmgt   d24, d4, d24
    fmov    x16, d24
    sub     x4, x4, x16, lsl #14
    ldr     d24, [x17, x5, lsl #3]
    fcmgt   d24, d5, d24
    fmov    x16, d24
    sub     x5, x5, x16, lsl #14
    ldr     d24, [x17, x6, lsl #3]
    fcmgt   d24, d6, d24
    fmov    x16, d24
    sub     x6, x6, x16, lsl #14
    ldr     d24, [x17, x7, lsl #3]
    fcmgt   d24, d7, d24
    fmov    x16, d24
    sub     x7, x7, x16, lsl #14
    ldr     d24, [x17, x8, lsl #3]
    fcmgt   d24, d16, d24
    fmov    x16, d24
    sub     x8, x8, x16, lsl #14
    ldr     d24, [x17, x9, lsl #3]
    fcmgt   d24, d17, d24
    fmov    x16, d24
    sub     x9, x9, x16, lsl #14
    ldr     d24, [x17, x10, lsl #3]
    fcmgt   d24, d18, d24
    fmov    x16, d24
    sub     x10, x10, x16, lsl #14
    ldr     d24, [x17, x11, lsl #3]
    fcmgt   d24, d19, d24
    fmov    x16, d24
    sub     x11, x11, x16, lsl #14
    ldr     d24, [x17, x12, lsl #3]
    fcmgt   d24, d20, d24
    fmov    x16, d24
    sub     x12, x12, x16, lsl #14
    ldr     d24, [x17, x13, lsl #3]
    fcmgt   d24, d21, d24
    fmov    x16, d24
    sub     x13, x13, x16, lsl #14
    ldr     d24, [x17, x14, lsl #3]
    fcmgt   d24, d22, d24
    fmov    x16, d24
    sub     x14, x14, x16, lsl #14
    ldr     d24, [x17, x15, lsl #3]
    fcmgt   d24, d23, d24
    fmov    x16, d24
    sub     x15, x15, x16, lsl #14
    mov     x16, #65536
    sub     x17, x17, x16           // &a[h/2 - 1], or a after h = 1
Lblock_13:                          // h = 2^13
    ldr     d24, [x17, x0, lsl #3]
    fcmgt   d24, d0, d24
    fmov    x16, d24
    sub     x0, x0, x16, lsl #13
    ldr     d24, [x17, x1, lsl #3]
    fcmgt   d24, d1, d24
    fmov    x16, d24
    sub     x1, x1, x16, lsl #13
    ldr     d24, [x17, x2, lsl #3]
    fcmgt   d24, d2, d24
    fmov    x16, d24
    sub     x2, x2, x16, lsl #13
    ldr     d24, [x17, x3, lsl #3]
    fcmgt   d24, d3, d24
    fmov    x16, d24
    sub     x3, x3, x16, lsl #13
    ldr     d24, [x17, x4, lsl #3]
    fcmgt   d24, d4, d24
    fmov    x16, d24
    sub     x4, x4, x16, lsl #13
    ldr     d24, [x17, x5, lsl #3]
    fcmgt   d24, d5, d24
    fmov    x16, d24
    sub     x5, x5, x16, lsl #13
    ldr     d24, [x17, x6, lsl #3]
    fcmgt   d24, d6, d24
    fmov    x16, d24
    sub     x6, x6, x16, lsl #13
    ldr     d24, [x17, x7, lsl #3]
    fcmgt   d24, d7, d24
    fmov    x16, d24
    sub     x7, x7, x16, lsl #13
    ldr     d24, [x17, x8, lsl #3]
    fcmgt   d24, d16, d24
    fmov    x16, d24
    sub     x8, x8, x16, lsl #13
    ldr     d24, [x17, x9, lsl #3]
    fcmgt   d24, d17, d24
    fmov    x16, d24
    sub     x9, x9, x16, lsl #13
    ldr     d24, [x17, x10, lsl #3]
    fcmgt   d24, d18, d24
    fmov    x16, d24
    sub     x10, x10, x16, lsl #13
    ldr     d24, [x17, x11, lsl #3]
    fcmgt   d24, d19, d24
    fmov    x16, d24
    sub     x11, x11, x16, lsl #13
    ldr     d24, [x17, x12, lsl #3]
    fcmgt   d24, d20, d24
    fmov    x16, d24
    sub     x12, x12, x16, lsl #13
    ldr     d24, [x17, x13, lsl #3]
    fcmgt   d24, d21, d24
    fmov    x16, d24
    sub     x13, x13, x16, lsl #13
    ldr     d24, [x17, x14, lsl #3]
    fcmgt   d24, d22, d24
    fmov    x16, d24
    sub     x14, x14, x16, lsl #13
    ldr     d24, [x17, x15, lsl #3]
    fcmgt   d24, d23, d24
    fmov    x16, d24
    sub     x15, x15, x16, lsl #13
    mov     x16, #32768
    sub     x17, x17, x16           // &a[h/2 - 1], or a after h = 1
Lblock_12:                          // h = 2^12
    ldr     d24, [x17, x0, lsl #3]
    fcmgt   d24, d0, d24
    fmov    x16, d24
    sub     x0, x0, x16, lsl #12
    ldr     d24, [x17, x1, lsl #3]
    fcmgt   d24, d1, d24
    fmov    x16, d24
    sub     x1, x1, x16, lsl #12
    ldr     d24, [x17, x2, lsl #3]
    fcmgt   d24, d2, d24
    fmov    x16, d24
    sub     x2, x2, x16, lsl #12
    ldr     d24, [x17, x3, lsl #3]
    fcmgt   d24, d3, d24
    fmov    x16, d24
    sub     x3, x3, x16, lsl #12
    ldr     d24, [x17, x4, lsl #3]
    fcmgt   d24, d4, d24
    fmov    x16, d24
    sub     x4, x4, x16, lsl #12
    ldr     d24, [x17, x5, lsl #3]
    fcmgt   d24, d5, d24
    fmov    x16, d24
    sub     x5, x5, x16, lsl #12
    ldr     d24, [x17, x6, lsl #3]
    fcmgt   d24, d6, d24
    fmov    x16, d24
    sub     x6, x6, x16, lsl #12
    ldr     d24, [x17, x7, lsl #3]
    fcmgt   d24, d7, d24
    fmov    x16, d24
    sub     x7, x7, x16, lsl #12
    ldr     d24, [x17, x8, lsl #3]
    fcmgt   d24, d16, d24
    fmov    x16, d24
    sub     x8, x8, x16, lsl #12
    ldr     d24, [x17, x9, lsl #3]
    fcmgt   d24, d17, d24
    fmov    x16, d24
    sub     x9, x9, x16, lsl #12
    ldr     d24, [x17, x10, lsl #3]
    fcmgt   d24, d18, d24
    fmov    x16, d24
    sub     x10, x10, x16, lsl #12
    ldr     d24, [x17, x11, lsl #3]
    fcmgt   d24, d19, d24
    fmov    x16, d24
    sub     x11, x11, x16, lsl #12
    ldr     d24, [x17, x12, lsl #3]
    fcmgt   d24, d20, d24
    fmov    x16, d24
    sub     x12, x12, x16, lsl #12
    ldr     d24, [x17, x13, lsl #3]
    fcmgt   d24, d21, d24
    fmov    x16, d24
    sub     x13, x13, x16, lsl #12
    ldr     d24, [x17, x14, lsl #3]
    fcmgt   d24, d22, d24
    fmov    x16, d24
    sub     x14, x14, x16, lsl #12
    ldr     d24, [x17, x15, lsl #3]
    fcmgt   d24, d23, d24
    fmov    x16, d24
    sub     x15, x15, x16, lsl #12
    mov     x16, #16384
    sub     x17, x17, x16           // &a[h/2 - 1], or a after h = 1
Lblock_11:                          // h = 2^11
    ldr     d24, [x17, x0, lsl #3]
    fcmgt   d24, d0, d24
    fmov    x16, d24
    sub     x0, x0, x16, lsl #11
    ldr     d24, [x17, x1, lsl #3]
    fcmgt   d24, d1, d24
    fmov    x16, d24
    sub     x1, x1, x16, lsl #11
    ldr     d24, [x17, x2, lsl #3]
    fcmgt   d24, d2, d24
    fmov    x16, d24
    sub     x2, x2, x16, lsl #11
    ldr     d24, [x17, x3, lsl #3]
    fcmgt   d24, d3, d24
    fmov    x16, d24
    sub     x3, x3, x16, lsl #11
    ldr     d24, [x17, x4, lsl #3]
    fcmgt   d24, d4, d24
    fmov    x16, d24
    sub     x4, x4, x16, lsl #11
    ldr     d24, [x17, x5, lsl #3]
    fcmgt   d24, d5, d24
    fmov    x16, d24
    sub     x5, x5, x16, lsl #11
    ldr     d24, [x17, x6, lsl #3]
    fcmgt   d24, d6, d24
    fmov    x16, d24
    sub     x6, x6, x16, lsl #11
    ldr     d24, [x17, x7, lsl #3]
    fcmgt   d24, d7, d24
    fmov    x16, d24
    sub     x7, x7, x16, lsl #11
    ldr     d24, [x17, x8, lsl #3]
    fcmgt   d24, d16, d24
    fmov    x16, d24
    sub     x8, x8, x16, lsl #11
    ldr     d24, [x17, x9, lsl #3]
    fcmgt   d24, d17, d24
    fmov    x16, d24
    sub     x9, x9, x16, lsl #11
    ldr     d24, [x17, x10, lsl #3]
    fcmgt   d24, d18, d24
    fmov    x16, d24
    sub     x10, x10, x16, lsl #11
    ldr     d24, [x17, x11, lsl #3]
    fcmgt   d24, d19, d24
    fmov    x16, d24
    sub     x11, x11, x16, lsl #11
    ldr     d24, [x17, x12, lsl #3]
    fcmgt   d24, d20, d24
    fmov    x16, d24
    sub     x12, x12, x16, lsl #11
    ldr     d24, [x17, x13, lsl #3]
    fcmgt   d24, d21, d24
    fmov    x16, d24
    sub     x13, x13, x16, lsl #11
    ldr     d24, [x17, x14, lsl #3]
    fcmgt   d24, d22, d24
    fmov    x16, d24
    sub     x14, x14, x16, lsl #11
    ldr     d24, [x17, x15, lsl #3]
    fcmgt   d24, d23, d24
    fmov    x16, d24
    sub     x15, x15, x16, lsl #11
    mov     x16, #8192
    sub     x17, x17, x16           // &a[h/2 - 1], or a after h = 1
Lblock_10:                          // h = 2^10
    ldr     d24, [x17, x0, lsl #3]
    fcmgt   d24, d0, d24
    fmov    x16, d24
    sub     x0, x0, x16, lsl #10
    ldr     d24, [x17, x1, lsl #3]
    fcmgt   d24, d1, d24
    fmov    x16, d24
    sub     x1, x1, x16, lsl #10
    ldr     d24, [x17, x2, lsl #3]
    fcmgt   d24, d2, d24
    fmov    x16, d24
    sub     x2, x2, x16, lsl #10
    ldr     d24, [x17, x3, lsl #3]
    fcmgt   d24, d3, d24
    fmov    x16, d24
    sub     x3, x3, x16, lsl #10
    ldr     d24, [x17, x4, lsl #3]
    fcmgt   d24, d4, d24
    fmov    x16, d24
    sub     x4, x4, x16, lsl #10
    ldr     d24, [x17, x5, lsl #3]
    fcmgt   d24, d5, d24
    fmov    x16, d24
    sub     x5, x5, x16, lsl #10
    ldr     d24, [x17, x6, lsl #3]
    fcmgt   d24, d6, d24
    fmov    x16, d24
    sub     x6, x6, x16, lsl #10
    ldr     d24, [x17, x7, lsl #3]
    fcmgt   d24, d7, d24
    fmov    x16, d24
    sub     x7, x7, x16, lsl #10
    ldr     d24, [x17, x8, lsl #3]
    fcmgt   d24, d16, d24
    fmov    x16, d24
    sub     x8, x8, x16, lsl #10
    ldr     d24, [x17, x9, lsl #3]
    fcmgt   d24, d17, d24
    fmov    x16, d24
    sub     x9, x9, x16, lsl #10
    ldr     d24, [x17, x10, lsl #3]
    fcmgt   d24, d18, d24
    fmov    x16, d24
    sub     x10, x10, x16, lsl #10
    ldr     d24, [x17, x11, lsl #3]
    fcmgt   d24, d19, d24
    fmov    x16, d24
    sub     x11, x11, x16, lsl #10
    ldr     d24, [x17, x12, lsl #3]
    fcmgt   d24, d20, d24
    fmov    x16, d24
    sub     x12, x12, x16, lsl #10
    ldr     d24, [x17, x13, lsl #3]
    fcmgt   d24, d21, d24
    fmov    x16, d24
    sub     x13, x13, x16, lsl #10
    ldr     d24, [x17, x14, lsl #3]
    fcmgt   d24, d22, d24
    fmov    x16, d24
    sub     x14, x14, x16, lsl #10
    ldr     d24, [x17, x15, lsl #3]
    fcmgt   d24, d23, d24
    fmov    x16, d24
    sub     x15, x15, x16, lsl #10
    mov     x16, #4096
    sub     x17, x17, x16           // &a[h/2 - 1], or a after h = 1
Lblock_9:                          // h = 2^9
    ldr     d24, [x17, x0, lsl #3]
    fcmgt   d24, d0, d24
    fmov    x16, d24
    sub     x0, x0, x16, lsl #9
    ldr     d24, [x17, x1, lsl #3]
    fcmgt   d24, d1, d24
    fmov    x16, d24
    sub     x1, x1, x16, lsl #9
    ldr     d24, [x17, x2, lsl #3]
    fcmgt   d24, d2, d24
    fmov    x16, d24
    sub     x2, x2, x16, lsl #9
    ldr     d24, [x17, x3, lsl #3]
    fcmgt   d24, d3, d24
    fmov    x16, d24
    sub     x3, x3, x16, lsl #9
    ldr     d24, [x17, x4, lsl #3]
    fcmgt   d24, d4, d24
    fmov    x16, d24
    sub     x4, x4, x16, lsl #9
    ldr     d24, [x17, x5, lsl #3]
    fcmgt   d24, d5, d24
    fmov    x16, d24
    sub     x5, x5, x16, lsl #9
    ldr     d24, [x17, x6, lsl #3]
    fcmgt   d24, d6, d24
    fmov    x16, d24
    sub     x6, x6, x16, lsl #9
    ldr     d24, [x17, x7, lsl #3]
    fcmgt   d24, d7, d24
    fmov    x16, d24
    sub     x7, x7, x16, lsl #9
    ldr     d24, [x17, x8, lsl #3]
    fcmgt   d24, d16, d24
    fmov    x16, d24
    sub     x8, x8, x16, lsl #9
    ldr     d24, [x17, x9, lsl #3]
    fcmgt   d24, d17, d24
    fmov    x16, d24
    sub     x9, x9, x16, lsl #9
    ldr     d24, [x17, x10, lsl #3]
    fcmgt   d24, d18, d24
    fmov    x16, d24
    sub     x10, x10, x16, lsl #9
    ldr     d24, [x17, x11, lsl #3]
    fcmgt   d24, d19, d24
    fmov    x16, d24
    sub     x11, x11, x16, lsl #9
    ldr     d24, [x17, x12, lsl #3]
    fcmgt   d24, d20, d24
    fmov    x16, d24
    sub     x12, x12, x16, lsl #9
    ldr     d24, [x17, x13, lsl #3]
    fcmgt   d24, d21, d24
    fmov    x16, d24
    sub     x13, x13, x16, lsl #9
    ldr     d24, [x17, x14, lsl #3]
    fcmgt   d24, d22, d24
    fmov    x16, d24
    sub     x14, x14, x16, lsl #9
    ldr     d24, [x17, x15, lsl #3]
    fcmgt   d24, d23, d24
    fmov    x16, d24
    sub     x15, x15, x16, lsl #9
    mov     x16, #2048
    sub     x17, x17, x16           // &a[h/2 - 1], or a after h = 1
Lblock_8:                          // h = 2^8
    ldr     d24, [x17, x0, lsl #3]
    fcmgt   d24, d0, d24
    fmov    x16, d24
    sub     x0, x0, x16, lsl #8
    ldr     d24, [x17, x1, lsl #3]
    fcmgt   d24, d1, d24
    fmov    x16, d24
    sub     x1, x1, x16, lsl #8
    ldr     d24, [x17, x2, lsl #3]
    fcmgt   d24, d2, d24
    fmov    x16, d24
    sub     x2, x2, x16, lsl #8
    ldr     d24, [x17, x3, lsl #3]
    fcmgt   d24, d3, d24
    fmov    x16, d24
    sub     x3, x3, x16, lsl #8
    ldr     d24, [x17, x4, lsl #3]
    fcmgt   d24, d4, d24
    fmov    x16, d24
    sub     x4, x4, x16, lsl #8
    ldr     d24, [x17, x5, lsl #3]
    fcmgt   d24, d5, d24
    fmov    x16, d24
    sub     x5, x5, x16, lsl #8
    ldr     d24, [x17, x6, lsl #3]
    fcmgt   d24, d6, d24
    fmov    x16, d24
    sub     x6, x6, x16, lsl #8
    ldr     d24, [x17, x7, lsl #3]
    fcmgt   d24, d7, d24
    fmov    x16, d24
    sub     x7, x7, x16, lsl #8
    ldr     d24, [x17, x8, lsl #3]
    fcmgt   d24, d16, d24
    fmov    x16, d24
    sub     x8, x8, x16, lsl #8
    ldr     d24, [x17, x9, lsl #3]
    fcmgt   d24, d17, d24
    fmov    x16, d24
    sub     x9, x9, x16, lsl #8
    ldr     d24, [x17, x10, lsl #3]
    fcmgt   d24, d18, d24
    fmov    x16, d24
    sub     x10, x10, x16, lsl #8
    ldr     d24, [x17, x11, lsl #3]
    fcmgt   d24, d19, d24
    fmov    x16, d24
    sub     x11, x11, x16, lsl #8
    ldr     d24, [x17, x12, lsl #3]
    fcmgt   d24, d20, d24
    fmov    x16, d24
    sub     x12, x12, x16, lsl #8
    ldr     d24, [x17, x13, lsl #3]
    fcmgt   d24, d21, d24
    fmov    x16, d24
    sub     x13, x13, x16, lsl #8
    ldr     d24, [x17, x14, lsl #3]
    fcmgt   d24, d22, d24
    fmov    x16, d24
    sub     x14, x14, x16, lsl #8
    ldr     d24, [x17, x15, lsl #3]
    fcmgt   d24, d23, d24
    fmov    x16, d24
    sub     x15, x15, x16, lsl #8
    mov     x16, #1024
    sub     x17, x17, x16           // &a[h/2 - 1], or a after h = 1
Lblock_7:                          // h = 2^7
    ldr     d24, [x17, x0, lsl #3]
    fcmgt   d24, d0, d24
    fmov    x16, d24
    sub     x0, x0, x16, lsl #7
    ldr     d24, [x17, x1, lsl #3]
    fcmgt   d24, d1, d24
    fmov    x16, d24
    sub     x1, x1, x16, lsl #7
    ldr     d24, [x17, x2, lsl #3]
    fcmgt   d24, d2, d24
    fmov    x16, d24
    sub     x2, x2, x16, lsl #7
    ldr     d24, [x17, x3, lsl #3]
    fcmgt   d24, d3, d24
    fmov    x16, d24
    sub     x3, x3, x16, lsl #7
    ldr     d24, [x17, x4, lsl #3]
    fcmgt   d24, d4, d24
    fmov    x16, d24
    sub     x4, x4, x16, lsl #7
    ldr     d24, [x17, x5, lsl #3]
    fcmgt   d24, d5, d24
    fmov    x16, d24
    sub     x5, x5, x16, lsl #7
    ldr     d24, [x17, x6, lsl #3]
    fcmgt   d24, d6, d24
    fmov    x16, d24
    sub     x6, x6, x16, lsl #7
    ldr     d24, [x17, x7, lsl #3]
    fcmgt   d24, d7, d24
    fmov    x16, d24
    sub     x7, x7, x16, lsl #7
    ldr     d24, [x17, x8, lsl #3]
    fcmgt   d24, d16, d24
    fmov    x16, d24
    sub     x8, x8, x16, lsl #7
    ldr     d24, [x17, x9, lsl #3]
    fcmgt   d24, d17, d24
    fmov    x16, d24
    sub     x9, x9, x16, lsl #7
    ldr     d24, [x17, x10, lsl #3]
    fcmgt   d24, d18, d24
    fmov    x16, d24
    sub     x10, x10, x16, lsl #7
    ldr     d24, [x17, x11, lsl #3]
    fcmgt   d24, d19, d24
    fmov    x16, d24
    sub     x11, x11, x16, lsl #7
    ldr     d24, [x17, x12, lsl #3]
    fcmgt   d24, d20, d24
    fmov    x16, d24
    sub     x12, x12, x16, lsl #7
    ldr     d24, [x17, x13, lsl #3]
    fcmgt   d24, d21, d24
    fmov    x16, d24
    sub     x13, x13, x16, lsl #7
    ldr     d24, [x17, x14, lsl #3]
    fcmgt   d24, d22, d24
    fmov    x16, d24
    sub     x14, x14, x16, lsl #7
    ldr     d24, [x17, x15, lsl #3]
    fcmgt   d24, d23, d24
    fmov    x16, d24
    sub     x15, x15, x16, lsl #7
    mov     x16, #512
    sub     x17, x17, x16           // &a[h/2 - 1], or a after h = 1
Lblock_6:                          // h = 2^6
    ldr     d24, [x17, x0, lsl #3]
    fcmgt   d24, d0, d24
    fmov    x16, d24
    sub     x0, x0, x16, lsl #6
    ldr     d24, [x17, x1, lsl #3]
    fcmgt   d24, d1, d24
    fmov    x16, d24
    sub     x1, x1, x16, lsl #6
    ldr     d24, [x17, x2, lsl #3]
    fcmgt   d24, d2, d24
    fmov    x16, d24
    sub     x2, x2, x16, lsl #6
    ldr     d24, [x17, x3, lsl #3]
    fcmgt   d24, d3, d24
    fmov    x16, d24
    sub     x3, x3, x16, lsl #6
    ldr     d24, [x17, x4, lsl #3]
    fcmgt   d24, d4, d24
    fmov    x16, d24
    sub     x4, x4, x16, lsl #6
    ldr     d24, [x17, x5, lsl #3]
    fcmgt   d24, d5, d24
    fmov    x16, d24
    sub     x5, x5, x16, lsl #6
    ldr     d24, [x17, x6, lsl #3]
    fcmgt   d24, d6, d24
    fmov    x16, d24
    sub     x6, x6, x16, lsl #6
    ldr     d24, [x17, x7, lsl #3]
    fcmgt   d24, d7, d24
    fmov    x16, d24
    sub     x7, x7, x16, lsl #6
    ldr     d24, [x17, x8, lsl #3]
    fcmgt   d24, d16, d24
    fmov    x16, d24
    sub     x8, x8, x16, lsl #6
    ldr     d24, [x17, x9, lsl #3]
    fcmgt   d24, d17, d24
    fmov    x16, d24
    sub     x9, x9, x16, lsl #6
    ldr     d24, [x17, x10, lsl #3]
    fcmgt   d24, d18, d24
    fmov    x16, d24
    sub     x10, x10, x16, lsl #6
    ldr     d24, [x17, x11, lsl #3]
    fcmgt   d24, d19, d24
    fmov    x16, d24
    sub     x11, x11, x16, lsl #6
    ldr     d24, [x17, x12, lsl #3]
    fcmgt   d24, d20, d24
    fmov    x16, d24
    sub     x12, x12, x16, lsl #6
    ldr     d24, [x17, x13, lsl #3]
    fcmgt   d24, d21, d24
    fmov    x16, d24
    sub     x13, x13, x16, lsl #6
    ldr     d24, [x17, x14, lsl #3]
    fcmgt   d24, d22, d24
    fmov    x16, d24
    sub     x14, x14, x16, lsl #6
    ldr     d24, [x17, x15, lsl #3]
    fcmgt   d24, d23, d24
    fmov    x16, d24
    sub     x15, x15, x16, lsl #6
    mov     x16, #256
    sub     x17, x17, x16           // &a[h/2 - 1], or a after h = 1
Lblock_5:                          // h = 2^5
    ldr     d24, [x17, x0, lsl #3]
    fcmgt   d24, d0, d24
    fmov    x16, d24
    sub     x0, x0, x16, lsl #5
    ldr     d24, [x17, x1, lsl #3]
    fcmgt   d24, d1, d24
    fmov    x16, d24
    sub     x1, x1, x16, lsl #5
    ldr     d24, [x17, x2, lsl #3]
    fcmgt   d24, d2, d24
    fmov    x16, d24
    sub     x2, x2, x16, lsl #5
    ldr     d24, [x17, x3, lsl #3]
    fcmgt   d24, d3, d24
    fmov    x16, d24
    sub     x3, x3, x16, lsl #5
    ldr     d24, [x17, x4, lsl #3]
    fcmgt   d24, d4, d24
    fmov    x16, d24
    sub     x4, x4, x16, lsl #5
    ldr     d24, [x17, x5, lsl #3]
    fcmgt   d24, d5, d24
    fmov    x16, d24
    sub     x5, x5, x16, lsl #5
    ldr     d24, [x17, x6, lsl #3]
    fcmgt   d24, d6, d24
    fmov    x16, d24
    sub     x6, x6, x16, lsl #5
    ldr     d24, [x17, x7, lsl #3]
    fcmgt   d24, d7, d24
    fmov    x16, d24
    sub     x7, x7, x16, lsl #5
    ldr     d24, [x17, x8, lsl #3]
    fcmgt   d24, d16, d24
    fmov    x16, d24
    sub     x8, x8, x16, lsl #5
    ldr     d24, [x17, x9, lsl #3]
    fcmgt   d24, d17, d24
    fmov    x16, d24
    sub     x9, x9, x16, lsl #5
    ldr     d24, [x17, x10, lsl #3]
    fcmgt   d24, d18, d24
    fmov    x16, d24
    sub     x10, x10, x16, lsl #5
    ldr     d24, [x17, x11, lsl #3]
    fcmgt   d24, d19, d24
    fmov    x16, d24
    sub     x11, x11, x16, lsl #5
    ldr     d24, [x17, x12, lsl #3]
    fcmgt   d24, d20, d24
    fmov    x16, d24
    sub     x12, x12, x16, lsl #5
    ldr     d24, [x17, x13, lsl #3]
    fcmgt   d24, d21, d24
    fmov    x16, d24
    sub     x13, x13, x16, lsl #5
    ldr     d24, [x17, x14, lsl #3]
    fcmgt   d24, d22, d24
    fmov    x16, d24
    sub     x14, x14, x16, lsl #5
    ldr     d24, [x17, x15, lsl #3]
    fcmgt   d24, d23, d24
    fmov    x16, d24
    sub     x15, x15, x16, lsl #5
    mov     x16, #128
    sub     x17, x17, x16           // &a[h/2 - 1], or a after h = 1
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
    sub     x17, x17, x16           // &a[h/2 - 1], or a after h = 1
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
    mov     x16, #32
    sub     x17, x17, x16           // &a[h/2 - 1], or a after h = 1
Lblock_2:                          // h = 2^2
    ldr     d24, [x17, x0, lsl #3]
    fcmgt   d24, d0, d24
    fmov    x16, d24
    sub     x0, x0, x16, lsl #2
    ldr     d24, [x17, x1, lsl #3]
    fcmgt   d24, d1, d24
    fmov    x16, d24
    sub     x1, x1, x16, lsl #2
    ldr     d24, [x17, x2, lsl #3]
    fcmgt   d24, d2, d24
    fmov    x16, d24
    sub     x2, x2, x16, lsl #2
    ldr     d24, [x17, x3, lsl #3]
    fcmgt   d24, d3, d24
    fmov    x16, d24
    sub     x3, x3, x16, lsl #2
    ldr     d24, [x17, x4, lsl #3]
    fcmgt   d24, d4, d24
    fmov    x16, d24
    sub     x4, x4, x16, lsl #2
    ldr     d24, [x17, x5, lsl #3]
    fcmgt   d24, d5, d24
    fmov    x16, d24
    sub     x5, x5, x16, lsl #2
    ldr     d24, [x17, x6, lsl #3]
    fcmgt   d24, d6, d24
    fmov    x16, d24
    sub     x6, x6, x16, lsl #2
    ldr     d24, [x17, x7, lsl #3]
    fcmgt   d24, d7, d24
    fmov    x16, d24
    sub     x7, x7, x16, lsl #2
    ldr     d24, [x17, x8, lsl #3]
    fcmgt   d24, d16, d24
    fmov    x16, d24
    sub     x8, x8, x16, lsl #2
    ldr     d24, [x17, x9, lsl #3]
    fcmgt   d24, d17, d24
    fmov    x16, d24
    sub     x9, x9, x16, lsl #2
    ldr     d24, [x17, x10, lsl #3]
    fcmgt   d24, d18, d24
    fmov    x16, d24
    sub     x10, x10, x16, lsl #2
    ldr     d24, [x17, x11, lsl #3]
    fcmgt   d24, d19, d24
    fmov    x16, d24
    sub     x11, x11, x16, lsl #2
    ldr     d24, [x17, x12, lsl #3]
    fcmgt   d24, d20, d24
    fmov    x16, d24
    sub     x12, x12, x16, lsl #2
    ldr     d24, [x17, x13, lsl #3]
    fcmgt   d24, d21, d24
    fmov    x16, d24
    sub     x13, x13, x16, lsl #2
    ldr     d24, [x17, x14, lsl #3]
    fcmgt   d24, d22, d24
    fmov    x16, d24
    sub     x14, x14, x16, lsl #2
    ldr     d24, [x17, x15, lsl #3]
    fcmgt   d24, d23, d24
    fmov    x16, d24
    sub     x15, x15, x16, lsl #2
    mov     x16, #16
    sub     x17, x17, x16           // &a[h/2 - 1], or a after h = 1
Lblock_1:                          // h = 2^1
    ldr     d24, [x17, x0, lsl #3]
    fcmgt   d24, d0, d24
    fmov    x16, d24
    sub     x0, x0, x16, lsl #1
    ldr     d24, [x17, x1, lsl #3]
    fcmgt   d24, d1, d24
    fmov    x16, d24
    sub     x1, x1, x16, lsl #1
    ldr     d24, [x17, x2, lsl #3]
    fcmgt   d24, d2, d24
    fmov    x16, d24
    sub     x2, x2, x16, lsl #1
    ldr     d24, [x17, x3, lsl #3]
    fcmgt   d24, d3, d24
    fmov    x16, d24
    sub     x3, x3, x16, lsl #1
    ldr     d24, [x17, x4, lsl #3]
    fcmgt   d24, d4, d24
    fmov    x16, d24
    sub     x4, x4, x16, lsl #1
    ldr     d24, [x17, x5, lsl #3]
    fcmgt   d24, d5, d24
    fmov    x16, d24
    sub     x5, x5, x16, lsl #1
    ldr     d24, [x17, x6, lsl #3]
    fcmgt   d24, d6, d24
    fmov    x16, d24
    sub     x6, x6, x16, lsl #1
    ldr     d24, [x17, x7, lsl #3]
    fcmgt   d24, d7, d24
    fmov    x16, d24
    sub     x7, x7, x16, lsl #1
    ldr     d24, [x17, x8, lsl #3]
    fcmgt   d24, d16, d24
    fmov    x16, d24
    sub     x8, x8, x16, lsl #1
    ldr     d24, [x17, x9, lsl #3]
    fcmgt   d24, d17, d24
    fmov    x16, d24
    sub     x9, x9, x16, lsl #1
    ldr     d24, [x17, x10, lsl #3]
    fcmgt   d24, d18, d24
    fmov    x16, d24
    sub     x10, x10, x16, lsl #1
    ldr     d24, [x17, x11, lsl #3]
    fcmgt   d24, d19, d24
    fmov    x16, d24
    sub     x11, x11, x16, lsl #1
    ldr     d24, [x17, x12, lsl #3]
    fcmgt   d24, d20, d24
    fmov    x16, d24
    sub     x12, x12, x16, lsl #1
    ldr     d24, [x17, x13, lsl #3]
    fcmgt   d24, d21, d24
    fmov    x16, d24
    sub     x13, x13, x16, lsl #1
    ldr     d24, [x17, x14, lsl #3]
    fcmgt   d24, d22, d24
    fmov    x16, d24
    sub     x14, x14, x16, lsl #1
    ldr     d24, [x17, x15, lsl #3]
    fcmgt   d24, d23, d24
    fmov    x16, d24
    sub     x15, x15, x16, lsl #1
    mov     x16, #8
    sub     x17, x17, x16           // &a[h/2 - 1], or a after h = 1
Lblock_0:                          // h = 2^0
    ldr     d24, [x17, x0, lsl #3]
    fcmgt   d24, d0, d24
    fmov    x16, d24
    sub     x0, x0, x16, lsl #0
    ldr     d24, [x17, x1, lsl #3]
    fcmgt   d24, d1, d24
    fmov    x16, d24
    sub     x1, x1, x16, lsl #0
    ldr     d24, [x17, x2, lsl #3]
    fcmgt   d24, d2, d24
    fmov    x16, d24
    sub     x2, x2, x16, lsl #0
    ldr     d24, [x17, x3, lsl #3]
    fcmgt   d24, d3, d24
    fmov    x16, d24
    sub     x3, x3, x16, lsl #0
    ldr     d24, [x17, x4, lsl #3]
    fcmgt   d24, d4, d24
    fmov    x16, d24
    sub     x4, x4, x16, lsl #0
    ldr     d24, [x17, x5, lsl #3]
    fcmgt   d24, d5, d24
    fmov    x16, d24
    sub     x5, x5, x16, lsl #0
    ldr     d24, [x17, x6, lsl #3]
    fcmgt   d24, d6, d24
    fmov    x16, d24
    sub     x6, x6, x16, lsl #0
    ldr     d24, [x17, x7, lsl #3]
    fcmgt   d24, d7, d24
    fmov    x16, d24
    sub     x7, x7, x16, lsl #0
    ldr     d24, [x17, x8, lsl #3]
    fcmgt   d24, d16, d24
    fmov    x16, d24
    sub     x8, x8, x16, lsl #0
    ldr     d24, [x17, x9, lsl #3]
    fcmgt   d24, d17, d24
    fmov    x16, d24
    sub     x9, x9, x16, lsl #0
    ldr     d24, [x17, x10, lsl #3]
    fcmgt   d24, d18, d24
    fmov    x16, d24
    sub     x10, x10, x16, lsl #0
    ldr     d24, [x17, x11, lsl #3]
    fcmgt   d24, d19, d24
    fmov    x16, d24
    sub     x11, x11, x16, lsl #0
    ldr     d24, [x17, x12, lsl #3]
    fcmgt   d24, d20, d24
    fmov    x16, d24
    sub     x12, x12, x16, lsl #0
    ldr     d24, [x17, x13, lsl #3]
    fcmgt   d24, d21, d24
    fmov    x16, d24
    sub     x13, x13, x16, lsl #0
    ldr     d24, [x17, x14, lsl #3]
    fcmgt   d24, d22, d24
    fmov    x16, d24
    sub     x14, x14, x16, lsl #0
    ldr     d24, [x17, x15, lsl #3]
    fcmgt   d24, d23, d24
    fmov    x16, d24
    sub     x15, x15, x16, lsl #0
    mov     x16, #0
    sub     x17, x17, x16           // &a[h/2 - 1], or a after h = 1
Lfinal:                             // x17 = a
    ldr     d24, [x17, x0, lsl #3]
    fcmgt   d24, d0, d24
    fmov    x16, d24
    sub     x0, x0, x16
    ldr     d24, [x17, x1, lsl #3]
    fcmgt   d24, d1, d24
    fmov    x16, d24
    sub     x1, x1, x16
    ldr     d24, [x17, x2, lsl #3]
    fcmgt   d24, d2, d24
    fmov    x16, d24
    sub     x2, x2, x16
    ldr     d24, [x17, x3, lsl #3]
    fcmgt   d24, d3, d24
    fmov    x16, d24
    sub     x3, x3, x16
    ldr     d24, [x17, x4, lsl #3]
    fcmgt   d24, d4, d24
    fmov    x16, d24
    sub     x4, x4, x16
    ldr     d24, [x17, x5, lsl #3]
    fcmgt   d24, d5, d24
    fmov    x16, d24
    sub     x5, x5, x16
    ldr     d24, [x17, x6, lsl #3]
    fcmgt   d24, d6, d24
    fmov    x16, d24
    sub     x6, x6, x16
    ldr     d24, [x17, x7, lsl #3]
    fcmgt   d24, d7, d24
    fmov    x16, d24
    sub     x7, x7, x16
    ldr     d24, [x17, x8, lsl #3]
    fcmgt   d24, d16, d24
    fmov    x16, d24
    sub     x8, x8, x16
    ldr     d24, [x17, x9, lsl #3]
    fcmgt   d24, d17, d24
    fmov    x16, d24
    sub     x9, x9, x16
    ldr     d24, [x17, x10, lsl #3]
    fcmgt   d24, d18, d24
    fmov    x16, d24
    sub     x10, x10, x16
    ldr     d24, [x17, x11, lsl #3]
    fcmgt   d24, d19, d24
    fmov    x16, d24
    sub     x11, x11, x16
    ldr     d24, [x17, x12, lsl #3]
    fcmgt   d24, d20, d24
    fmov    x16, d24
    sub     x12, x12, x16
    ldr     d24, [x17, x13, lsl #3]
    fcmgt   d24, d21, d24
    fmov    x16, d24
    sub     x13, x13, x16
    ldr     d24, [x17, x14, lsl #3]
    fcmgt   d24, d22, d24
    fmov    x16, d24
    sub     x14, x14, x16
    ldr     d24, [x17, x15, lsl #3]
    fcmgt   d24, d23, d24
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
    add     v25.2d, v25.2d, v30.2d
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


    // ---- Large n: the array is far bigger than the cache, so what matters is
    // how many probes are in flight at once. Needles are taken 128 at a time
    // and the block is swept once per level, with the bases kept in `out`.
    // x0 a, x3 n, x4 k, x9 block's needles, x10 block's out, x11 last block's
    // needles, x12 out - v, x13 h, x17 &a[h - 1], x16 passes left, x8 NaN seen
Lbig:
    dup     v27.2d, v27.d[0]
    dup     v28.2d, x6
    sub     x9, x3, #128
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
    add     x16, x4, #1             // k levels and the last probe
Lbig_pass:
    add     x17, x0, x13, lsl #3
    sub     x17, x17, #8            // &a[h - 1]
    add     x5, x0, x13, lsl #2
    sub     x5, x5, #8              // &a[h/2 - 1], the next pass's probes
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
    mov     x14, x9
    mov     x15, x10
    mov     x7, #128
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
    subs    x7, x7, #1
    b.ne    Lbig_nf_loop
    b       Lbig_next


Lnan_group:
    fmov    x0, d26
    mov     x3, v26.d[1]
    fmov    x1, d25
    mov     x2, v25.d[1]
    mov     x4, #16
    mov     x8, #1                  // come back to Lnext afterwards
    b       Lscalar
Lsmall:
    cbz     x3, Ldone
    mov     x4, x3
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
