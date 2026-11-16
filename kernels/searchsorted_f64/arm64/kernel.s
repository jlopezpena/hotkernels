// void searchsorted_f64(const double *a, const double *v, int64_t *out, size_t n);
//
// For each needle v[j], out[j] = first index i with !less(a[i], v[j]), where
// less(x, y) = x < y || (y is NaN && x is not NaN).
//
// Eight needles are searched at once with a branchless lower bound. For every
// needle the search keeps `base` and a shared `len`, with the answer always in
// [base, base + len]:
//     half = len >> 1
//     if (a[base + half - 1] < key) base += half
//     len -= half
// until len == 1, then one last probe of a[base]. `len` depends only on n, so
// the eight searches stay in step and their loads overlap.
//
// A needle that is NaN makes every `<` false, so a group containing one is
// handed to the plain one-at-a-time search at Lscalar, as are all needles when
// n < 8. Leftover needles (n not a multiple of 8) are covered by one more
// group that ends exactly at the last needle and recomputes a few results.
//
// Registers:
//   x0 a, x1 next needle, x2 next out, x3 n, x4 needles left
//   x5-x12 the eight bases, x13 len, x14 half, x15 &a[half - 1]
//   x16, x17 scratch;  d0-d7 keys, d16-d23 scratch
.text
.globl _searchsorted_f64
.p2align 2
_searchsorted_f64:
    mov     x4, x3
Lmain:
    cmp     x4, #8
    b.lo    Ltail
Lgroup:
    ldp     d0, d1, [x1]
    ldp     d2, d3, [x1, #16]
    ldp     d4, d5, [x1, #32]
    ldp     d6, d7, [x1, #48]
    // fmax returns NaN if either operand is NaN.
    fmax    d16, d0, d1
    fmax    d17, d2, d3
    fmax    d18, d4, d5
    fmax    d19, d6, d7
    fmax    d16, d16, d17
    fmax    d18, d18, d19
    fmax    d16, d16, d18
    fcmp    d16, d16
    b.vs    Lnan_group
    mov     x5, #0
    mov     x6, #0
    mov     x7, #0
    mov     x8, #0
    mov     x9, #0
    mov     x10, #0
    mov     x11, #0
    mov     x12, #0
    mov     x13, x3                 // len = n (at least 8 here)
Lstep:
    lsr     x14, x13, #1            // half
    add     x15, x0, x14, lsl #3
    sub     x13, x13, x14           // len -= half
    sub     x15, x15, #8            // &a[half - 1]
    ldr     d16, [x15, x5, lsl #3]
    fcmp    d16, d0
    add     x16, x5, x14
    csel    x5, x16, x5, mi
    ldr     d17, [x15, x6, lsl #3]
    fcmp    d17, d1
    add     x17, x6, x14
    csel    x6, x17, x6, mi
    ldr     d18, [x15, x7, lsl #3]
    fcmp    d18, d2
    add     x16, x7, x14
    csel    x7, x16, x7, mi
    ldr     d19, [x15, x8, lsl #3]
    fcmp    d19, d3
    add     x17, x8, x14
    csel    x8, x17, x8, mi
    ldr     d20, [x15, x9, lsl #3]
    fcmp    d20, d4
    add     x16, x9, x14
    csel    x9, x16, x9, mi
    ldr     d21, [x15, x10, lsl #3]
    fcmp    d21, d5
    add     x17, x10, x14
    csel    x10, x17, x10, mi
    ldr     d22, [x15, x11, lsl #3]
    fcmp    d22, d6
    add     x16, x11, x14
    csel    x11, x16, x11, mi
    ldr     d23, [x15, x12, lsl #3]
    fcmp    d23, d7
    add     x17, x12, x14
    csel    x12, x17, x12, mi
    cmp     x13, #1
    b.hi    Lstep
    // len == 1: the answer is base or base + 1.
    ldr     d16, [x0, x5, lsl #3]
    fcmp    d16, d0
    cinc    x5, x5, mi
    ldr     d17, [x0, x6, lsl #3]
    fcmp    d17, d1
    cinc    x6, x6, mi
    ldr     d18, [x0, x7, lsl #3]
    fcmp    d18, d2
    cinc    x7, x7, mi
    ldr     d19, [x0, x8, lsl #3]
    fcmp    d19, d3
    cinc    x8, x8, mi
    ldr     d20, [x0, x9, lsl #3]
    fcmp    d20, d4
    cinc    x9, x9, mi
    ldr     d21, [x0, x10, lsl #3]
    fcmp    d21, d5
    cinc    x10, x10, mi
    ldr     d22, [x0, x11, lsl #3]
    fcmp    d22, d6
    cinc    x11, x11, mi
    ldr     d23, [x0, x12, lsl #3]
    fcmp    d23, d7
    cinc    x12, x12, mi
    stp     x5, x6, [x2]
    stp     x7, x8, [x2, #16]
    stp     x9, x10, [x2, #32]
    stp     x11, x12, [x2, #48]
    add     x1, x1, #64
    add     x2, x2, #64
    sub     x4, x4, #8
    cmp     x4, #8
    b.hs    Lgroup
Ltail:
    cbz     x4, Ldone
    cmp     x3, #8
    b.lo    Lscalar_rest
    // 1 to 7 needles left and n >= 8: redo the last eight needles as one group.
    sub     x5, x4, #8
    add     x1, x1, x5, lsl #3
    add     x2, x2, x5, lsl #3
    mov     x4, #8
    b       Lgroup
Ldone:
    ret

Lnan_group:
    mov     x17, #8
    b       Lscalar
Lscalar_rest:
    mov     x17, x4
    // Plain binary search for the next x17 needles, any value. Rare path.
Lscalar:
    ldr     d0, [x1], #8            // key
    mov     x5, #0                  // lo
    mov     x6, x3                  // hi
    fcmp    d0, d0
    b.vs    Lnan_search
Lsearch:
    cmp     x5, x6
    b.hs    Lstore
    sub     x7, x6, x5
    add     x7, x5, x7, lsr #1      // mid
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
    sub     x4, x4, #1
    subs    x17, x17, #1
    b.ne    Lscalar
    b       Lmain
