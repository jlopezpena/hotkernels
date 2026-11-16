// void isin_i64(const int64_t *x, const int64_t *test, bool *out, size_t n);
// x0 = x (n values), x1 = test (exactly 8 values), x2 = out (n bytes), x3 = n.
//
// 16 elements per iteration. The 8 test values are broadcast into v16-v23.
// Each vector of two elements is compared with all of them and the results are
// combined with an OR tree; the eight 64-bit lane masks pairs are then narrowed
// to 16 bytes and shifted down to 0 or 1.
//
// The leftover (n mod 16) is handled by one more block that ends exactly at
// the end of the arrays and overlaps the previous one. Fewer than 16 elements
// take a scalar loop.

// \res = all-ones in each 64-bit lane of \src that equals a test value.
// Uses v1-v6.
.macro MEMBER res, src
    cmeq    v1.2d, \src\().2d, v16.2d
    cmeq    v2.2d, \src\().2d, v17.2d
    cmeq    v3.2d, \src\().2d, v18.2d
    cmeq    v4.2d, \src\().2d, v19.2d
    orr     v1.16b, v1.16b, v2.16b
    orr     v3.16b, v3.16b, v4.16b
    cmeq    v2.2d, \src\().2d, v20.2d
    cmeq    v4.2d, \src\().2d, v21.2d
    cmeq    v5.2d, \src\().2d, v22.2d
    cmeq    v6.2d, \src\().2d, v23.2d
    orr     v2.16b, v2.16b, v4.16b
    orr     v5.16b, v5.16b, v6.16b
    orr     v1.16b, v1.16b, v3.16b
    orr     v2.16b, v2.16b, v5.16b
    orr     \res\().16b, v1.16b, v2.16b
.endm

.text
.globl _isin_i64
.p2align 2
_isin_i64:
    cmp     x3, #16
    b.lo    Lsmall
    ld1r    {v16.2d}, [x1]
    add     x8, x1, #8
    ld1r    {v17.2d}, [x8]
    add     x8, x1, #16
    ld1r    {v18.2d}, [x8]
    add     x8, x1, #24
    ld1r    {v19.2d}, [x8]
    add     x8, x1, #32
    ld1r    {v20.2d}, [x8]
    add     x8, x1, #40
    ld1r    {v21.2d}, [x8]
    add     x8, x1, #48
    ld1r    {v22.2d}, [x8]
    add     x8, x1, #56
    ld1r    {v23.2d}, [x8]
    add     x9, x0, x3, lsl #3      // end of x
    add     x10, x2, x3             // end of out
    sub     x3, x3, #16             // elements left after the block below
Lloop:
    ldp     q0, q7, [x0]
    MEMBER  v24, v0
    MEMBER  v25, v7
    ldp     q0, q7, [x0, #32]
    MEMBER  v26, v0
    MEMBER  v27, v7
    ldp     q0, q7, [x0, #64]
    MEMBER  v28, v0
    MEMBER  v29, v7
    ldp     q0, q7, [x0, #96]
    MEMBER  v30, v0
    MEMBER  v31, v7
    uzp1    v24.4s, v24.4s, v25.4s
    uzp1    v26.4s, v26.4s, v27.4s
    uzp1    v28.4s, v28.4s, v29.4s
    uzp1    v30.4s, v30.4s, v31.4s
    uzp1    v24.8h, v24.8h, v26.8h
    uzp1    v28.8h, v28.8h, v30.8h
    uzp1    v24.16b, v24.16b, v28.16b
    ushr    v24.16b, v24.16b, #7    // 0xFF -> 1
    str     q24, [x2], #16
    add     x0, x0, #128
    subs    x3, x3, #16
    b.hs    Lloop
    cmn     x3, #16                 // x3 = leftover - 16
    b.eq    Ldone
    sub     x0, x9, #128            // last 16 elements, overlapping
    sub     x2, x10, #16
    mov     x3, #0
    b       Lloop

Lsmall:
    cbz     x3, Ldone
    ldp     x8, x9, [x1]
    ldp     x10, x11, [x1, #16]
    ldp     x12, x13, [x1, #32]
    ldp     x14, x15, [x1, #48]
Lscalar:
    ldr     x4, [x0], #8
    cmp     x4, x8
    ccmp    x4, x9, #4, ne          // #4 sets Z, so an earlier match is kept
    ccmp    x4, x10, #4, ne
    ccmp    x4, x11, #4, ne
    ccmp    x4, x12, #4, ne
    ccmp    x4, x13, #4, ne
    ccmp    x4, x14, #4, ne
    ccmp    x4, x15, #4, ne
    cset    w5, eq
    strb    w5, [x2], #1
    subs    x3, x3, #1
    b.ne    Lscalar
Ldone:
    ret
