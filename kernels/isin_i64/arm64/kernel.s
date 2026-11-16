// void isin_i64(const int64_t *x, const int64_t *test, bool *out, size_t n);
// x0 = x (n values), x1 = test (exactly 8 values), x2 = out (n bytes), x3 = n.
//
// 16 elements per iteration, in one of two ways, chosen once from the test
// values.
//
// Narrow keys (Lnarrow, Lbiased), when there is a base for which every test
// value minus base is below 0xFFFF. Base 0 is tried first, then
// test[0] - 0x7FFF, which works when all 8 test values lie within 32,767 of
// test[0]. Each element (minus the base, if it is not 0) is narrowed from 64
// to 16 bits with unsigned saturation (uqxtn twice), which keeps values below
// 0xFFFF exact and turns everything else into 0xFFFF. An element is in the set
// exactly when its 16-bit key equals a test key, because no test key is
// 0xFFFF. The 8 compares and the OR tree then run on vectors of eight keys
// instead of two elements.
//
// Wide (Lwide), for any other test values: each vector of two elements is
// compared with the 8 broadcast test values at 64 bits.
//
// Either way the lane masks are narrowed to 16 bytes and shifted down to 0 or
// 1. The leftover (n mod 16) is handled by one more block that ends exactly at
// the end of the arrays and overlaps the previous one. Fewer than 16 elements
// take a scalar loop.

// \res = all-ones in each lane (of arrangement \sz) of \src that equals one of
// the values broadcast in v16-v23. \a-\f are scratch registers.
.macro MEMBER res, src, sz, a, b, c, d, e, f
    cmeq    \a\().\sz, \src\().\sz, v16.\sz
    cmeq    \b\().\sz, \src\().\sz, v17.\sz
    cmeq    \c\().\sz, \src\().\sz, v18.\sz
    cmeq    \d\().\sz, \src\().\sz, v19.\sz
    orr     \a\().16b, \a\().16b, \b\().16b
    orr     \c\().16b, \c\().16b, \d\().16b
    cmeq    \b\().\sz, \src\().\sz, v20.\sz
    cmeq    \d\().\sz, \src\().\sz, v21.\sz
    cmeq    \e\().\sz, \src\().\sz, v22.\sz
    cmeq    \f\().\sz, \src\().\sz, v23.\sz
    orr     \b\().16b, \b\().16b, \d\().16b
    orr     \e\().16b, \e\().16b, \f\().16b
    orr     \a\().16b, \a\().16b, \c\().16b
    orr     \b\().16b, \b\().16b, \e\().16b
    orr     \res\().16b, \a\().16b, \b\().16b
.endm

// Narrow four vectors of two 64-bit values (\a, \b, \c, \d, in order) to eight
// 16-bit values in \a, saturating. Clobbers \c.
.macro NARROW8 a, b, c, d
    uqxtn   \a\().2s, \a\().2d
    uqxtn2  \a\().4s, \b\().2d
    uqxtn   \c\().2s, \c\().2d
    uqxtn2  \c\().4s, \d\().2d
    uqxtn   \a\().4h, \a\().4s
    uqxtn2  \a\().8h, \c\().4s
.endm

// Broadcast the eight 16-bit keys in \src into v16-v23.
.macro DUPKEYS src
    dup     v16.8h, \src\().h[0]
    dup     v17.8h, \src\().h[1]
    dup     v18.8h, \src\().h[2]
    dup     v19.8h, \src\().h[3]
    dup     v20.8h, \src\().h[4]
    dup     v21.8h, \src\().h[5]
    dup     v22.8h, \src\().h[6]
    dup     v23.8h, \src\().h[7]
.endm

// The narrow-key loop, ending in a branch to Ldone. With \biased, the base in
// v31 is subtracted from every element first.
.macro NARROWLOOP name, biased
\name:
    ldp     q0, q1, [x0]
    ldp     q2, q3, [x0, #32]
    ldp     q4, q5, [x0, #64]
    ldp     q6, q7, [x0, #96]
.if \biased
    sub     v0.2d, v0.2d, v31.2d
    sub     v1.2d, v1.2d, v31.2d
    sub     v2.2d, v2.2d, v31.2d
    sub     v3.2d, v3.2d, v31.2d
    sub     v4.2d, v4.2d, v31.2d
    sub     v5.2d, v5.2d, v31.2d
    sub     v6.2d, v6.2d, v31.2d
    sub     v7.2d, v7.2d, v31.2d
.endif
    NARROW8 v0, v1, v2, v3          // keys of elements 0-7
    NARROW8 v4, v5, v6, v7          // keys of elements 8-15
    MEMBER  v24, v0, 8h, v1, v2, v3, v5, v6, v7
    MEMBER  v25, v4, 8h, v1, v2, v3, v5, v6, v7
    uzp1    v24.16b, v24.16b, v25.16b
    ushr    v24.16b, v24.16b, #7    // 0xFF -> 1
    str     q24, [x2], #16
    add     x0, x0, #128
    subs    x3, x3, #16
    b.hs    \name
    cmn     x3, #16                 // x3 = leftover - 16
    b.eq    Ldone
    sub     x0, x9, #128            // last 16 elements, overlapping
    sub     x2, x10, #16
    mov     x3, #0
    b       \name
.endm

.text
.globl _isin_i64
.p2align 2
_isin_i64:
    cmp     x3, #16
    b.lo    Lsmall
    add     x9, x0, x3, lsl #3      // end of x
    add     x10, x2, x3             // end of out
    sub     x3, x3, #16             // elements left after the first block

    // Test keys: the test values narrowed like the elements. A key of 0xFFFF
    // means that value is out of range for that base. Try base 0 first.
    ldp     q0, q1, [x1]
    ldp     q2, q3, [x1, #32]
    movi    v30.16b, #0xff
    NARROW8 v0, v1, v2, v3
    cmeq    v1.8h, v0.8h, v30.8h
    umaxv   h1, v1.8h
    fmov    w8, s1
    cbnz    w8, Lbiased_setup
    DUPKEYS v0
    NARROWLOOP Lnarrow, 0

Lbiased_setup:
    ldp     q0, q1, [x1]
    ldp     q2, q3, [x1, #32]
    ldr     x8, [x1]
    sub     x8, x8, #0x7000
    sub     x8, x8, #0xfff          // base = test[0] - 0x7FFF
    dup     v31.2d, x8
    sub     v0.2d, v0.2d, v31.2d
    sub     v1.2d, v1.2d, v31.2d
    sub     v2.2d, v2.2d, v31.2d
    sub     v3.2d, v3.2d, v31.2d
    NARROW8 v0, v1, v2, v3
    cmeq    v1.8h, v0.8h, v30.8h
    umaxv   h1, v1.8h
    fmov    w8, s1
    cbnz    w8, Lwide_setup
    DUPKEYS v0
    NARROWLOOP Lbiased, 1

Lwide_setup:
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
Lwide:
    ldp     q0, q7, [x0]
    MEMBER  v24, v0, 2d, v1, v2, v3, v4, v5, v6
    MEMBER  v25, v7, 2d, v1, v2, v3, v4, v5, v6
    ldp     q0, q7, [x0, #32]
    MEMBER  v26, v0, 2d, v1, v2, v3, v4, v5, v6
    MEMBER  v27, v7, 2d, v1, v2, v3, v4, v5, v6
    ldp     q0, q7, [x0, #64]
    MEMBER  v28, v0, 2d, v1, v2, v3, v4, v5, v6
    MEMBER  v29, v7, 2d, v1, v2, v3, v4, v5, v6
    ldp     q0, q7, [x0, #96]
    MEMBER  v30, v0, 2d, v1, v2, v3, v4, v5, v6
    MEMBER  v31, v7, 2d, v1, v2, v3, v4, v5, v6
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
    b.hs    Lwide
    cmn     x3, #16                 // x3 = leftover - 16
    b.eq    Ldone
    sub     x0, x9, #128            // last 16 elements, overlapping
    sub     x2, x10, #16
    mov     x3, #0
    b       Lwide

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
