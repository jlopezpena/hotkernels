// void isin_i64(const int64_t *x, const int64_t *test, bool *out, size_t n);
// x0 = x (n values), x1 = test (exactly 8 values), x2 = out (n bytes), x3 = n.
//
// Scalar: each element is compared with the 8 test values through a chain of
// conditional compares, and the result is written as 0 or 1.
.text
.globl _isin_i64
.p2align 2
_isin_i64:
    cbz     x3, Ldone
    ldp     x8, x9, [x1]
    ldp     x10, x11, [x1, #16]
    ldp     x12, x13, [x1, #32]
    ldp     x14, x15, [x1, #48]
Lloop:
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
    b.ne    Lloop
Ldone:
    ret
