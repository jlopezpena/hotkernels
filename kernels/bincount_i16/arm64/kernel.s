// void bincount_i16(const int16_t *x, int64_t *counts, size_t n, size_t bins);
//
// x0 = x, x1 = counts, x2 = n, x3 = bins. counts[v] = how many of the n
// elements at x equal v, for v = 0..bins-1. An element outside that range,
// negative or at or above bins, is ignored.
//
// The element is sign-extended to 64 bits and compared with bins as an
// unsigned number, so a negative element is a huge one and fails the same
// test as one that is too large.

.text
.globl _bincount_i16
.p2align 2
_bincount_i16:
    // Zero the bins counters.
    mov     x4, x1
    mov     x5, x3
Lzero:
    str     xzr, [x4], #8
    subs    x5, x5, #1
    b.ne    Lzero

    cbz     x2, Ldone
Lloop:
    ldrsh   x6, [x0], #2
    cmp     x6, x3
    b.hs    Lskip
    ldr     x7, [x1, x6, lsl #3]
    add     x7, x7, #1
    str     x7, [x1, x6, lsl #3]
Lskip:
    subs    x2, x2, #1
    b.ne    Lloop
Ldone:
    ret
