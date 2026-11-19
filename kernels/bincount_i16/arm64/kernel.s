// void bincount_i16(const int16_t *x, int64_t *counts, size_t n, size_t bins);
//
// x0 = x, x1 = counts, x2 = n, x3 = bins. counts[v] = how many of the n
// elements at x equal v, for v = 0..bins-1. An element outside that range,
// negative or at or above bins, is ignored.
//
// The element is sign-extended to 64 bits and compared with bins as an
// unsigned number, so a negative element is a huge one and fails the same
// test as one that is too large.

// Count the sign-extended element in \reg into counts, if it is below bins.
.macro COUNT reg, tmp
    cmp     \reg, x3
    b.hs    1f
    ldr     \tmp, [x1, \reg, lsl #3]
    add     \tmp, \tmp, #1
    str     \tmp, [x1, \reg, lsl #3]
1:
.endm

.text
.globl _bincount_i16
.p2align 2
_bincount_i16:
    // Clear the bins counters: 64 bytes per step, the last 64 bytes ending
    // exactly at the end of counts and overlapping the step before.
    cmp     x3, #8
    b.lo    Lclearfew
    movi    v0.2d, #0
    mov     x4, x1
    add     x5, x1, x3, lsl #3
    sub     x5, x5, #64             // the last eight counters
Lclear:
    stp     q0, q0, [x4]
    stp     q0, q0, [x4, #32]
    add     x4, x4, #64
    cmp     x4, x5
    b.lo    Lclear
    stp     q0, q0, [x5]
    stp     q0, q0, [x5, #32]
    b       Lcount
Lclearfew:                          // 1 to 7 counters
    mov     x5, x3
Lclearone:
    subs    x5, x5, #1
    str     xzr, [x1, x5, lsl #3]
    b.ne    Lclearone

Lcount:
    subs    x2, x2, #8
    b.lo    Ltail
    // Eight elements per step: two 8-byte loads, each element taken out of
    // its word sign-extended.
Lloop8:
    ldp     x6, x7, [x0], #16
    sxth    x8, w6
    sbfx    x9, x6, #16, #16
    sbfx    x10, x6, #32, #16
    asr     x11, x6, #48
    COUNT   x8, x12
    COUNT   x9, x13
    COUNT   x10, x14
    COUNT   x11, x15
    sxth    x8, w7
    sbfx    x9, x7, #16, #16
    sbfx    x10, x7, #32, #16
    asr     x11, x7, #48
    COUNT   x8, x12
    COUNT   x9, x13
    COUNT   x10, x14
    COUNT   x11, x15
    subs    x2, x2, #8
    b.hs    Lloop8
Ltail:
    ands    x2, x2, #7              // 0 to 7 elements left
    b.eq    Ldone
Lone:
    ldrsh   x6, [x0], #2
    COUNT   x6, x7
    subs    x2, x2, #1
    b.ne    Lone
Ldone:
    ret
