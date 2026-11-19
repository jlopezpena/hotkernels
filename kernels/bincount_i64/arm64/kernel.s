// void bincount_i64(const int64_t *x, int64_t *counts, size_t n, size_t bins);
//
// x0 = x, x1 = counts, x2 = n, x3 = bins. counts[v] = how many of the n
// elements at x equal v, for v = 0..bins-1. An element outside that range,
// negative or at or above bins, is ignored.
//
// Each element is compared with bins as an unsigned 64-bit number, so a
// negative element is a huge one and fails the same test as one that is
// too large.

// Count the element in \reg into counts, if it is below bins.
.macro COUNT reg, tmp
    cmp     \reg, x3
    b.hs    1f
    ldr     \tmp, [x1, \reg, lsl #3]
    add     \tmp, \tmp, #1
    str     \tmp, [x1, \reg, lsl #3]
1:
.endm

.text
.globl _bincount_i64
.p2align 2
_bincount_i64:
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
    // Eight elements per step, from four pair loads.
Lloop8:
    ldp     x6, x7, [x0]
    ldp     x8, x9, [x0, #16]
    COUNT   x6, x12
    COUNT   x7, x13
    COUNT   x8, x14
    COUNT   x9, x15
    ldp     x6, x7, [x0, #32]
    ldp     x8, x9, [x0, #48]
    add     x0, x0, #64
    COUNT   x6, x12
    COUNT   x7, x13
    COUNT   x8, x14
    COUNT   x9, x15
    subs    x2, x2, #8
    b.hs    Lloop8
Ltail:
    ands    x2, x2, #7              // 0 to 7 elements left
    b.eq    Ldone
Lone:
    ldr     x6, [x0], #8
    COUNT   x6, x7
    subs    x2, x2, #1
    b.ne    Lone
Ldone:
    ret
