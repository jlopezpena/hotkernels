// void bincount_i16(const int16_t *x, int64_t *counts, size_t n, size_t bins);
//
// x0 = x, x1 = counts, x2 = n, x3 = bins. counts[v] = how many of the n
// elements at x equal v, for v = 0..bins-1. An element outside that range,
// negative or at or above bins, is ignored.
//
// The element is sign-extended to 64 bits and compared with bins as an
// unsigned number, so a negative element is a huge one and fails the same
// test as one that is too large.

// The table path is for at least this many counters (its sum works four at
// a time) and at most this many: four tables of 32-bit counters, 16 KB of
// stack.
.set TABLES_MIN_BINS, 4
.set TABLES_MAX_BINS, 1024
// It is taken from this many elements per counter, where clearing and
// summing the tables is repaid.
.set TABLES_MIN_PER_BIN, 8

// Count the sign-extended element in \reg into the 32-bit table at \base, if
// it is below bins.
.macro COUNTT reg, base, tmp
    cmp     \reg, x3
    b.hs    1f
    ldr     \tmp, [\base, \reg, lsl #2]
    add     \tmp, \tmp, #1
    str     \tmp, [\base, \reg, lsl #2]
1:
.endm

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
    // The four tables: 4 <= bins <= 1024, n >= 8 * bins and n < 2^32.
    cmp     x3, #TABLES_MIN_BINS
    b.lo    Lsingle
    cmp     x3, #TABLES_MAX_BINS
    b.hi    Lsingle
    lsr     x4, x2, #32
    cbnz    x4, Lsingle
    cmp     x2, x3, lsl #3
    b.hs    Ltables
Lsingle:
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

    // ---- Four tables ------------------------------------------------------
    // Element k of each group of four is counted in table k, 32-bit counters
    // on the stack, so neighbouring elements never update the same memory.
    // A table is bins counters rounded up to whole 16 bytes; the padding is
    // cleared and never counted. counts is not cleared: the sum of the four
    // tables is stored over it. The counters cannot overflow: n < 2^32.
Ltables:
    add     x17, x3, #3
    and     x17, x17, #0xfffffffffffffffc
    lsl     x17, x17, #2            // bytes of one table, a multiple of 16
    sub     sp, sp, x17, lsl #2
    add     x4, sp, x17             // table 1
    add     x5, x4, x17             // table 2
    add     x16, x5, x17            // table 3
    // Clear the four tables, 64 bytes per step: four times a multiple of 16.
    movi    v0.2d, #0
    mov     x6, sp
    add     x7, x16, x17
Lcleartables:
    stp     q0, q0, [x6]
    stp     q0, q0, [x6, #32]
    add     x6, x6, #64
    cmp     x6, x7
    b.lo    Lcleartables

    subs    x2, x2, #8              // at least eight elements: n >= 8 * bins
Ltables8:
    ldp     x6, x7, [x0], #16
    sxth    x8, w6
    sbfx    x9, x6, #16, #16
    sbfx    x10, x6, #32, #16
    asr     x11, x6, #48
    COUNTT  x8, sp, w12
    COUNTT  x9, x4, w13
    COUNTT  x10, x5, w14
    COUNTT  x11, x16, w15
    sxth    x8, w7
    sbfx    x9, x7, #16, #16
    sbfx    x10, x7, #32, #16
    asr     x11, x7, #48
    COUNTT  x8, sp, w12
    COUNTT  x9, x4, w13
    COUNTT  x10, x5, w14
    COUNTT  x11, x16, w15
    subs    x2, x2, #8
    b.hs    Ltables8

    // counts[v] = the four tables' counters v, four counters per step; the
    // last step ends exactly at bins and may overlap the one before.
    mov     x6, #0                  // counter index
    sub     x7, x3, #4              // the last step's index
Lsum:
    cmp     x6, x7
    csel    x6, x6, x7, lo
    lsl     x8, x6, #2
    ldr     q1, [sp, x8]
    ldr     q2, [x4, x8]
    ldr     q3, [x5, x8]
    ldr     q4, [x16, x8]
    add     v1.4s, v1.4s, v2.4s
    add     v3.4s, v3.4s, v4.4s
    add     v1.4s, v1.4s, v3.4s
    uxtl    v5.2d, v1.2s
    uxtl2   v6.2d, v1.4s
    add     x9, x1, x6, lsl #3
    stp     q5, q6, [x9]
    add     x6, x6, #4
    cmp     x6, x3
    b.lo    Lsum
    add     sp, sp, x17, lsl #2
    b       Ltail                   // 0 to 7 elements left, into counts
