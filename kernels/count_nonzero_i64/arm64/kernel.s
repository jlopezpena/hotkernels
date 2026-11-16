// size_t count_nonzero_i64(const int64_t *x, size_t n);
//
// x0 = x, x1 = n. Returns how many of the n elements at x are not zero.
//
// Elements are handled in groups of sixteen (eight vectors). A 64-bit element
// is non-zero exactly when the larger of its two 32-bit halves is non-zero,
// and `umaxp` computes that for two vectors at once, packing four elements
// into one vector of 32-bit lanes. The same step again gives eight elements
// in 16-bit lanes, and again sixteen elements in bytes:
//
//     8 vectors  --4 umaxp.4s-->  4  --2 umaxp.8h-->  2  --1 umaxp.16b-->  1
//
// One `cmtst` turns the sixteen bytes into -1 (non-zero) or 0, and one
// `sadalp` adds them, in pairs, into 16-bit lanes of an accumulator. That is
// nine vector instructions per eight vectors where compare-and-subtract needs
// sixteen, so the loop is limited by the three loads the core can do per
// cycle and not by its four vector ports.
//
// The accumulators hold minus the count. A lane changes by at most 2 per
// group, and the counts are moved to a general register before a lane can
// reach 65,536.

// Steps of the main loop (32 elements, one group into each accumulator)
// between flushes. Each lane is then at most 2 * 8192 = 16,384, the two
// accumulators added together at most 32,768, and the 4 more that the last
// groups can add still fit in 16 bits.
.set BLOCK_STEPS, 8192

// v0-v7 = eight vectors (sixteen elements; absent ones must be zero).
// Subtracts the number of non-zero elements from the 16-bit lanes of v\acc.
.macro COUNT16 acc
    umaxp   v0.4s, v0.4s, v1.4s
    umaxp   v2.4s, v2.4s, v3.4s
    umaxp   v4.4s, v4.4s, v5.4s
    umaxp   v6.4s, v6.4s, v7.4s
    umaxp   v0.8h, v0.8h, v2.8h
    umaxp   v4.8h, v4.8h, v6.8h
    umaxp   v0.16b, v0.16b, v4.16b
    cmtst   v0.16b, v0.16b, v0.16b
    sadalp  v\acc\().8h, v0.16b
.endm

// Loads the sixteen elements at \base + \off into v0-v7.
.macro LOAD16 base, off
    ldp     q0, q1, [\base, #\off]
    ldp     q2, q3, [\base, #\off + 32]
    ldp     q4, q5, [\base, #\off + 64]
    ldp     q6, q7, [\base, #\off + 96]
.endm

// \dst = minus the sum of the 16-bit lanes of v16 and v17, plus x6.
// Clobbers v0, v16 and x7.
.macro REDUCE dst
    add     v16.8h, v16.8h, v17.8h
    neg     v16.8h, v16.8h
    uaddlv  s0, v16.8h
    fmov    w7, s0
    add     \dst, x7, x6
.endm

.text
.globl _count_nonzero_i64
.p2align 2
_count_nonzero_i64:
    movi    v16.2d, #0              // minus the count, 16-bit lanes
    movi    v17.2d, #0
    mov     x6, #0                  // counts already flushed
    cmp     x1, #32
    b.hs    Lbig

    // ---- Fewer than 32 elements left; the low five bits of x1 say how many.
Ltail:
    tbz     x1, #4, Ltail15
    LOAD16  x0, 0
    add     x0, x0, #128
    COUNT16 16
Ltail15:
    tst     x1, #15
    b.ne    Lladder
    REDUCE  x0
    ret

    // 1..15 elements left: load what is there into v0-v7, leave the rest
    // zero, and count the group as usual. Zero elements add nothing.
Lladder:
    movi    v0.2d, #0
    movi    v1.2d, #0
    movi    v2.2d, #0
    movi    v3.2d, #0
    movi    v4.2d, #0
    movi    v5.2d, #0
    movi    v6.2d, #0
    movi    v7.2d, #0
    tbz     x1, #3, Lladder4
    ldp     q0, q1, [x0]
    ldp     q2, q3, [x0, #32]
    add     x0, x0, #64
Lladder4:
    tbz     x1, #2, Lladder2
    ldp     q4, q5, [x0], #32
Lladder2:
    tbz     x1, #1, Lladder1
    ldr     q6, [x0], #16
Lladder1:
    tbz     x1, #0, Lladder0
    ldr     d7, [x0]                // one element; the upper half of v7 is zeroed
Lladder0:
    COUNT16 17
    REDUCE  x0
    ret

    // ---- 32 elements or more.
Lbig:
    lsr     x4, x1, #5              // steps left
Lblock:
    mov     x5, #BLOCK_STEPS
    cmp     x4, x5
    csel    x5, x4, x5, lo          // steps in this block
    sub     x4, x4, x5
Lstep:
    LOAD16  x0, 0
    COUNT16 16
    LOAD16  x0, 128
    add     x0, x0, #256
    COUNT16 17
    subs    x5, x5, #1
    b.ne    Lstep
    cbnz    x4, Lflush
    // Last block: its counts stay in the accumulators.
    tst     x1, #31
    b.ne    Ltail
    REDUCE  x0
    ret
Lflush:
    REDUCE  x6
    movi    v16.2d, #0
    movi    v17.2d, #0
    b       Lblock
