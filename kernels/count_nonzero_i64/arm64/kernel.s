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
// between flushes. Each lane is then at most 2 * 8192 = 16,384 and the two
// accumulators added together at most 32,768, which fits in 16 bits.
.set BLOCK_STEPS, 8192
// From this many elements (16 MB) the array is larger than the 12 MB L2 cache
// and the main loop also prefetches. Below it the prefetches only cost time.
.set HUGE_MIN_N, 2097152
// How far ahead of the loads the prefetches reach, in bytes.
.set PREFETCH_AHEAD, 262144

// v0-v7 = eight vectors (sixteen elements; absent ones must be zero).
// Leaves sixteen bytes in v\dst: -1 for a non-zero element, 0 for a zero one.
.macro BYTES16 dst
    umaxp   v0.4s, v0.4s, v1.4s
    umaxp   v2.4s, v2.4s, v3.4s
    umaxp   v4.4s, v4.4s, v5.4s
    umaxp   v6.4s, v6.4s, v7.4s
    umaxp   v0.8h, v0.8h, v2.8h
    umaxp   v4.8h, v4.8h, v6.8h
    umaxp   v0.16b, v0.16b, v4.16b
    cmtst   v\dst\().16b, v0.16b, v0.16b
.endm

// The same, subtracting the number of non-zero elements from the 16-bit
// lanes of v\acc.
.macro COUNT16 acc
    BYTES16 0
    sadalp  v\acc\().8h, v0.16b
.endm

// Loads the sixteen elements at \base + \off into v0-v7.
.macro LOAD16 base, off
    ldp     q0, q1, [\base, #\off]
    ldp     q2, q3, [\base, #\off + 32]
    ldp     q4, q5, [\base, #\off + 64]
    ldp     q6, q7, [\base, #\off + 96]
.endm

// The same with `ldnp`, the load that hints the data will not be needed again
// soon. Used in the main loop: it costs nothing while the array fits in the
// caches and is a few percent faster once it does not.
.macro LOAD16_STREAM base, off
    ldnp    q0, q1, [\base, #\off]
    ldnp    q2, q3, [\base, #\off + 32]
    ldnp    q4, q5, [\base, #\off + 64]
    ldnp    q6, q7, [\base, #\off + 96]
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
    mov     x6, #0                  // count so far
    cmp     x1, #32
    b.hs    Lbig

    // ---- Fewer than 32 elements left; the low five bits of x1 say how many.
    //      At most two groups, so their bytes are simply added: the sum of
    //      the sixteen bytes is minus the count and fits in a signed byte.
Ltail:
    movi    v16.2d, #0
    tbz     x1, #4, Ltail15
    LOAD16  x0, 0
    add     x0, x0, #128
    BYTES16 16
Ltail15:
    tst     x1, #15
    b.ne    Lladder
Lfinish:
    addv    b0, v16.16b
    smov    w7, v0.b[0]
    sub     x0, x6, w7, sxtw
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
    BYTES16 0
    add     v16.16b, v16.16b, v0.16b
    b       Lfinish

    // ---- 32 elements or more.
Lbig:
    movi    v16.2d, #0              // minus the count, 16-bit lanes
    movi    v17.2d, #0
    lsr     x4, x1, #5              // steps left
Lblock:
    mov     x5, #BLOCK_STEPS
    cmp     x4, x5
    csel    x5, x4, x5, lo          // steps in this block
    sub     x4, x4, x5
    cmp     x1, #HUGE_MIN_N
    b.hs    Lhugestep
Lstep:
    LOAD16_STREAM x0, 0
    COUNT16 16
    LOAD16_STREAM x0, 128
    add     x0, x0, #256
    COUNT16 17
    subs    x5, x5, #1
    b.ne    Lstep
Lblockdone:
    cbnz    x4, Lflush
    tst     x1, #31
    b.ne    Lbigtail
    REDUCE  x0
    ret
Lbigtail:
    REDUCE  x6
    b       Ltail
Lflush:
    REDUCE  x6
    movi    v16.2d, #0
    movi    v17.2d, #0
    b       Lblock

    // The same loop for arrays that do not fit in the L2 cache: every step
    // also asks for two of the four cache lines it will read PREFETCH_AHEAD
    // bytes later (the hardware prefetcher brings their neighbours). Near the
    // end of the array these addresses lie beyond it; a prefetch never
    // faults and reads nothing the program can see.
Lhugestep:
    add     x7, x0, #PREFETCH_AHEAD
    prfm    pldl2keep, [x7]
    prfm    pldl2keep, [x7, #128]
    LOAD16_STREAM x0, 0
    COUNT16 16
    LOAD16_STREAM x0, 128
    add     x0, x0, #256
    COUNT16 17
    subs    x5, x5, #1
    b.ne    Lhugestep
    b       Lblockdone
