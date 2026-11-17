// size_t count_nonzero_f64(const double *x, size_t n);
//
// x0 = x, x1 = n. Returns how many of the n elements at x are not zero.
// NaN counts as non-zero; +0.0 and -0.0 are both zero.
//
// A double is zero exactly when its low 63 bits are zero (the sign bit is the
// only difference between +0.0 and -0.0), and every NaN has some of those
// bits set. So the test is done on the bits: `cmtst` against a mask of the
// low 63 bits (kept in v31) gives a 64-bit lane of -1 for a non-zero element
// and 0 for a zero one. It does not depend on the floating-point mode.
//
// Elements are handled in groups of sixteen (eight vectors). After the eight
// `cmtst`, `uzp1` keeps the low half of every lane of two vectors at once,
// which packs four elements into one vector of 32-bit lanes. The same step
// again gives eight elements in 16-bit lanes, and again sixteen in bytes:
//
//     8 masks  --4 uzp1.4s-->  4  --2 uzp1.8h-->  2  --1 uzp1.16b-->  1
//
// One `sadalp` adds the sixteen bytes, in pairs, into 16-bit lanes of an
// accumulator. That is sixteen vector instructions per eight vectors, two
// per vector, where the compiler's loop needs three.
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
// From this many elements, and below HUGE_MIN_N, the main loop is the hybrid
// one (see Lhybrid). Below it, the leftover elements a step of 18 leaves
// behind cost more than the hybrid loop saves.
.set HYBRID_MIN_N, 512
// Elements per step of the hybrid loop: one group of sixteen in vector
// registers and two in general registers.
.set HYBRID_STEP, 18
// The low 63 bits of a double: everything but the sign.
.set MAGNITUDE_BITS, 0x7fffffffffffffff

// v0-v7 = eight vectors (sixteen elements; absent ones must be zero),
// v31 = the 63-bit mask.
// Leaves sixteen bytes in v\dst: -1 for a non-zero element, 0 for a zero one.
.macro BYTES16 dst
    cmtst   v0.2d, v0.2d, v31.2d
    cmtst   v1.2d, v1.2d, v31.2d
    cmtst   v2.2d, v2.2d, v31.2d
    cmtst   v3.2d, v3.2d, v31.2d
    cmtst   v4.2d, v4.2d, v31.2d
    cmtst   v5.2d, v5.2d, v31.2d
    cmtst   v6.2d, v6.2d, v31.2d
    cmtst   v7.2d, v7.2d, v31.2d
    uzp1    v0.4s, v0.4s, v1.4s
    uzp1    v2.4s, v2.4s, v3.4s
    uzp1    v4.4s, v4.4s, v5.4s
    uzp1    v6.4s, v6.4s, v7.4s
    uzp1    v0.8h, v0.8h, v2.8h
    uzp1    v4.8h, v4.8h, v6.8h
    uzp1    v\dst\().16b, v0.16b, v4.16b
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
.globl _count_nonzero_f64
.p2align 2
_count_nonzero_f64:
    movi    v31.2d, #0xffffffffffffffff
    ushr    v31.2d, v31.2d, #1      // 0x7fffffffffffffff in both lanes
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
    cmp     x1, #HYBRID_MIN_N
    b.hs    Lhybridmin
Lvector:
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

    // ---- HYBRID_MIN_N elements or more.
Lhybridmin:
    cmp     x1, #HUGE_MIN_N
    b.hs    Lvector

    // Hybrid loop, for arrays that fit in the caches. The vector loop is
    // limited by the four vector ports (sixteen instructions per group, four
    // cycles) while the core can start eight instructions per cycle, and a
    // group with its loads and loop control uses only 26 of those 32. The
    // rest is spent on two more elements per step, loaded into general
    // registers and tested there, which use no vector port: 18 elements in
    // the time of 16.
Lhybrid:
    mov     x7, #HYBRID_STEP
    udiv    x4, x1, x7              // steps left
    msub    x2, x4, x7, x1          // elements left after the last step, 0..17
    mov     x8, #0                  // count of the elements tested in general registers
Lhblock:
    mov     x5, #BLOCK_STEPS
    cmp     x4, x5
    csel    x5, x4, x5, lo          // steps in this block
    sub     x4, x4, x5
Lhstep:
    ldp     x9, x10, [x0, #128]
    LOAD16_STREAM x0, 0
    add     x0, x0, #(8 * HYBRID_STEP)
    COUNT16 16
    tst     x9, #MAGNITUDE_BITS
    cinc    x8, x8, ne
    tst     x10, #MAGNITUDE_BITS
    cinc    x8, x8, ne
    subs    x5, x5, #1
    b.ne    Lhstep
    REDUCE  x6                      // v17 is zero here
    cbnz    x4, Lhflush
    add     x6, x6, x8
    mov     x1, x2                  // Ltail reads the count from the low bits of x1
    b       Ltail
Lhflush:
    movi    v16.2d, #0
    b       Lhblock
