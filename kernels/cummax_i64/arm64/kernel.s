// void cummax_i64(const int64_t *x, int64_t *out, size_t n);
// out[i] = the largest of x[0..i], compared as signed numbers.
// x0 = x, x1 = out, x2 = n. out does not overlap x.
//
// NEON has no maximum of 64-bit lanes, and the scalar maximum is a cmp and a
// csel. So the kernel does not compute the maximum where one vector compare
// shows there is nothing to compute. It works in blocks of 16 elements, with M
// (x3) the maximum before the block, and has three ways through a block:
//
//   flat      no element of the block is above M: one cmgt per vector, and
//             the block is M sixteen times.
//   rising    no element of the block is below the element before it, and the
//             element before the block is M: one cmgt per vector against an
//             overlapping load one element back, and the block is a copy.
//   general   anything else: groups of four in the integer registers without a
//             branch. The maxima of x0, x0..x1, x0..x2, x0..x3 do not depend on
//             M, so the chain from group to group is one cmp and one csel.
//
// The flat and rising loops run until a block fails their test; that block is
// done the general way. After a block that failed the flat test the flat test
// is tried again at once. After a general block the next one is tried flat if
// M did not change in it, rising if its last element is M, and is done the
// general way without a test otherwise.
//
// x2 counts the elements left minus 16 while blocks run.

// Four elements at [x0 + \o] the general way; x3 = M before and after.
.macro GROUP o
    ldp     x4, x5, [x0, #\o]
    ldp     x6, x7, [x0, #\o + 16]
    cmp     x5, x4
    csel    x5, x5, x4, gt          // max of x0..x1
    cmp     x6, x5
    csel    x6, x6, x5, gt          // max of x0..x2
    cmp     x7, x6
    csel    x7, x7, x6, gt          // max of x0..x3
    cmp     x4, x3
    csel    x4, x4, x3, gt
    cmp     x5, x3
    csel    x5, x5, x3, gt
    cmp     x6, x3
    csel    x6, x6, x3, gt
    cmp     x7, x3
    csel    x3, x7, x3, gt
    stp     x4, x5, [x1, #\o]
    stp     x6, x3, [x1, #\o + 16]
.endm

// A block the general way, and the count of what is left: the flags say
// whether another block follows (hs).
.macro GENERAL_BLOCK
    GROUP   0
    GROUP   32
    GROUP   64
    GROUP   96
    add     x0, x0, #128
    add     x1, x1, #128
    subs    x2, x2, #16
.endm

// The flat test: branches to \fail if an element of the block at x0 is above
// M, which v31 holds in both lanes.
.macro FLAT_TEST fail
    ldp     q0, q1, [x0]
    ldp     q2, q3, [x0, #32]
    ldp     q4, q5, [x0, #64]
    ldp     q6, q7, [x0, #96]
    cmgt    v0.2d, v0.2d, v31.2d
    cmgt    v1.2d, v1.2d, v31.2d
    cmgt    v2.2d, v2.2d, v31.2d
    cmgt    v3.2d, v3.2d, v31.2d
    cmgt    v4.2d, v4.2d, v31.2d
    cmgt    v5.2d, v5.2d, v31.2d
    cmgt    v6.2d, v6.2d, v31.2d
    cmgt    v7.2d, v7.2d, v31.2d
    orr     v0.16b, v0.16b, v1.16b
    orr     v2.16b, v2.16b, v3.16b
    orr     v4.16b, v4.16b, v5.16b
    orr     v6.16b, v6.16b, v7.16b
    orr     v0.16b, v0.16b, v2.16b
    orr     v4.16b, v4.16b, v6.16b
    orr     v0.16b, v0.16b, v4.16b
    umaxv   s0, v0.4s
    fmov    w9, s0
    cbnz    w9, \fail
.endm

// A flat block: M sixteen times.
.macro FLAT_STORE
    stp     q31, q31, [x1]
    stp     q31, q31, [x1, #32]
    stp     q31, q31, [x1, #64]
    stp     q31, q31, [x1, #96]
    add     x0, x0, #128
    add     x1, x1, #128
    subs    x2, x2, #16
.endm

// max(a, b) into a, in place: t is a temporary.
.macro VMAX a, b, t
    cmgt    \t\().2d, \b\().2d, \a\().2d
    bit     \a\().16b, \b\().16b, \t\().16b
.endm

// The first block of x in the vector registers: a prefix maximum in four
// steps, each vector against the one 1, 2 and 4 vectors back, after the
// first step against the input one element back (x[0] for x[0], since there
// is nothing before it). Sets x3 = M after the block.
.macro FIRST_BLOCK
    ldp     q0, q1, [x0]
    ldp     q2, q3, [x0, #32]
    ldp     q4, q5, [x0, #64]
    ldp     q6, q7, [x0, #96]
    ld1r    {v16.2d}, [x0]
    ldur    q17, [x0, #8]
    ldur    q18, [x0, #24]
    ldur    q19, [x0, #40]
    ldur    q20, [x0, #56]
    ldur    q21, [x0, #72]
    ldur    q22, [x0, #88]
    ldur    q23, [x0, #104]
    VMAX    v0, v16, v24
    VMAX    v1, v17, v25
    VMAX    v2, v18, v26
    VMAX    v3, v19, v27
    VMAX    v4, v20, v28
    VMAX    v5, v21, v29
    VMAX    v6, v22, v30
    VMAX    v7, v23, v24
    VMAX    v7, v6, v24
    VMAX    v6, v5, v30
    VMAX    v5, v4, v29
    VMAX    v4, v3, v28
    VMAX    v3, v2, v27
    VMAX    v2, v1, v26
    VMAX    v1, v0, v25
    VMAX    v7, v5, v24
    VMAX    v6, v4, v30
    VMAX    v5, v3, v29
    VMAX    v4, v2, v28
    VMAX    v3, v1, v27
    VMAX    v2, v0, v26
    VMAX    v7, v3, v24
    VMAX    v6, v2, v30
    VMAX    v5, v1, v29
    VMAX    v4, v0, v28
    stp     q0, q1, [x1]
    stp     q2, q3, [x1, #32]
    stp     q4, q5, [x1, #64]
    stp     q6, q7, [x1, #96]
    mov     x3, v7.d[1]
    add     x0, x0, #128
    add     x1, x1, #128
    subs    x2, x2, #16
.endm

.text
.globl _cummax_i64
.p2align 2
_cummax_i64:
    cbz     x2, Ldone
    ldr     x3, [x0]                // M
    subs    x2, x2, #16
    b.lo    Ltail
    mov     x8, x3
    FIRST_BLOCK
    b       Lgeneral_next

    // ---- General: one block, then choose how to try the next.
Lgeneral:
    mov     x8, x3                  // M before the block
    GENERAL_BLOCK
Lgeneral_next:
    b.lo    Ltail
    cmp     x3, x8
    b.eq    Lflat_enter
    ldur    x9, [x0, #-8]
    cmp     x3, x9
    b.ne    Lgeneral

    // ---- Rising. The element before the block is M.
Lrising:
    ldp     q0, q1, [x0]
    ldp     q2, q3, [x0, #32]
    ldp     q4, q5, [x0, #64]
    ldp     q6, q7, [x0, #96]
    ldur    q16, [x0, #-8]
    ext     v17.16b, v0.16b, v1.16b, #8
    ldur    q18, [x0, #24]
    ext     v19.16b, v2.16b, v3.16b, #8
    ldur    q20, [x0, #56]
    ext     v21.16b, v4.16b, v5.16b, #8
    ldur    q22, [x0, #88]
    ext     v23.16b, v6.16b, v7.16b, #8
    cmgt    v16.2d, v16.2d, v0.2d   // an element below the one before it
    cmgt    v17.2d, v17.2d, v1.2d
    cmgt    v18.2d, v18.2d, v2.2d
    cmgt    v19.2d, v19.2d, v3.2d
    cmgt    v20.2d, v20.2d, v4.2d
    cmgt    v21.2d, v21.2d, v5.2d
    cmgt    v22.2d, v22.2d, v6.2d
    cmgt    v23.2d, v23.2d, v7.2d
    orr     v16.16b, v16.16b, v17.16b
    orr     v18.16b, v18.16b, v19.16b
    orr     v20.16b, v20.16b, v21.16b
    orr     v22.16b, v22.16b, v23.16b
    orr     v16.16b, v16.16b, v18.16b
    orr     v20.16b, v20.16b, v22.16b
    orr     v16.16b, v16.16b, v20.16b
    umaxv   s16, v16.4s
    fmov    w9, s16
    cbnz    w9, Lrising_fail
    stp     q0, q1, [x1]
    stp     q2, q3, [x1, #32]
    stp     q4, q5, [x1, #64]
    stp     q6, q7, [x1, #96]
    add     x0, x0, #128
    add     x1, x1, #128
    subs    x2, x2, #16
    b.hs    Lrising
    ldur    x3, [x0, #-8]
    b       Ltail
Lrising_fail:
    ldur    x3, [x0, #-8]
    b       Lgeneral

    // ---- A block of the flat loop that holds a new maximum: the general way,
    //      then the flat test again at once, since in data whose maximum
    //      rarely changes the next block is flat. If it is not, the blocks
    //      after it go to the general loop.
Lflat_fail:
    GENERAL_BLOCK
    b.lo    Ltail
    ldur    x9, [x0, #-8]
    cmp     x3, x9
    b.eq    Lrising
    dup     v31.2d, x3
    FLAT_TEST Lgeneral
    FLAT_STORE
    b.hs    Lflat
    b       Ltail

    // ---- Flat. v31 = M in both lanes.
Lflat_enter:
    dup     v31.2d, x3
Lflat:
    FLAT_TEST Lflat_fail
    FLAT_STORE
    b.hs    Lflat

    // ---- Fewer than 16 elements left: groups of four, then one at a time.
Ltail:
    adds    x2, x2, #16
    b.eq    Ldone
    subs    x2, x2, #4
    b.lo    Lsingles
Ltail_group:
    GROUP   0
    add     x0, x0, #32
    add     x1, x1, #32
    subs    x2, x2, #4
    b.hs    Ltail_group
Lsingles:
    adds    x2, x2, #4
    b.eq    Ldone
Lsingle:
    ldr     x4, [x0], #8
    cmp     x4, x3
    csel    x3, x4, x3, gt
    str     x3, [x1], #8
    subs    x2, x2, #1
    b.ne    Lsingle
Ldone:
    ret
