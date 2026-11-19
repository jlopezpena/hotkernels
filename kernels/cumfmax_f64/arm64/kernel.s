// void cumfmax_f64(const double *x, double *out, size_t n);
// out[i] = the largest of x[0..i] that is not NaN: np.fmax.accumulate.
// x0 = x, x1 = out, x2 = n. out does not overlap x.
//
// NumPy's ARM64 loop is one fmaxnm per element from x[0]. fmaxnm ignores a
// quiet NaN operand, and of two quiet NaNs returns one of them; on the
// spec's domain (no -0.0, one NaN bit pattern, no signalling NaN) it is
// therefore associative and commutative to the bit, so any order of maxima
// over x[0..i] gives NumPy's out[i], the all-NaN prefix included (x[0]'s
// NaN until the first number).
//
// M, the running maximum, is in both lanes of v31, x[0] at first. Blocks of
// 16 elements from x[0], so that the blocks keep the alignment of the
// buffers; the first block uses x[0] where the others use the element
// before them. A block goes one of three ways:
//
//   flat      the block's maximum, a tree of seven fmaxnm and a pairwise
//             one, does not change M: the block is M sixteen times.
//   rising    the element before the block is M and no element of the block
//             is below the one before it: the block is a copy.
//   general   a prefix maximum (GENERAL).
//
// The blocks start general and stay general while each changes M. After a
// general block that does not change M, they are tried flat; after one that
// ends on M, rising. A block that fails the flat test is done the general
// way with the next tried flat again; one that fails the rising test is
// done the general way.
//
// Fewer than 16 left: pairs, one vector at a time against the element
// before, then a single element. Fewer than 16 in all: x[0], then the same
// from x[1].
//
// x2 counts the elements left minus 16 while blocks run.

// The block at x0 into v0 to v7.
.macro LOAD
    ldp     q0, q1, [x0]
    ldp     q2, q3, [x0, #32]
    ldp     q4, q5, [x0, #64]
    ldp     q6, q7, [x0, #96]
.endm

// The block in v0 to v7, the general way; stores it and moves on to the next.
//   1. each vector against the input one element back: lane j holds the
//      maximum of a run of two that ends at its own element;
//   2. Sklansky steps over the eight vectors (the odd ones against the one
//      before, 2 and 3 against 1, 6 and 7 against 5, 4 to 7 against 3): the
//      runs meet in each step, and after the last every lane holds the
//      maximum from the element before the block up to its own. That
//      element is already in M, so including it changes nothing;
//   3. every vector against M, and the next M is M against the block's last
//      lane: the chain from block to block is one fmaxnm.
// Leaves the flags of the count: hs if another block follows.
// With first=1 the block is x[0..15]: there is no element before it, and
// x[0] (M, in v31) stands in for it.
.macro GENERAL first=0
.if \first
    mov     v16.16b, v31.16b
.else
    ldur    q16, [x0, #-8]
.endif
    ldur    q17, [x0, #8]
    ldur    q18, [x0, #24]
    ldur    q19, [x0, #40]
    ldur    q20, [x0, #56]
    ldur    q21, [x0, #72]
    ldur    q22, [x0, #88]
    ldur    q23, [x0, #104]
    fmaxnm  v0.2d, v0.2d, v16.2d
    fmaxnm  v1.2d, v1.2d, v17.2d
    fmaxnm  v2.2d, v2.2d, v18.2d
    fmaxnm  v3.2d, v3.2d, v19.2d
    fmaxnm  v4.2d, v4.2d, v20.2d
    fmaxnm  v5.2d, v5.2d, v21.2d
    fmaxnm  v6.2d, v6.2d, v22.2d
    fmaxnm  v7.2d, v7.2d, v23.2d
    fmaxnm  v1.2d, v1.2d, v0.2d
    fmaxnm  v3.2d, v3.2d, v2.2d
    fmaxnm  v5.2d, v5.2d, v4.2d
    fmaxnm  v7.2d, v7.2d, v6.2d
    fmaxnm  v2.2d, v2.2d, v1.2d
    fmaxnm  v3.2d, v3.2d, v1.2d
    fmaxnm  v6.2d, v6.2d, v5.2d
    fmaxnm  v7.2d, v7.2d, v5.2d
    fmaxnm  v4.2d, v4.2d, v3.2d
    fmaxnm  v5.2d, v5.2d, v3.2d
    fmaxnm  v6.2d, v6.2d, v3.2d
    fmaxnm  v7.2d, v7.2d, v3.2d
    dup     v30.2d, v7.d[1]         // the block's maximum
    fmaxnm  v0.2d, v0.2d, v31.2d
    fmaxnm  v1.2d, v1.2d, v31.2d
    fmaxnm  v2.2d, v2.2d, v31.2d
    fmaxnm  v3.2d, v3.2d, v31.2d
    fmaxnm  v4.2d, v4.2d, v31.2d
    fmaxnm  v5.2d, v5.2d, v31.2d
    fmaxnm  v6.2d, v6.2d, v31.2d
    fmaxnm  v7.2d, v7.2d, v31.2d
    fmaxnm  v31.2d, v31.2d, v30.2d  // the next M
    stp     q0, q1, [x1]
    stp     q2, q3, [x1, #32]
    stp     q4, q5, [x1, #64]
    stp     q6, q7, [x1, #96]
    add     x0, x0, #128
    add     x1, x1, #128
    subs    x2, x2, #16
.endm

.text
.globl _cumfmax_f64
.p2align 2
_cumfmax_f64:
    cbz     x2, Ldone
    ld1r    {v31.2d}, [x0]          // M = x[0]
    subs    x2, x2, #16
    b.lo    Lshort
    LOAD
    mov     v29.16b, v31.16b
    GENERAL 1
    b       Lgeneral_next
Lshort:
    // Fewer than 16 elements: x[0], then the tail from x[1].
    str     d31, [x1], #8
    add     x0, x0, #8
    sub     x2, x2, #1
    b       Ltail

    // ---- General, while each block changes M. v29 = M before the block.
Lgeneral:
    LOAD
Lgeneral_loaded:
    mov     v29.16b, v31.16b
    GENERAL
Lgeneral_next:
    b.lo    Ltail
    fcmp    d29, d31                // unordered, and so general, while M is NaN
    b.eq    Lflat
    ldur    d28, [x0, #-8]
    fcmp    d28, d31                // the block ends on M
    b.ne    Lgeneral

    // ---- Rising: the element before the block is M, and no element of the
    //      block is below the one before it (a NaN fails the test): the
    //      block is a copy. v31 is not kept up to date here.
Lrising:
    LOAD
    ldur    q16, [x0, #-8]
    ext     v17.16b, v0.16b, v1.16b, #8
    ldur    q18, [x0, #24]
    ext     v19.16b, v2.16b, v3.16b, #8
    ldur    q20, [x0, #56]
    ext     v21.16b, v4.16b, v5.16b, #8
    ldur    q22, [x0, #88]
    ext     v23.16b, v6.16b, v7.16b, #8
    fcmge   v16.2d, v0.2d, v16.2d
    fcmge   v17.2d, v1.2d, v17.2d
    fcmge   v18.2d, v2.2d, v18.2d
    fcmge   v19.2d, v3.2d, v19.2d
    fcmge   v20.2d, v4.2d, v20.2d
    fcmge   v21.2d, v5.2d, v21.2d
    fcmge   v22.2d, v6.2d, v22.2d
    fcmge   v23.2d, v7.2d, v23.2d
    and     v16.16b, v16.16b, v17.16b
    and     v18.16b, v18.16b, v19.16b
    and     v20.16b, v20.16b, v21.16b
    and     v22.16b, v22.16b, v23.16b
    and     v16.16b, v16.16b, v18.16b
    and     v20.16b, v20.16b, v22.16b
    and     v16.16b, v16.16b, v20.16b
    uminv   s16, v16.4s
    fmov    w9, s16
    cbz     w9, Lrising_fail
    stp     q0, q1, [x1]
    stp     q2, q3, [x1, #32]
    stp     q4, q5, [x1, #64]
    stp     q6, q7, [x1, #96]
    add     x0, x0, #128
    add     x1, x1, #128
    subs    x2, x2, #16
    b.hs    Lrising
    sub     x9, x0, #8
    ld1r    {v31.2d}, [x9]
    b       Ltail
Lrising_fail:
    sub     x9, x0, #8
    ld1r    {v31.2d}, [x9]
    b       Lgeneral_loaded

    // ---- Flat: the block's maximum against M.
Lflat:
    LOAD
    fmaxnm  v24.2d, v0.2d, v1.2d
    fmaxnm  v25.2d, v2.2d, v3.2d
    fmaxnm  v26.2d, v4.2d, v5.2d
    fmaxnm  v27.2d, v6.2d, v7.2d
    fmaxnm  v24.2d, v24.2d, v25.2d
    fmaxnm  v26.2d, v26.2d, v27.2d
    fmaxnm  v24.2d, v24.2d, v26.2d
    fmaxnmp d24, v24.2d
    fmaxnm  d24, d24, d31
    fcmp    d24, d31
    b.ne    Lflat_fail
    stp     q31, q31, [x1]
    stp     q31, q31, [x1, #32]
    stp     q31, q31, [x1, #64]
    stp     q31, q31, [x1, #96]
    add     x0, x0, #128
    add     x1, x1, #128
    subs    x2, x2, #16
    b.hs    Lflat
    b       Ltail
Lflat_fail:
    GENERAL
    b.hs    Lflat

    // ---- Fewer than 16 elements left: pairs, then one.
Ltail:
    adds    x2, x2, #16
    b.eq    Ldone
    subs    x2, x2, #2
    b.lo    Lsingle
Lpair:
    ldr     q0, [x0], #16
    ldur    q16, [x0, #-24]
    fmaxnm  v0.2d, v0.2d, v16.2d
    fmaxnm  v0.2d, v0.2d, v31.2d
    str     q0, [x1], #16
    dup     v31.2d, v0.d[1]
    subs    x2, x2, #2
    b.hs    Lpair
    tbz     x2, #0, Ldone           // x2 is -2 or -1 here
Lsingle:
    ldr     d0, [x0]
    fmaxnm  d0, d0, d31
    str     d0, [x1]
Ldone:
    ret
