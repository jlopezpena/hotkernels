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
// out[0] = x[0] = M, the running maximum, in both lanes of v31. Then blocks
// of 16 elements, from x[1] on, so the element before each block exists:
//
//   1. each vector against the input one element back: lane j holds the
//      maximum of a run of two that ends at its own element;
//   2. Sklansky steps over the eight vectors (the odd ones against the one
//      before, 2 and 3 against 1, 6 and 7 against 5, 4 to 7 against 3): the
//      runs meet in each step, and after the last every lane holds the
//      maximum from the element before the block up to its own. That
//      element is already in M, so including it changes nothing;
//   3. every vector against M, and the next M is M against the block's last
//      lane: the chain from block to block is one fmaxnm.
//
// Fewer than 16 left: pairs, one vector at a time, then a single element.
//
// x2 counts the elements left minus 16 while blocks run.

.text
.globl _cumfmax_f64
.p2align 2
_cumfmax_f64:
    cbz     x2, Ldone
    ld1r    {v31.2d}, [x0], #8      // M
    str     d31, [x1], #8
    subs    x2, x2, #17
    b.lo    Ltail
Lblock:
    ldp     q0, q1, [x0]
    ldp     q2, q3, [x0, #32]
    ldp     q4, q5, [x0, #64]
    ldp     q6, q7, [x0, #96]
    ldur    q16, [x0, #-8]
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
    b.hs    Lblock

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
