// double fmax_reduce_f64(const double *x, size_t n);
//
// x0 = x, x1 = n (at least 1). Returns the largest element that is not a NaN,
// and a NaN only if every element is one: np.fmax.reduce(x).
//
// `fmaxnm` is the IEEE maxNum: a quiet NaN against a number gives the number.
// NumPy's own ARM64 loop is the chain m = fmaxnm(m, x[i]) from m = x[0].
//
// This kernel takes the elements in another order, and some of them twice.
// That gives NumPy's bits only within the spec's domain, which the caller
// guarantees:
//   - a zero maximum has no -0.0 beside it (fmaxnm gives +0.0 for +0.0 and
//     -0.0 in either order anyway);
//   - an all-NaN run has one bit pattern, so whichever NaN survives is it
//     (two quiet NaNs give the first operand; FPCR.DN is clear on macOS);
//   - no NaN is signalling (fmaxnm would turn one into a quiet NaN result).
//
// Paths, by n, none with a loop below 33 elements: the first elements and
// the last ones are loaded as vectors that overlap in the middle.
//   1         x[0]
//   2..4      the first two and the last two
//   5..8      the first four and the last four
//   9..16     the first eight and the last eight
//   17..32    the first sixteen and the last sixteen
//   33..      eight accumulators of two lanes, 16 elements per step; the
//             first step is the first 16 elements, the last step the last 16,
//             overlapping the step before it

.text
.globl _fmax_reduce_f64
.p2align 2
_fmax_reduce_f64:
    add     x2, x0, x1, lsl #3          // end of the array
    sub     x3, x1, #1
    cmp     x3, #32
    b.hs    Lbig
    tbnz    x3, #4, L17to32
    tbnz    x3, #3, L9to16
    tbnz    x3, #2, L5to8
    cbz     x3, Lone
    // 2..4
    ldr     q0, [x0]
    ldur    q1, [x2, #-16]
    fmaxnm  v0.2d, v0.2d, v1.2d
    fmaxnmp d0, v0.2d
    ret
Lone:
    ldr     d0, [x0]
    ret

L5to8:
    ldp     q0, q1, [x0]
    ldp     q2, q3, [x2, #-32]
    fmaxnm  v0.2d, v0.2d, v1.2d
    fmaxnm  v2.2d, v2.2d, v3.2d
    fmaxnm  v0.2d, v0.2d, v2.2d
    fmaxnmp d0, v0.2d
    ret

L9to16:
    ldp     q0, q1, [x0]
    ldp     q2, q3, [x0, #32]
    ldp     q4, q5, [x2, #-64]
    ldp     q6, q7, [x2, #-32]
    fmaxnm  v0.2d, v0.2d, v1.2d
    fmaxnm  v2.2d, v2.2d, v3.2d
    fmaxnm  v4.2d, v4.2d, v5.2d
    fmaxnm  v6.2d, v6.2d, v7.2d
    fmaxnm  v0.2d, v0.2d, v2.2d
    fmaxnm  v4.2d, v4.2d, v6.2d
    fmaxnm  v0.2d, v0.2d, v4.2d
    fmaxnmp d0, v0.2d
    ret

L17to32:
    ldp     q0, q1, [x0]
    ldp     q2, q3, [x0, #32]
    ldp     q4, q5, [x0, #64]
    ldp     q6, q7, [x0, #96]
    ldp     q16, q17, [x2, #-128]
    ldp     q18, q19, [x2, #-96]
    ldp     q20, q21, [x2, #-64]
    ldp     q22, q23, [x2, #-32]
    b       Lfold16

Lbig:
    // Eight accumulators v0..v7 start as the first 16 elements.
    ldp     q0, q1, [x0]
    ldp     q2, q3, [x0, #32]
    ldp     q4, q5, [x0, #64]
    ldp     q6, q7, [x0, #96]
    add     x0, x0, #128
    sub     x4, x2, #128                // start of the last step
    cmp     x0, x4
    b.hs    Llast
Lloop:
    ldp     q16, q17, [x0]
    ldp     q18, q19, [x0, #32]
    ldp     q20, q21, [x0, #64]
    ldp     q22, q23, [x0, #96]
    add     x0, x0, #128
    fmaxnm  v0.2d, v0.2d, v16.2d
    fmaxnm  v1.2d, v1.2d, v17.2d
    fmaxnm  v2.2d, v2.2d, v18.2d
    fmaxnm  v3.2d, v3.2d, v19.2d
    fmaxnm  v4.2d, v4.2d, v20.2d
    fmaxnm  v5.2d, v5.2d, v21.2d
    fmaxnm  v6.2d, v6.2d, v22.2d
    fmaxnm  v7.2d, v7.2d, v23.2d
    cmp     x0, x4
    b.lo    Lloop
Llast:
    ldp     q16, q17, [x4]
    ldp     q18, q19, [x4, #32]
    ldp     q20, q21, [x4, #64]
    ldp     q22, q23, [x4, #96]
Lfold16:
    // v0..v7 and v16..v23 to one value.
    fmaxnm  v0.2d, v0.2d, v16.2d
    fmaxnm  v1.2d, v1.2d, v17.2d
    fmaxnm  v2.2d, v2.2d, v18.2d
    fmaxnm  v3.2d, v3.2d, v19.2d
    fmaxnm  v4.2d, v4.2d, v20.2d
    fmaxnm  v5.2d, v5.2d, v21.2d
    fmaxnm  v6.2d, v6.2d, v22.2d
    fmaxnm  v7.2d, v7.2d, v23.2d
    fmaxnm  v0.2d, v0.2d, v1.2d
    fmaxnm  v2.2d, v2.2d, v3.2d
    fmaxnm  v4.2d, v4.2d, v5.2d
    fmaxnm  v6.2d, v6.2d, v7.2d
    fmaxnm  v0.2d, v0.2d, v2.2d
    fmaxnm  v4.2d, v4.2d, v6.2d
    fmaxnm  v0.2d, v0.2d, v4.2d
    fmaxnmp d0, v0.2d
    ret
