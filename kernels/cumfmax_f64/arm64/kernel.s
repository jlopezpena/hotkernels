// void cumfmax_f64(const double *x, double *out, size_t n);
// out[i] = the largest of x[0..i] that is not NaN: np.fmax.accumulate.
// x0 = x, x1 = out, x2 = n. out does not overlap x.
//
// One fmaxnm per element from x[0], NumPy's own ARM64 loop: a quiet NaN
// operand is ignored, and while every element so far is NaN the running
// value is that NaN.

.text
.globl _cumfmax_f64
.p2align 2
_cumfmax_f64:
    cbz     x2, Ldone
    ldr     d0, [x0], #8            // M
    str     d0, [x1], #8
    subs    x2, x2, #1
    b.eq    Ldone
Lloop:
    ldr     d1, [x0], #8
    fmaxnm  d0, d0, d1
    str     d0, [x1], #8
    subs    x2, x2, #1
    b.ne    Lloop
Ldone:
    ret
