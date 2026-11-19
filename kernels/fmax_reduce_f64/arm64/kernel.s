// double fmax_reduce_f64(const double *x, size_t n);
//
// x0 = x, x1 = n (at least 1). Returns the largest element that is not a NaN,
// and a NaN only if every element is one: np.fmax.reduce(x).
//
// `fmaxnm` is the IEEE maxNum: a quiet NaN against a number gives the number.
// A chain m = fmaxnm(m, x[i]) from m = x[0] is NumPy's own loop on ARM64.

.text
.globl _fmax_reduce_f64
.p2align 2
_fmax_reduce_f64:
    ldr     d0, [x0], #8
    subs    x1, x1, #1
    b.eq    Ldone
Lloop:
    ldr     d1, [x0], #8
    fmaxnm  d0, d0, d1
    subs    x1, x1, #1
    b.ne    Lloop
Ldone:
    ret
