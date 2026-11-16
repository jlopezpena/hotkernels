// void cumsum_f64(const double *x, double *out, size_t n);
// out[i] = x[0] + ... + x[i], added strictly left to right.
//
// Straightforward scalar version: one fadd per element on a single chain. The
// first element is copied, not added to zero, so a leading -0.0 stays -0.0.
.text
.globl _cumsum_f64
.p2align 2
_cumsum_f64:
    cbz     x2, Ldone
    ldr     d0, [x0], #8            // running sum = x[0]
    str     d0, [x1], #8
    subs    x2, x2, #1
    b.eq    Ldone
Lloop:
    ldr     d1, [x0], #8
    fadd    d0, d0, d1              // sum + x[i], same operand order as the C
    str     d0, [x1], #8
    subs    x2, x2, #1
    b.ne    Lloop
Ldone:
    ret
