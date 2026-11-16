// void maskfill_f64(double *x, const bool *mask, size_t n, double value);
// x0 = x (in place), x1 = mask (bytes, 0 or 1), x2 = n, d0 = value.
//
// Scalar and branch-free: every element is loaded, replaced by value when its
// mask byte is set, and stored back. Unselected elements are rewritten with
// their own bits.
.text
.globl _maskfill_f64
.p2align 2
_maskfill_f64:
    cbz     x2, Ldone
    fmov    x3, d0                  // value bits
Lloop:
    ldrb    w4, [x1], #1
    ldr     x5, [x0]
    cmp     w4, #0
    csel    x5, x3, x5, ne
    str     x5, [x0], #8
    subs    x2, x2, #1
    b.ne    Lloop
Ldone:
    ret
