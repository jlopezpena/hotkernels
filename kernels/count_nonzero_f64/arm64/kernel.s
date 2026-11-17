// size_t count_nonzero_f64(const double *x, size_t n);
//
// x0 = x, x1 = n. Returns how many of the n elements at x are not zero.
// NaN counts as non-zero; +0.0 and -0.0 are both zero.
//
// A double is zero exactly when its low 63 bits are zero (the sign bit is the
// only difference between +0.0 and -0.0), and every NaN has some of those
// bits set. So the test is done on the bits: `cmtst` against a mask of the
// low 63 bits gives -1 per non-zero lane, and subtracting that from a 64-bit
// accumulator counts it. An odd last element is tested in a general register.

.text
.globl _count_nonzero_f64
.p2align 2
_count_nonzero_f64:
    movi    v16.2d, #0              // two 64-bit counters
    movi    v31.2d, #0xffffffffffffffff
    ushr    v31.2d, v31.2d, #1      // 0x7fffffffffffffff in both lanes
    subs    x2, x1, #2
    b.lo    Lodd
Lloop:
    ldr     q0, [x0], #16
    cmtst   v0.2d, v0.2d, v31.2d
    sub     v16.2d, v16.2d, v0.2d
    subs    x2, x2, #2
    b.hs    Lloop
Lodd:
    addp    d0, v16.2d
    fmov    x3, d0
    tbz     x1, #0, Ldone
    ldr     x4, [x0]
    tst     x4, #0x7fffffffffffffff
    cinc    x3, x3, ne
Ldone:
    mov     x0, x3
    ret
