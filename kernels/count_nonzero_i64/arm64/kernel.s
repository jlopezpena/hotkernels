// size_t count_nonzero_i64(const int64_t *x, size_t n);
//
// x0 = x, x1 = n. Returns how many of the n elements at x are not zero.
//
// Two elements per step: `cmtst` sets a lane to all ones (-1) where the
// element is non-zero, and subtracting that from a 64-bit accumulator counts
// it. An odd last element is tested in a general register.

.text
.globl _count_nonzero_i64
.p2align 2
_count_nonzero_i64:
    movi    v16.2d, #0              // two 64-bit counters
    subs    x2, x1, #2
    b.lo    Lodd
Lloop:
    ldr     q0, [x0], #16
    cmtst   v0.2d, v0.2d, v0.2d
    sub     v16.2d, v16.2d, v0.2d
    subs    x2, x2, #2
    b.hs    Lloop
Lodd:
    addp    d0, v16.2d
    fmov    x3, d0
    tbz     x1, #0, Ldone
    ldr     x4, [x0]
    cmp     x4, #0
    cinc    x3, x3, ne
Ldone:
    mov     x0, x3
    ret
