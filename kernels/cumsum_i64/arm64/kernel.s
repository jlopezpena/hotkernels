// void cumsum_i64(const int64_t *x, int64_t *out, size_t n);
// out[i] = x[0] + ... + x[i], wrapping on overflow.
//
// Simplest correct version: one add per element on a single chain.
.text
.globl _cumsum_i64
.p2align 2
_cumsum_i64:
    cbz     x2, Ldone
    mov     x3, #0                  // running sum
Lloop:
    ldr     x4, [x0], #8
    add     x3, x3, x4
    str     x3, [x1], #8
    subs    x2, x2, #1
    b.ne    Lloop
Ldone:
    ret
