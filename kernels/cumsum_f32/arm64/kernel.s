// void cumsum_f32(const float *x, float *out, size_t n);
// out[i] = x[0] + ... + x[i], added strictly left to right.
//
// Straightforward scalar version: one fadd per element on a single chain. The
// first element is copied, not added to zero, so a leading -0.0 stays -0.0.
.text
.globl _cumsum_f32
.p2align 2
_cumsum_f32:
    cbz     x2, Ldone
    ldr     s0, [x0], #4            // running sum = x[0]
    str     s0, [x1], #4
    subs    x2, x2, #1
    b.eq    Ldone
Lloop:
    ldr     s1, [x0], #4
    fadd    s0, s0, s1              // sum + x[i], same operand order as the C
    str     s0, [x1], #4
    subs    x2, x2, #1
    b.ne    Lloop
Ldone:
    ret
