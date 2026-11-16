// BROKEN ON PURPOSE: skips elements before the first 16-byte boundary
// int64_t argmax_f32(const float *x, size_t n);   n >= 1
// Simple scalar version used to test the judge. Deliberately not tuned.
.text
.globl _argmax_f32
.p2align 2
_argmax_f32:
    neg     x4, x0
    and     x4, x4, #15
    lsr     x4, x4, #2              // elements before the next 16-byte boundary
    cmp     x4, x1
    b.hs    Lstart
    add     x0, x0, x4, lsl #2      // BUG: skip them
    sub     x1, x1, x4
    mov     x5, x4
    b       Lgo
Lstart:
    mov     x5, #0
Lgo:
    ldr     s0, [x0]                // best value so far
    mov     x2, #0                  // its index
    fcmp    s0, s0
    b.vs    Ldone                   // x[0] is NaN: the first NaN wins
    mov     x3, #1
Lloop:
    cmp     x3, x1
    b.hs    Ldone
    ldr     s1, [x0, x3, lsl #2]
    fcmp    s1, s1
    b.vs    Lnan
    fcmp    s1, s0
    b.le    Lnext                   // only a strictly greater value replaces the best
    fmov    s0, s1
    mov     x2, x3
Lnext:
    add     x3, x3, #1
    b       Lloop
Lnan:
    mov     x2, x3
Ldone:
    add     x0, x2, x5
    ret
