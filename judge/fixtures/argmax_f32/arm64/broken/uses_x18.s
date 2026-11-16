// BROKEN ON PURPOSE: uses the reserved register x18
// int64_t argmax_f32(const float *x, size_t n);   n >= 1
// Simple scalar version used to test the judge. Deliberately not tuned.
.text
.globl _argmax_f32
.p2align 2
_argmax_f32:
    ldr     s0, [x0]                // best value so far
    mov     x2, #0                  // its index
    fcmp    s0, s0
    b.vs    Ldone                   // x[0] is NaN: the first NaN wins
    mov     x18, #1
Lloop:
    cmp     x18, x1
    b.hs    Ldone
    ldr     s1, [x0, x18, lsl #2]
    fcmp    s1, s1
    b.vs    Lnan
    fcmp    s1, s0
    b.le    Lnext                   // only a strictly greater value replaces the best
    fmov    s0, s1
    mov     x2, x18
Lnext:
    add     x18, x18, #1
    b       Lloop
Lnan:
    mov     x2, x18
Ldone:
    mov     x0, x2
    ret
