// int64_t argmax_f64(const double *x, size_t n);   n >= 1
// Scalar baseline: one element per step.
.text
.globl _argmax_f64
.p2align 2
_argmax_f64:
    ldr     d0, [x0]                // best value so far
    mov     x2, #0                  // its index
    fcmp    d0, d0
    b.vs    Ldone                   // x[0] is NaN: the first NaN wins
    mov     x3, #1
Lloop:
    cmp     x3, x1
    b.hs    Ldone
    ldr     d1, [x0, x3, lsl #3]
    fcmp    d1, d1
    b.vs    Lnan
    fcmp    d1, d0
    b.le    Lnext                   // only a strictly greater value replaces the best
    fmov    d0, d1
    mov     x2, x3
Lnext:
    add     x3, x3, #1
    b       Lloop
Lnan:
    mov     x2, x3
Ldone:
    mov     x0, x2
    ret
