// BROKEN ON PURPOSE: writes every element, then advances only when the mask is set.
// The result is right, but one element past the used output gets overwritten.
// size_t compress_f32(const float *x, const bool *mask, float *out, size_t n);
.text
.globl _compress_f32
.p2align 2
_compress_f32:
    mov     x4, #0                  // i
    mov     x5, #0                  // number written
Lloop:
    cmp     x4, x3
    b.hs    Ldone
    ldr     w7, [x0, x4, lsl #2]
    str     w7, [x2, x5, lsl #2]    // BUG: unconditional write
    ldrb    w6, [x1, x4]
    add     x5, x5, x6
    add     x4, x4, #1
    b       Lloop
Ldone:
    mov     x0, x5
    ret
