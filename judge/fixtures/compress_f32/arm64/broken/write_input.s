// BROKEN ON PURPOSE: writes to the mask, which is an input
// size_t compress_f32(const float *x, const bool *mask, float *out, size_t n);
// Simple scalar version used to test the judge. Deliberately not tuned.
.text
.globl _compress_f32
.p2align 2
_compress_f32:
    mov     x4, #0                  // i
    mov     x5, #0                  // number written
Lloop:
    cmp     x4, x3
    b.hs    Ldone
    ldrb    w6, [x1, x4]
    strb    w6, [x1, x4]            // BUG: write to an input
    cbz     w6, Lnext
    ldr     w7, [x0, x4, lsl #2]
    str     w7, [x2, x5, lsl #2]
    add     x5, x5, #1
Lnext:
    add     x4, x4, #1
    b       Lloop
Ldone:
    mov     x0, x5
    ret
