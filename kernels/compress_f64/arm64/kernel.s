// size_t compress_f64(const double *x, const bool *mask, double *out, size_t n);
// Copies x[i] to out wherever mask[i] is true. Returns how many were copied.
//
// Scalar baseline: store every element at out[k], advance k by the mask byte.
.text
.globl _compress_f64
.p2align 2
_compress_f64:
    mov     x4, #0                  // i
    mov     x5, #0                  // k, number kept
    cbz     x3, Ldone
Lloop:
    ldr     x7, [x0, x4, lsl #3]
    ldrb    w6, [x1, x4]
    str     x7, [x2, x5, lsl #3]    // inside the capacity because k <= i
    add     x4, x4, #1
    add     x5, x5, x6
    cmp     x4, x3
    b.lo    Lloop
Ldone:
    mov     x0, x5
    ret
