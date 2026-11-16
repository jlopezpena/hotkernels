// size_t nonzero_bool(const bool *mask, int64_t *out, size_t n);
// Writes the index of every true mask byte to out, ascending. Returns how many.
//
// Scalar baseline: store every index at out[k], advance k by the mask byte.
.text
.globl _nonzero_bool
.p2align 2
_nonzero_bool:
    mov     x4, #0                  // i
    mov     x5, #0                  // k, number of indices kept
    cbz     x2, Ldone
Lloop:
    ldrb    w6, [x0, x4]
    str     x4, [x1, x5, lsl #3]    // inside the capacity because k <= i
    add     x4, x4, #1
    add     x5, x5, x6
    cmp     x4, x2
    b.lo    Lloop
Ldone:
    mov     x0, x5
    ret
