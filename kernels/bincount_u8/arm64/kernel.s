// void bincount_u8(const uint8_t *x, int64_t *counts, size_t n);
// counts has exactly 256 slots; it arrives uninitialised.
//
// x0 = x, x1 = counts, x2 = n
.text
.globl _bincount_u8
.p2align 2
_bincount_u8:
    // Zero the 256 counters (2048 bytes), 64 bytes per step.
    movi    v0.2d, #0
    mov     x3, x1
    add     x4, x1, #2048
Lzero:
    stp     q0, q0, [x3]
    stp     q0, q0, [x3, #32]
    add     x3, x3, #64
    cmp     x3, x4
    b.lo    Lzero

    cbz     x2, Ldone
    add     x5, x0, x2              // end of x
Lloop:
    ldrb    w6, [x0], #1
    ldr     x7, [x1, x6, lsl #3]
    add     x7, x7, #1
    str     x7, [x1, x6, lsl #3]
    cmp     x0, x5
    b.lo    Lloop
Ldone:
    ret
