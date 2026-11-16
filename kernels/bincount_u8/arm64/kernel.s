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

    subs    x2, x2, #8
    b.lo    Ltail
    // Eight bytes per step: one 64-bit load, then each byte is extracted
    // with a bit-field move.
Lloop8:
    ldr     x6, [x0], #8
    and     x7, x6, #0xff
    ubfx    x8, x6, #8, #8
    ubfx    x9, x6, #16, #8
    ubfx    x10, x6, #24, #8
    ldr     x15, [x1, x7, lsl #3]
    add     x15, x15, #1
    str     x15, [x1, x7, lsl #3]
    ldr     x16, [x1, x8, lsl #3]
    add     x16, x16, #1
    str     x16, [x1, x8, lsl #3]
    ldr     x17, [x1, x9, lsl #3]
    add     x17, x17, #1
    str     x17, [x1, x9, lsl #3]
    ldr     x3, [x1, x10, lsl #3]
    add     x3, x3, #1
    str     x3, [x1, x10, lsl #3]
    ubfx    x11, x6, #32, #8
    ubfx    x12, x6, #40, #8
    ubfx    x13, x6, #48, #8
    lsr     x14, x6, #56
    ldr     x15, [x1, x11, lsl #3]
    add     x15, x15, #1
    str     x15, [x1, x11, lsl #3]
    ldr     x16, [x1, x12, lsl #3]
    add     x16, x16, #1
    str     x16, [x1, x12, lsl #3]
    ldr     x17, [x1, x13, lsl #3]
    add     x17, x17, #1
    str     x17, [x1, x13, lsl #3]
    ldr     x3, [x1, x14, lsl #3]
    add     x3, x3, #1
    str     x3, [x1, x14, lsl #3]
    subs    x2, x2, #8
    b.hs    Lloop8
Ltail:
    adds    x2, x2, #8              // 0..7 bytes left
    b.eq    Ldone
Lloop1:
    ldrb    w6, [x0], #1
    ldr     x7, [x1, x6, lsl #3]
    add     x7, x7, #1
    str     x7, [x1, x6, lsl #3]
    subs    x2, x2, #1
    b.ne    Lloop1
Ldone:
    ret
