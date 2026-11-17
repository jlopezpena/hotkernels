// void cast_u8_f32(const uint8_t *x, float *out, size_t n);
// out[i] = (float)x[i]
//
// x0 = x, x1 = out, x2 = n.
//
// Sixteen bytes become four vectors of four floats: tbl spreads four bytes
// into the low byte of four zeroed 32-bit lanes (index 255 selects zero),
// ucvtf converts the lanes. Every value from 0 to 255 is exact in a float,
// so no rounding mode or other FPCR setting can change the result.
//
// A tail that is not a whole 16-byte block is one more block that ends
// exactly at the end of the array and overlaps the previous one. 4 to 15
// elements are a front piece and an overlapping back piece.

// tbl indices: row k spreads bytes 4k to 4k+3.
.section __TEXT,__literal16,16byte_literals
.p2align 4
Lindex:
    .byte 0, 255, 255, 255, 1, 255, 255, 255, 2, 255, 255, 255, 3, 255, 255, 255
    .byte 4, 255, 255, 255, 5, 255, 255, 255, 6, 255, 255, 255, 7, 255, 255, 255
    .byte 8, 255, 255, 255, 9, 255, 255, 255, 10, 255, 255, 255, 11, 255, 255, 255
    .byte 12, 255, 255, 255, 13, 255, 255, 255, 14, 255, 255, 255, 15, 255, 255, 255

.text
.globl _cast_u8_f32
.p2align 2
_cast_u8_f32:
    adrp    x9, Lindex@PAGE
    add     x9, x9, Lindex@PAGEOFF
    ldp     q0, q1, [x9]            // indices for bytes 0-3 and 4-7
    subs    x8, x2, #16             // x8 = elements left after this block
    b.lo    Lsmall
    ldp     q2, q3, [x9, #32]       // indices for bytes 8-11 and 12-15
Lblock:
    ldr     q4, [x0], #16
    tbl.16b v5, { v4 }, v0
    tbl.16b v6, { v4 }, v1
    tbl.16b v7, { v4 }, v2
    tbl.16b v4, { v4 }, v3
    ucvtf.4s v5, v5
    ucvtf.4s v6, v6
    ucvtf.4s v7, v7
    ucvtf.4s v4, v4
    stp     q5, q6, [x1]
    stp     q7, q4, [x1, #32]
    add     x1, x1, #64
    subs    x8, x8, #16
    b.hs    Lblock
    adds    x8, x8, #16             // elements left: 0 to 15
    b.eq    Ldone
    // Step back so that one more block ends exactly at the end.
    add     x0, x0, x8
    add     x1, x1, x8, lsl #2
    sub     x0, x0, #16
    sub     x1, x1, #64
    mov     x8, #0
    b       Lblock

Lsmall:                             // n < 16
    cmp     x2, #4
    b.lo    Lscalar
    add     x9, x0, x2              // end of x
    add     x10, x1, x2, lsl #2     // end of out
    tbz     x2, #3, Lfour
    // 8 <= n < 16: the first eight bytes and the last eight, overlapping.
    ldr     d4, [x0]
    ldur    d5, [x9, #-8]
    tbl.16b v6, { v4 }, v0
    tbl.16b v7, { v4 }, v1
    tbl.16b v16, { v5 }, v0
    tbl.16b v17, { v5 }, v1
    ucvtf.4s v6, v6
    ucvtf.4s v7, v7
    ucvtf.4s v16, v16
    ucvtf.4s v17, v17
    stp     q6, q7, [x1]
    stp     q16, q17, [x10, #-32]
    ret
Lfour:
    // 4 <= n < 8: the first four bytes and the last four, overlapping.
    ldr     s4, [x0]
    ldur    s5, [x9, #-4]
    tbl.16b v4, { v4 }, v0
    tbl.16b v5, { v5 }, v0
    ucvtf.4s v4, v4
    ucvtf.4s v5, v5
    str     q4, [x1]
    stur    q5, [x10, #-16]
Ldone:
    ret

Lscalar:                            // n < 4
    cbz     x2, Ldone
Lone:
    ldrb    w9, [x0], #1
    ucvtf   s4, w9
    str     s4, [x1], #4
    subs    x2, x2, #1
    b.ne    Lone
    ret
