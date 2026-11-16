// size_t compress_f32(const float *x, const bool *mask, float *out, size_t n);
// Copies x[i] to out wherever mask[i] is true. Returns how many were copied.
//
// Eight elements per step. The eight mask bytes (each 0 or 1) are loaded as one
// 64-bit word. Multiplications turn that word into (a) an 8-bit mask, split
// into two nibbles that index a table of byte-shuffle patterns, and (b) the
// selected counts, already scaled to bytes. Each half is compacted with `tbl`
// and stored whole at the write pointer; lanes past the selected ones are
// rubbish that the next store overwrites (`out` is a scratch buffer).
//
// A full 16-byte store at the write pointer is inside the capacity because the
// write position never runs ahead of the read position.
.text
.globl _compress_f32
.p2align 2
_compress_f32:
    mov     x5, x2                      // x5 = write pointer
    subs    x4, x3, #8                  // x4 = elements left after this block
    b.lo    Ltail_setup
    adrp    x9, Lshuffle@PAGE
    add     x9, x9, Lshuffle@PAGEOFF
    mov     x10, #0x4080                // 0x0102040810204080: gathers the low
    movk    x10, #0x1020, lsl #16       // bit of each byte into the top byte
    movk    x10, #0x0408, lsl #32
    movk    x10, #0x0102, lsl #48
    mov     x11, #0x0404040404040404    // running sums of the bytes, times 4
Lloop8:
    ldr     x6, [x1], #8                // eight mask bytes
    ldp     q0, q1, [x0], #32
    mul     x7, x6, x10                 // 8-bit mask in the top byte
    mul     x8, x6, x11                 // byte j = 4 * (count in elements 0..j)
    ubfx    x13, x7, #56, #4            // low nibble of the mask
    lsr     x14, x7, #60                // high nibble
    ubfx    x12, x8, #24, #8            // 4 * (count in the low four)
    lsr     x8, x8, #56                 // 4 * (count in all eight)
    ldr     q2, [x9, x13, lsl #4]
    ldr     q3, [x9, x14, lsl #4]
    tbl     v0.16b, {v0.16b}, v2.16b
    tbl     v1.16b, {v1.16b}, v3.16b
    str     q0, [x5]
    str     q1, [x5, x12]
    add     x5, x5, x8
    subs    x4, x4, #8
    b.hs    Lloop8
Ltail_setup:
    adds    x4, x4, #8                  // 0 to 7 elements left
    b.eq    Ldone
Ltail:
    ldr     w7, [x0], #4
    ldrb    w6, [x1], #1
    str     w7, [x5]
    add     x5, x5, x6, lsl #2
    subs    x4, x4, #1
    b.ne    Ltail
Ldone:
    sub     x0, x5, x2
    lsr     x0, x0, #2
    ret

// Shuffle pattern for each 4-bit mask: the selected lanes first, in order.
.section __TEXT,__const
.p2align 4
Lshuffle:
    .byte  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15   // 0000
    .byte  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15   // 1000
    .byte  4,  5,  6,  7,  0,  1,  2,  3,  8,  9, 10, 11, 12, 13, 14, 15   // 0100
    .byte  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15   // 1100
    .byte  8,  9, 10, 11,  0,  1,  2,  3,  4,  5,  6,  7, 12, 13, 14, 15   // 0010
    .byte  0,  1,  2,  3,  8,  9, 10, 11,  4,  5,  6,  7, 12, 13, 14, 15   // 1010
    .byte  4,  5,  6,  7,  8,  9, 10, 11,  0,  1,  2,  3, 12, 13, 14, 15   // 0110
    .byte  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15   // 1110
    .byte 12, 13, 14, 15,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11   // 0001
    .byte  0,  1,  2,  3, 12, 13, 14, 15,  4,  5,  6,  7,  8,  9, 10, 11   // 1001
    .byte  4,  5,  6,  7, 12, 13, 14, 15,  0,  1,  2,  3,  8,  9, 10, 11   // 0101
    .byte  0,  1,  2,  3,  4,  5,  6,  7, 12, 13, 14, 15,  8,  9, 10, 11   // 1101
    .byte  8,  9, 10, 11, 12, 13, 14, 15,  0,  1,  2,  3,  4,  5,  6,  7   // 0011
    .byte  0,  1,  2,  3,  8,  9, 10, 11, 12, 13, 14, 15,  4,  5,  6,  7   // 1011
    .byte  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,  0,  1,  2,  3   // 0111
    .byte  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15   // 1111
