// void cumsum_bool(const bool *mask, int64_t *out, size_t n);
// out[i] = how many of mask[0..i] are true. Every mask byte is 0 or 1.
//
// x0 = mask, x1 = out, x2 = n.
//
// Sixteen elements are one step. Their running count within the block, P, is
// formed as sixteen bytes without the count before the block:
//   - a multiply of each 32-bit lane by 0x01010101 puts in byte k of a lane
//     the sum of its bytes 0..k (no byte passes 4, so nothing carries);
//   - a `tbl` copies the top byte of lanes 0 and 2 to every byte of lanes 1
//     and 3, and an add makes the counts run over eight bytes;
//   - a `tbl` copies byte 7 to bytes 8..15, and an add makes them run over 16.
// Four `tbl` then put four bytes of P each in the low bytes of four 32-bit
// lanes, and `uaddw`/`uaddw2` widen those to 64 bits while adding T, the count
// before the block in both lanes: the eight output vectors. The next T is T
// plus byte 15 of P, one add per sixteen elements on the carried chain.
.section __TEXT,__const
.p2align 4
Lconst:
    // Group totals to the next group: byte 3 to bytes 4..7, byte 11 to 12..15.
    .byte 0xff, 0xff, 0xff, 0xff, 3, 3, 3, 3, 0xff, 0xff, 0xff, 0xff, 11, 11, 11, 11
    // The low half's total to the high half.
    .byte 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 7, 7, 7, 7, 7, 7, 7, 7
    // Four bytes to the low bytes of four 32-bit lanes.
    .byte 0, 0xff, 0xff, 0xff, 1, 0xff, 0xff, 0xff, 2, 0xff, 0xff, 0xff, 3, 0xff, 0xff, 0xff
    .byte 4, 0xff, 0xff, 0xff, 5, 0xff, 0xff, 0xff, 6, 0xff, 0xff, 0xff, 7, 0xff, 0xff, 0xff
    .byte 8, 0xff, 0xff, 0xff, 9, 0xff, 0xff, 0xff, 10, 0xff, 0xff, 0xff, 11, 0xff, 0xff, 0xff
    .byte 12, 0xff, 0xff, 0xff, 13, 0xff, 0xff, 0xff, 14, 0xff, 0xff, 0xff, 15, 0xff, 0xff, 0xff
    // The block's total to the low byte of both 64-bit lanes.
    .byte 15, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 15, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff

.text
.globl _cumsum_bool
.p2align 2
_cumsum_bool:
    cmp     x2, #16
    b.lo    Lsmall
    adrp    x9, Lconst@PAGE
    add     x9, x9, Lconst@PAGEOFF
    ldp     q24, q25, [x9]
    ldp     q26, q27, [x9, #32]
    ldp     q28, q29, [x9, #64]
    ldr     q30, [x9, #96]
    movi    v31.16b, #1                 // 0x01010101 in every 32-bit lane
    movi    v16.2d, #0                  // T: the count so far, in both lanes
    sub     x2, x2, #16
Lblock:
    ldr     q0, [x0], #16
    mul     v0.4s, v0.4s, v31.4s        // counts within each group of four
    tbl     v1.16b, {v0.16b}, v24.16b
    add     v0.16b, v0.16b, v1.16b      // within each half
    tbl     v1.16b, {v0.16b}, v25.16b
    add     v0.16b, v0.16b, v1.16b      // P: within the block
    tbl     v1.16b, {v0.16b}, v26.16b
    tbl     v2.16b, {v0.16b}, v27.16b
    tbl     v3.16b, {v0.16b}, v28.16b
    tbl     v4.16b, {v0.16b}, v29.16b
    tbl     v5.16b, {v0.16b}, v30.16b   // the block's total, both lanes
    uaddw   v6.2d, v16.2d, v1.2s
    uaddw2  v7.2d, v16.2d, v1.4s
    uaddw   v17.2d, v16.2d, v2.2s
    uaddw2  v18.2d, v16.2d, v2.4s
    uaddw   v19.2d, v16.2d, v3.2s
    uaddw2  v20.2d, v16.2d, v3.4s
    uaddw   v21.2d, v16.2d, v4.2s
    uaddw2  v22.2d, v16.2d, v4.4s
    add     v16.2d, v16.2d, v5.2d       // the next T
    stp     q6, q7, [x1]
    stp     q17, q18, [x1, #32]
    stp     q19, q20, [x1, #64]
    stp     q21, q22, [x1, #96]
    add     x1, x1, #128
    subs    x2, x2, #16
    b.hs    Lblock
    adds    x2, x2, #16                 // 0 to 15 elements left
    b.eq    Ldone
    fmov    x3, d16                     // the count so far
    b       Lone
Lsmall:
    cbz     x2, Ldone
    mov     x3, #0
Lone:
    ldrb    w4, [x0], #1
    add     x3, x3, x4
    str     x3, [x1], #8
    subs    x2, x2, #1
    b.ne    Lone
Ldone:
    ret
