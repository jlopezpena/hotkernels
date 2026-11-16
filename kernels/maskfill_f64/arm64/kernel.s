// void maskfill_f64(double *x, const bool *mask, size_t n, double value);
// x0 = x (in place), x1 = mask (bytes, 0 or 1), x2 = n, d0 = value.
//
// Branch-free blend, 16 elements per iteration: 16 mask bytes are turned into
// 0x00/0xFF bytes, each pair of bytes is spread over a vector of two 64-bit
// lanes with tbl, and bsl selects between the broadcast value and the loaded
// data. Unselected elements are stored back with their own bits.
//
// The leftover (n mod 16) is handled by one more block that ends exactly at the
// end of the arrays and overlaps the previous one; applying the blend twice
// gives the same result. Fewer than 16 elements take a scalar loop.
.section __TEXT,__const
.p2align 4
Lidx:
    .byte 0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1
    .byte 2,2,2,2,2,2,2,2, 3,3,3,3,3,3,3,3
    .byte 4,4,4,4,4,4,4,4, 5,5,5,5,5,5,5,5
    .byte 6,6,6,6,6,6,6,6, 7,7,7,7,7,7,7,7
    .byte 8,8,8,8,8,8,8,8, 9,9,9,9,9,9,9,9
    .byte 10,10,10,10,10,10,10,10, 11,11,11,11,11,11,11,11
    .byte 12,12,12,12,12,12,12,12, 13,13,13,13,13,13,13,13
    .byte 14,14,14,14,14,14,14,14, 15,15,15,15,15,15,15,15

.text
.globl _maskfill_f64
.p2align 2
_maskfill_f64:
    cmp     x2, #16
    b.lo    Lsmall
    adrp    x8, Lidx@PAGE
    add     x8, x8, Lidx@PAGEOFF
    ldp     q16, q17, [x8]
    ldp     q18, q19, [x8, #32]
    ldp     q20, q21, [x8, #64]
    ldp     q22, q23, [x8, #96]
    dup     v0.2d, v0.d[0]          // value in both lanes
    add     x9, x0, x2, lsl #3      // end of x
    add     x10, x1, x2             // end of mask
    sub     x3, x2, #16             // elements left after the block below
Lloop:
    ldr     q1, [x1], #16
    ldp     q24, q25, [x0]
    ldp     q26, q27, [x0, #32]
    ldp     q28, q29, [x0, #64]
    ldp     q30, q31, [x0, #96]
    cmtst   v1.16b, v1.16b, v1.16b  // 0xFF where the mask byte is set
    tbl     v2.16b, {v1.16b}, v16.16b
    tbl     v3.16b, {v1.16b}, v17.16b
    tbl     v4.16b, {v1.16b}, v18.16b
    tbl     v5.16b, {v1.16b}, v19.16b
    tbl     v6.16b, {v1.16b}, v20.16b
    tbl     v7.16b, {v1.16b}, v21.16b
    bsl     v2.16b, v0.16b, v24.16b // mask ? value : x
    bsl     v3.16b, v0.16b, v25.16b
    bsl     v4.16b, v0.16b, v26.16b
    bsl     v5.16b, v0.16b, v27.16b
    bsl     v6.16b, v0.16b, v28.16b
    bsl     v7.16b, v0.16b, v29.16b
    tbl     v24.16b, {v1.16b}, v22.16b
    tbl     v25.16b, {v1.16b}, v23.16b
    bsl     v24.16b, v0.16b, v30.16b
    bsl     v25.16b, v0.16b, v31.16b
    stp     q2, q3, [x0]
    stp     q4, q5, [x0, #32]
    stp     q6, q7, [x0, #64]
    stp     q24, q25, [x0, #96]
    add     x0, x0, #128
    subs    x3, x3, #16
    b.hs    Lloop
    cmn     x3, #16                 // x3 = leftover - 16
    b.eq    Ldone
    sub     x0, x9, #128            // last 16 elements, overlapping
    sub     x1, x10, #16
    mov     x3, #0
    b       Lloop

Lsmall:
    cbz     x2, Ldone
    fmov    x3, d0                  // value bits
Lscalar:
    ldrb    w4, [x1], #1
    ldr     x5, [x0]
    cmp     w4, #0
    csel    x5, x3, x5, ne
    str     x5, [x0], #8
    subs    x2, x2, #1
    b.ne    Lscalar
Ldone:
    ret
