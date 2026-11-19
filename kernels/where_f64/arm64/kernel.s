// void where_f64(const bool *c, const double *a, const double *b, double *out, size_t n);
// x0 = c, x1 = a, x2 = b, x3 = out, x4 = n. out[i] = c[i] ? a[i] : b[i].
//
// The elements are moved and never computed with: only loads, stores, tbl
// and bsl touch the data of a and b, so a NaN keeps its payload, a zero its
// sign, and 64-bit integers sent through as bit patterns arrive unchanged.
//
// A byte of c is true when it is not zero, whatever its value.
//
// Sixteen elements are one block. Their sixteen bytes of c are loaded once
// and turned into 0xff (true) or 0x00 (false) by cmtst; eight tbl then spread
// each pair of those bytes over the two 64-bit lanes of a vector, and bsl
// takes each lane from a or from b.
//
// The elements left after the whole blocks are done by one more block that
// ends exactly at the end of the four buffers. It writes again some elements
// the block before it wrote, with the same values: out never overlaps an
// input. Under sixteen elements a scalar loop with csel does the work.

.section __TEXT,__const
.p2align 4
// tbl indices: vector k of a block takes mask bytes 2k and 2k+1, eight times each.
Lidx:
    .byte 0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1
    .byte 2,2,2,2,2,2,2,2, 3,3,3,3,3,3,3,3
    .byte 4,4,4,4,4,4,4,4, 5,5,5,5,5,5,5,5
    .byte 6,6,6,6,6,6,6,6, 7,7,7,7,7,7,7,7
    .byte 8,8,8,8,8,8,8,8, 9,9,9,9,9,9,9,9
    .byte 10,10,10,10,10,10,10,10, 11,11,11,11,11,11,11,11
    .byte 12,12,12,12,12,12,12,12, 13,13,13,13,13,13,13,13
    .byte 14,14,14,14,14,14,14,14, 15,15,15,15,15,15,15,15

// Elements per block.
.set BLOCK_N, 16

// One block. \c, \a, \b, \o = registers that point at it in c, a, b and out.
// v16 to v23 hold the tbl indices.
.macro BLOCK c, a, b, o
    ldr     q0, [\c]
    ldp     q24, q25, [\a]
    ldp     q28, q29, [\b]
    ldp     q26, q27, [\a, #32]
    ldp     q30, q31, [\b, #32]
    cmtst   v0.16b, v0.16b, v0.16b      // 0xff where c is true
    tbl     v1.16b, {v0.16b}, v16.16b
    tbl     v2.16b, {v0.16b}, v17.16b
    tbl     v3.16b, {v0.16b}, v18.16b
    tbl     v4.16b, {v0.16b}, v19.16b
    bsl     v1.16b, v24.16b, v28.16b    // a where the lane mask is set, else b
    bsl     v2.16b, v25.16b, v29.16b
    bsl     v3.16b, v26.16b, v30.16b
    bsl     v4.16b, v27.16b, v31.16b
    stp     q1, q2, [\o]
    stp     q3, q4, [\o, #32]
    ldp     q24, q25, [\a, #64]
    ldp     q28, q29, [\b, #64]
    ldp     q26, q27, [\a, #96]
    ldp     q30, q31, [\b, #96]
    tbl     v1.16b, {v0.16b}, v20.16b
    tbl     v2.16b, {v0.16b}, v21.16b
    tbl     v3.16b, {v0.16b}, v22.16b
    tbl     v4.16b, {v0.16b}, v23.16b
    bsl     v1.16b, v24.16b, v28.16b
    bsl     v2.16b, v25.16b, v29.16b
    bsl     v3.16b, v26.16b, v30.16b
    bsl     v4.16b, v27.16b, v31.16b
    stp     q1, q2, [\o, #64]
    stp     q3, q4, [\o, #96]
.endm

.text
.globl _where_f64
.p2align 2
_where_f64:
    cmp     x4, #BLOCK_N
    b.lo    Lsmall
    adrp    x8, Lidx@PAGE
    add     x8, x8, Lidx@PAGEOFF
    ldp     q16, q17, [x8]
    ldp     q18, q19, [x8, #32]
    ldp     q20, q21, [x8, #64]
    ldp     q22, q23, [x8, #96]
    // The last block, in each buffer.
    add     x10, x0, x4
    sub     x10, x10, #BLOCK_N
    add     x11, x1, x4, lsl #3
    sub     x11, x11, #BLOCK_N * 8
    add     x12, x2, x4, lsl #3
    sub     x12, x12, #BLOCK_N * 8
    add     x13, x3, x4, lsl #3
    sub     x13, x13, #BLOCK_N * 8
    cmp     x0, x10
    b.hs    Llast
Lblock:
    BLOCK   x0, x1, x2, x3
    add     x0, x0, #BLOCK_N
    add     x1, x1, #BLOCK_N * 8
    add     x2, x2, #BLOCK_N * 8
    add     x3, x3, #BLOCK_N * 8
    cmp     x0, x10
    b.lo    Lblock
Llast:
    BLOCK   x10, x11, x12, x13
    ret

Lsmall:
    cbz     x4, Ldone
Lone:
    ldrb    w5, [x0], #1
    ldr     x6, [x1], #8
    ldr     x7, [x2], #8
    cmp     w5, #0
    csel    x6, x6, x7, ne
    str     x6, [x3], #8
    subs    x4, x4, #1
    b.ne    Lone
Ldone:
    ret
