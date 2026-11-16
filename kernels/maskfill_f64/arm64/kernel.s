// void maskfill_f64(double *x, const bool *mask, size_t n, double value);
// x0 = x (in place), x1 = mask (bytes, each exactly 0 or 1), x2 = n, d0 = value.
//
// Works in blocks of 16 elements and switches between two ways of handling a
// block, depending on how many of its mask bytes are set.
//
// Blend (dense masks). 16 mask bytes are turned into 0x00/0xFF bytes, each
// pair of bytes is spread over a vector of two 64-bit lanes with tbl, and bsl
// selects between the broadcast value and the loaded data. No data-dependent
// branch. Unselected elements are stored back with their own bits.
//
// Sparse stores (at most one selected element in each half of the block).
// The 16 mask bytes are read as two 64-bit words. A word with at most one
// byte set satisfies w & (w - 1) == 0, and clz gives the position of that
// byte. Each half then does exactly one 8-byte store of value: to its selected
// element, or, when it has none, to the element that half selected most
// recently (which already holds value). Nothing is loaded from x and no unselected element is
// written, and there is still no branch per element.
//
// Every block reached by Ltest is tested; a block that fails the test starts a
// run of blended blocks that is not tested. The run is 1 block long, then 4,
// 16 and 64 while each run is followed directly by another failing block, so
// a dense mask pays for one test per 64 blocks and a sparse mask loses little
// to an isolated dense block. The first block is always blended.
//
// The address of a selected element is not known until one is found, so after
// a run Lscan skips blocks with no mask byte set (touching nothing) and starts
// the address chains at the first block that has one.
//
// The leftover (n mod 16) is handled by one more block that ends exactly at
// the end of the arrays and overlaps the previous one; doing a block twice
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

// Blend the 16 elements at x0 with the 16 raw mask bytes in v1.
.macro BLEND
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
.endm

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
    add     x12, x0, x2, lsl #3     // end of x
    add     x13, x1, x2             // end of mask
    // x3 = (elements from x1 onwards) - 16, for the x1 that follows the
    // current run; negative when no full block is left.
    sub     x3, x2, #32
    mov     x14, #1                 // blocks in this run
    mov     x16, #1                 // length the last run asked for
    add     x15, x1, #16            // mask address where the last run ends

Lrun:                               // blend x14 blocks without testing them
    ldr     q1, [x1], #16
    BLEND
    add     x0, x0, #128
    subs    x14, x14, #1
    b.ne    Lrun
    tbnz    x3, #63, Ltail

Lscan:                              // skip blocks with no mask byte set
    ldp     x4, x5, [x1], #16
    orr     x6, x4, x5
    cbnz    x6, Lfound
    add     x0, x0, #128
    subs    x3, x3, #16
    b.hs    Lscan
Ltail:
    cmn     x3, #16                 // x3 = leftover - 16
    b.eq    Ldone
    sub     x0, x12, #128           // last 16 elements, overlapping
    sub     x1, x13, #16
    mov     x3, #0
    b       Lscan

Lfound:
    // Start the two address chains (see Ltest) from this block, which has a
    // selected element. Each half's chain is given the other half's element;
    // if that half is empty the value is wrong, but then this half is not
    // empty and Ltest replaces it before it is used.
    clz     x6, x5
    sub     x10, x0, x6
    add     x10, x10, #64           // second half's element, for the first half
    clz     x7, x4
    sub     x8, x0, x7
    sub     x8, x8, #64             // first half's element, for the second half
    b       Ltest

Lloop:
    ldp     x4, x5, [x1], #16
Ltest:
    // For a word w of 0/1 bytes with one byte set, at byte k: clz(w) = 63 - 8k,
    // so x0 - clz(w) is the address of element k minus 63 (first half) or
    // minus 127 (second half). x10 and x8 hold that for the element each half
    // selected most recently, in the same biased form. subs leaves the carry
    // set exactly when w != 0, and w & (w - 1) is non-zero when more than one
    // byte is set.
    subs    x6, x4, #1
    clz     x7, x4
    sub     x7, x0, x7
    csel    x10, x7, x10, hs        // first half: its element, or the previous one
    tst     x6, x4
    b.ne    Ldense
    subs    x6, x5, #1
    clz     x7, x5
    sub     x7, x0, x7
    csel    x8, x7, x8, hs          // second half likewise
    tst     x6, x5
    b.ne    Ldense
    stur    d0, [x10, #63]
    stur    d0, [x8, #127]
    add     x0, x0, #128
    subs    x3, x3, #16
    b.hs    Lloop
    b       Ltail

Ldense:
    sub     x7, x1, #16             // this block's mask bytes
    cmp     x7, x15
    b.eq    Lgrow
    // Not directly after a run: blend this block alone and carry on testing.
    // x10 and x8 stay valid: with more than one byte set clz still finds a
    // selected element.
    ldur    q1, [x1, #-16]
    BLEND
    mov     x15, x1                 // counts as a run of one block
    mov     x16, #1
    add     x0, x0, #128
    subs    x3, x3, #16
    b.hs    Lloop
    b       Ltail
Lgrow:
    // Directly after a run: blend four times as many blocks as that run asked
    // for (at most 64), but never more than the full blocks that are left.
    mov     x1, x7                  // back to this block
    lsl     x6, x16, #2
    mov     x7, #64
    cmp     x6, #64
    csel    x16, x6, x7, lo
    lsr     x14, x3, #4
    add     x14, x14, #1            // full blocks left, this one included
    cmp     x14, x16
    csel    x14, x14, x16, lo
    sub     x3, x3, x14, lsl #4
    add     x15, x1, x14, lsl #4
    b       Lrun

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
