// size_t compress_f64(const double *x, const bool *mask, double *out, size_t n);
// Copies x[i] to out wherever mask[i] is true. Returns how many were copied.
//
// Two ways of handling a stretch of elements, chosen by how many are selected.
// Both store more than they keep: what lands past the selected elements is
// rubbish that the next store overwrites or that stays past the returned count
// (`out` is a scratch buffer). Stores stay inside the capacity because the
// write position never runs ahead of the read position.
//
// Sparse (64 elements per step): the 64 mask bytes (each 0 or 1) are packed
// into a 64-bit word, first byte in the top bit, and counted. Then the first
// 6 set bits are found one after another with `clz`, with no branches, and
// the elements they name are copied. When fewer are set, `clz` gives 64 and
// x[64] is copied as rubbish, so this path runs only while at least 65
// elements are left. If more than 6 are selected, a second copy extracts 12;
// if more than 12, the dense code handles those 64 elements.
//
// Dense (8 elements per block, as four pairs; the stretch loop does two blocks
// per iteration): a pair (x[p], x[p+1]) needs only
// one 16-byte store at its output position:
//   mask[p] true:  store {x[p],   x[p+1]}   (x[p+1] is rubbish if unselected)
//   mask[p] false: store {x[p+1], x[p+2]}   (both may be rubbish)
// So the compaction is done by the load address: the pair is loaded from
// x + 8*(p + 1 - mask[p]). The eight mask bytes are loaded as one word m:
//   x16 - (m << 3) holds, in the low byte of each 16-bit field, the load
//   offset of one pair; m * 0x0808...08 holds, in byte j, eight times the
//   number of selected elements among 0..j, which gives every store offset.
// The pair load reads one element past the pair, so this loop runs only while
// at least nine elements are left. Exactly eight left are handled by a block
// whose last pair uses two scalar stores.
//
// After a dense stretch the fraction selected in it decides what handles the
// next elements: sparse if at most one in eight, else a longer dense stretch.
.text
.globl _compress_f64
.p2align 2
_compress_f64:
    mov     x5, x2                      // x5 = write pointer
    mov     x16, #0x0808                // field j = 0x0800 + 16*j + 8
    movk    x16, #0x0818, lsl #16
    movk    x16, #0x0828, lsl #32
    movk    x16, #0x0838, lsl #48
    mov     x17, #0x0808080808080808    // running sums of the bytes, times 8
    mov     x4, x3                      // x4 = elements left
    cmp     x3, #64
    b.lo    Lsmall
    adrp    x12, Lbit_shifts@PAGE
    ldr     q19, [x12, Lbit_shifts@PAGEOFF]
    add     x4, x1, x3
    sub     x4, x4, #65                 // x4 = last mask position a sparse step can start at
    mov     x3, #0x8000000000000000
    cmp     x1, x4
    b.hi    Lrest                       // exactly 64 elements
Lsparse:                                // at least 65 elements left
    ldp     q0, q1, [x1]
    ldp     q2, q3, [x1, #32]
    ushl    v0.16b, v0.16b, v19.16b     // byte j becomes bit (j mod 8)
    ushl    v1.16b, v1.16b, v19.16b
    ushl    v2.16b, v2.16b, v19.16b
    ushl    v3.16b, v3.16b, v19.16b
    addp    v0.16b, v0.16b, v1.16b      // three rounds of pairwise sums
    addp    v2.16b, v2.16b, v3.16b      // leave one byte per 8 mask bytes
    addp    v0.16b, v0.16b, v2.16b
    addp    v0.16b, v0.16b, v0.16b
    cnt     v1.8b, v0.8b
    fmov    x6, d0                      // bit j = mask byte j
    addv    b1, v1.8b
    rbit    x6, x6                      // bit 63-j = mask byte j
    fmov    w7, s1                      // number of selected elements
    cmp     w7, #6
    b.hi    Lsparse_more
    clz     x8, x6
    lsr     x12, x3, x8
    bic     x6, x6, x12
    ldr     d4, [x0, x8, lsl #3]
    clz     x8, x6
    lsr     x12, x3, x8
    bic     x6, x6, x12
    ldr     d5, [x0, x8, lsl #3]
    stp     d4, d5, [x5, #0]
    clz     x8, x6
    lsr     x12, x3, x8
    bic     x6, x6, x12
    ldr     d4, [x0, x8, lsl #3]
    clz     x8, x6
    lsr     x12, x3, x8
    bic     x6, x6, x12
    ldr     d5, [x0, x8, lsl #3]
    stp     d4, d5, [x5, #16]
    clz     x8, x6
    lsr     x12, x3, x8
    bic     x6, x6, x12
    ldr     d4, [x0, x8, lsl #3]
    clz     x8, x6
    ldr     d5, [x0, x8, lsl #3]
    stp     d4, d5, [x5, #32]
Lsparse_next:
    add     x5, x5, x7, lsl #3
    add     x0, x0, #512
    add     x1, x1, #64
    cmp     x1, x4
    b.ls    Lsparse
Lrest:                                  // fewer than 65 elements left
    add     x12, x4, #65
    sub     x4, x12, x1                 // x4 = elements left
    cmp     x4, #64
    b.ne    Lsmall
    // Exactly 64 left: sparse, with the rubbish reads kept inside x.
    ldp     q0, q1, [x1]
    ldp     q2, q3, [x1, #32]
    ushl    v0.16b, v0.16b, v19.16b     // byte j becomes bit (j mod 8)
    ushl    v1.16b, v1.16b, v19.16b
    ushl    v2.16b, v2.16b, v19.16b
    ushl    v3.16b, v3.16b, v19.16b
    addp    v0.16b, v0.16b, v1.16b      // three rounds of pairwise sums
    addp    v2.16b, v2.16b, v3.16b      // leave one byte per 8 mask bytes
    addp    v0.16b, v0.16b, v2.16b
    addp    v0.16b, v0.16b, v0.16b
    cnt     v1.8b, v0.8b
    fmov    x6, d0                      // bit j = mask byte j
    addv    b1, v1.8b
    rbit    x6, x6                      // bit 63-j = mask byte j
    fmov    w7, s1                      // number of selected elements
    cmp     w7, #6
    b.hi    Lsmall
    clz     x8, x6
    lsr     x12, x3, x8
    bic     x6, x6, x12
    and     x8, x8, #63
    ldr     d4, [x0, x8, lsl #3]
    clz     x8, x6
    lsr     x12, x3, x8
    bic     x6, x6, x12
    and     x8, x8, #63
    ldr     d5, [x0, x8, lsl #3]
    stp     d4, d5, [x5, #0]
    clz     x8, x6
    lsr     x12, x3, x8
    bic     x6, x6, x12
    and     x8, x8, #63
    ldr     d4, [x0, x8, lsl #3]
    clz     x8, x6
    lsr     x12, x3, x8
    bic     x6, x6, x12
    and     x8, x8, #63
    ldr     d5, [x0, x8, lsl #3]
    stp     d4, d5, [x5, #16]
    clz     x8, x6
    lsr     x12, x3, x8
    bic     x6, x6, x12
    and     x8, x8, #63
    ldr     d4, [x0, x8, lsl #3]
    clz     x8, x6
    and     x8, x8, #63
    ldr     d5, [x0, x8, lsl #3]
    stp     d4, d5, [x5, #32]
    add     x5, x5, x7, lsl #3
    b       Ldone
Lsparse_more:                           // more than 6 selected in this 64
    cmp     w7, #12
    b.hi    Lsparse_fail
    clz     x8, x6
    lsr     x12, x3, x8
    bic     x6, x6, x12
    ldr     d4, [x0, x8, lsl #3]
    clz     x8, x6
    lsr     x12, x3, x8
    bic     x6, x6, x12
    ldr     d5, [x0, x8, lsl #3]
    stp     d4, d5, [x5, #0]
    clz     x8, x6
    lsr     x12, x3, x8
    bic     x6, x6, x12
    ldr     d4, [x0, x8, lsl #3]
    clz     x8, x6
    lsr     x12, x3, x8
    bic     x6, x6, x12
    ldr     d5, [x0, x8, lsl #3]
    stp     d4, d5, [x5, #16]
    clz     x8, x6
    lsr     x12, x3, x8
    bic     x6, x6, x12
    ldr     d4, [x0, x8, lsl #3]
    clz     x8, x6
    lsr     x12, x3, x8
    bic     x6, x6, x12
    ldr     d5, [x0, x8, lsl #3]
    stp     d4, d5, [x5, #32]
    clz     x8, x6
    lsr     x12, x3, x8
    bic     x6, x6, x12
    ldr     d4, [x0, x8, lsl #3]
    clz     x8, x6
    lsr     x12, x3, x8
    bic     x6, x6, x12
    ldr     d5, [x0, x8, lsl #3]
    stp     d4, d5, [x5, #48]
    clz     x8, x6
    lsr     x12, x3, x8
    bic     x6, x6, x12
    ldr     d4, [x0, x8, lsl #3]
    clz     x8, x6
    lsr     x12, x3, x8
    bic     x6, x6, x12
    ldr     d5, [x0, x8, lsl #3]
    stp     d4, d5, [x5, #64]
    clz     x8, x6
    lsr     x12, x3, x8
    bic     x6, x6, x12
    ldr     d4, [x0, x8, lsl #3]
    clz     x8, x6
    ldr     d5, [x0, x8, lsl #3]
    stp     d4, d5, [x5, #80]
    b       Lsparse_next
Lsparse_fail:
    mov     x14, #4
Ldense:                                 // run x14 steps of 16; at least 16*x14 + 1 elements are left
    add     x13, x5, x14, lsl #4        // write pointer if one in eight is selected
Ldense_loop:
    ldp     x6, x15, [x1], #16          // sixteen mask bytes
    sub     x7, x16, x6, lsl #3         // load offsets, one per 16-bit field
    mul     x8, x6, x17                 // store offsets, one per byte
    and     x9, x7, #0xff
    ubfx    x10, x7, #16, #8
    ubfx    x11, x7, #32, #8
    ubfx    x12, x7, #48, #8
    ldr     q0, [x0, x9]
    ldr     q1, [x0, x10]
    ldr     q2, [x0, x11]
    ldr     q3, [x0, x12]
    add     x0, x0, #64
    sub     x7, x16, x15, lsl #3
    mul     x15, x15, x17
    and     x9, x7, #0xff
    ubfx    x10, x7, #16, #8
    ubfx    x11, x7, #32, #8
    ubfx    x12, x7, #48, #8
    ldr     q4, [x0, x9]
    ldr     q5, [x0, x10]
    ldr     q6, [x0, x11]
    ldr     q7, [x0, x12]
    add     x0, x0, #64
    ubfx    x9, x8, #8, #8              // 8 * (count in elements 0-1)
    ubfx    x10, x8, #24, #8            // 8 * (count in elements 0-3)
    ubfx    x11, x8, #40, #8            // 8 * (count in elements 0-5)
    lsr     x8, x8, #56                 // 8 * (count in all eight)
    str     q0, [x5]
    str     q1, [x5, x9]
    str     q2, [x5, x10]
    str     q3, [x5, x11]
    add     x5, x5, x8
    ubfx    x9, x15, #8, #8
    ubfx    x10, x15, #24, #8
    ubfx    x11, x15, #40, #8
    lsr     x15, x15, #56
    str     q4, [x5]
    str     q5, [x5, x9]
    str     q6, [x5, x10]
    str     q7, [x5, x11]
    add     x5, x5, x15
    subs    x14, x14, #1
    b.ne    Ldense_loop
    cmp     x1, x4
    b.hi    Lrest
    cmp     x5, x13
    b.ls    Lsparse                     // at most one in eight was selected
    add     x14, x4, #64                // else a longer dense stretch
    sub     x14, x14, x1
    lsr     x14, x14, #4
    mov     x12, #32
    cmp     x14, x12
    csel    x14, x14, x12, lo
    b       Ldense
Lsmall:                                 // x4 = elements left, fewer than 65
    subs    x4, x4, #9                  // x4 = (elements left after a block) - 1
    b.lo    Llast8
Lsmall_loop:
    ldr     x6, [x1], #8                // eight mask bytes
    sub     x7, x16, x6, lsl #3         // load offsets, one per 16-bit field
    mul     x8, x6, x17                 // store offsets, one per byte
    and     x9, x7, #0xff
    ubfx    x10, x7, #16, #8
    ubfx    x11, x7, #32, #8
    ubfx    x12, x7, #48, #8
    ldr     q0, [x0, x9]
    ldr     q1, [x0, x10]
    ldr     q2, [x0, x11]
    ldr     q3, [x0, x12]
    add     x0, x0, #64
    ubfx    x9, x8, #8, #8              // 8 * (count in elements 0-1)
    ubfx    x10, x8, #24, #8            // 8 * (count in elements 0-3)
    ubfx    x11, x8, #40, #8            // 8 * (count in elements 0-5)
    lsr     x8, x8, #56                 // 8 * (count in all eight)
    str     q0, [x5]
    str     q1, [x5, x9]
    str     q2, [x5, x10]
    str     q3, [x5, x11]
    add     x5, x5, x8
    subs    x4, x4, #8
    b.hs    Lsmall_loop
Llast8:                                 // x4 = (elements left) - 9, left <= 8
    cmn     x4, #1
    b.ne    Ltail_setup
    // Exactly eight left: the last pair must not read x[8].
    ldr     x6, [x1], #8
    sub     x7, x16, x6, lsl #3
    mul     x8, x6, x17
    and     x9, x7, #0xff
    ubfx    x10, x7, #16, #8
    ubfx    x11, x7, #32, #8
    ldr     q0, [x0, x9]
    ldr     q1, [x0, x10]
    ldr     q2, [x0, x11]
    ldp     d3, d4, [x0, #48]
    ubfx    x13, x8, #8, #8
    ubfx    x14, x8, #24, #8
    ubfx    x15, x8, #40, #8            // output offset of element 6
    ubfx    x12, x8, #48, #8            // output offset of element 7
    lsr     x8, x8, #56
    str     q0, [x5]
    str     q1, [x5, x13]
    str     q2, [x5, x14]
    str     d3, [x5, x15]
    str     d4, [x5, x12]
    add     x5, x5, x8
    b       Ldone
Ltail_setup:
    adds    x4, x4, #9                  // 0 to 7 elements left
    b.eq    Ldone
Ltail:
    ldr     x7, [x0], #8
    ldrb    w6, [x1], #1
    str     x7, [x5]
    add     x5, x5, x6, lsl #3
    subs    x4, x4, #1
    b.ne    Ltail
Ldone:
    sub     x0, x5, x2
    lsr     x0, x0, #3
    ret

.section __TEXT,__const
.p2align 4
// Left-shift amounts that give each of 8 consecutive bytes its own bit.
Lbit_shifts:
    .byte 0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7
