// size_t compress_f64(const double *x, const bool *mask, double *out, size_t n);
// Copies x[i] to out wherever mask[i] is true. Returns how many were copied.
//
// Eight elements per step, as four pairs. A pair (x[p], x[p+1]) needs only one
// 16-byte store at its output position:
//   mask[p] true:  store {x[p],   x[p+1]}   (x[p+1] is rubbish if unselected)
//   mask[p] false: store {x[p+1], x[p+2]}   (both may be rubbish)
// Rubbish is overwritten by the next store or left past the returned count
// (`out` is a scratch buffer). So the compaction is done by the load address:
// the pair is loaded from x + 8*(p + 1 - mask[p]). No shuffle is needed.
//
// The eight mask bytes (each 0 or 1) are loaded as one 64-bit word m.
//   x16 - (m << 3) holds, in the low byte of each 16-bit field, the load
//   offset of one pair; m * 0x0808...08 holds, in byte j, eight times the
//   number of selected elements among 0..j, which gives every store offset.
// The write pointer advances once per eight elements.
//
// The pair load reads one element past the pair, so the main loop runs only
// while at least nine elements are left. Exactly eight left are handled by a
// block whose last pair uses two scalar stores. Stores stay inside the
// capacity because the write position never runs ahead of the read position.
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
    subs    x4, x3, #9                  // x4 = (elements left after a block) - 1
    b.lo    Llast8
Lloop8:
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
    ubfx    x13, x8, #8, #8             // 8 * (count in elements 0-1)
    ubfx    x14, x8, #24, #8            // 8 * (count in elements 0-3)
    ubfx    x15, x8, #40, #8            // 8 * (count in elements 0-5)
    lsr     x8, x8, #56                 // 8 * (count in all eight)
    str     q0, [x5]
    str     q1, [x5, x13]
    str     q2, [x5, x14]
    str     q3, [x5, x15]
    add     x5, x5, x8
    subs    x4, x4, #8
    b.hs    Lloop8
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
