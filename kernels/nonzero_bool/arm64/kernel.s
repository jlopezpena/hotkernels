// size_t nonzero_bool(const bool *mask, int64_t *out, size_t n);
// Writes the index of every true mask byte to out, ascending. Returns how many.
//
// Two ways of handling a stretch of mask bytes, chosen by how many are true.
//
// Sparse (64 bytes per step): the 64 bytes (each 0 or 1) are packed into a
// 64-bit word, one bit per byte, and the true ones are counted. If at most
// 8 are true, the positions of the 8 lowest set bits are extracted one
// after another, with no branches, and stored; the ones past the true count are
// rubbish that the next store overwrites (`out` is a scratch buffer). If more
// are true, the dense code handles those 64 bytes.
//
// Dense (8 bytes per step): the bytes are loaded as one 64-bit word. One
// multiplication packs them into an 8-bit number, another gives the true count
// (times 8). The 8-bit number indexes a table whose entry lists the positions
// of the true bytes, in order, as eight bytes. Those are widened to 64 bits,
// added to the index of the block's first element, and all eight are stored at
// the write pointer.
//
// After a dense stretch the fraction of true bytes in it decides what handles
// the next bytes: sparse if at most one in eight was true, else a longer dense
// stretch. Stores are always inside the capacity because the write position
// never runs ahead of the read position.
.text
.globl _nonzero_bool
.p2align 2
_nonzero_bool:
    mov     x5, x1                      // x5 = write pointer
    mov     x3, #0                      // x3 = index of the next mask byte
    mov     x4, x2                      // x4 = mask bytes left
    cmp     x2, #8
    b.lo    Ltail
    adrp    x9, Lpositions@PAGE
    add     x9, x9, Lpositions@PAGEOFF
    mov     x10, #0x4080                // 0x0102040810204080: gathers the low
    movk    x10, #0x1020, lsl #16       // bit of each byte into the top byte
    movk    x10, #0x0408, lsl #32
    movk    x10, #0x0102, lsl #48
    mov     x11, #0x0808080808080808    // sums the bytes, times 8, into the top byte
    mov     w12, #8
    dup     v18.2d, x12
    cmp     x4, #64
    b.hs    Lbig
    movi    v17.2d, #0
Lsmall:                                 // 8 to 63 bytes left; v17 = their index
    sub     x4, x4, #8
Lsmall_loop:
    ldr     x6, [x0], #8                // eight mask bytes
    mul     x7, x6, x10
    mul     x8, x6, x11
    lsr     x7, x7, #56                 // 8-bit mask
    lsr     x8, x8, #56                 // 8 * (number of true bytes)
    ldr     d0, [x9, x7, lsl #3]        // positions of the true bytes
    ushll   v0.8h, v0.8b, #0
    ushll   v1.4s, v0.4h, #0
    ushll2  v2.4s, v0.8h, #0
    uaddw   v3.2d, v17.2d, v1.2s
    uaddw2  v4.2d, v17.2d, v1.4s
    uaddw   v5.2d, v17.2d, v2.2s
    uaddw2  v6.2d, v17.2d, v2.4s
    stp     q3, q4, [x5]
    stp     q5, q6, [x5, #32]
    add     v17.2d, v17.2d, v18.2d
    add     x5, x5, x8
    subs    x4, x4, #8
    b.hs    Lsmall_loop
    fmov    x3, d17
    and     x4, x4, #7
    b       Ltail
Lbig:
    adrp    x12, Lbit_shifts@PAGE
    ldr     q19, [x12, Lbit_shifts@PAGEOFF]
Lsparse:                                // at least 64 bytes left
    ldp     q0, q1, [x0]
    ldp     q2, q3, [x0, #32]
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
    fmov    w7, s1                      // number of true bytes
    cmp     w7, #8
    b.hi    Lsparse_fail
    rbit    x8, x6
    clz     x8, x8                      // position of the lowest set bit (64 if none)
    add     x8, x8, x3
    sub     x12, x6, #1
    and     x6, x6, x12                 // clear the lowest set bit
    rbit    x15, x6
    clz     x15, x15                      // position of the lowest set bit (64 if none)
    add     x15, x15, x3
    stp     x8, x15, [x5, #0]
    sub     x12, x6, #1
    and     x6, x6, x12                 // clear the lowest set bit
    rbit    x8, x6
    clz     x8, x8                      // position of the lowest set bit (64 if none)
    add     x8, x8, x3
    sub     x12, x6, #1
    and     x6, x6, x12                 // clear the lowest set bit
    rbit    x15, x6
    clz     x15, x15                      // position of the lowest set bit (64 if none)
    add     x15, x15, x3
    stp     x8, x15, [x5, #16]
    sub     x12, x6, #1
    and     x6, x6, x12                 // clear the lowest set bit
    rbit    x8, x6
    clz     x8, x8                      // position of the lowest set bit (64 if none)
    add     x8, x8, x3
    sub     x12, x6, #1
    and     x6, x6, x12                 // clear the lowest set bit
    rbit    x15, x6
    clz     x15, x15                      // position of the lowest set bit (64 if none)
    add     x15, x15, x3
    stp     x8, x15, [x5, #32]
    sub     x12, x6, #1
    and     x6, x6, x12                 // clear the lowest set bit
    rbit    x8, x6
    clz     x8, x8                      // position of the lowest set bit (64 if none)
    add     x8, x8, x3
    sub     x12, x6, #1
    and     x6, x6, x12                 // clear the lowest set bit
    rbit    x15, x6
    clz     x15, x15                      // position of the lowest set bit (64 if none)
    add     x15, x15, x3
    stp     x8, x15, [x5, #48]
    add     x5, x5, x7, lsl #3
    add     x0, x0, #64
    add     x3, x3, #64
    sub     x4, x4, #64
    cmp     x4, #64
    b.hs    Lsparse
Lrest:                                  // fewer than 64 bytes left
    cmp     x4, #8
    b.lo    Ltail
    dup     v17.2d, x3
    b       Lsmall
Lsparse_fail:
    mov     x14, #8
Ldense:                                 // run x14 blocks of 8 bytes; x14 * 8 <= x4
    dup     v17.2d, x3                  // index of the block's first element, twice
    mov     x13, x5
    mov     x16, x14
Ldense_loop:
    ldr     x6, [x0], #8                // eight mask bytes
    mul     x7, x6, x10
    mul     x8, x6, x11
    lsr     x7, x7, #56                 // 8-bit mask
    lsr     x8, x8, #56                 // 8 * (number of true bytes)
    ldr     d0, [x9, x7, lsl #3]        // positions of the true bytes
    ushll   v0.8h, v0.8b, #0
    ushll   v1.4s, v0.4h, #0
    ushll2  v2.4s, v0.8h, #0
    uaddw   v3.2d, v17.2d, v1.2s
    uaddw2  v4.2d, v17.2d, v1.4s
    uaddw   v5.2d, v17.2d, v2.2s
    uaddw2  v6.2d, v17.2d, v2.4s
    stp     q3, q4, [x5]
    stp     q5, q6, [x5, #32]
    add     v17.2d, v17.2d, v18.2d
    add     x5, x5, x8
    subs    x14, x14, #1
    b.ne    Ldense_loop
    add     x3, x3, x16, lsl #3
    sub     x4, x4, x16, lsl #3
    cmp     x4, #64
    b.lo    Lrest
    sub     x13, x5, x13                // 8 * (true bytes in this stretch)
    cmp     x13, x16, lsl #3
    b.ls    Lsparse                     // at most one in eight was true
    lsr     x14, x4, #3                 // else a longer dense stretch
    mov     x15, #64
    cmp     x14, x15
    csel    x14, x14, x15, lo
    b       Ldense
Ltail:                                  // 0 to 7 bytes left
    cbz     x4, Ldone
Ltail_loop:
    ldrb    w6, [x0], #1
    str     x3, [x5]
    add     x3, x3, #1
    add     x5, x5, x6, lsl #3
    subs    x4, x4, #1
    b.ne    Ltail_loop
Ldone:
    sub     x0, x5, x1
    lsr     x0, x0, #3
    ret

.section __TEXT,__const
.p2align 4
// Left-shift amounts that give each of 8 consecutive bytes its own bit.
Lbit_shifts:
    .byte 0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7
// For each 8-bit mask (bit j is mask byte j): the positions of its set bits,
// in ascending order, one byte each. Unused bytes are zero. The comment shows
// mask bytes 0 to 7.
Lpositions:
    .byte 0,0,0,0,0,0,0,0  // 00000000
    .byte 0,0,0,0,0,0,0,0  // 10000000
    .byte 1,0,0,0,0,0,0,0  // 01000000
    .byte 0,1,0,0,0,0,0,0  // 11000000
    .byte 2,0,0,0,0,0,0,0  // 00100000
    .byte 0,2,0,0,0,0,0,0  // 10100000
    .byte 1,2,0,0,0,0,0,0  // 01100000
    .byte 0,1,2,0,0,0,0,0  // 11100000
    .byte 3,0,0,0,0,0,0,0  // 00010000
    .byte 0,3,0,0,0,0,0,0  // 10010000
    .byte 1,3,0,0,0,0,0,0  // 01010000
    .byte 0,1,3,0,0,0,0,0  // 11010000
    .byte 2,3,0,0,0,0,0,0  // 00110000
    .byte 0,2,3,0,0,0,0,0  // 10110000
    .byte 1,2,3,0,0,0,0,0  // 01110000
    .byte 0,1,2,3,0,0,0,0  // 11110000
    .byte 4,0,0,0,0,0,0,0  // 00001000
    .byte 0,4,0,0,0,0,0,0  // 10001000
    .byte 1,4,0,0,0,0,0,0  // 01001000
    .byte 0,1,4,0,0,0,0,0  // 11001000
    .byte 2,4,0,0,0,0,0,0  // 00101000
    .byte 0,2,4,0,0,0,0,0  // 10101000
    .byte 1,2,4,0,0,0,0,0  // 01101000
    .byte 0,1,2,4,0,0,0,0  // 11101000
    .byte 3,4,0,0,0,0,0,0  // 00011000
    .byte 0,3,4,0,0,0,0,0  // 10011000
    .byte 1,3,4,0,0,0,0,0  // 01011000
    .byte 0,1,3,4,0,0,0,0  // 11011000
    .byte 2,3,4,0,0,0,0,0  // 00111000
    .byte 0,2,3,4,0,0,0,0  // 10111000
    .byte 1,2,3,4,0,0,0,0  // 01111000
    .byte 0,1,2,3,4,0,0,0  // 11111000
    .byte 5,0,0,0,0,0,0,0  // 00000100
    .byte 0,5,0,0,0,0,0,0  // 10000100
    .byte 1,5,0,0,0,0,0,0  // 01000100
    .byte 0,1,5,0,0,0,0,0  // 11000100
    .byte 2,5,0,0,0,0,0,0  // 00100100
    .byte 0,2,5,0,0,0,0,0  // 10100100
    .byte 1,2,5,0,0,0,0,0  // 01100100
    .byte 0,1,2,5,0,0,0,0  // 11100100
    .byte 3,5,0,0,0,0,0,0  // 00010100
    .byte 0,3,5,0,0,0,0,0  // 10010100
    .byte 1,3,5,0,0,0,0,0  // 01010100
    .byte 0,1,3,5,0,0,0,0  // 11010100
    .byte 2,3,5,0,0,0,0,0  // 00110100
    .byte 0,2,3,5,0,0,0,0  // 10110100
    .byte 1,2,3,5,0,0,0,0  // 01110100
    .byte 0,1,2,3,5,0,0,0  // 11110100
    .byte 4,5,0,0,0,0,0,0  // 00001100
    .byte 0,4,5,0,0,0,0,0  // 10001100
    .byte 1,4,5,0,0,0,0,0  // 01001100
    .byte 0,1,4,5,0,0,0,0  // 11001100
    .byte 2,4,5,0,0,0,0,0  // 00101100
    .byte 0,2,4,5,0,0,0,0  // 10101100
    .byte 1,2,4,5,0,0,0,0  // 01101100
    .byte 0,1,2,4,5,0,0,0  // 11101100
    .byte 3,4,5,0,0,0,0,0  // 00011100
    .byte 0,3,4,5,0,0,0,0  // 10011100
    .byte 1,3,4,5,0,0,0,0  // 01011100
    .byte 0,1,3,4,5,0,0,0  // 11011100
    .byte 2,3,4,5,0,0,0,0  // 00111100
    .byte 0,2,3,4,5,0,0,0  // 10111100
    .byte 1,2,3,4,5,0,0,0  // 01111100
    .byte 0,1,2,3,4,5,0,0  // 11111100
    .byte 6,0,0,0,0,0,0,0  // 00000010
    .byte 0,6,0,0,0,0,0,0  // 10000010
    .byte 1,6,0,0,0,0,0,0  // 01000010
    .byte 0,1,6,0,0,0,0,0  // 11000010
    .byte 2,6,0,0,0,0,0,0  // 00100010
    .byte 0,2,6,0,0,0,0,0  // 10100010
    .byte 1,2,6,0,0,0,0,0  // 01100010
    .byte 0,1,2,6,0,0,0,0  // 11100010
    .byte 3,6,0,0,0,0,0,0  // 00010010
    .byte 0,3,6,0,0,0,0,0  // 10010010
    .byte 1,3,6,0,0,0,0,0  // 01010010
    .byte 0,1,3,6,0,0,0,0  // 11010010
    .byte 2,3,6,0,0,0,0,0  // 00110010
    .byte 0,2,3,6,0,0,0,0  // 10110010
    .byte 1,2,3,6,0,0,0,0  // 01110010
    .byte 0,1,2,3,6,0,0,0  // 11110010
    .byte 4,6,0,0,0,0,0,0  // 00001010
    .byte 0,4,6,0,0,0,0,0  // 10001010
    .byte 1,4,6,0,0,0,0,0  // 01001010
    .byte 0,1,4,6,0,0,0,0  // 11001010
    .byte 2,4,6,0,0,0,0,0  // 00101010
    .byte 0,2,4,6,0,0,0,0  // 10101010
    .byte 1,2,4,6,0,0,0,0  // 01101010
    .byte 0,1,2,4,6,0,0,0  // 11101010
    .byte 3,4,6,0,0,0,0,0  // 00011010
    .byte 0,3,4,6,0,0,0,0  // 10011010
    .byte 1,3,4,6,0,0,0,0  // 01011010
    .byte 0,1,3,4,6,0,0,0  // 11011010
    .byte 2,3,4,6,0,0,0,0  // 00111010
    .byte 0,2,3,4,6,0,0,0  // 10111010
    .byte 1,2,3,4,6,0,0,0  // 01111010
    .byte 0,1,2,3,4,6,0,0  // 11111010
    .byte 5,6,0,0,0,0,0,0  // 00000110
    .byte 0,5,6,0,0,0,0,0  // 10000110
    .byte 1,5,6,0,0,0,0,0  // 01000110
    .byte 0,1,5,6,0,0,0,0  // 11000110
    .byte 2,5,6,0,0,0,0,0  // 00100110
    .byte 0,2,5,6,0,0,0,0  // 10100110
    .byte 1,2,5,6,0,0,0,0  // 01100110
    .byte 0,1,2,5,6,0,0,0  // 11100110
    .byte 3,5,6,0,0,0,0,0  // 00010110
    .byte 0,3,5,6,0,0,0,0  // 10010110
    .byte 1,3,5,6,0,0,0,0  // 01010110
    .byte 0,1,3,5,6,0,0,0  // 11010110
    .byte 2,3,5,6,0,0,0,0  // 00110110
    .byte 0,2,3,5,6,0,0,0  // 10110110
    .byte 1,2,3,5,6,0,0,0  // 01110110
    .byte 0,1,2,3,5,6,0,0  // 11110110
    .byte 4,5,6,0,0,0,0,0  // 00001110
    .byte 0,4,5,6,0,0,0,0  // 10001110
    .byte 1,4,5,6,0,0,0,0  // 01001110
    .byte 0,1,4,5,6,0,0,0  // 11001110
    .byte 2,4,5,6,0,0,0,0  // 00101110
    .byte 0,2,4,5,6,0,0,0  // 10101110
    .byte 1,2,4,5,6,0,0,0  // 01101110
    .byte 0,1,2,4,5,6,0,0  // 11101110
    .byte 3,4,5,6,0,0,0,0  // 00011110
    .byte 0,3,4,5,6,0,0,0  // 10011110
    .byte 1,3,4,5,6,0,0,0  // 01011110
    .byte 0,1,3,4,5,6,0,0  // 11011110
    .byte 2,3,4,5,6,0,0,0  // 00111110
    .byte 0,2,3,4,5,6,0,0  // 10111110
    .byte 1,2,3,4,5,6,0,0  // 01111110
    .byte 0,1,2,3,4,5,6,0  // 11111110
    .byte 7,0,0,0,0,0,0,0  // 00000001
    .byte 0,7,0,0,0,0,0,0  // 10000001
    .byte 1,7,0,0,0,0,0,0  // 01000001
    .byte 0,1,7,0,0,0,0,0  // 11000001
    .byte 2,7,0,0,0,0,0,0  // 00100001
    .byte 0,2,7,0,0,0,0,0  // 10100001
    .byte 1,2,7,0,0,0,0,0  // 01100001
    .byte 0,1,2,7,0,0,0,0  // 11100001
    .byte 3,7,0,0,0,0,0,0  // 00010001
    .byte 0,3,7,0,0,0,0,0  // 10010001
    .byte 1,3,7,0,0,0,0,0  // 01010001
    .byte 0,1,3,7,0,0,0,0  // 11010001
    .byte 2,3,7,0,0,0,0,0  // 00110001
    .byte 0,2,3,7,0,0,0,0  // 10110001
    .byte 1,2,3,7,0,0,0,0  // 01110001
    .byte 0,1,2,3,7,0,0,0  // 11110001
    .byte 4,7,0,0,0,0,0,0  // 00001001
    .byte 0,4,7,0,0,0,0,0  // 10001001
    .byte 1,4,7,0,0,0,0,0  // 01001001
    .byte 0,1,4,7,0,0,0,0  // 11001001
    .byte 2,4,7,0,0,0,0,0  // 00101001
    .byte 0,2,4,7,0,0,0,0  // 10101001
    .byte 1,2,4,7,0,0,0,0  // 01101001
    .byte 0,1,2,4,7,0,0,0  // 11101001
    .byte 3,4,7,0,0,0,0,0  // 00011001
    .byte 0,3,4,7,0,0,0,0  // 10011001
    .byte 1,3,4,7,0,0,0,0  // 01011001
    .byte 0,1,3,4,7,0,0,0  // 11011001
    .byte 2,3,4,7,0,0,0,0  // 00111001
    .byte 0,2,3,4,7,0,0,0  // 10111001
    .byte 1,2,3,4,7,0,0,0  // 01111001
    .byte 0,1,2,3,4,7,0,0  // 11111001
    .byte 5,7,0,0,0,0,0,0  // 00000101
    .byte 0,5,7,0,0,0,0,0  // 10000101
    .byte 1,5,7,0,0,0,0,0  // 01000101
    .byte 0,1,5,7,0,0,0,0  // 11000101
    .byte 2,5,7,0,0,0,0,0  // 00100101
    .byte 0,2,5,7,0,0,0,0  // 10100101
    .byte 1,2,5,7,0,0,0,0  // 01100101
    .byte 0,1,2,5,7,0,0,0  // 11100101
    .byte 3,5,7,0,0,0,0,0  // 00010101
    .byte 0,3,5,7,0,0,0,0  // 10010101
    .byte 1,3,5,7,0,0,0,0  // 01010101
    .byte 0,1,3,5,7,0,0,0  // 11010101
    .byte 2,3,5,7,0,0,0,0  // 00110101
    .byte 0,2,3,5,7,0,0,0  // 10110101
    .byte 1,2,3,5,7,0,0,0  // 01110101
    .byte 0,1,2,3,5,7,0,0  // 11110101
    .byte 4,5,7,0,0,0,0,0  // 00001101
    .byte 0,4,5,7,0,0,0,0  // 10001101
    .byte 1,4,5,7,0,0,0,0  // 01001101
    .byte 0,1,4,5,7,0,0,0  // 11001101
    .byte 2,4,5,7,0,0,0,0  // 00101101
    .byte 0,2,4,5,7,0,0,0  // 10101101
    .byte 1,2,4,5,7,0,0,0  // 01101101
    .byte 0,1,2,4,5,7,0,0  // 11101101
    .byte 3,4,5,7,0,0,0,0  // 00011101
    .byte 0,3,4,5,7,0,0,0  // 10011101
    .byte 1,3,4,5,7,0,0,0  // 01011101
    .byte 0,1,3,4,5,7,0,0  // 11011101
    .byte 2,3,4,5,7,0,0,0  // 00111101
    .byte 0,2,3,4,5,7,0,0  // 10111101
    .byte 1,2,3,4,5,7,0,0  // 01111101
    .byte 0,1,2,3,4,5,7,0  // 11111101
    .byte 6,7,0,0,0,0,0,0  // 00000011
    .byte 0,6,7,0,0,0,0,0  // 10000011
    .byte 1,6,7,0,0,0,0,0  // 01000011
    .byte 0,1,6,7,0,0,0,0  // 11000011
    .byte 2,6,7,0,0,0,0,0  // 00100011
    .byte 0,2,6,7,0,0,0,0  // 10100011
    .byte 1,2,6,7,0,0,0,0  // 01100011
    .byte 0,1,2,6,7,0,0,0  // 11100011
    .byte 3,6,7,0,0,0,0,0  // 00010011
    .byte 0,3,6,7,0,0,0,0  // 10010011
    .byte 1,3,6,7,0,0,0,0  // 01010011
    .byte 0,1,3,6,7,0,0,0  // 11010011
    .byte 2,3,6,7,0,0,0,0  // 00110011
    .byte 0,2,3,6,7,0,0,0  // 10110011
    .byte 1,2,3,6,7,0,0,0  // 01110011
    .byte 0,1,2,3,6,7,0,0  // 11110011
    .byte 4,6,7,0,0,0,0,0  // 00001011
    .byte 0,4,6,7,0,0,0,0  // 10001011
    .byte 1,4,6,7,0,0,0,0  // 01001011
    .byte 0,1,4,6,7,0,0,0  // 11001011
    .byte 2,4,6,7,0,0,0,0  // 00101011
    .byte 0,2,4,6,7,0,0,0  // 10101011
    .byte 1,2,4,6,7,0,0,0  // 01101011
    .byte 0,1,2,4,6,7,0,0  // 11101011
    .byte 3,4,6,7,0,0,0,0  // 00011011
    .byte 0,3,4,6,7,0,0,0  // 10011011
    .byte 1,3,4,6,7,0,0,0  // 01011011
    .byte 0,1,3,4,6,7,0,0  // 11011011
    .byte 2,3,4,6,7,0,0,0  // 00111011
    .byte 0,2,3,4,6,7,0,0  // 10111011
    .byte 1,2,3,4,6,7,0,0  // 01111011
    .byte 0,1,2,3,4,6,7,0  // 11111011
    .byte 5,6,7,0,0,0,0,0  // 00000111
    .byte 0,5,6,7,0,0,0,0  // 10000111
    .byte 1,5,6,7,0,0,0,0  // 01000111
    .byte 0,1,5,6,7,0,0,0  // 11000111
    .byte 2,5,6,7,0,0,0,0  // 00100111
    .byte 0,2,5,6,7,0,0,0  // 10100111
    .byte 1,2,5,6,7,0,0,0  // 01100111
    .byte 0,1,2,5,6,7,0,0  // 11100111
    .byte 3,5,6,7,0,0,0,0  // 00010111
    .byte 0,3,5,6,7,0,0,0  // 10010111
    .byte 1,3,5,6,7,0,0,0  // 01010111
    .byte 0,1,3,5,6,7,0,0  // 11010111
    .byte 2,3,5,6,7,0,0,0  // 00110111
    .byte 0,2,3,5,6,7,0,0  // 10110111
    .byte 1,2,3,5,6,7,0,0  // 01110111
    .byte 0,1,2,3,5,6,7,0  // 11110111
    .byte 4,5,6,7,0,0,0,0  // 00001111
    .byte 0,4,5,6,7,0,0,0  // 10001111
    .byte 1,4,5,6,7,0,0,0  // 01001111
    .byte 0,1,4,5,6,7,0,0  // 11001111
    .byte 2,4,5,6,7,0,0,0  // 00101111
    .byte 0,2,4,5,6,7,0,0  // 10101111
    .byte 1,2,4,5,6,7,0,0  // 01101111
    .byte 0,1,2,4,5,6,7,0  // 11101111
    .byte 3,4,5,6,7,0,0,0  // 00011111
    .byte 0,3,4,5,6,7,0,0  // 10011111
    .byte 1,3,4,5,6,7,0,0  // 01011111
    .byte 0,1,3,4,5,6,7,0  // 11011111
    .byte 2,3,4,5,6,7,0,0  // 00111111
    .byte 0,2,3,4,5,6,7,0  // 10111111
    .byte 1,2,3,4,5,6,7,0  // 01111111
    .byte 0,1,2,3,4,5,6,7  // 11111111
