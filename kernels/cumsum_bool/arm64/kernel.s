// void cumsum_bool(const bool *mask, int64_t *out, size_t n);
// out[i] = how many of mask[0..i] are true. Every mask byte is 0 or 1.
//
// x0 = mask, x1 = out, x2 = n.
//
// Sixteen elements are one step, as two groups of eight. A group is loaded as
// one 64-bit number and multiplied by 0x0101010101010101: byte k of the
// product is the sum of bytes 0..k, the running count within the group (no
// byte passes 8, so nothing carries), and its top byte is the group's total.
// The count before the group, T, is kept in a general register, where the
// carried chain is one add of a cycle per group.
//
// The product and T are moved to vector registers. Two `tbl` put four bytes
// of the product each in the low bytes of four 32-bit lanes, and `uaddw` and
// `uaddw2` widen those to 64 bits while adding T: four output vectors a group.
//
// The loop loads and multiplies the next block between widening the current
// one and storing it. The work is the same; on this core the order is faster.
.section __TEXT,__const
.p2align 4
Lspread:
    // Four bytes to the low bytes of four 32-bit lanes.
    .byte 0, 0xff, 0xff, 0xff, 1, 0xff, 0xff, 0xff, 2, 0xff, 0xff, 0xff, 3, 0xff, 0xff, 0xff
    .byte 4, 0xff, 0xff, 0xff, 5, 0xff, 0xff, 0xff, 6, 0xff, 0xff, 0xff, 7, 0xff, 0xff, 0xff

.text
// Load a block: its byte counts to d0 and d1, the counts before its two
// groups to v16 and v17, and T moved past it.
.macro LOAD
    ldp     x4, x5, [x0], #16
    mul     x4, x4, x10                 // counts within the first eight
    mul     x5, x5, x10                 // and within the second
    dup     v16.2d, x3                  // T for the first eight
    fmov    d0, x4
    fmov    d1, x5
    add     x3, x3, x4, lsr #56
    dup     v17.2d, x3                  // T for the second eight
    add     x3, x3, x5, lsr #56
.endm

// The eight output vectors of the block that was loaded.
.macro WIDEN
    tbl     v2.16b, {v0.16b}, v26.16b
    tbl     v3.16b, {v0.16b}, v27.16b
    tbl     v4.16b, {v1.16b}, v26.16b
    tbl     v5.16b, {v1.16b}, v27.16b
    uaddw   v6.2d, v16.2d, v2.2s
    uaddw2  v7.2d, v16.2d, v2.4s
    uaddw   v18.2d, v16.2d, v3.2s
    uaddw2  v19.2d, v16.2d, v3.4s
    uaddw   v20.2d, v17.2d, v4.2s
    uaddw2  v21.2d, v17.2d, v4.4s
    uaddw   v22.2d, v17.2d, v5.2s
    uaddw2  v23.2d, v17.2d, v5.4s
.endm

.macro STORE
    stp     q6, q7, [x1]
    stp     q18, q19, [x1, #32]
    stp     q20, q21, [x1, #64]
    stp     q22, q23, [x1, #96]
    add     x1, x1, #128
.endm

.globl _cumsum_bool
.p2align 2
_cumsum_bool:
    mov     x3, #0                      // T: the count so far
    cmp     x2, #16
    b.lo    Ltail
    adrp    x9, Lspread@PAGE
    add     x9, x9, Lspread@PAGEOFF
    ldp     q26, q27, [x9]
    mov     x10, #0x0101010101010101
    sub     x2, x2, #16
    LOAD
    subs    x2, x2, #16
    b.lo    Llast
Lblock:
    WIDEN
    LOAD                                // the next block, ahead of the stores
    STORE
    subs    x2, x2, #16
    b.hs    Lblock
Llast:
    WIDEN
    STORE
    add     x2, x2, #16                 // 0 to 15 elements left
Ltail:
    cbz     x2, Ldone
Lone:
    ldrb    w4, [x0], #1
    add     x3, x3, x4
    str     x3, [x1], #8
    subs    x2, x2, #1
    b.ne    Lone
Ldone:
    ret
