// size_t any_rows_bool(const bool *m, bool *out, size_t n, size_t cols);
//
// x0 = m, x1 = out, x2 = n, x3 = cols. m holds n / cols whole rows of cols
// elements; out[r] = 1 if any element of row r is true, else 0. Returns the
// number of rows and writes exactly that many bytes. cols >= 1. Every byte of
// m is 0 or 1.
//
// One row at a time. Rows of 16 or more: 16-byte chunks or'ed together, the
// last ending at the row's end, then `umaxv`. Shorter rows: a byte at a time.

// The row at x0 to the 16 bytes of v\d, whose maximum is the row's result;
// x0 moves to the next row. Rows of 16 to 32: the first 16 bytes or'ed with
// the last 16 (x7 = cols - 16), which overlap.
.macro ROW32 d
    ldr     q\d, [x0]
    ldr     q31, [x0, x7]
    add     x0, x0, x3
    orr     v\d\().16b, v\d\().16b, v31.16b
.endm

// The same for rows of 33 to 64: the first 32 bytes and the last 32, at
// x8 = x0 + cols - 32.
.macro ROW64 d
    ldp     q\d, q30, [x0]
    ldp     q29, q31, [x8]
    add     x0, x0, x3
    add     x8, x8, x3
    orr     v\d\().16b, v\d\().16b, v30.16b
    orr     v29.16b, v29.16b, v31.16b
    orr     v\d\().16b, v\d\().16b, v29.16b
.endm

// Sixteen rows to the sixteen bytes of v0, one per row, in order. Each
// `umaxp` halves the bytes per row and keeps the rows of its first operand
// before those of its second.
.macro TREE16 row
    \row   0
    \row   1
    umaxp   v0.16b, v0.16b, v1.16b      // rows 0-1, 8 bytes each
    \row   2
    \row   3
    umaxp   v2.16b, v2.16b, v3.16b
    umaxp   v0.16b, v0.16b, v2.16b      // rows 0-3, 4 bytes each
    \row   4
    \row   5
    umaxp   v4.16b, v4.16b, v5.16b
    \row   6
    \row   7
    umaxp   v6.16b, v6.16b, v7.16b
    umaxp   v4.16b, v4.16b, v6.16b
    umaxp   v0.16b, v0.16b, v4.16b      // rows 0-7, 2 bytes each
    \row   16
    \row   17
    umaxp   v16.16b, v16.16b, v17.16b
    \row   18
    \row   19
    umaxp   v18.16b, v18.16b, v19.16b
    umaxp   v16.16b, v16.16b, v18.16b
    \row   20
    \row   21
    umaxp   v20.16b, v20.16b, v21.16b
    \row   22
    \row   23
    umaxp   v22.16b, v22.16b, v23.16b
    umaxp   v20.16b, v20.16b, v22.16b
    umaxp   v16.16b, v16.16b, v20.16b   // rows 8-15, 2 bytes each
    umaxp   v0.16b, v0.16b, v16.16b     // rows 0-15, 1 byte each
.endm

// Sixteen rows of four at x0 to the sixteen bytes of v0; x0 moves past them.
.macro FOUR16
    ldp     q0, q1, [x0]
    ldp     q2, q3, [x0, #32]
    add     x0, x0, #64
    umaxp   v0.16b, v0.16b, v1.16b
    umaxp   v2.16b, v2.16b, v3.16b
    umaxp   v0.16b, v0.16b, v2.16b
.endm

.text
.globl _any_rows_bool
.p2align 2
_any_rows_bool:
    udiv    x4, x2, x3                  // rows
    mov     x5, x4
    cbz     x4, Ldone
    cmp     x3, #4
    b.eq    Lfour
    cmp     x3, #16
    b.lo    Lshort
    cmp     x3, #64
    b.hi    Lgeneric
    cmp     x3, #32
    b.hi    Lmid64
    sub     x7, x3, #16                 // start of a row's last 16 bytes
    subs    x4, x4, #16
    b.lo    Lmid32_tail
Lmid32:
    TREE16  ROW32
    str     q0, [x1], #16
    subs    x4, x4, #16
    b.hs    Lmid32
Lmid32_tail:
    adds    x4, x4, #16
    b.eq    Ldone
    b       Lrow
Lmid64:
    sub     x7, x3, #32
    add     x8, x0, x7                  // start of the row's last 32 bytes
    subs    x4, x4, #16
    b.lo    Lmid64_tail
Lmid64_loop:
    TREE16  ROW64
    str     q0, [x1], #16
    subs    x4, x4, #16
    b.hs    Lmid64_loop
Lmid64_tail:
    adds    x4, x4, #16
    b.eq    Ldone
    sub     x7, x3, #16
    b       Lrow
Lgeneric:
    sub     x7, x3, #16                 // start of a row's last chunk
Lrow:
    ldr     q0, [x0, x7]
    mov     x6, #0
Lchunk:
    cmp     x6, x7
    b.hs    Lrow_end
    ldr     q1, [x0, x6]
    orr     v0.16b, v0.16b, v1.16b
    add     x6, x6, #16
    b       Lchunk
Lrow_end:
    umaxv   b0, v0.16b
    str     b0, [x1], #1
    add     x0, x0, x3
    subs    x4, x4, #1
    b.ne    Lrow
    b       Ldone
Lshort:
    mov     w8, #0
    mov     x6, #0
Lbyte:
    ldrb    w9, [x0, x6]
    orr     w8, w8, w9
    add     x6, x6, #1
    cmp     x6, x3
    b.lo    Lbyte
    strb    w8, [x1], #1
    add     x0, x0, x3
    subs    x4, x4, #1
    b.ne    Lshort
Ldone:
    mov     x0, x5
    ret

// Rows of four. A row is four bytes, so 64 bytes are sixteen rows and three
// `umaxp` reduce them to one byte per row, in order.
Lfour:
    subs    x4, x4, #32
    b.lo    Lfour_tail
Lfour32:
    ldp     q0, q1, [x0]
    ldp     q2, q3, [x0, #32]
    ldp     q4, q5, [x0, #64]
    ldp     q6, q7, [x0, #96]
    add     x0, x0, #128
    umaxp   v0.16b, v0.16b, v1.16b
    umaxp   v2.16b, v2.16b, v3.16b
    umaxp   v4.16b, v4.16b, v5.16b
    umaxp   v6.16b, v6.16b, v7.16b
    umaxp   v0.16b, v0.16b, v2.16b
    umaxp   v4.16b, v4.16b, v6.16b
    stp     q0, q4, [x1], #32
    subs    x4, x4, #32
    b.hs    Lfour32
Lfour_tail:
    adds    x4, x4, #32                 // 0 to 31 rows left
    b.eq    Ldone
    cmp     x4, #16
    b.lo    Lfour_last
    FOUR16
    str     q0, [x1], #16
    subs    x4, x4, #16
    b.eq    Ldone
Lfour_last:
    // 1 to 15 rows left. With 16 rows or more in all, one more block that
    // ends at the last row and stores some rows a second time.
    cmp     x5, #16
    b.lo    Lfour_one
    mov     x6, #16
    sub     x6, x6, x4                  // rows to step back
    sub     x0, x0, x6, lsl #2
    sub     x1, x1, x6
    FOUR16
    str     q0, [x1]
    b       Ldone
Lfour_one:
    ldr     w6, [x0], #4
    cmp     w6, #0
    cset    w6, ne
    strb    w6, [x1], #1
    subs    x4, x4, #1
    b.ne    Lfour_one
    b       Ldone
