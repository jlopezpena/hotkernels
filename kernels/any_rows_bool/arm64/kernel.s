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

.text
.globl _any_rows_bool
.p2align 2
_any_rows_bool:
    udiv    x4, x2, x3                  // rows
    mov     x5, x4
    cbz     x4, Ldone
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
