// size_t any_rows_bool(const bool *m, bool *out, size_t n, size_t cols);
//
// x0 = m, x1 = out, x2 = n, x3 = cols. m holds n / cols whole rows of cols
// elements; out[r] = 1 if any element of row r is true, else 0. Returns the
// number of rows and writes exactly that many bytes. cols >= 1. Every byte of
// m is 0 or 1.
//
// One row at a time. Rows of 16 or more: 16-byte chunks or'ed together, the
// last ending at the row's end, then `umaxv`. Shorter rows: a byte at a time.

.text
.globl _any_rows_bool
.p2align 2
_any_rows_bool:
    udiv    x4, x2, x3                  // rows
    mov     x5, x4
    cbz     x4, Ldone
    cmp     x3, #16
    b.lo    Lshort
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
