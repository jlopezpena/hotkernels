// size_t any_rows_bool(const bool *m, bool *out, size_t n, size_t cols);
//
// x0 = m, x1 = out, x2 = n, x3 = cols. m holds n / cols whole rows of cols
// elements; out[r] = 1 if any element of row r is true, else 0. Returns the
// number of rows and writes exactly that many bytes. cols >= 1. Every byte of
// m is 0 or 1.
//
// The vector paths rely on the bytes being 0 or 1: the result byte is the
// maximum of the row's bytes, stored as it is, so a byte above 1 would be
// stored as itself (widths 1, 4 and 16 to 64, and the last 64 bytes of a
// longer row). Widths 2, 3 and 5 to 15 compare with zero and write 0 or 1.
// A byte above 1 never changes another row's result.
//
// Paths, by cols, chosen once at entry:
//   4        a row is four bytes: three `umaxp` turn 64 bytes into sixteen
//            results, 32 rows per step. Rows left under a block: one more
//            block ending at the last row (stores some rows twice); under 16
//            rows in all, four rows per 16-byte load; under 4, one by one.
//   16..32   each row reduced to 16 bytes, its first 16 or'ed with its last
//            16 (overlapping); sixteen rows by fifteen `umaxp` to one 16-byte
//            store. Rows addressed from four positions and fixed offsets, so
//            no chain of adds runs through a block.
//   33..64   the same with the row's first 32 bytes or'ed with its last 32.
//   1        the result is the matrix: a copy, 32 bytes per step.
//   2..15    other than 4: a row in a general register, from two loads that
//            start at its start and end at its end (8, 4, or 2 and 1 bytes),
//            or'ed and compared with zero; four rows per step.
//   65..     one row at a time, 64 bytes per step, ending at the first step
//            that holds a True; the last 64 end at the row's end.
// The 16..64 paths take the rows left under a block as one more block that
// ends at the last row, and matrices of under 16 rows in groups of 1, 2, 4
// and 8 rows. From 8 MB (HUGE_N) the main loops use non-temporal loads.

// Row \i of a block of sixteen to the 16 bytes of v\d, whose maximum is the
// row's result. Rows of 16 to 32: the first 16 bytes or'ed with the last 16,
// which overlap. No position moves inside a block, so the loads do not wait
// on a chain of adds: row i is at p[i / 4] + o[i % 4], where p = x0, x9, x10,
// x11 are four rows apart and o = 0, x12, x13, x14 are 0, 1, 2 and 3 cols;
// its last 16 bytes at p[i / 4] + e[i % 4], e = x7, x15, x16, x17 the same
// plus cols - 16.
.macro ROW32 d, i
.if \i / 4 == 0
    BASE    x0, \d, \i
.elseif \i / 4 == 1
    BASE    x9, \d, \i
.elseif \i / 4 == 2
    BASE    x10, \d, \i
.else
    BASE    x11, \d, \i
.endif
    orr     v\d\().16b, v\d\().16b, v31.16b
.endm

// The two loads of ROW32 from base \p: v\d and v31.
.macro BASE p, d, i
.if \i % 4 == 0
    ldr     q\d, [\p]
    ldr     q31, [\p, x7]
.elseif \i % 4 == 1
    ldr     q\d, [\p, x12]
    ldr     q31, [\p, x15]
.elseif \i % 4 == 2
    ldr     q\d, [\p, x13]
    ldr     q31, [\p, x16]
.else
    ldr     q\d, [\p, x14]
    ldr     q31, [\p, x17]
.endif
.endm

// One row at x0 as ROW32 does it; x0 moves to the next row.
// Row \i of a block of sixteen for matrices larger than the second-level
// cache: one non-temporal pair of loads of the 32 bytes at the row's start,
// the second 16 masked to the row's last cols - 16 bytes by v30. Row i is at
// x0, x9, x10 or x11 by i % 4, each moved four rows (x2 = 4 cols) after use.
// The pair reads up to 15 bytes past the row's end: the loop stops while a
// whole block is still left.
.macro ROW32NT d, i
.if \i % 4 == 0
    NTPAIR  x0, \d
.elseif \i % 4 == 1
    NTPAIR  x9, \d
.elseif \i % 4 == 2
    NTPAIR  x10, \d
.else
    NTPAIR  x11, \d
.endif
.endm

// The loads of ROW32NT from position \p, which then moves four rows.
.macro NTPAIR p, d
    ldnp    q\d, q31, [\p]
    add     \p, \p, x2
    and     v31.16b, v31.16b, v30.16b
    orr     v\d\().16b, v\d\().16b, v31.16b
.endm

.macro ROW32_ONE d, i
    ldr     q\d, [x0]
    ldr     q31, [x0, x7]
    add     x0, x0, x3
    orr     v\d\().16b, v\d\().16b, v31.16b
.endm

// The same for rows of 33 to 64: the first 32 bytes and the last 32, at
// x8 = x0 + cols - 32.
.macro ROW64 d, i, ld=ldp
    \ld     q\d, q30, [x0]
    \ld     q29, q31, [x8]
    add     x0, x0, x3
    add     x8, x8, x3
    orr     v\d\().16b, v\d\().16b, v30.16b
    orr     v29.16b, v29.16b, v31.16b
    orr     v\d\().16b, v\d\().16b, v29.16b
.endm

// The same for rows of 65 to 128: the first 64 bytes and the last 64, at
// x8 = x0 + cols - 64.
.macro ROW128 d, i
    ldp     q\d, q30, [x0]
    ldp     q29, q31, [x0, #32]
    ldp     q24, q25, [x8]
    ldp     q26, q27, [x8, #32]
    add     x0, x0, x3
    add     x8, x8, x3
    orr     v\d\().16b, v\d\().16b, v30.16b
    orr     v29.16b, v29.16b, v31.16b
    orr     v24.16b, v24.16b, v25.16b
    orr     v26.16b, v26.16b, v27.16b
    orr     v\d\().16b, v\d\().16b, v29.16b
    orr     v24.16b, v24.16b, v26.16b
    orr     v\d\().16b, v\d\().16b, v24.16b
.endm

// ROW64 with non-temporal loads, for matrices larger than the second-level
// cache.
.macro ROW64NT d, i
    ROW64   \d, \i, ldnp
.endm

// Sixteen rows to the sixteen bytes of v0, one per row, in order. Each
// `umaxp` halves the bytes per row and keeps the rows of its first operand
// before those of its second.
.macro TREE16 row
    \row   0, 0
    \row   1, 1
    umaxp   v0.16b, v0.16b, v1.16b      // rows 0-1, 8 bytes each
    \row   2, 2
    \row   3, 3
    umaxp   v2.16b, v2.16b, v3.16b
    umaxp   v0.16b, v0.16b, v2.16b      // rows 0-3, 4 bytes each
    \row   4, 4
    \row   5, 5
    umaxp   v4.16b, v4.16b, v5.16b
    \row   6, 6
    \row   7, 7
    umaxp   v6.16b, v6.16b, v7.16b
    umaxp   v4.16b, v4.16b, v6.16b
    umaxp   v0.16b, v0.16b, v4.16b      // rows 0-7, 2 bytes each
    \row   16, 8
    \row   17, 9
    umaxp   v16.16b, v16.16b, v17.16b
    \row   18, 10
    \row   19, 11
    umaxp   v18.16b, v18.16b, v19.16b
    umaxp   v16.16b, v16.16b, v18.16b
    \row   20, 12
    \row   21, 13
    umaxp   v20.16b, v20.16b, v21.16b
    \row   22, 14
    \row   23, 15
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

// Fewer than sixteen rows, x4 of them, one row after another: one row if bit
// 0 of x4 is set, then two, four and eight by the higher bits, each group
// reduced like a block (or the pair by two `umaxv`) and stored as 1, 2, 4
// or 8 bytes. The smallest groups
// come first so that one row, the most common short matrix, takes one branch.
// Ends at Ldone.
.macro FEW row
    tbz     x4, #0, 1f
    \row   22, 0
    umaxv   b22, v22.16b
    str     b22, [x1], #1
    cmp     x4, #2
    b.lo    Ldone
1:
    tbz     x4, #1, 2f
    \row   20, 0
    \row   21, 0
    umaxv   b20, v20.16b                // two rows side by side: two
    umaxv   b21, v21.16b                // reductions in parallel are
    str     b20, [x1]                   // shorter than a chain of umaxp
    str     b21, [x1, #1]
    add     x1, x1, #2
    cmp     x4, #4
    b.lo    Ldone
2:
    tbz     x4, #2, 3f
    \row   16, 0
    \row   17, 0
    umaxp   v16.16b, v16.16b, v17.16b
    \row   18, 0
    \row   19, 0
    umaxp   v18.16b, v18.16b, v19.16b
    umaxp   v16.16b, v16.16b, v18.16b   // 4 rows, 4 bytes each
    umaxp   v16.16b, v16.16b, v16.16b
    umaxp   v16.16b, v16.16b, v16.16b   // 1 byte each, in the low 4
    str     s16, [x1], #4
3:
    tbz     x4, #3, Ldone
    \row   0, 0
    \row   1, 0
    umaxp   v0.16b, v0.16b, v1.16b
    \row   2, 0
    \row   3, 0
    umaxp   v2.16b, v2.16b, v3.16b
    umaxp   v0.16b, v0.16b, v2.16b
    \row   4, 0
    \row   5, 0
    umaxp   v4.16b, v4.16b, v5.16b
    \row   6, 0
    \row   7, 0
    umaxp   v6.16b, v6.16b, v7.16b
    umaxp   v4.16b, v4.16b, v6.16b
    umaxp   v0.16b, v0.16b, v4.16b      // 8 rows, 2 bytes each
    umaxp   v0.16b, v0.16b, v0.16b      // 1 byte each, in the low 8
    str     d0, [x1]
    b       Ldone
.endm

// From this many elements (8 MB) the main loops of rows of 4 and of 33 to 64
// use non-temporal loads: measured, they are 1.16x faster at 16,777,216
// elements and 0.88x at 1,048,576, which fits in the 12 MB second-level
// cache.
.set HUGE_N, 8388608

// The loop of the rows of 2 to 15: \l1 loads the row's first bytes into a
// \r1 register, \l2 its last bytes (at x13 = cols minus their size) into a
// \r2 register. x9, x10, x11 = 1, 2, 3 cols, x12 = 4 cols. Ends at Ldone.
.macro SHORT l1, r1, l2, r2
    add     x14, x13, x9
    add     x15, x13, x10
    add     x16, x13, x11
    subs    x4, x4, #4
    b.lo    2f
1:
    \l1     \r1\()6, [x0]
    \l2     \r2\()7, [x0, x13]
    \l1     \r1\()8, [x0, x9]
    \l2     \r2\()17, [x0, x14]
    orr     x6, x6, x7
    orr     x8, x8, x17
    cmp     x6, #0
    cset    w6, ne
    cmp     x8, #0
    cset    w8, ne
    strb    w6, [x1]
    strb    w8, [x1, #1]
    \l1     \r1\()6, [x0, x10]
    \l2     \r2\()7, [x0, x15]
    \l1     \r1\()8, [x0, x11]
    \l2     \r2\()17, [x0, x16]
    orr     x6, x6, x7
    orr     x8, x8, x17
    cmp     x6, #0
    cset    w6, ne
    cmp     x8, #0
    cset    w8, ne
    strb    w6, [x1, #2]
    strb    w8, [x1, #3]
    add     x0, x0, x12
    add     x1, x1, #4
    subs    x4, x4, #4
    b.hs    1b
2:
    adds    x4, x4, #4
    b.eq    Ldone
3:
    \l1     \r1\()6, [x0]
    \l2     \r2\()7, [x0, x13]
    orr     x6, x6, x7
    cmp     x6, #0
    cset    w6, ne
    strb    w6, [x1], #1
    add     x0, x0, x3
    subs    x4, x4, #1
    b.ne    3b
    b       Ldone
.endm

.text
.globl _any_rows_bool
.p2align 2
_any_rows_bool:
    cmp     x3, #4
    b.eq    Lfour
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
    b.lo    Lmid32_few
    mov     x12, x3                     // offsets of the rows of a block
    add     x13, x3, x3
    add     x14, x13, x3
    add     x15, x7, x12
    add     x16, x7, x13
    add     x17, x7, x14
    lsl     x6, x3, #4                  // 16 cols: one block
    add     x9, x0, x3, lsl #2
    add     x10, x9, x3, lsl #2
    add     x11, x10, x3, lsl #2
    cmp     x2, #HUGE_N
    b.hs    Lmid32_huge
Lmid32:
    TREE16  ROW32
    add     x0, x0, x6
    add     x9, x9, x6
    add     x10, x10, x6
    add     x11, x11, x6
    str     q0, [x1], #16
    subs    x4, x4, #16
    b.hs    Lmid32
    adds    x4, x4, #16                 // 0 to 15 rows left
    b.eq    Ldone
    // One more block that ends at the last row and stores some rows a
    // second time.
    mov     x6, #16
    sub     x6, x6, x4                  // rows to step back
    msub    x0, x6, x3, x0
    sub     x1, x1, x6
    add     x9, x0, x3, lsl #2
    add     x10, x9, x3, lsl #2
    add     x11, x10, x3, lsl #2
    TREE16  ROW32
    str     q0, [x1]
    b       Ldone
Lmid32_few:                             // fewer than 16 rows in all
    add     x4, x4, #16
    FEW     ROW32_ONE
Lmid64:
    sub     x7, x3, #32
    add     x8, x0, x7                  // start of the row's last 32 bytes
    subs    x4, x4, #16
    b.lo    Lmid64_few
    cmp     x2, #HUGE_N
    b.hs    Lmid64_huge
Lmid64_loop:
    TREE16  ROW64
    str     q0, [x1], #16
    subs    x4, x4, #16
    b.hs    Lmid64_loop
Lmid64_rest:
    adds    x4, x4, #16
    b.eq    Ldone
    mov     x6, #16
    sub     x6, x6, x4
    msub    x0, x6, x3, x0
    msub    x8, x6, x3, x8
    sub     x1, x1, x6
    TREE16  ROW64
    str     q0, [x1]
    b       Ldone
Lmid64_few:
    add     x4, x4, #16
    FEW     ROW64
Lgeneric:
    cmp     x3, #128
    b.ls    Lmid128
    // Rows of more than 128, one at a time, 64 bytes per step, and a row ends
    // at the first step that holds a True: its four vectors are or'ed, one
    // `umaxp` halves them to 8 bytes, and those go to a general register to
    // test. The last 64 bytes end at the row's end and overlap the step
    // before.
    sub     x7, x3, #64                 // where a row's last 64 bytes start
Llong_row:
    mov     x6, x0
    add     x10, x0, x7
Llong_step:
    cmp     x6, x10
    b.hs    Llong_last
    ldp     q0, q1, [x6]
    ldp     q2, q3, [x6, #32]
    add     x6, x6, #64
    orr     v0.16b, v0.16b, v1.16b
    orr     v2.16b, v2.16b, v3.16b
    orr     v0.16b, v0.16b, v2.16b
    umaxp   v0.16b, v0.16b, v0.16b
    fmov    x9, d0
    cbz     x9, Llong_step
    mov     w9, #1
    b       Llong_store
Llong_last:
    ldp     q0, q1, [x10]
    ldp     q2, q3, [x10, #32]
    orr     v0.16b, v0.16b, v1.16b
    orr     v2.16b, v2.16b, v3.16b
    orr     v0.16b, v0.16b, v2.16b
    umaxv   b0, v0.16b
    fmov    w9, s0
Llong_store:
    strb    w9, [x1], #1
    add     x0, x0, x3
    subs    x4, x4, #1
    b.ne    Llong_row
    b       Ldone
Lshort:
    // Rows of 1 to 15 other than 4. A row is reduced in a general register:
    // two loads that start at the row's start and end at its end (they
    // overlap when the row is shorter than both), or'ed, compared with zero.
    // Four rows per step, addressed from one position and fixed offsets.
    cmp     x3, #1
    b.eq    Lone
    mov     x9, x3                      // offsets of rows 1, 2 and 3
    add     x10, x3, x3
    add     x11, x10, x3
    lsl     x12, x3, #2                 // 4 cols: one step
    cmp     x3, #8
    b.hs    Lshort8
    cmp     x3, #4
    b.hs    Lshort4
    // Rows of 2 and 3: the first two bytes and the last byte.
    sub     x13, x3, #1
    SHORT   ldrh, w, ldrb, w
Lshort4:                                // rows of 5 to 7: the first 4 bytes and the last 4
    sub     x13, x3, #4
    SHORT   ldr, w, ldr, w
Lshort8:                                // rows of 8 to 15: the first 8 bytes and the last 8
    sub     x13, x3, #8
    SHORT   ldr, x, ldr, x

// Rows of one: the result is the matrix itself, every byte being 0 or 1.
Lone:
    mov     x5, x2
    cmp     x2, #32
    b.lo    Lone_small
    sub     x6, x2, #32
    add     x8, x0, x6
    ldp     q0, q1, [x8]                // the last 32, stored at the end
    add     x7, x1, x6
    sub     x2, x2, #32
Lone_loop:
    ldp     q2, q3, [x0], #32
    stp     q2, q3, [x1], #32
    subs    x2, x2, #32
    b.gt    Lone_loop
    stp     q0, q1, [x7]
    b       Ldone
Lone_small:                             // 1 to 31 bytes: two overlapping copies
    sub     x6, x2, #16
    tbz     x2, #4, 1f
    ldr     q0, [x0]
    ldr     q1, [x0, x6]
    str     q0, [x1]
    str     q1, [x1, x6]
    b       Ldone
1:  sub     x6, x2, #8
    tbz     x2, #3, 2f
    ldr     x7, [x0]
    ldr     x8, [x0, x6]
    str     x7, [x1]
    str     x8, [x1, x6]
    b       Ldone
2:  sub     x6, x2, #4
    tbz     x2, #2, 3f
    ldr     w7, [x0]
    ldr     w8, [x0, x6]
    str     w7, [x1]
    str     w8, [x1, x6]
    b       Ldone
3:  ldrb    w7, [x0], #1                // 1 to 3 bytes
    strb    w7, [x1], #1
    subs    x2, x2, #1
    b.ne    3b
    b       Ldone
Ldone:
    mov     x0, x5
    ret

// The main loops again with non-temporal loads, for matrices larger than the
// second-level cache.
Lmid32_huge:                            // x4 = rows left - 16
    adrp    x8, Lfirst_bytes@PAGE
    add     x8, x8, Lfirst_bytes@PAGEOFF
    sub     x8, x8, x7
    ldr     q30, [x8, #16]              // the first cols - 16 bytes set
    lsl     x2, x3, #2                  // 4 cols
    add     x9, x0, x3                  // rows 1, 2 and 3
    add     x10, x9, x3
    add     x11, x10, x3
    cmp     x4, #16
    b.lo    Lmid32_resume
Lmid32_huge_loop:
    TREE16  ROW32NT
    str     q0, [x1], #16
    sub     x4, x4, #16
    cmp     x4, #16
    b.hs    Lmid32_huge_loop
Lmid32_resume:                          // at least one whole block left
    add     x9, x0, x3, lsl #2
    add     x10, x9, x3, lsl #2
    add     x11, x10, x3, lsl #2
    b       Lmid32
Lmid64_huge:
    TREE16  ROW64NT
    str     q0, [x1], #16
    subs    x4, x4, #16
    b.hs    Lmid64_huge
    b       Lmid64_rest
Lfour32_huge:
    ldnp    q0, q1, [x0]
    ldnp    q2, q3, [x0, #32]
    ldnp    q4, q5, [x0, #64]
    ldnp    q6, q7, [x0, #96]
    add     x0, x0, #128
    umaxp   v0.16b, v0.16b, v1.16b
    umaxp   v2.16b, v2.16b, v3.16b
    umaxp   v4.16b, v4.16b, v5.16b
    umaxp   v6.16b, v6.16b, v7.16b
    umaxp   v0.16b, v0.16b, v2.16b
    umaxp   v4.16b, v4.16b, v6.16b
    stp     q0, q4, [x1], #32
    subs    x4, x4, #32
    b.hs    Lfour32_huge
    b       Lfour_tail

// Rows of four. A row is four bytes, so 64 bytes are sixteen rows and three
// `umaxp` reduce them to one byte per row, in order.
Lfour:
    lsr     x4, x2, #2                  // rows, without a division
    mov     x5, x4
    cmp     x4, #16
    b.lo    Lfour_few
    subs    x4, x4, #32
    b.lo    Lfour_tail
    cmp     x2, #HUGE_N
    b.hs    Lfour32_huge
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
    b.lo    Lfour_few
    mov     x6, #16
    sub     x6, x6, x4                  // rows to step back
    sub     x0, x0, x6, lsl #2
    sub     x1, x1, x6
    FOUR16
    str     q0, [x1]
    b       Ldone
Lfour_few:
    // Fewer than 16 rows in all. With four or more, four rows per 16-byte
    // load, the last four ending at the last row.
    cmp     x4, #4
    b.lo    Lfour_under4
    add     x6, x0, x4, lsl #2
    sub     x6, x6, #16                 // the last four rows
    add     x7, x1, x4
    sub     x7, x7, #4                  // and their results
Lfour4:
    ldr     q0, [x0], #16
    umaxp   v0.16b, v0.16b, v0.16b
    umaxp   v0.16b, v0.16b, v0.16b
    str     s0, [x1], #4
    subs    x4, x4, #4
    b.hi    Lfour4_more
    b       Ldone
Lfour4_more:
    cmp     x4, #4
    b.hs    Lfour4
    ldr     q0, [x6]
    umaxp   v0.16b, v0.16b, v0.16b
    umaxp   v0.16b, v0.16b, v0.16b
    str     s0, [x7]
    b       Ldone
Lfour_under4:
    cbz     x4, Ldone
Lfour_one:
    ldr     w6, [x0], #4
    cmp     w6, #0
    cset    w6, ne
    strb    w6, [x1], #1
    subs    x4, x4, #1
    b.ne    Lfour_one
    b       Ldone

Lmid128:
    // Rows of 65 to 128 as rows of 33 to 64 are done, with no exit: a row
    // is two steps at most, and on sparse rows the exit after the first was
    // a branch that could not be predicted.
    sub     x7, x3, #64
    add     x8, x0, x7                  // start of the row's last 64 bytes
    subs    x4, x4, #16
    b.lo    Lmid128_few
Lmid128_loop:
    TREE16  ROW128
    str     q0, [x1], #16
    subs    x4, x4, #16
    b.hs    Lmid128_loop
    adds    x4, x4, #16
    b.eq    Ldone
    mov     x6, #16
    sub     x6, x6, x4
    msub    x0, x6, x3, x0
    msub    x8, x6, x3, x8
    sub     x1, x1, x6
    TREE16  ROW128
    str     q0, [x1]
    b       Ldone
Lmid128_few:
    add     x4, x4, #16
    FEW     ROW128

.section __TEXT,__const
.p2align 4
// 16 bytes set and 16 clear: the 16 bytes at Lfirst_bytes + 16 - k have
// their first k set.
Lfirst_bytes:
    .fill 16, 1, 0xff
    .fill 16, 1, 0
