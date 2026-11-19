// size_t unique_i64(const int64_t *x, int64_t *out, size_t n, int64_t lo, size_t span);
//
// x0 = x, x1 = out, x2 = n, x3 = lo, x4 = span. Writes the distinct values
// of the n elements at x that lie in [lo, lo + span) to out, in ascending
// order, and returns how many. An element is inside the range when
// d = x - lo, taken as an unsigned 64-bit number, is below span; every
// other element is ignored, so nothing outside out[0..span) is ever
// written, whatever the data. lo + span - 1 is at most the largest int64.
//
// Two paths, chosen by span:
//
//   bits    span <= 64. Value lo + d is bit d of a 64-bit set; nothing is
//           written to out until the values are read out of the set.
//           From CHUNK elements on, NEON takes two elements per vector:
//           1 << d by ushl, or'ed into four sets, and d itself or'ed into
//           four more. Over a chunk of CHUNK elements the or of every d is
//           below 64 exactly when each d is; then each d is a bit of its
//           own, and a d from span to 63 sets a bit at or above span,
//           which the mask at the end drops. A chunk whose or is 64 or
//           more holds an element whose shift was meaningless (ushl reads
//           only the low byte of d), and is read again one element at a
//           time with the exact test. The last chunk ends at n and
//           overlaps the one before it, which a set does not mind. Fewer
//           than CHUNK elements are read one at a time.
//   table   span > 64. The last span bytes of out are a table of one byte
//           per value: cleared, a one stored at byte d for each element
//           inside the range, then read out in order into the front of
//           out, eight bytes at a time where all eight are zero. The
//           values written never reach a byte of the table not yet read
//           (see Lread_start).
//
// No stack, no constants.

// Elements per chunk of the vector bits path, and the fewest it is taken for.
.set CHUNK, 64
// Largest span of the bits path: one bit per value of a 64-bit register.
.set BITS_SPAN, 64

.text
.globl _unique_i64
.p2align 2
_unique_i64:
    cmp     x4, #BITS_SPAN
    b.hi    Ltable

    // ---- bits path: x8 = the set of values found.
    mov     x8, #0
    mov     x9, #1
    cmp     x2, #CHUNK
    b.lo    Lbits_tail

    dup     v31.2d, x3              // lo in both lanes
    dup     v30.2d, x9              // 1 in both lanes
    mov     x10, #0                 // start of the chunk, in elements
    sub     x11, x2, #CHUNK         // start of the last chunk
Lchunk:
    add     x12, x0, x10, lsl #3    // the chunk's first element
    movi    v0.16b, #0
    movi    v1.16b, #0
    movi    v2.16b, #0
    movi    v3.16b, #0
    movi    v4.16b, #0
    movi    v5.16b, #0
    movi    v6.16b, #0
    movi    v7.16b, #0
    mov     x13, #(CHUNK / 8)
Lchunk_step:
    ldp     q16, q17, [x12]
    ldp     q18, q19, [x12, #32]
    add     x12, x12, #64
    sub     v16.2d, v16.2d, v31.2d
    sub     v17.2d, v17.2d, v31.2d
    sub     v18.2d, v18.2d, v31.2d
    sub     v19.2d, v19.2d, v31.2d
    ushl    v20.2d, v30.2d, v16.2d
    ushl    v21.2d, v30.2d, v17.2d
    ushl    v22.2d, v30.2d, v18.2d
    ushl    v23.2d, v30.2d, v19.2d
    orr     v4.16b, v4.16b, v16.16b
    orr     v5.16b, v5.16b, v17.16b
    orr     v6.16b, v6.16b, v18.16b
    orr     v7.16b, v7.16b, v19.16b
    orr     v0.16b, v0.16b, v20.16b
    orr     v1.16b, v1.16b, v21.16b
    orr     v2.16b, v2.16b, v22.16b
    orr     v3.16b, v3.16b, v23.16b
    subs    x13, x13, #1
    b.ne    Lchunk_step

    // The or of every d of the chunk, and of its sets.
    orr     v4.16b, v4.16b, v5.16b
    orr     v6.16b, v6.16b, v7.16b
    orr     v4.16b, v4.16b, v6.16b
    orr     v0.16b, v0.16b, v1.16b
    orr     v2.16b, v2.16b, v3.16b
    orr     v0.16b, v0.16b, v2.16b
    mov     x14, v4.d[0]
    mov     x15, v4.d[1]
    orr     x14, x14, x15
    cmp     x14, #BITS_SPAN
    b.hs    Lchunk_exact
    mov     x14, v0.d[0]
    mov     x15, v0.d[1]
    orr     x8, x8, x14
    orr     x8, x8, x15
Lchunk_next:
    cmp     x10, x11
    b.eq    Lbits_done
    add     x10, x10, #CHUNK
    cmp     x10, x11
    csel    x10, x10, x11, lo
    b       Lchunk

    // A chunk with an element far outside the range: read it again with the
    // exact test.
Lchunk_exact:
    add     x12, x0, x10, lsl #3
    mov     x13, #CHUNK
Lexact_step:
    ldr     x14, [x12], #8
    sub     x14, x14, x3
    cmp     x14, x4
    lsl     x15, x9, x14
    csel    x15, x15, xzr, lo
    orr     x8, x8, x15
    subs    x13, x13, #1
    b.ne    Lexact_step
    b       Lchunk_next

    // Fewer than CHUNK elements: one at a time, with the exact test.
Lbits_tail:
    cbz     x2, Lbits_done
Ltail_step:
    ldr     x14, [x0], #8
    sub     x14, x14, x3
    cmp     x14, x4
    lsl     x15, x9, x14
    csel    x15, x15, xzr, lo
    orr     x8, x8, x15
    subs    x2, x2, #1
    b.ne    Ltail_step

    // Drop the bits at or above span: all ones shifted right by 64 - span,
    // which is 0 to 63 for a span of 1 to 64.
Lbits_done:
    mov     x14, #64
    sub     x14, x14, x4
    mov     x15, #-1
    lsr     x15, x15, x14
    and     x8, x8, x15
    // Read the set out, lowest bit first: value lo + (number of the bit).
    mov     x0, #0
    cbz     x8, Lbits_ret
Lread:
    rbit    x14, x8
    clz     x14, x14
    add     x15, x3, x14
    str     x15, [x1, x0, lsl #3]
    add     x0, x0, #1
    sub     x14, x8, #1
    and     x8, x8, x14
    cbnz    x8, Lread
Lbits_ret:
    ret

    // ---- table path: span > 64, so out has at least 65 elements, 520 bytes.
    // The table is one byte per value, the last span bytes of out: byte d
    // at x6 + d, where x6 = out + 7 * span bytes.
Ltable:
    add     x5, x1, x4, lsl #3      // end of out
    sub     x6, x5, x4              // the table
    // Clear the table, 32 bytes per step; the last step ends at its end.
    movi    v0.16b, #0
    sub     x11, x5, #32
    mov     x12, x6
Lclear:
    stp     q0, q0, [x12], #32
    cmp     x12, x11
    b.lo    Lclear
    stp     q0, q0, [x11]

    // Mark: byte d = 1 for each element inside the range. Four elements per
    // step from two ldp, each with its own test and branch, which the data
    // the binding passes (every element inside) makes predictable.
    mov     w9, #1
    lsr     x13, x2, #2             // steps of four
    and     x2, x2, #3              // the rest, one at a time
    cbz     x13, Lmark_rest
Lmark4:
    ldp     x14, x15, [x0]
    ldp     x16, x17, [x0, #16]
    add     x0, x0, #32
    sub     x14, x14, x3
    sub     x15, x15, x3
    sub     x16, x16, x3
    sub     x17, x17, x3
    cmp     x14, x4
    b.hs    1f
    strb    w9, [x6, x14]
1:  cmp     x15, x4
    b.hs    2f
    strb    w9, [x6, x15]
2:  cmp     x16, x4
    b.hs    3f
    strb    w9, [x6, x16]
3:  cmp     x17, x4
    b.hs    4f
    strb    w9, [x6, x17]
4:  subs    x13, x13, #1
    b.ne    Lmark4
Lmark_rest:
    cbz     x2, Lread_start
Lmark:
    ldr     x14, [x0], #8
    sub     x14, x14, x3
    cmp     x14, x4
    b.hs    Lmark_next
    strb    w9, [x6, x14]
Lmark_next:
    subs    x2, x2, #1
    b.ne    Lmark

    // Read the table out: value lo + j goes to out[count], and count grows
    // by byte j, 0 or 1. Eight bytes are loaded at a time and skipped when
    // all are zero. A store to out[count] covers bytes 8 * count to
    // 8 * count + 7 of out, and count is at most j, so a step whose eight
    // bytes start at j <= span - 8 writes below byte 8 * j + 64, which is
    // at most 7 * span + j + 8: no byte of the table that is still to be
    // read. The last span % 8 bytes are read one at a time, where the same
    // holds for j <= span - 1.
Lread_start:
    mov     x0, #0                  // count
    mov     x10, #0                 // j
    sub     x11, x4, #8             // the last j a step of eight may start at
Lread8:
    cmp     x10, x11
    b.hi    Lread_tail
    ldr     x14, [x6, x10]
    cbz     x14, Lread8_next
    add     x15, x3, x10            // lo + j
    .irp k, 0, 1, 2, 3, 4, 5, 6, 7
    str     x15, [x1, x0, lsl #3]
    and     x16, x14, #0xff
    add     x0, x0, x16
    lsr     x14, x14, #8
    add     x15, x15, #1
    .endr
Lread8_next:
    add     x10, x10, #8
    b       Lread8
Lread_tail:
    cmp     x10, x4
    b.hs    Ltable_ret
    ldrb    w14, [x6, x10]
    add     x15, x3, x10
    str     x15, [x1, x0, lsl #3]
    add     x0, x0, x14
    add     x10, x10, #1
    b       Lread_tail
Ltable_ret:
    ret
