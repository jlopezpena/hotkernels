// void bincount_u8(const uint8_t *x, int64_t *counts, size_t n);
// counts has exactly 256 slots; it arrives uninitialised.
//
// x0 = x, x1 = counts, x2 = n
//
// Two paths:
//   small n: zero counts, then increment counts directly.
//   large n: four tables of 32-bit counters on the stack. Byte j of the
//            input goes to table j % 4, so neighbouring elements never
//            increment the same memory location. The tables are summed
//            into counts at the end.

// Below this many elements the fixed cost of the stack tables is not repaid.
.set BIG_MIN_N, 8192
// The stack tables hold 32-bit counters, so they are flushed into counts
// after at most this many elements. A multiple of 16.
.set CHUNK, 0x80000000
// Bytes of stack tables: 4 tables x 256 counters x 4 bytes.
.set TABLES, 4096
// Stack slots above the tables.
.set REMAINING, TABLES          // elements not yet assigned to a chunk
.set FLUSHED, TABLES + 8        // non-zero once counts holds a flushed chunk

.text
.globl _bincount_u8
.p2align 2
_bincount_u8:
    cmp     x2, #BIG_MIN_N
    b.hs    Lbig

    // ---- Small path -------------------------------------------------------
    // Zero the 256 counters (2048 bytes), 64 bytes per step.
    movi    v0.2d, #0
    mov     x3, x1
    add     x4, x1, #2048
Lzero:
    stp     q0, q0, [x3]
    stp     q0, q0, [x3, #32]
    add     x3, x3, #64
    cmp     x3, x4
    b.lo    Lzero

    subs    x2, x2, #8
    b.lo    Ltail
    // Eight bytes per step: one 64-bit load, then each byte is extracted
    // with a bit-field move.
Lloop8:
    ldr     x6, [x0], #8
    and     x7, x6, #0xff
    ubfx    x8, x6, #8, #8
    ubfx    x9, x6, #16, #8
    ubfx    x10, x6, #24, #8
    ldr     x15, [x1, x7, lsl #3]
    add     x15, x15, #1
    str     x15, [x1, x7, lsl #3]
    ldr     x16, [x1, x8, lsl #3]
    add     x16, x16, #1
    str     x16, [x1, x8, lsl #3]
    ldr     x17, [x1, x9, lsl #3]
    add     x17, x17, #1
    str     x17, [x1, x9, lsl #3]
    ldr     x3, [x1, x10, lsl #3]
    add     x3, x3, #1
    str     x3, [x1, x10, lsl #3]
    ubfx    x11, x6, #32, #8
    ubfx    x12, x6, #40, #8
    ubfx    x13, x6, #48, #8
    lsr     x14, x6, #56
    ldr     x15, [x1, x11, lsl #3]
    add     x15, x15, #1
    str     x15, [x1, x11, lsl #3]
    ldr     x16, [x1, x12, lsl #3]
    add     x16, x16, #1
    str     x16, [x1, x12, lsl #3]
    ldr     x17, [x1, x13, lsl #3]
    add     x17, x17, #1
    str     x17, [x1, x13, lsl #3]
    ldr     x3, [x1, x14, lsl #3]
    add     x3, x3, #1
    str     x3, [x1, x14, lsl #3]
    subs    x2, x2, #8
    b.hs    Lloop8
Ltail:
    adds    x2, x2, #8              // 0..7 bytes left
    b.eq    Ldone
Lloop1:
    ldrb    w6, [x0], #1
    ldr     x7, [x1, x6, lsl #3]
    add     x7, x7, #1
    str     x7, [x1, x6, lsl #3]
    subs    x2, x2, #1
    b.ne    Lloop1
Ldone:
    ret

    // ---- Large path -------------------------------------------------------
Lbig:
    sub     sp, sp, #TABLES
    sub     sp, sp, #16
    mov     x17, x2                 // elements not yet assigned to a chunk
    str     xzr, [sp, #FLUSHED]
    movi    v0.2d, #0
    // Table bases: sp, x3, x4, x5.
    add     x3, sp, #1024
    add     x4, sp, #2048
    add     x5, sp, #3072
Lchunk:
    // This chunk: min(remaining, CHUNK), rounded down to whole 16-byte steps.
    mov     x2, #CHUNK
    cmp     x17, x2
    csel    x2, x17, x2, lo
    and     x2, x2, #0xfffffffffffffff0
    sub     x17, x17, x2
    str     x17, [sp, #REMAINING]

    // Zero the four tables.
    mov     x6, sp
    add     x7, sp, #TABLES
Lzero_tables:
    stp     q0, q0, [x6]
    stp     q0, q0, [x6, #32]
    add     x6, x6, #64
    cmp     x6, x7
    b.lo    Lzero_tables

    cbz     x2, Lreduce
    add     x16, x0, x2             // end of this chunk
    // Sixteen bytes per step, in four groups of four. Within a group each
    // byte goes to a different table, so the four loads, adds and stores
    // are independent of each other. x17 is free here: the remaining count
    // lives in its stack slot until the chunk is done.
Lmain:
    ldp     x6, x7, [x0]
    add     x0, x0, #16
    and     x8, x6, #0xff
    ubfx    x9, x6, #8, #8
    ubfx    x10, x6, #16, #8
    ubfx    x11, x6, #24, #8
    ldr     w12, [sp, x8, lsl #2]
    ldr     w13, [x3, x9, lsl #2]
    ldr     w14, [x4, x10, lsl #2]
    ldr     w15, [x5, x11, lsl #2]
    add     w12, w12, #1
    add     w13, w13, #1
    add     w14, w14, #1
    add     w15, w15, #1
    str     w12, [sp, x8, lsl #2]
    str     w13, [x3, x9, lsl #2]
    str     w14, [x4, x10, lsl #2]
    str     w15, [x5, x11, lsl #2]
    ubfx    x8, x6, #32, #8
    ubfx    x9, x6, #40, #8
    ubfx    x10, x6, #48, #8
    lsr     x11, x6, #56
    ldr     w12, [sp, x8, lsl #2]
    ldr     w13, [x3, x9, lsl #2]
    ldr     w14, [x4, x10, lsl #2]
    ldr     w15, [x5, x11, lsl #2]
    add     w12, w12, #1
    add     w13, w13, #1
    add     w14, w14, #1
    add     w15, w15, #1
    str     w12, [sp, x8, lsl #2]
    str     w13, [x3, x9, lsl #2]
    str     w14, [x4, x10, lsl #2]
    str     w15, [x5, x11, lsl #2]
    and     x8, x7, #0xff
    ubfx    x9, x7, #8, #8
    ubfx    x10, x7, #16, #8
    ubfx    x11, x7, #24, #8
    ldr     w12, [sp, x8, lsl #2]
    ldr     w13, [x3, x9, lsl #2]
    ldr     w14, [x4, x10, lsl #2]
    ldr     w15, [x5, x11, lsl #2]
    add     w12, w12, #1
    add     w13, w13, #1
    add     w14, w14, #1
    add     w15, w15, #1
    str     w12, [sp, x8, lsl #2]
    str     w13, [x3, x9, lsl #2]
    str     w14, [x4, x10, lsl #2]
    str     w15, [x5, x11, lsl #2]
    ubfx    x8, x7, #32, #8
    ubfx    x9, x7, #40, #8
    ubfx    x10, x7, #48, #8
    lsr     x11, x7, #56
    ldr     w12, [sp, x8, lsl #2]
    ldr     w13, [x3, x9, lsl #2]
    ldr     w14, [x4, x10, lsl #2]
    ldr     w15, [x5, x11, lsl #2]
    add     w12, w12, #1
    add     w13, w13, #1
    add     w14, w14, #1
    add     w15, w15, #1
    str     w12, [sp, x8, lsl #2]
    str     w13, [x3, x9, lsl #2]
    str     w14, [x4, x10, lsl #2]
    str     w15, [x5, x11, lsl #2]
    cmp     x0, x16
    b.lo    Lmain

Lreduce:
    // Sum the four tables. A chunk has fewer than 2^32 elements, so the
    // 32-bit sums cannot overflow. Then widen to 64 bits.
    mov     x6, sp                  // table 0 cursor; the others are at fixed offsets
    mov     x7, x1                  // counts cursor
    add     x8, sp, #1024           // end of table 0
    ldr     x9, [sp, #FLUSHED]
    cbnz    x9, Laccumulate
Lstore:
    ldr     q1, [x6]
    ldr     q2, [x6, #1024]
    ldr     q3, [x6, #2048]
    ldr     q4, [x6, #3072]
    add     v1.4s, v1.4s, v2.4s
    add     v3.4s, v3.4s, v4.4s
    add     v1.4s, v1.4s, v3.4s
    uxtl    v5.2d, v1.2s
    uxtl2   v6.2d, v1.4s
    stp     q5, q6, [x7], #32
    add     x6, x6, #16
    cmp     x6, x8
    b.lo    Lstore
    b       Lreduced
Laccumulate:
    ldr     q1, [x6]
    ldr     q2, [x6, #1024]
    ldr     q3, [x6, #2048]
    ldr     q4, [x6, #3072]
    ldp     q5, q6, [x7]
    add     v1.4s, v1.4s, v2.4s
    add     v3.4s, v3.4s, v4.4s
    add     v1.4s, v1.4s, v3.4s
    uaddw   v5.2d, v5.2d, v1.2s
    uaddw2  v6.2d, v6.2d, v1.4s
    stp     q5, q6, [x7], #32
    add     x6, x6, #16
    cmp     x6, x8
    b.lo    Laccumulate
Lreduced:
    mov     x9, #1
    str     x9, [sp, #FLUSHED]
    ldr     x17, [sp, #REMAINING]
    cmp     x17, #16
    b.hs    Lchunk

    add     sp, sp, #16
    add     sp, sp, #TABLES
    // 0..15 leftover bytes go straight into counts.
    cbz     x17, Lbig_done
Lbig_tail:
    ldrb    w6, [x0], #1
    ldr     x7, [x1, x6, lsl #3]
    add     x7, x7, #1
    str     x7, [x1, x6, lsl #3]
    subs    x17, x17, #1
    b.ne    Lbig_tail
Lbig_done:
    ret
