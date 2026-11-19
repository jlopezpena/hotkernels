// void bincount_i16(const int16_t *x, int64_t *counts, size_t n, size_t bins);
//
// x0 = x, x1 = counts, x2 = n, x3 = bins. counts[v] = how many of the n
// elements at x equal v, for v = 0..bins-1. An element outside that range,
// negative or at or above bins, is ignored.
//
// The element is sign-extended to 64 bits and compared with bins as an
// unsigned number, so a negative element is a huge one and fails the same
// test as one that is too large.

// The table path is for at least this many counters (its sum works four at
// a time) and at most this many: four tables of 32-bit counters, 16 KB of
// stack.
.set TABLES_MIN_BINS, 4
.set TABLES_MAX_BINS, 1024
// It is taken from this many elements per counter, where clearing and
// summing the tables is repaid.
.set TABLES_MIN_PER_BIN, 8
// A sample of the data is taken from SAMPLE + 8 elements: whether at least
// SAMPLE_REPEATS of the first SAMPLE equal the element four places on, which
// goes to the same table of four: one in eight. Uniform codes over 64 values
// have about 1 in 64, four distinct values about 16. On data that repeats
// the tables are taken from 2 elements per counter, and eight tables instead
// of four from EIGHT_MIN_N elements.
.set SAMPLE, 64
.set SAMPLE_REPEATS, 8
.set EIGHT_MIN_N, 8192

// The pair path is for at most this many counters: codes of six bits.
.set PAIR_CODES, 64
// It is taken from this many elements, where clearing and adding up the
// pair table is repaid.
.set PAIRS_MIN_N, 8192
// The pair table: PAIR_CODES rows of PAIR_CODES 32-bit counters, 16 KB.
// Row a1, column a0 counts the pairs (a0, a1). Above it, PAIR_CODES 32-bit
// row totals.
.set PAIR_TABLE, PAIR_CODES * PAIR_CODES * 4
.set ROW_TOTALS, PAIR_CODES * 4

// Count the sign-extended element in \reg into the 32-bit table at \base, if
// it is below bins.
.macro COUNTT reg, base, tmp
    cmp     \reg, x3
    b.hs    1f
    ldr     \tmp, [\base, \reg, lsl #2]
    add     \tmp, \tmp, #1
    str     \tmp, [\base, \reg, lsl #2]
1:
.endm

// Count the four elements of the word in \w, each known to be below bins,
// in the four tables sp, x4, x5 and x16.
.macro TABLE4 w
    and     x10, \w, #0xffff
    ubfx    x11, \w, #16, #16
    ubfx    x12, \w, #32, #16
    lsr     x13, \w, #48
    ldr     w14, [sp, x10, lsl #2]
    add     w14, w14, #1
    str     w14, [sp, x10, lsl #2]
    ldr     w15, [x4, x11, lsl #2]
    add     w15, w15, #1
    str     w15, [x4, x11, lsl #2]
    ldr     w14, [x5, x12, lsl #2]
    add     w14, w14, #1
    str     w14, [x5, x12, lsl #2]
    ldr     w15, [x16, x13, lsl #2]
    add     w15, w15, #1
    str     w15, [x16, x13, lsl #2]
.endm

// The same for four elements that are not all below bins: each is
// sign-extended and tested.
.macro TABLE4S w
    sxth    x10, \w
    sbfx    x11, \w, #16, #16
    sbfx    x12, \w, #32, #16
    asr     x13, \w, #48
    COUNTT  x10, sp, w14
    COUNTT  x11, x4, w15
    COUNTT  x12, x5, w14
    COUNTT  x13, x16, w15
.endm

// Count the four elements of the word in \w, each known to be below bins,
// in the tables at \b0 to \b3 (eight-table path).
.macro TABLE8 w, b0, b1, b2, b3
    and     x11, \w, #0xffff
    ubfx    x12, \w, #16, #16
    ubfx    x13, \w, #32, #16
    lsr     x14, \w, #48
    ldr     w15, [\b0, x11, lsl #2]
    ldr     w3, [\b1, x12, lsl #2]
    add     w15, w15, #1
    add     w3, w3, #1
    str     w15, [\b0, x11, lsl #2]
    str     w3, [\b1, x12, lsl #2]
    ldr     w15, [\b2, x13, lsl #2]
    ldr     w3, [\b3, x14, lsl #2]
    add     w15, w15, #1
    add     w3, w3, #1
    str     w15, [\b2, x13, lsl #2]
    str     w3, [\b3, x14, lsl #2]
.endm

// The same for four elements that are not all below bins: each is
// sign-extended and tested.
.macro TABLE8S w, b0, b1, b2, b3
    sxth    x11, \w
    sbfx    x12, \w, #16, #16
    sbfx    x13, \w, #32, #16
    asr     x14, \w, #48
    COUNTT  x11, \b0, w15
    COUNTT  x12, \b1, w15
    COUNTT  x13, \b2, w15
    COUNTT  x14, \b3, w15
.endm

// Count the sign-extended element in \reg into counts, if it is below bins.
.macro COUNT reg, tmp
    cmp     \reg, x3
    b.hs    1f
    ldr     \tmp, [x1, \reg, lsl #3]
    add     \tmp, \tmp, #1
    str     \tmp, [x1, \reg, lsl #3]
1:
.endm

.text
.globl _bincount_i16
.p2align 2
_bincount_i16:
    // Every path but the single table needs n < 2^32.
    mov     x15, #0                 // 1 for the pair path
    lsr     x4, x2, #32
    cbnz    x4, Lsingle
    // The pair table: bins <= 64 and n >= PAIRS_MIN_N. It counts the
    // elements it cannot pair into counts, so counts is cleared first.
    cmp     x3, #PAIR_CODES
    b.hi    Lnotpairs
    cmp     x2, #PAIRS_MIN_N
    b.lo    Lnotpairs
    mov     x15, #1
    b       Lsingle
Lnotpairs:
    // The tables: 4 <= bins <= 1024, and either n >= 8 * bins, or n >= 2 *
    // bins and a sample of the data repeats.
    cmp     x3, #TABLES_MIN_BINS
    b.lo    Lsingle
    cmp     x3, #TABLES_MAX_BINS
    b.hi    Lsingle
    cmp     x2, x3, lsl #1
    b.lo    Lsingle
    // From 8 elements per counter: four tables, or eight from EIGHT_MIN_N
    // elements if the data repeats. From 2: four tables if the data repeats.
    cmp     x2, x3, lsl #3
    b.lo    Lfew
    cmp     x2, #EIGHT_MIN_N
    b.lo    Ltables
Lfew:
    cmp     x2, #SAMPLE + 8
    b.lo    Lsampled
    // Does the data repeat: at least SAMPLE_REPEATS of the first SAMPLE
    // elements equal the one four places on.
    ldp     q0, q1, [x0]
    ldp     q2, q3, [x0, #32]
    ldp     q4, q5, [x0, #64]
    ldp     q6, q7, [x0, #96]
    ldur    q16, [x0, #8]
    ldur    q17, [x0, #24]
    ldur    q18, [x0, #40]
    ldur    q19, [x0, #56]
    ldur    q20, [x0, #72]
    ldur    q21, [x0, #88]
    ldur    q22, [x0, #104]
    ldur    q23, [x0, #120]
    cmeq    v0.8h, v0.8h, v16.8h    // a match is -1
    cmeq    v1.8h, v1.8h, v17.8h
    cmeq    v2.8h, v2.8h, v18.8h
    cmeq    v3.8h, v3.8h, v19.8h
    cmeq    v4.8h, v4.8h, v20.8h
    cmeq    v5.8h, v5.8h, v21.8h
    cmeq    v6.8h, v6.8h, v22.8h
    cmeq    v7.8h, v7.8h, v23.8h
    add     v0.8h, v0.8h, v1.8h
    add     v2.8h, v2.8h, v3.8h
    add     v4.8h, v4.8h, v5.8h
    add     v6.8h, v6.8h, v7.8h
    add     v0.8h, v0.8h, v2.8h
    add     v4.8h, v4.8h, v6.8h
    add     v0.8h, v0.8h, v4.8h
    neg     v0.8h, v0.8h
    addv    h0, v0.8h
    fmov    w11, s0
    cmp     w11, #SAMPLE_REPEATS
    b.lo    Lsampled                // no
    cmp     x2, x3, lsl #3
    b.lo    Ltables
    b       Leight
Lsampled:                           // data that does not repeat, or too few to tell
    cmp     x2, x3, lsl #3
    b.hs    Ltables
Lsingle:
    // Clear the bins counters: 64 bytes per step, the last 64 bytes ending
    // exactly at the end of counts and overlapping the step before.
    cmp     x3, #8
    b.lo    Lclearfew
    movi    v0.2d, #0
    mov     x4, x1
    add     x5, x1, x3, lsl #3
    sub     x5, x5, #64             // the last eight counters
Lclear:
    stp     q0, q0, [x4]
    stp     q0, q0, [x4, #32]
    add     x4, x4, #64
    cmp     x4, x5
    b.lo    Lclear
    stp     q0, q0, [x5]
    stp     q0, q0, [x5, #32]
    b       Lcount
Lclearfew:                          // 1 to 7 counters
    mov     x5, x3
Lclearone:
    subs    x5, x5, #1
    str     xzr, [x1, x5, lsl #3]
    b.ne    Lclearone

Lcount:
    cbnz    x15, Lpairs
    subs    x2, x2, #8
    b.lo    Ltail
    // Eight elements per step: two 8-byte loads, each element taken out of
    // its word sign-extended.
Lloop8:
    ldp     x6, x7, [x0], #16
    sxth    x8, w6
    sbfx    x9, x6, #16, #16
    sbfx    x10, x6, #32, #16
    asr     x11, x6, #48
    COUNT   x8, x12
    COUNT   x9, x13
    COUNT   x10, x14
    COUNT   x11, x15
    sxth    x8, w7
    sbfx    x9, x7, #16, #16
    sbfx    x10, x7, #32, #16
    asr     x11, x7, #48
    COUNT   x8, x12
    COUNT   x9, x13
    COUNT   x10, x14
    COUNT   x11, x15
    subs    x2, x2, #8
    b.hs    Lloop8
Ltail:
    ands    x2, x2, #7              // 0 to 7 elements left
    b.eq    Ldone
Lone:
    ldrsh   x6, [x0], #2
    COUNT   x6, x7
    subs    x2, x2, #1
    b.ne    Lone
Ldone:
    ret

    // ---- Four tables ------------------------------------------------------
    // Element k of each group of four is counted in table k, 32-bit counters
    // on the stack, so neighbouring elements never update the same memory.
    // A table is bins counters rounded up to whole 16 bytes; the padding is
    // cleared and never counted. counts is not cleared: the sum of the four
    // tables is stored over it. The counters cannot overflow: n < 2^32.
Ltables:
    add     x17, x3, #3
    and     x17, x17, #0xfffffffffffffffc
    lsl     x17, x17, #2            // bytes of one table, a multiple of 16
    sub     sp, sp, x17, lsl #2
    add     x4, sp, x17             // table 1
    add     x5, x4, x17             // table 2
    add     x16, x5, x17            // table 3
    // Clear the four tables, 64 bytes per step: four times a multiple of 16.
    movi    v0.2d, #0
    mov     x6, sp
    add     x7, x16, x17
Lcleartables:
    stp     q0, q0, [x6]
    stp     q0, q0, [x6, #32]
    add     x6, x6, #64
    cmp     x6, x7
    b.lo    Lcleartables

    // Sixteen elements per step. One vector test says whether all sixteen
    // are below bins, read as unsigned 16-bit numbers, where a negative
    // element is at least 32768 and bins at most 1024; if they are, each is
    // taken out of its word unsigned and counted with no test of its own.
    dup     v31.8h, w3
    subs    x2, x2, #16             // at least sixteen elements: n >= 8 * bins
Ltables16:
    ldp     q0, q1, [x0]
    cmhs    v0.8h, v0.8h, v31.8h
    cmhs    v1.8h, v1.8h, v31.8h
    orr     v0.16b, v0.16b, v1.16b
    umaxv   h0, v0.8h
    fmov    w8, s0
    cbnz    w8, Lslow16
    ldp     x6, x7, [x0]
    ldp     x8, x9, [x0, #16]
    TABLE4  x6
    TABLE4  x7
    TABLE4  x8
    TABLE4  x9
Lnext16:
    add     x0, x0, #32
    subs    x2, x2, #16
    b.hs    Ltables16

    // counts[v] = the four tables' counters v, four counters per step; the
    // last step ends exactly at bins and may overlap the one before.
    mov     x6, #0                  // counter index
    sub     x7, x3, #4              // the last step's index
Lsum:
    cmp     x6, x7
    csel    x6, x6, x7, lo
    lsl     x8, x6, #2
    ldr     q1, [sp, x8]
    ldr     q2, [x4, x8]
    ldr     q3, [x5, x8]
    ldr     q4, [x16, x8]
    add     v1.4s, v1.4s, v2.4s
    add     v3.4s, v3.4s, v4.4s
    add     v1.4s, v1.4s, v3.4s
    uxtl    v5.2d, v1.2s
    uxtl2   v6.2d, v1.4s
    add     x9, x1, x6, lsl #3
    stp     q5, q6, [x9]
    add     x6, x6, #4
    cmp     x6, x3
    b.lo    Lsum
    add     sp, sp, x17, lsl #2
    ands    x2, x2, #15             // 0 to 15 elements left, into counts
    b.eq    Ldone
    b       Lone

Lslow16:                            // one of the sixteen is not below bins
    ldp     x6, x7, [x0]
    ldp     x8, x9, [x0, #16]
    TABLE4S x6
    TABLE4S x7
    TABLE4S x8
    TABLE4S x9
    b       Lnext16

    // ---- Eight tables ----------------------------------------------------
    // As the four tables, for data that repeats: element k of each group of
    // eight is counted in table k.
Leight:
    add     x9, x3, #3
    and     x9, x9, #0xfffffffffffffffc
    lsl     x9, x9, #2              // bytes of one table, a multiple of 16
    lsl     x11, x9, #3
    sub     sp, sp, x11
    add     x4, sp, x9              // tables 1 to 7
    add     x5, x4, x9
    add     x6, x5, x9
    add     x7, x6, x9
    add     x8, x7, x9
    add     x16, x8, x9
    add     x17, x16, x9
    // Clear the tables, 64 bytes per step: eight times a multiple of 16.
    fmov    d30, x11                // kept for the end: no integer register is free
    movi    v0.2d, #0
    mov     x10, sp
    add     x9, sp, x11
L8_cleartable:
    stp     q0, q0, [x10]
    stp     q0, q0, [x10, #32]
    add     x10, x10, #64
    cmp     x10, x9
    b.lo    L8_cleartable

    // Sixteen elements per step. One vector test says whether all sixteen
    // are below bins, read as unsigned 16-bit numbers, where a negative
    // element is at least 32768 and bins at most 1024; if they are, each is
    // taken out of its word unsigned and counted with no test of its own.
    dup     v31.8h, w3
    subs    x2, x2, #16             // at least sixteen elements: n >= 8 * bins
L8_tables16:
    ldp     q0, q1, [x0]
    cmhs    v0.8h, v0.8h, v31.8h
    cmhs    v1.8h, v1.8h, v31.8h
    orr     v0.16b, v0.16b, v1.16b
    umaxv   h0, v0.8h
    fmov    w9, s0
    cbnz    w9, L8_slow16
    // bins is restored from v31 where it is needed: x3 is a scratch
    // register in the counting below.
    ldp     x9, x10, [x0]
    TABLE8  x9, sp, x4, x5, x6
    TABLE8  x10, x7, x8, x16, x17
    ldp     x9, x10, [x0, #16]
    TABLE8  x9, sp, x4, x5, x6
    TABLE8  x10, x7, x8, x16, x17
L8_next16:
    add     x0, x0, #32
    subs    x2, x2, #16
    b.hs    L8_tables16
    umov    w3, v31.h[0]            // bins, at most 1024

    // counts[v] = the tables' counters v, four counters per step; the
    // last step ends exactly at bins and may overlap the one before.
    mov     x9, #0                  // counter index
    sub     x10, x3, #4             // the last step's index
L8_sum:
    cmp     x9, x10
    csel    x9, x9, x10, lo
    lsl     x11, x9, #2
    ldr     q0, [sp, x11]
    ldr     q1, [x4, x11]
    ldr     q2, [x5, x11]
    ldr     q3, [x6, x11]
    ldr     q4, [x7, x11]
    ldr     q5, [x8, x11]
    ldr     q6, [x16, x11]
    ldr     q7, [x17, x11]
    add     v0.4s, v0.4s, v1.4s
    add     v2.4s, v2.4s, v3.4s
    add     v4.4s, v4.4s, v5.4s
    add     v6.4s, v6.4s, v7.4s
    add     v0.4s, v0.4s, v2.4s
    add     v4.4s, v4.4s, v6.4s
    add     v0.4s, v0.4s, v4.4s
    uxtl    v16.2d, v0.2s
    uxtl2   v17.2d, v0.4s
    add     x12, x1, x9, lsl #3
    stp     q16, q17, [x12]
    add     x9, x9, #4
    cmp     x9, x3
    b.lo    L8_sum
    fmov    x9, d30
    add     sp, sp, x9
    ands    x2, x2, #15             // 0 to 15 elements left, into counts
    b.eq    Ldone
    b       Lone

L8_slow16:                          // one of the sixteen is not below bins
    umov    w3, v31.h[0]            // bins, at most 1024
    ldp     x9, x10, [x0]
    TABLE8S x9, sp, x4, x5, x6
    TABLE8S x10, x7, x8, x16, x17
    ldp     x9, x10, [x0, #16]
    TABLE8S x9, sp, x4, x5, x6
    TABLE8S x10, x7, x8, x16, x17
    b       L8_next16

    // ---- The pair table ---------------------------------------------------
    // Four elements are read in one 8-byte word. If each is in 0..63, which
    // one test of the word against 0xffc0 in every element shows, the two
    // pairs (a0, a1) and (a2, a3) are counted in the table at a0 | a1 << 6
    // and a2 | a3 << 6; otherwise the four go one at a time into counts,
    // each tested against bins. A pair code at or above bins is counted in
    // the table and left out when the totals are added to counts. The table's
    // counters cannot overflow: n < 2^32, so there are fewer than 2^31 pairs.
Lpairs:
    sub     sp, sp, #(PAIR_TABLE >> 12), lsl #12
    sub     sp, sp, #ROW_TOTALS
    movi    v0.2d, #0
    mov     x6, sp
    add     x7, sp, #(PAIR_TABLE >> 12), lsl #12
Lclearpairs:
    stp     q0, q0, [x6]
    stp     q0, q0, [x6, #32]
    add     x6, x6, #64
    cmp     x6, x7
    b.lo    Lclearpairs

    subs    x2, x2, #8              // at least eight elements
Lpairs8:
    ldp     x6, x7, [x0], #16
    tst     x6, #0xffc0ffc0ffc0ffc0
    b.ne    Lslow6
    orr     x8, x6, x6, lsr #10     // a0 | a1 << 6 in bits 0..11, a2 | a3 << 6 in 32..43
    and     x9, x8, #0xfff
    ubfx    x10, x8, #32, #12
    ldr     w12, [sp, x9, lsl #2]
    add     w12, w12, #1
    str     w12, [sp, x9, lsl #2]
    ldr     w13, [sp, x10, lsl #2]
    add     w13, w13, #1
    str     w13, [sp, x10, lsl #2]
Lback6:
    tst     x7, #0xffc0ffc0ffc0ffc0
    b.ne    Lslow7
    orr     x8, x7, x7, lsr #10
    and     x9, x8, #0xfff
    ubfx    x10, x8, #32, #12
    ldr     w12, [sp, x9, lsl #2]
    add     w12, w12, #1
    str     w12, [sp, x9, lsl #2]
    ldr     w13, [sp, x10, lsl #2]
    add     w13, w13, #1
    str     w13, [sp, x10, lsl #2]
Lback7:
    subs    x2, x2, #8
    b.hs    Lpairs8

    // Column totals in v16..v31, four columns each, and row totals stored
    // above the table, one row of 64 counters per step.
    movi    v16.2d, #0
    movi    v17.2d, #0
    movi    v18.2d, #0
    movi    v19.2d, #0
    movi    v20.2d, #0
    movi    v21.2d, #0
    movi    v22.2d, #0
    movi    v23.2d, #0
    movi    v24.2d, #0
    movi    v25.2d, #0
    movi    v26.2d, #0
    movi    v27.2d, #0
    movi    v28.2d, #0
    movi    v29.2d, #0
    movi    v30.2d, #0
    movi    v31.2d, #0
    mov     x6, sp                  // row cursor
    add     x11, sp, #(PAIR_TABLE >> 12), lsl #12   // the row totals
    mov     x9, #0                  // row
Lrow:
    ld1     {v0.4s, v1.4s, v2.4s, v3.4s}, [x6], #64
    ld1     {v4.4s, v5.4s, v6.4s, v7.4s}, [x6], #64
    add     v16.4s, v16.4s, v0.4s
    add     v17.4s, v17.4s, v1.4s
    add     v18.4s, v18.4s, v2.4s
    add     v19.4s, v19.4s, v3.4s
    add     v20.4s, v20.4s, v4.4s
    add     v21.4s, v21.4s, v5.4s
    add     v22.4s, v22.4s, v6.4s
    add     v23.4s, v23.4s, v7.4s
    add     v0.4s, v0.4s, v1.4s
    add     v2.4s, v2.4s, v3.4s
    add     v4.4s, v4.4s, v5.4s
    add     v6.4s, v6.4s, v7.4s
    add     v0.4s, v0.4s, v2.4s
    add     v4.4s, v4.4s, v6.4s
    add     v0.4s, v0.4s, v4.4s
    addv    s0, v0.4s
    fmov    w12, s0
    ld1     {v0.4s, v1.4s, v2.4s, v3.4s}, [x6], #64
    ld1     {v4.4s, v5.4s, v6.4s, v7.4s}, [x6], #64
    add     v24.4s, v24.4s, v0.4s
    add     v25.4s, v25.4s, v1.4s
    add     v26.4s, v26.4s, v2.4s
    add     v27.4s, v27.4s, v3.4s
    add     v28.4s, v28.4s, v4.4s
    add     v29.4s, v29.4s, v5.4s
    add     v30.4s, v30.4s, v6.4s
    add     v31.4s, v31.4s, v7.4s
    add     v0.4s, v0.4s, v1.4s
    add     v2.4s, v2.4s, v3.4s
    add     v4.4s, v4.4s, v5.4s
    add     v6.4s, v6.4s, v7.4s
    add     v0.4s, v0.4s, v2.4s
    add     v4.4s, v4.4s, v6.4s
    add     v0.4s, v0.4s, v4.4s
    addv    s0, v0.4s
    fmov    w13, s0
    add     w12, w12, w13
    str     w12, [x11, x9, lsl #2]
    add     x9, x9, #1
    cmp     x9, #PAIR_CODES
    b.lo    Lrow
    // The column totals over the first row of the table, which is done with.
    mov     x6, sp
    st1     {v16.4s, v17.4s, v18.4s, v19.4s}, [x6], #64
    st1     {v20.4s, v21.4s, v22.4s, v23.4s}, [x6], #64
    st1     {v24.4s, v25.4s, v26.4s, v27.4s}, [x6], #64
    st1     {v28.4s, v29.4s, v30.4s, v31.4s}, [x6], #64
    // counts[v] += column total v + row total v, for v < bins.
    mov     x9, #0
Laddpairs:
    ldr     w12, [sp, x9, lsl #2]
    ldr     w13, [x11, x9, lsl #2]
    ldr     x14, [x1, x9, lsl #3]
    add     x12, x12, x13
    add     x14, x14, x12
    str     x14, [x1, x9, lsl #3]
    add     x9, x9, #1
    cmp     x9, x3
    b.lo    Laddpairs
    add     sp, sp, #ROW_TOTALS
    add     sp, sp, #(PAIR_TABLE >> 12), lsl #12
    b       Ltail                   // 0 to 7 elements left, into counts

Lslow6:                             // one of the four is not a pair code
    sxth    x8, w6
    sbfx    x9, x6, #16, #16
    sbfx    x10, x6, #32, #16
    asr     x11, x6, #48
    COUNT   x8, x12
    COUNT   x9, x13
    COUNT   x10, x14
    COUNT   x11, x12
    b       Lback6
Lslow7:
    sxth    x8, w7
    sbfx    x9, x7, #16, #16
    sbfx    x10, x7, #32, #16
    asr     x11, x7, #48
    COUNT   x8, x12
    COUNT   x9, x13
    COUNT   x10, x14
    COUNT   x11, x12
    b       Lback7
