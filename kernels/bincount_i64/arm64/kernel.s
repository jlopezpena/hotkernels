// void bincount_i64(const int64_t *x, int64_t *counts, size_t n, size_t bins);
//
// x0 = x, x1 = counts, x2 = n, x3 = bins. counts[v] = how many of the n
// elements at x equal v, for v = 0..bins-1. An element outside that range,
// negative or at or above bins, is ignored.
//
// Each element is compared with bins as an unsigned 64-bit number, so a
// negative element is a huge one and fails the same test as one that is
// too large, and so does one with any of its high bits set. No counter
// outside the tables and counts is ever touched, whatever the data.
//
// One counter is loaded, incremented and stored per element, and an
// element that repeats one shortly before it waits for that store. Paths,
// chosen once per call (all but the first need n < 2^32, so that no 32-bit
// counter can overflow):
//
//   one table      counts itself, cleared 64 bytes per step; each element
//                  tested on its own.
//   pair table     bins <= 64 and n >= PAIRS_MIN_N. Two elements per
//                  increment, in a 64 x 64 table of 32-bit counters on the
//                  stack; on data that repeats and from TWO_PAIRS_MIN_N, a
//                  second pair table takes the second pair of each four.
//   four tables    4 <= bins <= 1024, and n >= 8 * bins, or n >= 2 * bins
//                  on data that repeats. Element k of each group of four
//                  goes to table k of bins 32-bit counters on the stack;
//                  the four are tested together.
//   eight tables   as four, from EIGHT_MIN_N elements on data that repeats.
//
// "Repeats" is a sample: SAMPLE_REPEATS or more of the first SAMPLE
// elements equal the element four places on. Every path is right on any
// data; the sample only picks the faster one.
//
// Stack: the pair path takes 32,768 + 256 bytes below the stack pointer,
// the eight tables up to 32 KB (bins 1,024), the four up to 16 KB. The
// stack pointer is moved down by the whole frame at once and the tables
// are cleared from the bottom up, so the first store lands at the bottom
// of the frame, up to 32,768 + 256 bytes below the caller's stack pointer;
// the pages in between are not touched first.

// The table paths are for at least this many counters (their sums work four
// at a time) and at most this many: eight tables of 32-bit counters, 32 KB
// of stack. It is taken from 2^TABLES_SHIFT elements per counter.
.set TABLES_MIN_BINS, 4
.set TABLES_MAX_BINS, 1024
.set TABLES_SHIFT, 3
// On data that repeats, from 2^REPEAT_SHIFT elements per counter.
.set REPEAT_SHIFT, 1

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
// From this many elements, on data that repeats, a second pair table: the
// first pair of each four is counted in the first, the second in the
// second, so that neighbouring equal pairs do not wait for each other.
// Space for both tables is always taken; the row totals are above them.
.set TWO_PAIRS_MIN_N, 32768
.set PAIR_TABLES, 2 * PAIR_TABLE

// A sample of the data: whether at least SAMPLE_REPEATS of the first SAMPLE
// elements equal the element four places on, which goes to the same table
// of four: one in eight. Uniform codes over 64 values have about 0.5 in 32,
// four distinct values about 8. On data that repeats, eight tables instead
// of four from EIGHT_MIN_N elements.
.set SAMPLE, 32
.set SAMPLE_REPEATS, 4
.set EIGHT_MIN_N, 8192

// Count the element in \reg into the 32-bit table at \base, if it is below
// bins.
.macro COUNTT reg, base, tmp
    cmp     \reg, x3
    b.hs    1f
    ldr     \tmp, [\base, \reg, lsl #2]
    add     \tmp, \tmp, #1
    str     \tmp, [\base, \reg, lsl #2]
1:
.endm

// Flags: lo if each of x6 to x9 is below bins, hs if any is not.
.macro INSIDE4
    cmp     x6, x3
    ccmp    x7, x3, #2, lo          // 2: carry set, hs, once one is outside
    ccmp    x8, x3, #2, lo
    ccmp    x9, x3, #2, lo
.endm

// Count x6 to x9, each known to be below bins, in the four tables sp, x4,
// x5 and x16.
.macro TABLE4
    ldr     w12, [sp, x6, lsl #2]
    add     w12, w12, #1
    str     w12, [sp, x6, lsl #2]
    ldr     w13, [x4, x7, lsl #2]
    add     w13, w13, #1
    str     w13, [x4, x7, lsl #2]
    ldr     w14, [x5, x8, lsl #2]
    add     w14, w14, #1
    str     w14, [x5, x8, lsl #2]
    ldr     w15, [x16, x9, lsl #2]
    add     w15, w15, #1
    str     w15, [x16, x9, lsl #2]
.endm

// The same for four that are not all below bins: each is tested.
.macro TABLE4S
    COUNTT  x6, sp, w12
    COUNTT  x7, x4, w13
    COUNTT  x8, x5, w14
    COUNTT  x9, x16, w15
.endm

// Flags: lo if each of \a to \d is below bins, hs if any is not.
.macro INSIDE a, b, c, d
    cmp     \a, x3
    ccmp    \b, x3, #2, lo
    ccmp    \c, x3, #2, lo
    ccmp    \d, x3, #2, lo
.endm

// Count \a to \d, each known to be below bins, in the tables at \b0 to
// \b3 (eight-table path).
.macro TABLE8 a, b, c, d, b0, b1, b2, b3
    ldr     w13, [\b0, \a, lsl #2]
    ldr     w14, [\b1, \b, lsl #2]
    add     w13, w13, #1
    add     w14, w14, #1
    str     w13, [\b0, \a, lsl #2]
    str     w14, [\b1, \b, lsl #2]
    ldr     w13, [\b2, \c, lsl #2]
    ldr     w15, [\b3, \d, lsl #2]
    add     w13, w13, #1
    add     w15, w15, #1
    str     w13, [\b2, \c, lsl #2]
    str     w15, [\b3, \d, lsl #2]
.endm

// The same for four that are not all below bins: each is tested.
.macro TABLE8S a, b, c, d, b0, b1, b2, b3
    COUNTT  \a, \b0, w13
    COUNTT  \b, \b1, w14
    COUNTT  \c, \b2, w13
    COUNTT  \d, \b3, w15
.endm

// w11 = how many of the first SAMPLE elements equal the one four places
// on. Needs SAMPLE + 4 elements.
.macro REPEATS
    ldp     q0, q1, [x0]
    ldp     q2, q3, [x0, #32]
    ldp     q4, q5, [x0, #64]
    ldp     q6, q7, [x0, #96]
    ldp     q16, q17, [x0, #128]
    ldp     q18, q19, [x0, #160]
    ldp     q20, q21, [x0, #192]
    ldp     q22, q23, [x0, #224]
    ldp     q24, q25, [x0, #256]
    cmeq    v0.2d, v0.2d, v2.2d     // a match is -1
    cmeq    v1.2d, v1.2d, v3.2d
    cmeq    v2.2d, v2.2d, v4.2d
    cmeq    v3.2d, v3.2d, v5.2d
    cmeq    v4.2d, v4.2d, v6.2d
    cmeq    v5.2d, v5.2d, v7.2d
    cmeq    v6.2d, v6.2d, v16.2d
    cmeq    v7.2d, v7.2d, v17.2d
    cmeq    v16.2d, v16.2d, v18.2d
    cmeq    v17.2d, v17.2d, v19.2d
    cmeq    v18.2d, v18.2d, v20.2d
    cmeq    v19.2d, v19.2d, v21.2d
    cmeq    v20.2d, v20.2d, v22.2d
    cmeq    v21.2d, v21.2d, v23.2d
    cmeq    v22.2d, v22.2d, v24.2d
    cmeq    v23.2d, v23.2d, v25.2d
    add     v0.2d, v0.2d, v1.2d
    add     v2.2d, v2.2d, v3.2d
    add     v4.2d, v4.2d, v5.2d
    add     v6.2d, v6.2d, v7.2d
    add     v16.2d, v16.2d, v17.2d
    add     v18.2d, v18.2d, v19.2d
    add     v20.2d, v20.2d, v21.2d
    add     v22.2d, v22.2d, v23.2d
    add     v0.2d, v0.2d, v2.2d
    add     v4.2d, v4.2d, v6.2d
    add     v16.2d, v16.2d, v18.2d
    add     v20.2d, v20.2d, v22.2d
    add     v0.2d, v0.2d, v4.2d
    add     v16.2d, v16.2d, v20.2d
    add     v0.2d, v0.2d, v16.2d
    addp    d0, v0.2d
    fmov    x11, d0
    neg     x11, x11
.endm

// Count the element in \reg into counts, if it is below bins.
.macro COUNT reg, tmp
    cmp     \reg, x3
    b.hs    1f
    ldr     \tmp, [x1, \reg, lsl #3]
    add     \tmp, \tmp, #1
    str     \tmp, [x1, \reg, lsl #3]
1:
.endm

.text
.globl _bincount_i64
.p2align 2
_bincount_i64:
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
    cmp     x2, #TWO_PAIRS_MIN_N
    b.lo    Lsingle
    REPEATS
    cmp     x11, #SAMPLE_REPEATS
    cinc    x15, x15, hs            // 2: two pair tables
    b       Lsingle
Lnotpairs:
    // The tables: 4 <= bins <= 1024, and either n >= 8 * bins, or n >= 2 *
    // bins and a sample of the data repeats. From 8 elements per counter:
    // four tables, or eight from EIGHT_MIN_N elements if the data repeats.
    cmp     x3, #TABLES_MIN_BINS
    b.lo    Lsingle
    cmp     x3, #TABLES_MAX_BINS
    b.hi    Lsingle
    cmp     x2, x3, lsl #REPEAT_SHIFT
    b.lo    Lsingle
    cmp     x2, x3, lsl #TABLES_SHIFT
    b.lo    Lfew
    cmp     x2, #EIGHT_MIN_N
    b.lo    Ltables
Lfew:
    cmp     x2, #SAMPLE + 8
    b.lo    Lsampled
    REPEATS
    cmp     x11, #SAMPLE_REPEATS
    b.lo    Lsampled                // the data does not repeat
    cmp     x2, x3, lsl #TABLES_SHIFT
    b.lo    Ltables
    b       Leight
Lsampled:                           // data that does not repeat, or too few to tell
    cmp     x2, x3, lsl #TABLES_SHIFT
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
    // Eight elements per step, from four pair loads.
Lloop8:
    ldp     x6, x7, [x0]
    ldp     x8, x9, [x0, #16]
    COUNT   x6, x12
    COUNT   x7, x13
    COUNT   x8, x14
    COUNT   x9, x15
    ldp     x6, x7, [x0, #32]
    ldp     x8, x9, [x0, #48]
    add     x0, x0, #64
    COUNT   x6, x12
    COUNT   x7, x13
    COUNT   x8, x14
    COUNT   x9, x15
    subs    x2, x2, #8
    b.hs    Lloop8
Ltail:
    ands    x2, x2, #7              // 0 to 7 elements left
    b.eq    Ldone
Lone:
    ldr     x6, [x0], #8
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

    subs    x2, x2, #8              // at least eight elements: n >= 8 * bins
Ltables8:
    ldp     x6, x7, [x0]
    ldp     x8, x9, [x0, #16]
    INSIDE4
    b.hs    Lslowt0
    TABLE4
Lbackt0:
    ldp     x6, x7, [x0, #32]
    ldp     x8, x9, [x0, #48]
    INSIDE4
    b.hs    Lslowt1
    TABLE4
Lbackt1:
    add     x0, x0, #64
    subs    x2, x2, #8
    b.hs    Ltables8

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
    b       Ltail                   // 0 to 7 elements left, into counts

Lslowt0:                            // one of the four is not below bins
    TABLE4S
    b       Lbackt0
Lslowt1:
    TABLE4S
    b       Lbackt1

    // ---- The pair table ---------------------------------------------------
    // Four elements per step. If each is in 0..63, which one test of the
    // four or'ed together shows (the or is below 64, as an unsigned number,
    // only if each is), the two pairs (a0, a1) and (a2, a3) are counted in
    // the table at a0 | a1 << 6 and a2 | a3 << 6; otherwise the four go one
    // at a time into counts, each tested against bins. A pair code at or
    // above bins is counted in the table and left out when the totals are
    // added to counts. The table's counters cannot overflow: n < 2^32, so
    // there are fewer than 2^31 pairs.
Lpairs:
    sub     sp, sp, #(PAIR_TABLES >> 12), lsl #12
    sub     sp, sp, #ROW_TOTALS
    mov     x16, sp                 // the table of the second pairs
    cmp     x15, #2
    b.ne    1f
    add     x16, sp, #(PAIR_TABLE >> 12), lsl #12
1:
    movi    v0.2d, #0
    mov     x6, sp
    add     x7, x16, #(PAIR_TABLE >> 12), lsl #12
Lclearpairs:
    stp     q0, q0, [x6]
    stp     q0, q0, [x6, #32]
    add     x6, x6, #64
    cmp     x6, x7
    b.lo    Lclearpairs

    subs    x2, x2, #4              // at least four elements
Lpairs4:
    ldp     x6, x7, [x0]
    ldp     x8, x9, [x0, #16]
    orr     x10, x6, x7
    orr     x11, x8, x9
    orr     x10, x10, x11
    cmp     x10, #PAIR_CODES
    b.hs    Lslow4
    orr     x10, x6, x7, lsl #6
    orr     x11, x8, x9, lsl #6
    ldr     w12, [sp, x10, lsl #2]
    add     w12, w12, #1
    str     w12, [sp, x10, lsl #2]
    ldr     w13, [x16, x11, lsl #2]
    add     w13, w13, #1
    str     w13, [x16, x11, lsl #2]
Lback4:
    add     x0, x0, #32
    subs    x2, x2, #4
    b.hs    Lpairs4

    // Add the second table, if there is one, into the first.
    mov     x6, sp
    cmp     x16, x6
    b.eq    Lmerged
    mov     x7, x16
Lmergepairs:
    ld1     {v0.4s, v1.4s, v2.4s, v3.4s}, [x6]
    ld1     {v4.4s, v5.4s, v6.4s, v7.4s}, [x7], #64
    add     v0.4s, v0.4s, v4.4s
    add     v1.4s, v1.4s, v5.4s
    add     v2.4s, v2.4s, v6.4s
    add     v3.4s, v3.4s, v7.4s
    st1     {v0.4s, v1.4s, v2.4s, v3.4s}, [x6], #64
    cmp     x6, x16
    b.lo    Lmergepairs
Lmerged:
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
    add     x11, sp, #(PAIR_TABLES >> 12), lsl #12  // the row totals
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
    add     sp, sp, #(PAIR_TABLES >> 12), lsl #12
    ands    x2, x2, #3              // 0 to 3 elements left, into counts
    b.eq    Ldone
    b       Lone

Lslow4:                             // one of the four is not a pair code
    COUNT   x6, x12
    COUNT   x7, x13
    COUNT   x8, x14
    COUNT   x9, x12
    b       Lback4

    // ---- Eight tables ----------------------------------------------------
    // As the four tables, for data that repeats: element k of each group of
    // eight is counted in table k. Eight table bases need every free integer
    // register, so the frame size is kept in d30.
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
    fmov    d30, x11
    // Clear the tables, 64 bytes per step: eight times a multiple of 16.
    movi    v0.2d, #0
    mov     x10, sp
    add     x9, sp, x11
L8_clear:
    stp     q0, q0, [x10]
    stp     q0, q0, [x10, #32]
    add     x10, x10, #64
    cmp     x10, x9
    b.lo    L8_clear

    subs    x2, x2, #8              // at least eight elements
L8_loop:
    ldp     x9, x10, [x0]
    ldp     x11, x12, [x0, #16]
    INSIDE  x9, x10, x11, x12
    b.hs    L8_slow0
    TABLE8  x9, x10, x11, x12, sp, x4, x5, x6
L8_back0:
    ldp     x9, x10, [x0, #32]
    ldp     x11, x12, [x0, #48]
    INSIDE  x9, x10, x11, x12
    b.hs    L8_slow1
    TABLE8  x9, x10, x11, x12, x7, x8, x16, x17
L8_back1:
    add     x0, x0, #64
    subs    x2, x2, #8
    b.hs    L8_loop

    // counts[v] = the eight tables' counters v, four counters per step; the
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
    b       Ltail                   // 0 to 7 elements left, into counts

L8_slow0:                           // one of the four is not below bins
    TABLE8S x9, x10, x11, x12, sp, x4, x5, x6
    b       L8_back0
L8_slow1:
    TABLE8S x9, x10, x11, x12, x7, x8, x16, x17
    b       L8_back1
