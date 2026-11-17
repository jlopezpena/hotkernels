// void cumsum_f64(const double *x, double *out, size_t n);
// out[i] = x[0] + ... + x[i] with the additions regrouped, so finite results
// differ from the left-to-right sum by rounding only. NaN and infinity appear
// exactly where the left-to-right sum has them. out[0] is a copy of x[0].
//
// Fast path. NEON, two elements per vector. Consecutive output vectors differ by
//     out[i..i+1] - out[i-2..i-1] = x[i..i+1] + x[i-1..i],
// the input vector plus the input shifted back one element (an overlapping
// load, or for the first vector of a block an ext from the previous input
// vector; before the array both are -0.0, which changes nothing it is added
// to). A block is four vectors. Its four steps are summed from the inputs only
// (st0, st0+st1, st0+..+st3), and v16, the last output vector, is added to each:
//     out0 = v16 + st0             out1 = v16 + (st0 + st1)
//     out2 = out1 + st2            out3 = v16 + ((st0 + st1) + (st2 + st3))
// so the carried chain is one fadd per eight elements, not one per element.
// Even and odd elements are two separate running sums; each out[i] is still
// the sum of x[0..i] in some order.
//
// Why non-finite positions match the left-to-right sum, for every input.
// The array is done in chunks of CHUNK elements. In a chunk every output vector
// is squared and accumulated (fmla into v18-v21), and the chunk is accepted
// only if that sum of squares is finite. If it is finite then:
//   1. Every output of the chunk is finite and at most sqrt(DBL_MAX) in size.
//   2. Every input and every intermediate of the chunk is finite: each one is
//      an operand, directly or through further additions, of an output of its
//      own block (st0 of out0; st1 and st0+st1 of out1; st2 of out2; st3,
//      st2+st3 and the block total of out3), and an addition with a NaN or an
//      infinite operand never gives a finite result.
//   3. So each output is a rounded sum of x[0..i] in some order with only
//      finite values on the way, and so is the left-to-right sum for as long
//      as it has stayed finite. Two such sums of i+1 terms differ by less than
//      (i+1) * 2^-52 * DBL_MAX. The left-to-right sum is therefore within that
//      distance of a value no larger than sqrt(DBL_MAX), cannot overflow at
//      this element, and by induction is finite over the whole chunk.
// Both results are finite on an accepted chunk, so no position differs.
// If the sum of squares is not finite, the chunk is done again one element at
// a time from the last output of the chunk before it (Lredo), watching every
// element:
//   - x[i] finite and |sum| <= 2^1023: store it. By the argument of point 3
//     the left-to-right sum is finite here too.
//   - x[i] is NaN or infinite: the left-to-right sum was finite up to here
//     (previous case), so from this element on it is not finite and which of
//     NaN, +inf and -inf it is depends only on the inputs from here on, never
//     on the finite value before. The rest of the array is finished with the
//     plain left-to-right loop (Lplain), which gives the same kinds.
//   - x[i] finite but |sum| > 2^1023 or overflowed: finite data close to the
//     largest double. Whether the left-to-right sum overflows here can depend
//     on its exact rounding, so the whole array is redone from element 0 with
//     the plain loop (Lrestart), which is the reference computation itself.
// A redone chunk that stayed in the first case is followed by the fast path
// again. The price of the square test is that running sums above about 1e152
// in size, though finite, take the slow path.
//
// Short arrays. Fewer than SMALL elements are added strictly left to right
// (Lsmall), which is the reference order, so nothing needs checking. That
// chain costs 3 cycles of latency per element, but it is 46 operations for 16
// elements against 62 for two blocks plus the check, and when calls do not
// depend on each other the processor overlaps them, so the count is what
// matters. Measured, the chain wins up to 28 elements and loses from 32.
//
// Registers: x0 x, x1 out, x2 elements after this chunk, x3 chunk counter,
// x7 start of out, x9 elements in this chunk, x11 bits of x[0],
// v16 last output vector, v17 last input vector, v18-v21 sums of squares.
.set CHUNK, 1024
// Below this many elements the plain chain is used (see Lsmall).
.set SMALL, 32
.text
.globl _cumsum_f64
.p2align 2
_cumsum_f64:
    cmp     x2, #SMALL
    b.lo    Lsmall
    ldr     x11, [x0]               // out[0] is stored from this at the end, as a copy
    mov     x7, x1
    movi    v16.2d, #0
    fneg    v16.2d, v16.2d          // [-0.0, -0.0]: -0.0 + v is v for every v, -0.0 included
    mov     v17.16b, v16.16b
Lchunk:
    movi    v18.2d, #0
    movi    v19.2d, #0
    movi    v20.2d, #0
    movi    v21.2d, #0
    mov     x3, #CHUNK
    cmp     x2, x3
    csel    x3, x2, x3, lo          // elements in this chunk
    mov     x9, x3
    sub     x2, x2, x3
    subs    x3, x3, #16
    b.lo    Lrest
Lmain:
    ldp     q0, q1, [x0]
    ldp     q2, q3, [x0, #32]
    ext     v4.16b, v17.16b, v0.16b, #8     // [x[i-1], x[i]]
    ldur    q5, [x0, #8]
    ldur    q6, [x0, #24]
    ldur    q7, [x0, #40]
    fadd    v4.2d, v4.2d, v0.2d     // st0
    fadd    v5.2d, v5.2d, v1.2d     // st1
    fadd    v6.2d, v6.2d, v2.2d     // st2
    fadd    v7.2d, v7.2d, v3.2d     // st3
    fadd    v5.2d, v5.2d, v4.2d     // st0 + st1
    fadd    v7.2d, v7.2d, v6.2d     // st2 + st3
    fadd    v7.2d, v7.2d, v5.2d     // block total
    fadd    v4.2d, v4.2d, v16.2d    // out0
    fadd    v5.2d, v5.2d, v16.2d    // out1
    fadd    v16.2d, v16.2d, v7.2d   // out3, the next carry
    fadd    v6.2d, v6.2d, v5.2d     // out2
    stp     q4, q5, [x1]
    stp     q6, q16, [x1, #32]
    fmla    v18.2d, v4.2d, v4.2d
    fmla    v19.2d, v5.2d, v5.2d
    fmla    v20.2d, v6.2d, v6.2d
    fmla    v21.2d, v16.2d, v16.2d
    ldp     q0, q1, [x0, #64]
    ldp     q2, q17, [x0, #96]
    ext     v4.16b, v3.16b, v0.16b, #8
    ldur    q5, [x0, #72]
    ldur    q6, [x0, #88]
    ldur    q7, [x0, #104]
    fadd    v4.2d, v4.2d, v0.2d
    fadd    v5.2d, v5.2d, v1.2d
    fadd    v6.2d, v6.2d, v2.2d
    fadd    v7.2d, v7.2d, v17.2d
    fadd    v5.2d, v5.2d, v4.2d
    fadd    v7.2d, v7.2d, v6.2d
    fadd    v7.2d, v7.2d, v5.2d
    fadd    v4.2d, v4.2d, v16.2d
    fadd    v5.2d, v5.2d, v16.2d
    fadd    v16.2d, v16.2d, v7.2d
    fadd    v6.2d, v6.2d, v5.2d
    stp     q4, q5, [x1, #64]
    stp     q6, q16, [x1, #96]
    fmla    v18.2d, v4.2d, v4.2d
    fmla    v19.2d, v5.2d, v5.2d
    fmla    v20.2d, v6.2d, v6.2d
    fmla    v21.2d, v16.2d, v16.2d
    add     x0, x0, #128
    add     x1, x1, #128
    subs    x3, x3, #16
    b.hs    Lmain
Lrest:
    tbz     x3, #3, Ltail           // bit 3 of the count: 8 or more left
    ldp     q0, q1, [x0]
    ldp     q2, q3, [x0, #32]
    ext     v4.16b, v17.16b, v0.16b, #8
    ldur    q5, [x0, #8]
    ldur    q6, [x0, #24]
    ldur    q7, [x0, #40]
    fadd    v4.2d, v4.2d, v0.2d
    fadd    v5.2d, v5.2d, v1.2d
    fadd    v6.2d, v6.2d, v2.2d
    fadd    v7.2d, v7.2d, v3.2d
    fadd    v5.2d, v5.2d, v4.2d
    fadd    v7.2d, v7.2d, v6.2d
    fadd    v7.2d, v7.2d, v5.2d
    fadd    v4.2d, v4.2d, v16.2d
    fadd    v5.2d, v5.2d, v16.2d
    fadd    v16.2d, v16.2d, v7.2d
    fadd    v6.2d, v6.2d, v5.2d
    stp     q4, q5, [x1]
    stp     q6, q16, [x1, #32]
    fmla    v18.2d, v4.2d, v4.2d
    fmla    v19.2d, v5.2d, v5.2d
    fmla    v20.2d, v6.2d, v6.2d
    fmla    v21.2d, v16.2d, v16.2d
    add     x0, x0, #64
    add     x1, x1, #64
Ltail:
    ands    x3, x3, #7              // 0 to 7 elements left; only the last chunk has any
    b.eq    Lcheck
    mov     d0, v16.d[1]            // running sum = last output (-0.0 before the array)
Ltail_loop:
    ldr     d1, [x0], #8
    fadd    d0, d0, d1              // a scalar result clears the upper half of v0
    str     d0, [x1], #8
    fmla    v18.2d, v0.2d, v0.2d    // so this adds [sum^2, 0]
    subs    x3, x3, #1
    b.ne    Ltail_loop
Lcheck:
    fadd    v18.2d, v18.2d, v19.2d
    fadd    v20.2d, v20.2d, v21.2d
    fadd    v18.2d, v18.2d, v20.2d
    faddp   d0, v18.2d              // sum of the squares of every output of the chunk
    fsub    d0, d0, d0              // 0.0 if that is finite, NaN if not
    fcmp    d0, d0
    b.vs    Lredo
    cbnz    x2, Lchunk
Lfinish:
    str     x11, [x7]               // out[0] = x[0], bit for bit
Lret:
    ret

// The chunk just done held a NaN, an infinity or a value above sqrt(DBL_MAX).
// Do its x9 elements again one at a time, as the header describes.
Lredo:
    sub     x0, x0, x9, lsl #3
    sub     x1, x1, x9, lsl #3
    mov     x10, #0x7fe0000000000000
    fmov    d3, x10                 // 2^1023
    mov     x10, #0x8000000000000000
    fmov    d0, x10                 // -0.0, the sum before element 0
    cmp     x1, x7
    b.eq    Lredo_loop
    ldur    d0, [x1, #-8]           // the sum so far: the last output before the chunk
Lredo_loop:
    ldr     d1, [x0], #8
    fadd    d0, d0, d1
    fabs    d2, d0
    fcmp    d2, d3
    b.hi    Lredo_big               // above 2^1023, infinite or NaN
    str     d0, [x1], #8
    subs    x9, x9, #1
    b.ne    Lredo_loop
    cbz     x2, Lfinish
    ldur    q16, [x1, #-16]         // back to the fast path: last two outputs
    ldur    q17, [x0, #-16]         // and last two inputs
    b       Lchunk
Lredo_big:
    fsub    d2, d1, d1              // NaN if x[i] is NaN or infinite
    fcmp    d2, d2
    b.vc    Lrestart                // x[i] is finite: the sum is too close to overflow
    str     d0, [x1], #8            // the sum is not finite from here on
    add     x9, x9, x2              // all the elements left, this one included
    subs    x9, x9, #1
    b.eq    Lfinish
Lplain:                             // the reference loop: one fadd per element, in order
    ldr     d1, [x0], #8
    fadd    d0, d0, d1
    str     d0, [x1], #8
    subs    x9, x9, #1
    b.ne    Lplain
    b       Lfinish
Lrestart:                           // start again at element 0 with the reference loop
    sub     x10, x1, x7             // bytes of output before the element just loaded
    add     x9, x9, x2
    add     x9, x9, x10, lsr #3     // n
    sub     x0, x0, #8
    sub     x0, x0, x10
    mov     x1, x7
    mov     x10, #0x8000000000000000
    fmov    d0, x10                 // -0.0 + x[0] is x[0]
    b       Lplain

// Fewer than SMALL elements: one fadd chain in the reference order, elements
// loaded and stored in pairs. Exact, so no check.
Lsmall:
    cmp     x2, #8
    b.lo    Lshort
    ldp     d1, d2, [x0]
    ldp     d3, d4, [x0, #16]
    ldp     d5, d6, [x0, #32]
    ldp     d7, d16, [x0, #48]
    fadd    d2, d1, d2              // x[0] + x[1]; x[0] itself is stored as loaded
    fadd    d3, d2, d3
    fadd    d4, d3, d4
    fadd    d5, d4, d5
    fadd    d6, d5, d6
    fadd    d7, d6, d7
    fadd    d0, d7, d16
    stp     d1, d2, [x1]
    stp     d3, d4, [x1, #16]
    stp     d5, d6, [x1, #32]
    stp     d7, d0, [x1, #48]
    add     x0, x0, #64
    add     x1, x1, #64
    subs    x2, x2, #16             // 8 done; at least 8 more?
    b.lo    Lsmall_rest
Lsmall_block:
    ldp     d1, d2, [x0]
    ldp     d3, d4, [x0, #16]
    ldp     d5, d6, [x0, #32]
    ldp     d7, d16, [x0, #48]
    fadd    d1, d0, d1              // sum + x[i], same operand order as the C
    fadd    d2, d1, d2
    fadd    d3, d2, d3
    fadd    d4, d3, d4
    fadd    d5, d4, d5
    fadd    d6, d5, d6
    fadd    d7, d6, d7
    fadd    d0, d7, d16
    stp     d1, d2, [x1]
    stp     d3, d4, [x1, #16]
    stp     d5, d6, [x1, #32]
    stp     d7, d0, [x1, #48]
    add     x0, x0, #64
    add     x1, x1, #64
    subs    x2, x2, #8
    b.hs    Lsmall_block
Lsmall_rest:
    tst     x2, #7                  // 0 to 7 elements left; the running sum is in d0
    b.eq    Lsmall_done
    tbz     x2, #2, Lsmall_no4
    ldp     d1, d2, [x0]
    ldp     d3, d4, [x0, #16]
    fadd    d1, d0, d1
    fadd    d2, d1, d2
    fadd    d3, d2, d3
    fadd    d0, d3, d4
    stp     d1, d2, [x1]
    stp     d3, d0, [x1, #16]
    add     x0, x0, #32
    add     x1, x1, #32
Lsmall_no4:
    tbz     x2, #1, Lsmall_no2
    ldp     d1, d2, [x0]
    fadd    d1, d0, d1
    fadd    d0, d1, d2
    stp     d1, d0, [x1]
    add     x0, x0, #16
    add     x1, x1, #16
Lsmall_no2:
    tbz     x2, #0, Lsmall_done
    ldr     d1, [x0]
    fadd    d0, d0, d1
    str     d0, [x1]
Lsmall_done:
    ret
Lshort:                             // fewer than 8 elements
    cbz     x2, Lsmall_done
    ldr     d0, [x0], #8            // running sum = x[0], copied
    str     d0, [x1], #8
    subs    x2, x2, #1
    b.eq    Lsmall_done
Lshort_loop:
    ldr     d1, [x0], #8
    fadd    d0, d0, d1
    str     d0, [x1], #8
    subs    x2, x2, #1
    b.ne    Lshort_loop
    ret
