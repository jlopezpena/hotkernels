// void cumsum_f32(const float *x, float *out, size_t n);
// out[i] = x[0] + ... + x[i] with the additions regrouped, so finite results
// differ from the left-to-right sum by rounding only. NaN and infinity appear
// exactly where the left-to-right sum has them. out[0] is a copy of x[0].
//
// Fast path. NEON, four elements per vector. Consecutive output vectors differ
// by the sum of the four elements ending at each lane,
//     out[i..i+3] - out[i-4..i-1] = W = x[i..i+3] + x[i-1..i+2] + x[i-2..i+1] + x[i-3..i].
// W is built in two additions: p = x[i..i+3] + x[i-1..i+2] (the input vector
// plus an overlapping load one element back, or an ext from the previous input
// vector for the first vector of a block), then W = p + (p two elements back),
// the second operand being an ext across the previous p and this one. Before
// the array every value is -0.0, which changes nothing it is added to.
// A block is four vectors. Its four W are summed from the inputs only, and
// v16, the last output vector, is added to each:
//     out0 = v16 + W0              out1 = v16 + (W0 + W1)
//     out2 = out1 + W2             out3 = v16 + ((W0 + W1) + (W2 + W3))
// so the carried chain is one fadd per 16 elements, not one per element. The
// four lanes are four separate running sums; each out[i] is still the sum of
// x[0..i] in some order.
//
// Why non-finite positions match the left-to-right sum, for every input.
// The array is done in chunks of CHUNK elements. In a chunk every output vector
// is squared and accumulated (fmla into v18-v21), and the chunk is accepted
// only if that sum of squares is finite. If it is finite then:
//   1. Every output of the chunk is finite and at most sqrt(FLT_MAX) < 2^64.
//   2. Every input and every intermediate of the chunk is finite: each one is
//      an operand, directly or through further additions, of an output of its
//      own block (p0 and W0 of out0; p1, W1 and W0+W1 of out1; p2 and W2 of
//      out2; p3, W3, W2+W3 and the block total of out3), and an addition with
//      a NaN or an infinite operand never gives a finite result.
//   3. The left-to-right sum is finite on the chunk too. With S_j the exact
//      sum of x[0..j]: every accepted output is at most T = 2^64 and is a
//      rounded sum whose intermediate values are outputs or differences of
//      partial sums, so |S_j| <= T * (1 + j * 2^-23). The left-to-right sum
//      s_j picks up at most 2^-24 * |s_k| of rounding error at each step k, so
//      |s_j| <= |S_j| * e^(j * 2^-24). For j below 2^29 that is under
//      T * 65 * e^32 < 2^117, far from overflow.
// Both results are finite on an accepted chunk, so no position differs.
// If the sum of squares is not finite, the chunk is done again one element at
// a time from the last output of the chunk before it (Lredo), watching every
// element:
//   - x[i] finite and |sum| <= 2^64: store it. Point 3 applies to it as well.
//   - x[i] is NaN or infinite: the left-to-right sum was finite up to here
//     (previous case), so from this element on it is not finite and which of
//     NaN, +inf and -inf it is depends only on the inputs from here on, never
//     on the finite value before. The rest of the array is finished with the
//     plain left-to-right loop (Lplain), which gives the same kinds.
//   - x[i] finite but |sum| > 2^64 or overflowed: point 3 no longer covers
//     the left-to-right sum, so the whole array is redone from element 0 with
//     the plain loop (Lrestart), which is the reference computation itself.
// A redone chunk that stayed in the first case is followed by the fast path
// again. Point 3 needs fewer than 2^29 elements: over a longer array the
// rounding of the left-to-right float sum could in principle carry it to
// overflow from any starting size. Arrays of 2^29 elements or more are
// therefore done entirely with the plain loop. The price of the square test
// is that running sums above about 4e18 in size, though finite, take the slow
// path.
//
// Short arrays. Fewer than SMALL elements are added strictly left to right
// (Lsmall), which is the reference order, so nothing needs checking. That
// chain costs 3 cycles of latency per element, but it is fewer operations than
// a block plus the check, and when calls do not depend on each other the
// processor overlaps them, so the count is what matters. Measured, the chain
// wins at 16 elements, ties at 20 and loses from 24.
//
// Registers: x0 x, x1 out, x2 elements after this chunk, x3 chunk counter,
// x7 start of out, x9 elements in this chunk, w11 bits of x[0],
// v16 last output vector, v17 last input vector, v23 last p, v18-v21 sums of
// squares, v0-v7 inputs and p of the block being done, v24-v27 its W and
// outputs (v22, v28-v30 for the second block of a loop iteration).
.set CHUNK, 1024
// Arrays of at least 2^LONG_LOG2 elements use the plain loop (point 3 above).
.set LONG_LOG2, 29
// Below this many elements the plain chain is used (see Lsmall).
.set SMALL, 20
.text
.globl _cumsum_f32
.p2align 2
_cumsum_f32:
    cmp     x2, #SMALL
    b.lo    Lsmall
    lsr     x10, x2, #LONG_LOG2
    cbnz    x10, Llong
    ldr     w11, [x0]               // out[0] is stored from this at the end, as a copy
    mov     x7, x1
    movi    v16.4s, #0x80, lsl #24  // -0.0 in every lane: -0.0 + v is v for every v
    mov     v17.16b, v16.16b
    mov     v23.16b, v16.16b
Lchunk:
    movi    v18.2d, #0
    movi    v19.2d, #0
    movi    v20.2d, #0
    movi    v21.2d, #0
    mov     x3, #CHUNK
    cmp     x2, x3
    csel    x9, x2, x3, lo          // elements in this chunk
    sub     x2, x2, x9
    subs    x3, x9, #32
    b.lo    Lrest
Lmain:
    // Both blocks' loads and block sums come first (the W of the second block
    // in v22, v28, v29, v30); the additions that wait for the carry follow.
    ldp     q0, q1, [x0]
    ldp     q2, q3, [x0, #32]
    ext     v4.16b, v17.16b, v0.16b, #12    // x[i-1..i+2]
    ldur    q5, [x0, #12]
    ldur    q6, [x0, #28]
    ldur    q7, [x0, #44]
    fadd    v4.4s, v4.4s, v0.4s     // p0
    fadd    v5.4s, v5.4s, v1.4s     // p1
    fadd    v6.4s, v6.4s, v2.4s     // p2
    fadd    v7.4s, v7.4s, v3.4s     // p3
    ext     v24.16b, v23.16b, v4.16b, #8    // p two elements back
    ext     v25.16b, v4.16b, v5.16b, #8
    ext     v26.16b, v5.16b, v6.16b, #8
    ext     v27.16b, v6.16b, v7.16b, #8
    fadd    v24.4s, v24.4s, v4.4s   // W0
    fadd    v25.4s, v25.4s, v5.4s   // W1
    fadd    v26.4s, v26.4s, v6.4s   // W2
    fadd    v27.4s, v27.4s, v7.4s   // W3
    fadd    v25.4s, v25.4s, v24.4s  // W0 + W1
    fadd    v27.4s, v27.4s, v26.4s  // W2 + W3
    fadd    v27.4s, v27.4s, v25.4s  // block total
    ldp     q0, q1, [x0, #64]
    ldp     q2, q17, [x0, #96]
    ext     v4.16b, v3.16b, v0.16b, #12
    ldur    q5, [x0, #76]
    ldur    q6, [x0, #92]
    ldur    q23, [x0, #108]
    fadd    v4.4s, v4.4s, v0.4s
    fadd    v5.4s, v5.4s, v1.4s
    fadd    v6.4s, v6.4s, v2.4s
    fadd    v23.4s, v23.4s, v17.4s
    ext     v22.16b, v7.16b, v4.16b, #8
    ext     v28.16b, v4.16b, v5.16b, #8
    ext     v29.16b, v5.16b, v6.16b, #8
    ext     v30.16b, v6.16b, v23.16b, #8
    fadd    v22.4s, v22.4s, v4.4s
    fadd    v28.4s, v28.4s, v5.4s
    fadd    v29.4s, v29.4s, v6.4s
    fadd    v30.4s, v30.4s, v23.4s
    fadd    v28.4s, v28.4s, v22.4s
    fadd    v30.4s, v30.4s, v29.4s
    fadd    v30.4s, v30.4s, v28.4s
    fadd    v24.4s, v24.4s, v16.4s  // out0
    fadd    v25.4s, v25.4s, v16.4s  // out1
    fadd    v16.4s, v16.4s, v27.4s  // out3, the next carry
    fadd    v26.4s, v26.4s, v25.4s  // out2
    stp     q24, q25, [x1]
    stp     q26, q16, [x1, #32]
    fmla    v18.4s, v24.4s, v24.4s
    fmla    v19.4s, v25.4s, v25.4s
    fmla    v20.4s, v26.4s, v26.4s
    fmla    v21.4s, v16.4s, v16.4s
    fadd    v22.4s, v22.4s, v16.4s
    fadd    v28.4s, v28.4s, v16.4s
    fadd    v16.4s, v16.4s, v30.4s
    fadd    v29.4s, v29.4s, v28.4s
    stp     q22, q28, [x1, #64]
    stp     q29, q16, [x1, #96]
    fmla    v18.4s, v22.4s, v22.4s
    fmla    v19.4s, v28.4s, v28.4s
    fmla    v20.4s, v29.4s, v29.4s
    fmla    v21.4s, v16.4s, v16.4s
    add     x0, x0, #128
    add     x1, x1, #128
    subs    x3, x3, #32
    b.hs    Lmain
Lrest:
    tbz     x3, #4, Lvectors        // bit 4 of the count: 16 or more left
    ldp     q0, q1, [x0]
    ext     v4.16b, v17.16b, v0.16b, #12
    ldp     q2, q17, [x0, #32]
    ldur    q5, [x0, #12]
    ldur    q6, [x0, #28]
    fadd    v4.4s, v4.4s, v0.4s
    fadd    v5.4s, v5.4s, v1.4s
    fadd    v6.4s, v6.4s, v2.4s
    ext     v24.16b, v23.16b, v4.16b, #8
    ldur    q23, [x0, #44]
    fadd    v23.4s, v23.4s, v17.4s
    ext     v25.16b, v4.16b, v5.16b, #8
    ext     v26.16b, v5.16b, v6.16b, #8
    ext     v27.16b, v6.16b, v23.16b, #8
    fadd    v24.4s, v24.4s, v4.4s
    fadd    v25.4s, v25.4s, v5.4s
    fadd    v26.4s, v26.4s, v6.4s
    fadd    v27.4s, v27.4s, v23.4s
    fadd    v25.4s, v25.4s, v24.4s
    fadd    v27.4s, v27.4s, v26.4s
    fadd    v27.4s, v27.4s, v25.4s
    fadd    v24.4s, v24.4s, v16.4s
    fadd    v25.4s, v25.4s, v16.4s
    fadd    v16.4s, v16.4s, v27.4s
    fadd    v26.4s, v26.4s, v25.4s
    stp     q24, q25, [x1]
    stp     q26, q16, [x1, #32]
    fmla    v18.4s, v24.4s, v24.4s
    fmla    v19.4s, v25.4s, v25.4s
    fmla    v20.4s, v26.4s, v26.4s
    fmla    v21.4s, v16.4s, v16.4s
    add     x0, x0, #64
    add     x1, x1, #64
Lvectors:
    ands    x3, x3, #15             // 0 to 15 elements left; only the last chunk has any
    b.eq    Lcheck
    lsr     x10, x3, #2             // whole vectors among them
    cbz     x10, Lscalars
Lvector_loop:                       // one vector at a time: out = v16 + W
    ldr     q0, [x0], #16
    ext     v4.16b, v17.16b, v0.16b, #12
    fadd    v4.4s, v4.4s, v0.4s     // p
    ext     v24.16b, v23.16b, v4.16b, #8
    fadd    v24.4s, v24.4s, v4.4s   // W
    fadd    v16.4s, v16.4s, v24.4s
    str     q16, [x1], #16
    fmla    v18.4s, v16.4s, v16.4s
    mov     v17.16b, v0.16b
    mov     v23.16b, v4.16b
    subs    x10, x10, #1
    b.ne    Lvector_loop
Lscalars:
    ands    x3, x3, #3              // 0 to 3 elements left
    b.eq    Lcheck
    mov     s0, v16.s[3]            // running sum = last output (-0.0 before the array)
Lscalar_loop:
    ldr     s1, [x0], #4
    fadd    s0, s0, s1              // a scalar result clears the rest of v0
    str     s0, [x1], #4
    fmla    v18.4s, v0.4s, v0.4s    // so this adds [sum^2, 0, 0, 0]
    subs    x3, x3, #1
    b.ne    Lscalar_loop
Lcheck:
    fadd    v18.4s, v18.4s, v19.4s
    fadd    v20.4s, v20.4s, v21.4s
    fadd    v18.4s, v18.4s, v20.4s
    // The four lanes hold sums of squares: +0.0 or more, or infinite, or NaN.
    // As unsigned integers the bits of a finite one are below those of +inf
    // (0x7f800000) and those of an infinity or a NaN are not, so the largest
    // bit pattern has an all-ones exponent exactly when some lane is not finite.
    umaxv   s0, v18.4s
    fmov    w10, s0
    lsr     w10, w10, #23           // sign and exponent
    cmp     w10, #0xff
    b.hs    Lredo
    cbnz    x2, Lchunk
Lfinish:
    str     w11, [x7]               // out[0] = x[0], bit for bit
Lret:
    ret

// The chunk just done held a NaN, an infinity or a value above sqrt(FLT_MAX).
// Do its x9 elements again one at a time, as the header describes.
Lredo:
    sub     x0, x0, x9, lsl #2
    sub     x1, x1, x9, lsl #2
    mov     w10, #0x5f800000
    fmov    s3, w10                 // 2^64
    mov     w10, #0x80000000
    fmov    s0, w10                 // -0.0, the sum before element 0
    cmp     x1, x7
    b.eq    Lredo_loop
    ldur    s0, [x1, #-4]           // the sum so far: the last output before the chunk
Lredo_loop:
    ldr     s1, [x0], #4
    fadd    s0, s0, s1
    fabs    s2, s0
    fcmp    s2, s3
    b.hi    Lredo_big               // above 2^64, infinite or NaN
    str     s0, [x1], #4
    subs    x9, x9, #1
    b.ne    Lredo_loop
    cbz     x2, Lfinish
    ldur    q16, [x1, #-16]         // back to the fast path: last four outputs,
    ldur    q17, [x0, #-16]         // last four inputs
    ldur    q23, [x0, #-20]
    fadd    v23.4s, v23.4s, v17.4s  // and their p
    b       Lchunk
Lredo_big:
    fsub    s2, s1, s1              // NaN if x[i] is NaN or infinite
    fcmp    s2, s2
    b.vc    Lrestart                // x[i] is finite: the sum is too large to trust
    str     s0, [x1], #4            // the sum is not finite from here on
    add     x9, x9, x2              // all the elements left, this one included
    subs    x9, x9, #1
    b.eq    Lfinish
Lplain:                             // the reference loop: one fadd per element, in order
    ldr     s1, [x0], #4
    fadd    s0, s0, s1
    str     s0, [x1], #4
    subs    x9, x9, #1
    b.ne    Lplain
    b       Lfinish
Lrestart:                           // start again at element 0 with the reference loop
    sub     x10, x1, x7             // bytes of output before the element just loaded
    add     x9, x9, x2
    add     x9, x9, x10, lsr #2     // n
    sub     x0, x0, #4
    sub     x0, x0, x10
    mov     x1, x7
Lrestart_sum:
    mov     w10, #0x80000000
    fmov    s0, w10                 // -0.0 + x[0] is x[0]
    b       Lplain
Llong:                              // 2^29 elements or more: the reference loop throughout
    ldr     w11, [x0]
    mov     x7, x1
    mov     x9, x2
    b       Lrestart_sum

// Fewer than SMALL elements: one fadd chain in the reference order, elements
// loaded and stored in pairs. Exact, so no check.
Lsmall:
    cmp     x2, #8
    b.lo    Lshort
    ldp     s1, s2, [x0]
    ldp     s3, s4, [x0, #8]
    ldp     s5, s6, [x0, #16]
    ldp     s7, s16, [x0, #24]
    fadd    s2, s1, s2              // x[0] + x[1]; x[0] itself is stored as loaded
    fadd    s3, s2, s3
    fadd    s4, s3, s4
    fadd    s5, s4, s5
    fadd    s6, s5, s6
    fadd    s7, s6, s7
    fadd    s0, s7, s16
    stp     s1, s2, [x1]
    stp     s3, s4, [x1, #8]
    stp     s5, s6, [x1, #16]
    stp     s7, s0, [x1, #24]
    add     x0, x0, #32
    add     x1, x1, #32
    subs    x2, x2, #16             // 8 done; at least 8 more?
    b.lo    Lsmall_rest
Lsmall_block:
    ldp     s1, s2, [x0]
    ldp     s3, s4, [x0, #8]
    ldp     s5, s6, [x0, #16]
    ldp     s7, s16, [x0, #24]
    fadd    s1, s0, s1              // sum + x[i], same operand order as the C
    fadd    s2, s1, s2
    fadd    s3, s2, s3
    fadd    s4, s3, s4
    fadd    s5, s4, s5
    fadd    s6, s5, s6
    fadd    s7, s6, s7
    fadd    s0, s7, s16
    stp     s1, s2, [x1]
    stp     s3, s4, [x1, #8]
    stp     s5, s6, [x1, #16]
    stp     s7, s0, [x1, #24]
    add     x0, x0, #32
    add     x1, x1, #32
    subs    x2, x2, #8
    b.hs    Lsmall_block
Lsmall_rest:
    tst     x2, #7                  // 0 to 7 elements left; the running sum is in s0
    b.eq    Lsmall_done
    tbz     x2, #2, Lsmall_no4
    ldp     s1, s2, [x0]
    ldp     s3, s4, [x0, #8]
    fadd    s1, s0, s1
    fadd    s2, s1, s2
    fadd    s3, s2, s3
    fadd    s0, s3, s4
    stp     s1, s2, [x1]
    stp     s3, s0, [x1, #8]
    add     x0, x0, #16
    add     x1, x1, #16
Lsmall_no4:
    tbz     x2, #1, Lsmall_no2
    ldp     s1, s2, [x0]
    fadd    s1, s0, s1
    fadd    s0, s1, s2
    stp     s1, s0, [x1]
    add     x0, x0, #8
    add     x1, x1, #8
Lsmall_no2:
    tbz     x2, #0, Lsmall_done
    ldr     s1, [x0]
    fadd    s0, s0, s1
    str     s0, [x1]
Lsmall_done:
    ret
Lshort:                             // fewer than 8 elements
    cbz     x2, Lsmall_done
    ldr     s0, [x0], #4            // running sum = x[0], copied
    str     s0, [x1], #4
    subs    x2, x2, #1
    b.eq    Lsmall_done
Lshort_loop:
    ldr     s1, [x0], #4
    fadd    s0, s0, s1
    str     s0, [x1], #4
    subs    x2, x2, #1
    b.ne    Lshort_loop
    ret
