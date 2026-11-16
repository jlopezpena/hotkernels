// void cumsum_i64(const int64_t *x, int64_t *out, size_t n);
// out[i] = x[0] + ... + x[i], wrapping on overflow.
//
// NEON, two elements per vector. Consecutive output vectors differ by
//     out[i..i+1] - out[i-2..i-1] = x[i..i+1] + x[i-1..i],
// the sum of the input vector and the input shifted back by one element. The
// shifted vector is a second, overlapping load from x (or, for the first vector
// of a block, an ext from the previous input vector, which also covers i = 0).
//
// Blocks of four vectors. The four steps of a block are prefix-summed from the
// block's inputs only, so they do not wait for the running sum; v16 carries the
// last output vector and advances by one add per block.
.text
.globl _cumsum_i64
.p2align 2
_cumsum_i64:
    cmp     x2, #8
    b.lo    Lsmall
    movi    v16.2d, #0              // [out[i-2], out[i-1]], zero before the start
    movi    v17.2d, #0              // [x[i-2], x[i-1]], zero before the start
    cmp     x2, #8, lsl #12         // from 32,768 elements the loop at Llarge is the faster one
    b.hs    Llarge
    subs    x2, x2, #16
    b.lo    Lrest
Lmain:
    ldp     q0, q1, [x0]
    ldp     q2, q3, [x0, #32]
    ext     v4.16b, v17.16b, v0.16b, #8
    ldur    q5, [x0, #8]
    ldur    q6, [x0, #24]
    ldur    q7, [x0, #40]
    add     v4.2d, v4.2d, v0.2d     // step 0
    add     v5.2d, v5.2d, v1.2d     // step 1
    add     v6.2d, v6.2d, v2.2d     // step 2
    add     v7.2d, v7.2d, v3.2d     // step 3
    add     v5.2d, v5.2d, v4.2d     // steps 0..1
    add     v7.2d, v7.2d, v6.2d     // steps 2..3
    add     v4.2d, v4.2d, v16.2d    // out vector 0
    add     v7.2d, v7.2d, v5.2d     // steps 0..3
    add     v5.2d, v5.2d, v16.2d    // out vector 1
    add     v6.2d, v6.2d, v5.2d     // out vector 2
    add     v16.2d, v16.2d, v7.2d    // out vector 3, the next carry
    stp     q4, q5, [x1]
    stp     q6, q16, [x1, #32]
    ldp     q0, q1, [x0, #64]
    ldp     q2, q17, [x0, #96]
    ext     v4.16b, v3.16b, v0.16b, #8
    ldur    q5, [x0, #72]
    ldur    q6, [x0, #88]
    ldur    q7, [x0, #104]
    add     v4.2d, v4.2d, v0.2d
    add     v5.2d, v5.2d, v1.2d
    add     v6.2d, v6.2d, v2.2d
    add     v7.2d, v7.2d, v17.2d
    add     v5.2d, v5.2d, v4.2d
    add     v7.2d, v7.2d, v6.2d
    add     v4.2d, v4.2d, v16.2d
    add     v7.2d, v7.2d, v5.2d
    add     v5.2d, v5.2d, v16.2d
    add     v6.2d, v6.2d, v5.2d
    add     v16.2d, v16.2d, v7.2d
    stp     q4, q5, [x1, #64]
    stp     q6, q16, [x1, #96]
    add     x0, x0, #128
    add     x1, x1, #128
    subs    x2, x2, #16
    b.hs    Lmain
Lrest:
    tbz     x2, #3, Lneon_done      // bit 3 of the count: 8 or more left
    ldp     q0, q1, [x0]
    ldp     q2, q3, [x0, #32]
    ext     v4.16b, v17.16b, v0.16b, #8
    ldur    q5, [x0, #8]
    ldur    q6, [x0, #24]
    ldur    q7, [x0, #40]
    add     v4.2d, v4.2d, v0.2d
    add     v5.2d, v5.2d, v1.2d
    add     v6.2d, v6.2d, v2.2d
    add     v7.2d, v7.2d, v3.2d
    add     v5.2d, v5.2d, v4.2d
    add     v7.2d, v7.2d, v6.2d
    add     v4.2d, v4.2d, v16.2d
    add     v7.2d, v7.2d, v5.2d
    add     v5.2d, v5.2d, v16.2d
    add     v6.2d, v6.2d, v5.2d
    add     v16.2d, v16.2d, v7.2d
    stp     q4, q5, [x1]
    stp     q6, q16, [x1, #32]
    add     x0, x0, #64
    add     x1, x1, #64
Lneon_done:
    tst     x2, #7
    b.eq    Ldone                   // nothing left: skip the move out of the vector register
    mov     x3, v16.d[1]            // running sum = last output
    b       Lblock4
Lsmall:
    mov     x3, #0
Lblock4:
    tbz     x2, #2, Ltail           // bit 2 of the count: 4 or more left
    ldp     x4, x5, [x0]
    ldp     x6, x7, [x0, #16]
    add     x0, x0, #32
    add     x5, x4, x5              // a+b
    add     x7, x6, x7              // c+d
    add     x4, x4, x3              // out0 = sum + a
    add     x7, x5, x7              // a+b+c+d
    add     x5, x5, x3              // out1 = sum + a+b
    add     x6, x5, x6              // out2 = out1 + c
    add     x3, x3, x7              // out3 = sum + a+b+c+d
    stp     x4, x5, [x1]
    stp     x6, x3, [x1, #16]
    add     x1, x1, #32
Ltail:
    ands    x2, x2, #3              // 0 to 3 elements left
    b.eq    Ldone
Ltail_loop:
    ldr     x4, [x0], #8
    add     x3, x3, x4
    str     x3, [x1], #8
    subs    x2, x2, #1
    b.ne    Ltail_loop
Ldone:
    ret
// Same loop with one overlapping load per block replaced by an ext. It has one
// more vector operation and one load fewer per block, which measured slower up
// to 16,384 elements and faster from 65,536 to 1,048,576.
Llarge:
    sub     x2, x2, #16
Lmain_large:
    ldp     q0, q1, [x0]
    ldp     q2, q3, [x0, #32]
    ext     v4.16b, v17.16b, v0.16b, #8
    ext     v5.16b, v0.16b, v1.16b, #8
    ldur    q6, [x0, #24]
    ldur    q7, [x0, #40]
    add     v4.2d, v4.2d, v0.2d     // step 0
    add     v5.2d, v5.2d, v1.2d     // step 1
    add     v6.2d, v6.2d, v2.2d     // step 2
    add     v7.2d, v7.2d, v3.2d     // step 3
    add     v5.2d, v5.2d, v4.2d     // steps 0..1
    add     v7.2d, v7.2d, v6.2d     // steps 2..3
    add     v4.2d, v4.2d, v16.2d    // out vector 0
    add     v7.2d, v7.2d, v5.2d     // steps 0..3
    add     v5.2d, v5.2d, v16.2d    // out vector 1
    add     v6.2d, v6.2d, v5.2d     // out vector 2
    add     v16.2d, v16.2d, v7.2d    // out vector 3, the next carry
    stp     q4, q5, [x1]
    stp     q6, q16, [x1, #32]
    ldp     q0, q1, [x0, #64]
    ldp     q2, q17, [x0, #96]
    ext     v4.16b, v3.16b, v0.16b, #8
    ext     v5.16b, v0.16b, v1.16b, #8
    ldur    q6, [x0, #88]
    ldur    q7, [x0, #104]
    add     v4.2d, v4.2d, v0.2d
    add     v5.2d, v5.2d, v1.2d
    add     v6.2d, v6.2d, v2.2d
    add     v7.2d, v7.2d, v17.2d
    add     v5.2d, v5.2d, v4.2d
    add     v7.2d, v7.2d, v6.2d
    add     v4.2d, v4.2d, v16.2d
    add     v7.2d, v7.2d, v5.2d
    add     v5.2d, v5.2d, v16.2d
    add     v6.2d, v6.2d, v5.2d
    add     v16.2d, v16.2d, v7.2d
    stp     q4, q5, [x1, #64]
    stp     q6, q16, [x1, #96]
    add     x0, x0, #128
    add     x1, x1, #128
    subs    x2, x2, #16
    b.hs    Lmain_large
    b       Lrest
