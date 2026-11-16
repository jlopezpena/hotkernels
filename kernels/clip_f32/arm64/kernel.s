// void clip_f32(const float *x, float *out, size_t n, float lo, float hi);
// out[i] = x[i] < lo ? lo : (x[i] > hi ? hi : x[i])
//
// x0 = x, x1 = out, x2 = n, s0 = lo, s1 = hi.
//
// Compare and select, exactly as the C expression: two fcmgt and two bit per
// vector. Both compares are false for a NaN element, so it is copied
// unchanged, bit for bit; the same holds for a NaN bound. Nothing here is
// arithmetic, so signed zeros and signalling NaNs come out as they went in.
//
// A tail that is not a whole 16-element block is one more block that ends
// exactly at the end of the array and overlaps the previous one. It is
// computed from the input again, never from output already written.
.text
.globl _clip_f32
.p2align 2
_clip_f32:
    dup.4s  v2, v0[0]               // lo in every lane
    dup.4s  v3, v1[0]               // hi in every lane
    subs    x8, x2, #16             // x8 = elements left after this block
    b.lo    Lsmall
Lblock:
    ldp     q4, q5, [x0]
    ldp     q6, q7, [x0, #32]
    fcmgt.4s v16, v2, v4            // x < lo
    fcmgt.4s v17, v2, v5
    fcmgt.4s v18, v2, v6
    fcmgt.4s v19, v2, v7
    fcmgt.4s v20, v4, v3            // x > hi
    fcmgt.4s v21, v5, v3
    fcmgt.4s v22, v6, v3
    fcmgt.4s v23, v7, v3
    bit.16b v4, v3, v20             // hi where x > hi
    bit.16b v5, v3, v21
    bit.16b v6, v3, v22
    bit.16b v7, v3, v23
    bit.16b v4, v2, v16             // lo where x < lo; applied last, as in C
    bit.16b v5, v2, v17
    bit.16b v6, v2, v18
    bit.16b v7, v2, v19
    stp     q4, q5, [x1]
    stp     q6, q7, [x1, #32]
    add     x0, x0, #64
    add     x1, x1, #64
    subs    x8, x8, #16
    b.hs    Lblock
    adds    x8, x8, #16             // elements left: 0 to 15
    b.eq    Ldone
    // Step back so that one more block ends exactly at the end.
    add     x0, x0, x8, lsl #2
    add     x1, x1, x8, lsl #2
    sub     x0, x0, #64
    sub     x1, x1, #64
    mov     x8, #0
    b       Lblock

Lsmall:                             // n < 16
    cmp     x2, #4
    b.lo    Lscalar
    add     x9, x0, x2, lsl #2      // end of x
    add     x10, x1, x2, lsl #2     // end of out
    tbz     x2, #3, Lfour
    // 8 <= n < 16: the first eight elements and the last eight, overlapping.
    ldp     q4, q5, [x0]
    ldp     q6, q7, [x9, #-32]
    fcmgt.4s v16, v2, v4
    fcmgt.4s v17, v2, v5
    fcmgt.4s v18, v2, v6
    fcmgt.4s v19, v2, v7
    fcmgt.4s v20, v4, v3
    fcmgt.4s v21, v5, v3
    fcmgt.4s v22, v6, v3
    fcmgt.4s v23, v7, v3
    bit.16b v4, v3, v20
    bit.16b v5, v3, v21
    bit.16b v6, v3, v22
    bit.16b v7, v3, v23
    bit.16b v4, v2, v16
    bit.16b v5, v2, v17
    bit.16b v6, v2, v18
    bit.16b v7, v2, v19
    stp     q4, q5, [x1]
    stp     q6, q7, [x10, #-32]
    ret
Lfour:
    // 4 <= n < 8: the first four elements and the last four, overlapping.
    ldr     q4, [x0]
    ldur    q5, [x9, #-16]
    fcmgt.4s v16, v2, v4
    fcmgt.4s v17, v2, v5
    fcmgt.4s v20, v4, v3
    fcmgt.4s v21, v5, v3
    bit.16b v4, v3, v20
    bit.16b v5, v3, v21
    bit.16b v4, v2, v16
    bit.16b v5, v2, v17
    str     q4, [x1]
    stur    q5, [x10, #-16]
Ldone:
    ret

Lscalar:                            // n < 4
    cbz     x2, Ldone
Lone:
    ldr     s4, [x0], #4
    fcmp    s4, s1
    fcsel   s5, s1, s4, gt          // x > hi ? hi : x
    fcmp    s4, s0
    fcsel   s4, s0, s5, mi          // x < lo ? lo : that
    str     s4, [x1], #4
    subs    x2, x2, #1
    b.ne    Lone
    ret
