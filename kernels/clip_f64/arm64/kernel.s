// void clip_f64(const double *x, double *out, size_t n, double lo, double hi);
// out[i] = x[i] < lo ? lo : (x[i] > hi ? hi : x[i])
//
// x0 = x, x1 = out, x2 = n, d0 = lo, d1 = hi.
//
// Compare and select, exactly as the C expression: two fcmgt and two bit per
// vector. Both compares are false for a NaN element, so it is copied
// unchanged, bit for bit; the same holds for a NaN bound. Nothing here is
// arithmetic, so signed zeros and signalling NaNs come out as they went in.
//
// A tail that is not a whole 8-element block is one more block that ends
// exactly at the end of the array and overlaps the previous one. It is
// computed from the input again, never from output already written.

.text
.globl _clip_f64
.p2align 2
_clip_f64:
    dup.2d  v2, v0[0]               // lo in every lane
    dup.2d  v3, v1[0]               // hi in every lane
    subs    x8, x2, #8              // x8 = elements left after this block
    b.lo    Lsmall
Lblock:
    ldp     q4, q5, [x0]
    ldp     q6, q7, [x0, #32]
    fcmgt.2d v16, v2, v4            // x < lo
    fcmgt.2d v17, v2, v5
    fcmgt.2d v18, v2, v6
    fcmgt.2d v19, v2, v7
    fcmgt.2d v20, v4, v3            // x > hi
    fcmgt.2d v21, v5, v3
    fcmgt.2d v22, v6, v3
    fcmgt.2d v23, v7, v3
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
    subs    x8, x8, #8
    b.hs    Lblock
    adds    x8, x8, #8              // elements left: 0 to 7
    b.eq    Ldone
    // Step back so that one more block ends exactly at the end.
    add     x0, x0, x8, lsl #3
    add     x1, x1, x8, lsl #3
    sub     x0, x0, #64
    sub     x1, x1, #64
    mov     x8, #0
    b       Lblock

Lsmall:                             // n < 8
    cmp     x2, #2
    b.lo    Lscalar
    add     x9, x0, x2, lsl #3      // end of x
    add     x10, x1, x2, lsl #3     // end of out
    tbz     x2, #2, Ltwo
    // 4 <= n < 8: the first four elements and the last four, overlapping.
    ldp     q4, q5, [x0]
    ldp     q6, q7, [x9, #-32]
    fcmgt.2d v16, v2, v4
    fcmgt.2d v17, v2, v5
    fcmgt.2d v18, v2, v6
    fcmgt.2d v19, v2, v7
    fcmgt.2d v20, v4, v3
    fcmgt.2d v21, v5, v3
    fcmgt.2d v22, v6, v3
    fcmgt.2d v23, v7, v3
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
Ltwo:
    // n is 2 or 3: the first two elements and the last two, overlapping.
    ldr     q4, [x0]
    ldur    q5, [x9, #-16]
    fcmgt.2d v16, v2, v4
    fcmgt.2d v17, v2, v5
    fcmgt.2d v20, v4, v3
    fcmgt.2d v21, v5, v3
    bit.16b v4, v3, v20
    bit.16b v5, v3, v21
    bit.16b v4, v2, v16
    bit.16b v5, v2, v17
    str     q4, [x1]
    stur    q5, [x10, #-16]
Ldone:
    ret

Lscalar:                            // n is 0 or 1
    cbz     x2, Ldone
    ldr     d4, [x0]
    fcmp    d4, d1
    fcsel   d5, d1, d4, gt          // x > hi ? hi : x
    fcmp    d4, d0
    fcsel   d4, d0, d5, mi          // x < lo ? lo : that
    str     d4, [x1]
    ret
