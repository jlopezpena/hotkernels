// void cumsum_i64(const int64_t *x, int64_t *out, size_t n);
// out[i] = x[0] + ... + x[i], wrapping on overflow.
//
// Blocks of four. Inside a block the partial sums a, a+b, a+b+c, a+b+c+d are
// built from the block's inputs only, so they do not wait for the running sum
// and the core can work on several blocks at once. The running sum is added
// when storing and advances by one add per block. Integer addition is
// associative, so the regrouping is exact, including wrap-around.
//
// The main loop does two such blocks per iteration to halve the loop overhead.
.text
.globl _cumsum_i64
.p2align 2
_cumsum_i64:
    mov     x3, #0                  // running sum of everything before the block
    subs    x2, x2, #8
    b.lo    Lblock4
Lblock8:
    ldp     x4, x5, [x0]            // a, b
    ldp     x6, x7, [x0, #16]       // c, d
    ldp     x8, x9, [x0, #32]       // e, f
    ldp     x10, x11, [x0, #48]     // g, h
    add     x0, x0, #64
    add     x5, x4, x5              // a+b
    add     x7, x6, x7              // c+d
    add     x9, x8, x9              // e+f
    add     x11, x10, x11           // g+h
    add     x4, x4, x3              // out0 = sum + a
    add     x7, x5, x7              // a+b+c+d
    add     x5, x5, x3              // out1 = sum + a+b
    add     x11, x9, x11            // e+f+g+h
    add     x6, x5, x6              // out2 = out1 + c
    add     x12, x3, x7             // out3 = sum + a+b+c+d
    stp     x4, x5, [x1]
    add     x8, x8, x12             // out4 = out3 + e
    add     x9, x9, x12             // out5 = out3 + e+f
    stp     x6, x12, [x1, #16]
    add     x10, x9, x10            // out6 = out5 + g
    add     x3, x12, x11            // out7 = out3 + e+f+g+h, the next running sum
    stp     x8, x9, [x1, #32]
    stp     x10, x3, [x1, #48]
    add     x1, x1, #64
    subs    x2, x2, #8
    b.hs    Lblock8
Lblock4:
    tbz     x2, #2, Ltail           // x2 is (elements left) - 8; bit 2 set means 4 or more left
    ldp     x4, x5, [x0]
    ldp     x6, x7, [x0, #16]
    add     x0, x0, #32
    add     x5, x4, x5
    add     x7, x6, x7
    add     x4, x4, x3
    add     x7, x5, x7
    add     x5, x5, x3
    add     x6, x5, x6
    add     x3, x3, x7
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
