// void cumsum_i64(const int64_t *x, int64_t *out, size_t n);
// out[i] = x[0] + ... + x[i], wrapping on overflow.
//
// Blocks of four. Inside a block the partial sums a, a+b, a+b+c, a+b+c+d are
// built from the block's inputs only, so they do not wait for the running sum
// and the core can work on several blocks at once. The running sum is added
// when storing and advances by one add per block. Integer addition is
// associative, so the regrouping is exact, including wrap-around.
.text
.globl _cumsum_i64
.p2align 2
_cumsum_i64:
    mov     x3, #0                  // running sum of everything before the block
    subs    x2, x2, #4
    b.lo    Ltail
Lblock:
    ldp     x4, x5, [x0]            // a, b
    ldp     x6, x7, [x0, #16]       // c, d
    add     x0, x0, #32
    add     x5, x4, x5              // a+b
    add     x7, x6, x7              // c+d
    add     x4, x4, x3              // out0 = sum + a
    add     x7, x5, x7              // a+b+c+d
    add     x5, x5, x3              // out1 = sum + a+b
    add     x6, x5, x6              // out2 = out1 + c
    add     x3, x3, x7              // out3 = sum + a+b+c+d, the next running sum
    stp     x4, x5, [x1]
    stp     x6, x3, [x1, #16]
    add     x1, x1, #32
    subs    x2, x2, #4
    b.hs    Lblock
Ltail:
    adds    x2, x2, #4              // 0 to 3 elements left
    b.eq    Ldone
Ltail_loop:
    ldr     x4, [x0], #8
    add     x3, x3, x4
    str     x3, [x1], #8
    subs    x2, x2, #1
    b.ne    Ltail_loop
Ldone:
    ret
