// void cummax_i64(const int64_t *x, int64_t *out, size_t n);
// out[i] = the largest of x[0..i], compared as signed numbers.
// x0 = x, x1 = out, x2 = n. out does not overlap x.
.text
.globl _cummax_i64
.p2align 2
_cummax_i64:
    cbz     x2, Ldone
    ldr     x3, [x0]                // M
Lloop:
    ldr     x4, [x0], #8
    cmp     x4, x3
    csel    x3, x4, x3, gt
    str     x3, [x1], #8
    subs    x2, x2, #1
    b.ne    Lloop
Ldone:
    ret
