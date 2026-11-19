// void cumsum_bool(const bool *mask, int64_t *out, size_t n);
// out[i] = how many of mask[0..i] are true. Every mask byte is 0 or 1.
//
// x0 = mask, x1 = out, x2 = n. One element at a time.
.text
.globl _cumsum_bool
.p2align 2
_cumsum_bool:
    cbz     x2, Ldone
    mov     x3, #0                      // the count so far
Lone:
    ldrb    w4, [x0], #1
    add     x3, x3, x4
    str     x3, [x1], #8
    subs    x2, x2, #1
    b.ne    Lone
Ldone:
    ret
