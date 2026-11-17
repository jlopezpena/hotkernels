// BROKEN ON PURPOSE: clears at least 16 buckets, the capacity in the default
// scalar set, even when the table it was given is smaller
// void tally_u8(const uint8_t *x, int64_t *counts, size_t n, size_t nbuckets);
// Simple scalar version used to test the judge. Deliberately not tuned.
.text
.globl _tally_u8
.p2align 2
_tally_u8:
    mov     x8, #16
    cmp     x3, x8
    csel    x8, x3, x8, hi          // BUG: max(nbuckets, 16) entries are cleared
    mov     x4, #0                  // bucket being cleared
Lclear:
    cmp     x4, x8
    b.hs    Lcount
    str     xzr, [x1, x4, lsl #3]
    add     x4, x4, #1
    b       Lclear
Lcount:
    cbz     x2, Ldone
Lloop:
    ldrb    w5, [x0], #1            // element, zero-extended
    udiv    x6, x5, x3
    msub    x6, x6, x3, x5          // element % nbuckets
    ldr     x7, [x1, x6, lsl #3]
    add     x7, x7, #1
    str     x7, [x1, x6, lsl #3]
    subs    x2, x2, #1
    b.ne    Lloop
Ldone:
    ret
