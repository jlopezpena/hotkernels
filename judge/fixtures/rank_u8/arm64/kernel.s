// void rank_u8(const uint8_t *a, const uint8_t *v, int64_t *out, size_t n, size_t needles);
// Simple scalar version used to test the judge. Deliberately not tuned.
.text
.globl _rank_u8
.p2align 2
_rank_u8:
    cbz     x4, Ldone
Lneedle:
    ldrb    w5, [x1], #1            // needle, zero-extended
    mov     x6, #0                  // elements smaller than it
    mov     x7, #0                  // index into a
    cbz     x3, Lstore
Lelement:
    ldrb    w8, [x0, x7]
    cmp     w8, w5
    cinc    x6, x6, lo
    add     x7, x7, #1
    cmp     x7, x3
    b.lo    Lelement
Lstore:
    str     x6, [x2], #8
    subs    x4, x4, #1
    b.ne    Lneedle
Ldone:
    ret
