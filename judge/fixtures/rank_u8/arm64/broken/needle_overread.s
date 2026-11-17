// BROKEN ON PURPOSE: loads each needle as two bytes and uses the low one, so
// the last needle reads one byte past the end of v. The length of v is the
// scalar `needles`, not n: this is caught only if the judge places v against
// the guard page by that length.
.text
.globl _rank_u8
.p2align 2
_rank_u8:
    cbz     x4, Ldone
Lneedle:
    ldrh    w5, [x1], #1
    and     w5, w5, #0xff            // needle, zero-extended
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
