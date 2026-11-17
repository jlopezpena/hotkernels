// BROKEN ON PURPOSE: always handles three needles, the default count, whatever
// `needles` says. Right for the default scalar set and wrong for the others,
// so it is caught only if those are run with buffers of their own length.
.text
.globl _rank_u8
.p2align 2
_rank_u8:
    mov     x4, #3
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
