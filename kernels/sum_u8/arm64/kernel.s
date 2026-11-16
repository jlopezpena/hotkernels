// uint64_t sum_u8(const uint8_t *x, size_t n);
//
// x0 = x, x1 = n. Returns the sum of the n bytes at x.
//
// Sixteen bytes per step: `uaddlv` adds the sixteen bytes of a vector into
// one 16-bit value, which is moved to a general register and added to the
// running total. Leftover bytes are added one at a time.

.text
.globl _sum_u8
.p2align 2
_sum_u8:
    mov     x2, #0                  // running total
    subs    x1, x1, #16
    b.lo    Ltail
Lloop:
    ldr     q0, [x0], #16
    uaddlv  h0, v0.16b              // at most 16 * 255, fits in 16 bits
    fmov    w3, s0
    add     x2, x2, x3
    subs    x1, x1, #16
    b.hs    Lloop
Ltail:
    adds    x1, x1, #16             // leftover count, 0 to 15
    b.eq    Ldone
Lbyte:
    ldrb    w3, [x0], #1
    add     x2, x2, x3
    subs    x1, x1, #1
    b.ne    Lbyte
Ldone:
    mov     x0, x2
    ret
