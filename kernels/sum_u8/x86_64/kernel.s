.intel_syntax noprefix
# uint64_t sum_u8(const uint8_t *x, size_t n);
#
# rdi = x, rsi = n. Returns the sum of the n bytes at x.
#
# The work is done by `vpsadbw`: against a register of zeros it adds the 32
# bytes of a vector into four 64-bit lanes, eight bytes each. A lane grows by
# at most 8 * 255 per step, so nothing can overflow.

.text
.globl sum_u8
.type sum_u8, @function
.p2align 4
sum_u8:
    xor     eax, eax
    cmp     rsi, 32
    jb      .Ltail
    vpxor   xmm0, xmm0, xmm0            # zeros to compare against
    vpxor   xmm1, xmm1, xmm1            # four 64-bit sums
.Lstep:
    vpsadbw ymm2, ymm0, [rdi]
    vpaddq  ymm1, ymm1, ymm2
    add     rdi, 32
    sub     rsi, 32
    cmp     rsi, 32
    jae     .Lstep
    vextracti128 xmm2, ymm1, 1
    vpaddq  xmm1, xmm1, xmm2
    vpshufd xmm2, xmm1, 0xee
    vpaddq  xmm1, xmm1, xmm2
    vmovq   rax, xmm1
    vzeroupper
.Ltail:
    # 0..31 bytes left, one at a time.
    test    rsi, rsi
    jz      .Ldone
.Lbyte:
    movzx   ecx, byte ptr [rdi]
    add     rax, rcx
    inc     rdi
    dec     rsi
    jnz     .Lbyte
.Ldone:
    ret
.size sum_u8, .-sum_u8

.section .note.GNU-stack,"",@progbits
