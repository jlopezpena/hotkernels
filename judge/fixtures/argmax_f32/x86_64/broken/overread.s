# BROKEN ON PURPOSE: reads 16 bytes past the end of the input
# int64_t argmax_f32(const float *x, size_t n);   n >= 1
# Simple scalar version used to test the judge. Deliberately not tuned.
    .intel_syntax noprefix
    .text
    .globl  argmax_f32
    .type   argmax_f32, @function
    .p2align 4
argmax_f32:
    vmovups xmm7, xmmword ptr [rdi + rsi*4] # BUG: load past the end
    vmovss  xmm0, dword ptr [rdi]           # best value so far
    xor     eax, eax                        # its index
    vucomiss xmm0, xmm0
    jp      .Ldone                          # x[0] is NaN: the first NaN wins
    mov     ecx, 1
.Lloop:
    cmp     rcx, rsi
    jae     .Ldone
    vmovss  xmm1, dword ptr [rdi + rcx*4]
    vucomiss xmm1, xmm1
    jp      .Lnan
    vucomiss xmm1, xmm0
    jbe     .Lnext                          # only a strictly greater value replaces the best
    vmovaps xmm0, xmm1
    mov     rax, rcx
.Lnext:
    inc     rcx
    jmp     .Lloop
.Lnan:
    mov     rax, rcx
.Ldone:
    ret

    .section .note.GNU-stack,"",@progbits
