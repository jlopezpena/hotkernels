# BROKEN ON PURPOSE: uses rbx without saving it
# int64_t argmax_f32(const float *x, size_t n);   n >= 1
# Simple scalar version used to test the judge. Deliberately not tuned.
    .intel_syntax noprefix
    .text
    .globl  argmax_f32
    .type   argmax_f32, @function
    .p2align 4
argmax_f32:
    vmovss  xmm0, dword ptr [rdi]           # best value so far
    xor     eax, eax                        # its index
    vucomiss xmm0, xmm0
    jp      .Ldone                          # x[0] is NaN: the first NaN wins
    mov     ebx, 1
.Lloop:
    cmp     rbx, rsi
    jae     .Ldone
    vmovss  xmm1, dword ptr [rdi + rbx*4]
    vucomiss xmm1, xmm1
    jp      .Lnan
    vucomiss xmm1, xmm0
    jbe     .Lnext                          # only a strictly greater value replaces the best
    vmovaps xmm0, xmm1
    mov     rax, rbx
.Lnext:
    inc     rbx
    jmp     .Lloop
.Lnan:
    mov     rax, rbx
.Ldone:
    ret

    .section .note.GNU-stack,"",@progbits
