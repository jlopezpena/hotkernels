# BROKEN ON PURPOSE: skips elements before the first 16-byte boundary
# int64_t argmax_f32(const float *x, size_t n);   n >= 1
# Simple scalar version used to test the judge. Deliberately not tuned.
    .intel_syntax noprefix
    .text
    .globl  argmax_f32
    .type   argmax_f32, @function
    .p2align 4
argmax_f32:
    mov     r8, rdi
    neg     r8
    and     r8, 15
    shr     r8, 2                           # elements before the next 16-byte boundary
    xor     r9d, r9d
    cmp     r8, rsi
    jae     .Lgo
    lea     rdi, [rdi + r8*4]               # BUG: skip them
    sub     rsi, r8
    mov     r9, r8
.Lgo:
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
    add     rax, r9
    ret

    .section .note.GNU-stack,"",@progbits
