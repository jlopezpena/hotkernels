# BROKEN ON PURPOSE: correct, but far slower than naive C
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
    mov     ecx, 1
    mov     r8d, 0x3f800000                 # 1.0
    vmovd   xmm2, r8d
    vmovd   xmm3, r8d
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
    vdivss  xmm2, xmm2, xmm3                # pointless slow work on every element
    vdivss  xmm2, xmm2, xmm3
    vdivss  xmm2, xmm2, xmm3
    inc     rcx
    jmp     .Lloop
.Lnan:
    mov     rax, rcx
.Ldone:
    ret

    .section .note.GNU-stack,"",@progbits
