# BROKEN ON PURPOSE: writes every element, then advances only when the mask is set.
# The result is right, but one element past the used output gets overwritten.
# size_t compress_f32(const float *x, const bool *mask, float *out, size_t n);
    .intel_syntax noprefix
    .text
    .globl  compress_f32
    .type   compress_f32, @function
    .p2align 4
compress_f32:
    xor     r8d, r8d                        # i
    xor     eax, eax                        # number written
.Lloop:
    cmp     r8, rcx
    jae     .Ldone
    mov     r10d, dword ptr [rdi + r8*4]
    mov     dword ptr [rdx + rax*4], r10d   # BUG: unconditional write
    movzx   r9d, byte ptr [rsi + r8]
    add     rax, r9
.Lnext:
    inc     r8
    jmp     .Lloop
.Ldone:
    ret

    .section .note.GNU-stack,"",@progbits
