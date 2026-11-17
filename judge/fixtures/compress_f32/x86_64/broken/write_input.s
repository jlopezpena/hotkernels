# BROKEN ON PURPOSE: writes to the mask, which is an input
# size_t compress_f32(const float *x, const bool *mask, float *out, size_t n);
# Simple scalar version used to test the judge. Deliberately not tuned.
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
    movzx   r9d, byte ptr [rsi + r8]
    mov     byte ptr [rsi + r8], r9b        # BUG: write to an input
    test    r9d, r9d
    jz      .Lnext
    mov     r10d, dword ptr [rdi + r8*4]
    mov     dword ptr [rdx + rax*4], r10d
    inc     rax
.Lnext:
    inc     r8
    jmp     .Lloop
.Ldone:
    ret

    .section .note.GNU-stack,"",@progbits
