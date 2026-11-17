# size_t compress_scratch_f32(const float *x, const bool *mask, float *out, size_t n);
# Simple scalar version used to test the judge. Deliberately not tuned.
#
# Stores every element, then advances only when the mask is set. This leaves a
# stray element past the returned count, which the scratch output allows. It is
# the same code that compress_f32 rejects as broken/write_past_used.s.
    .intel_syntax noprefix
    .text
    .globl  compress_scratch_f32
    .type   compress_scratch_f32, @function
    .p2align 4
compress_scratch_f32:
    xor     r8d, r8d                        # i
    xor     eax, eax                        # number written
.Lloop:
    cmp     r8, rcx
    jae     .Ldone
    mov     r10d, dword ptr [rdi + r8*4]
    mov     dword ptr [rdx + rax*4], r10d   # unconditional store, inside the capacity
    movzx   r9d, byte ptr [rsi + r8]
    add     rax, r9
.Lnext:
    inc     r8
    jmp     .Lloop
.Ldone:
    ret

    .section .note.GNU-stack,"",@progbits
