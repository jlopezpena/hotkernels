# BROKEN ON PURPOSE: correct result, then one store just past the capacity.
# A scratch buffer allows writes anywhere inside the capacity, never beyond it.
# size_t compress_scratch_f32(const float *x, const bool *mask, float *out, size_t n);
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
    mov     dword ptr [rdx + rcx*4], 0      # BUG: writes out[n], one element past the capacity
    ret

    .section .note.GNU-stack,"",@progbits
