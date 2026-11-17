# BROKEN ON PURPOSE: clears at least 16 buckets, the capacity in the default
# scalar set, even when the table it was given is smaller
# void tally_u8(const uint8_t *x, int64_t *counts, size_t n, size_t nbuckets);
# Simple scalar version used to test the judge. Deliberately not tuned.
    .intel_syntax noprefix
    .text
    .globl  tally_u8
    .type   tally_u8, @function
    .p2align 4
tally_u8:
    mov     r8, rdx                         # n; rdx is needed by the division
    mov     r10d, 16
    cmp     rcx, r10
    cmova   r10, rcx                        # BUG: max(nbuckets, 16) entries are cleared
    xor     r9d, r9d                        # bucket being cleared
.Lclear:
    cmp     r9, r10
    jae     .Lcount
    mov     qword ptr [rsi + r9*8], 0
    inc     r9
    jmp     .Lclear
.Lcount:
    test    r8, r8
    jz      .Ldone
.Lloop:
    movzx   eax, byte ptr [rdi]             # element, zero-extended
    inc     rdi
    xor     edx, edx
    div     rcx                             # rdx = element % nbuckets
    inc     qword ptr [rsi + rdx*8]
    dec     r8
    jnz     .Lloop
.Ldone:
    ret

    .section .note.GNU-stack,"",@progbits
