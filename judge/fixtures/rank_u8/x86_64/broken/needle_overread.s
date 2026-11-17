# BROKEN ON PURPOSE: loads each needle as two bytes and uses the low one, so
# the last needle reads one byte past the end of v. The length of v is the
# scalar `needles`, not n: this is caught only if the judge places v against
# the guard page by that length.
# void rank_u8(const uint8_t *a, const uint8_t *v, int64_t *out, size_t n, size_t needles);
    .intel_syntax noprefix
    .text
    .globl  rank_u8
    .type   rank_u8, @function
    .p2align 4
rank_u8:
    test    r8, r8
    jz      .Ldone
.Lneedle:
    movzx   eax, word ptr [rsi]             # two bytes: one too many
    and     eax, 0xff
    inc     rsi
    xor     r9d, r9d                        # elements smaller than it
    xor     r10d, r10d                      # index into a
    test    rcx, rcx
    jz      .Lstore
.Lelement:
    movzx   r11d, byte ptr [rdi + r10]
    cmp     r11d, eax                       # sets the carry when the element is below the needle
    adc     r9, 0
    inc     r10
    cmp     r10, rcx
    jb      .Lelement
.Lstore:
    mov     qword ptr [rdx], r9
    add     rdx, 8
    dec     r8
    jnz     .Lneedle
.Ldone:
    ret

    .section .note.GNU-stack,"",@progbits
