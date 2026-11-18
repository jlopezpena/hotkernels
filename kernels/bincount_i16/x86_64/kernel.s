.intel_syntax noprefix
# void bincount_i16(const int16_t *x, int64_t *counts, size_t n, size_t bins);
#
# rdi = x, rsi = counts, rdx = n, rcx = bins. counts[v] = how many of the n
# elements at x equal v, for v = 0..bins-1. An element outside that range,
# negative or at or above bins, is ignored.
#
# The element is sign-extended to 64 bits and compared with bins as an
# unsigned number, so a negative element is a huge one and fails the same
# test as one that is too large. Every element takes the test, so no counter
# outside counts is ever touched, whatever the data.

.text
.globl bincount_i16
.type bincount_i16, @function
.p2align 4
bincount_i16:
    mov     r8, rdi                     # x
    mov     r9, rcx                     # bins
    # Clear the bins counters.
    mov     rdi, rsi
    xor     eax, eax
    rep stosq
    test    rdx, rdx
    jz      .Ldone
    lea     rdx, [r8 + rdx * 2]         # end of x
.Lelement:
    movsx   rax, word ptr [r8]
    add     r8, 2
    cmp     rax, r9
    jae     .Lskip
    inc     qword ptr [rsi + rax * 8]
.Lskip:
    cmp     r8, rdx
    jne     .Lelement
.Ldone:
    ret
.size bincount_i16, .-bincount_i16

.section .note.GNU-stack,"",@progbits
