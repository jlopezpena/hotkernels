.intel_syntax noprefix
# void bincount_i64(const int64_t *x, int64_t *counts, size_t n, size_t bins);
#
# rdi = x, rsi = counts, rdx = n, rcx = bins. counts[v] = how many of the n
# elements at x equal v, for v = 0..bins-1. An element outside that range,
# negative or at or above bins, is ignored.
#
# The element is compared with bins as an unsigned number, so a negative
# element is a huge one and fails the same test as one that is too large.
# Every element takes the test, so no counter outside counts is ever
# touched, whatever the data.
#
# The counters are cleared with 32-byte stores, the last ones ending exactly
# at the end of counts and overlapping the ones before; from CLEAR_REP_MIN
# counters with `rep stosq`.

# From this many counters `rep stosq` clears them; below it, a loop of
# 32-byte stores, which has no start-up cost.
.set CLEAR_REP_MIN, 512

# Count the element at [rdi + \offset]: test against bins, increment its
# counter.
.macro COUNT_ELEMENT offset
    mov     rax, [rdi + \offset]
    cmp     rax, r9
    jae     1f
    inc     qword ptr [rsi + rax * 8]
1:
.endm

.text
.globl bincount_i64
.type bincount_i64, @function
.p2align 4
bincount_i64:
    mov     r9, rcx                     # bins
    # Clear the bins counters.
    cmp     rcx, CLEAR_REP_MIN
    jae     .Lclearrep
    cmp     rcx, 4
    jb      .Lclearfew
    vpxor   xmm0, xmm0, xmm0
    cmp     rcx, 8
    jb      .Lclearfour
    lea     r8, [rsi + rcx * 8 - 64]    # the last eight counters
    mov     rax, rsi
.Lclearstep:
    vmovdqu [rax], ymm0
    vmovdqu [rax + 32], ymm0
    add     rax, 64
    cmp     rax, r8
    jb      .Lclearstep
    vmovdqu [r8], ymm0
    vmovdqu [r8 + 32], ymm0
    vzeroupper
    jmp     .Lcount
.Lclearfour:                            # 4 to 7 counters: the first four and the last four
    vmovdqu [rsi], ymm0
    vmovdqu [rsi + rcx * 8 - 32], ymm0
    vzeroupper
    jmp     .Lcount
.Lclearfew:                             # 1 to 3 counters
    mov     qword ptr [rsi + rcx * 8 - 8], 0
    dec     rcx
    jnz     .Lclearfew
    jmp     .Lcount
.Lclearrep:
    mov     r8, rdi
    mov     rdi, rsi
    xor     eax, eax
    rep stosq
    mov     rdi, r8

.Lcount:
    mov     r8, rdx
    shr     r8, 2                       # steps of four elements
    jz      .Llast
.p2align 4
.Lfour:
    COUNT_ELEMENT 0
    COUNT_ELEMENT 8
    COUNT_ELEMENT 16
    COUNT_ELEMENT 24
    add     rdi, 32
    dec     r8
    jnz     .Lfour
.Llast:
    and     edx, 3                      # 0 to 3 elements left
    jz      .Ldone
.Lone:
    COUNT_ELEMENT 0
    add     rdi, 8
    dec     edx
    jnz     .Lone
.Ldone:
    ret
.size bincount_i64, .-bincount_i64

.section .note.GNU-stack,"",@progbits
