.intel_syntax noprefix
# size_t count_nonzero_f64(const double *x, size_t n);
#
# rdi = x, rsi = n. Returns how many of the n elements are not zero.
#
# The zeros are counted and subtracted from n.
# An element is zero when it compares equal to 0.0: +0.0 and -0.0 are, a NaN
# is not. `vcmpeqpd` against a register of zeros says exactly that, four
# elements at a time, as all ones in the lanes that are zero.
# All ones is -1, so subtracting the compare's result from an accumulator of
# 64-bit lanes adds one per zero. That is two instructions per four elements,
# the compare with its load folded in and the subtract, and the processor
# does two 32-byte loads per cycle: eight elements per cycle while the data
# is in the first-level cache.
#
# The loads are aligned to 32 bytes. A buffer from NumPy usually starts 16
# bytes into a 64-byte cache line, where every second 32-byte load would
# straddle two lines, which costs a cycle each.
#
# Paths, by n:
#   0..3    one element at a time
#   4..     sixteen per step into four accumulators, four per step, then the
#           last 0..3 one at a time; from ALIGN_MIN_N elements, 0..3 elements
#           first, up to a 32-byte boundary

# From this many elements the loads are aligned. Below it the array is a few
# cache lines and the head costs more than the straddling loads.
.set ALIGN_MIN_N, 64

.text
.globl count_nonzero_f64
.type count_nonzero_f64, @function
.p2align 4
count_nonzero_f64:
    xor     eax, eax                    # zeros seen so far
    mov     rdx, rsi                    # n, for the result
    cmp     rsi, 4
    jb      .Ltail
    cmp     rsi, ALIGN_MIN_N
    jb      .Laligned

    # Head: one at a time until x is a multiple of 32. If x is not a multiple
    # of 8 this runs three times and the loads stay unaligned.
.Lhead:
    test    dil, 31
    jz      .Laligned
    mov     r8, [rdi]
    add     r8, r8                      # the sign bit out: +0.0 and -0.0 are both zero
    cmp     r8, 1
    adc     rax, 0                      # one more zero
    add     rdi, 8
    dec     rsi
    test    dil, 24
    jnz     .Lhead
.Laligned:
    vpxor   xmm0, xmm0, xmm0
    vpxor   xmm1, xmm1, xmm1
    vpxor   xmm2, xmm2, xmm2
    vpxor   xmm3, xmm3, xmm3
    vpxor   xmm4, xmm4, xmm4
    mov     rcx, rsi
    shr     rcx, 4                      # steps of sixteen
    jz      .Lfours
.p2align 6
.Lsixteen:
    vcmpeqpd ymm5, ymm0, [rdi]
    vcmpeqpd ymm6, ymm0, [rdi + 32]
    vcmpeqpd ymm7, ymm0, [rdi + 64]
    vcmpeqpd ymm8, ymm0, [rdi + 96]
    sub     rdi, -128
    vpsubq  ymm1, ymm1, ymm5
    vpsubq  ymm2, ymm2, ymm6
    vpsubq  ymm3, ymm3, ymm7
    vpsubq  ymm4, ymm4, ymm8
    dec     rcx
    jnz     .Lsixteen
    vpaddq  ymm1, ymm1, ymm2
    vpaddq  ymm3, ymm3, ymm4
    vpaddq  ymm1, ymm1, ymm3
.Lfours:
    test    sil, 12
    jz      .Lreduce
    mov     ecx, esi
    shr     ecx, 2
    and     ecx, 3                      # steps of four
.Lfour:
    vcmpeqpd ymm5, ymm0, [rdi]
    add     rdi, 32
    vpsubq  ymm1, ymm1, ymm5
    dec     ecx
    jnz     .Lfour
.Lreduce:
    vextracti128 xmm2, ymm1, 1
    vpaddq  xmm1, xmm1, xmm2
    vpshufd xmm2, xmm1, 0xee
    vpaddq  xmm1, xmm1, xmm2
    vmovq   rcx, xmm1
    add     rax, rcx
    vzeroupper
    and     esi, 3                      # 0..3 left

.Ltail:
    test    rsi, rsi
    jz      .Ldone
.Lone:
    mov     r8, [rdi]
    add     r8, r8                      # the sign bit out: +0.0 and -0.0 are both zero
    cmp     r8, 1
    adc     rax, 0                      # one more zero
    add     rdi, 8
    dec     rsi
    jnz     .Lone
.Ldone:
    sub     rdx, rax                    # n minus the zeros
    mov     rax, rdx
    ret
.size count_nonzero_f64, .-count_nonzero_f64

.section .note.GNU-stack,"",@progbits
