.intel_syntax noprefix
# void searchsorted_f64(const double *a, const double *v, int64_t *out, size_t n, size_t needles);
#
# rdi = a, rsi = v, rdx = out, rcx = n, r8 = needles. For each needle in v,
# writes the first index i such that a[i] is not less than the needle (side='left').
# a holds n values sorted ascending, NaN last.
#
# One search is a fixed number of steps that depends only on n, with no
# branch on the data:
#
#     lo = 0, len = n
#     while len > 0:
#         half = (len + 1) / 2
#         if a[lo + half - 1] < needle: lo += half
#         len -= half
#
# The answer is always in lo .. lo + len, and lo + len never grows, so every
# probe is inside a whatever the data, sorted or not. A NaN in a (they are last) compares false and is never passed, which is right for
# every needle that is not NaN.
#
# Sixteen searches run in step: four registers of four positions each. A step
# is, per register, one `vgatherqpd` for the four probes, one compare with
# the needles, and one add of `half` under the compare's mask. Nothing in a
# step waits for another search, so when a is too large for the caches the
# sixteen probes of a step are sixteen cache misses in flight together, where
# a search done alone has one.
#
# A NaN needle. NumPy orders NaN after everything and equal to itself, so its
# answer is the index of the first NaN of a, or n. The compare above is false
# for it at every step and leaves 0. So the needles are also tested for NaN as
# they are read, and only if one was seen (.Lfixnan) is that index found, with
# one more search of a for its first NaN, and written over the answer of every
# NaN needle.
#
# Paths, by the number of needles:
#   0        nothing
#   1..3     one search at a time, in general registers
#   4..15    four in step, the last group overlapping the one before it
#   16..     sixteen in step, the last group overlapping the one before it
#
# `vgatherqpd` is fast on this processor and slow on some others (older AMD,
# and Intel processors that carry the gather mitigation).
#
# Registers: r9 len, rax the address of a[half - 1], r10 byte offset of the
# group, r11 byte offset of the last group. ymm0-ymm3 lo, ymm4-ymm7 needles,
# ymm8 half in every lane, ymm9 the gather's mask, ymm10-ymm13 the probes,
# ymm15 collects "this needle is a NaN".

# One step for the four searches in \lo against the needles in \key. A gather
# clears its mask register, and it merges into its destination, which is
# therefore one more input: each of the four uses of this macro has a
# destination of its own, so that none waits for another.
.macro PROBE lo, key, tmp
    vpcmpeqd ymm9, ymm9, ymm9
    vgatherqpd \tmp, [rax + \lo * 8], ymm9
    vcmpltpd \tmp, \tmp, \key
    vpand   \tmp, \tmp, ymm8
    vpaddq  \lo, \lo, \tmp
.endm

# The start of a step: half into rax and ymm8, len reduced, and rax turned
# into the address of a[half - 1]. Leaves the zero flag set when this is the
# last step; nothing that follows in a step changes the flags.
.macro HALVE
    lea     rax, [r9 + 1]
    shr     rax, 1
    vmovq   xmm8, rax
    vpbroadcastq ymm8, xmm8
    sub     r9, rax
    lea     rax, [rdi + rax * 8 - 8]
.endm

.text
.globl searchsorted_f64
.type searchsorted_f64, @function
.p2align 4
searchsorted_f64:
    test    r8, r8
    jz      .Lreturn
    cmp     r8, 4
    jb      .Lfew
    vpxor   xmm15, xmm15, xmm15
    xor     r10d, r10d
    cmp     r8, 16
    jb      .Lquads

    # ---- Sixteen in step.
    lea     r11, [r8 * 8 - 128]
.Lgroup16:
    vmovupd ymm4, [rsi + r10]
    vmovupd ymm5, [rsi + r10 + 32]
    vmovupd ymm6, [rsi + r10 + 64]
    vmovupd ymm7, [rsi + r10 + 96]
    vpxor   xmm0, xmm0, xmm0
    vpxor   xmm1, xmm1, xmm1
    vpxor   xmm2, xmm2, xmm2
    vpxor   xmm3, xmm3, xmm3
    vcmpunordpd ymm10, ymm4, ymm5       # a NaN in either
    vcmpunordpd ymm11, ymm6, ymm7
    vorpd   ymm15, ymm15, ymm10
    vorpd   ymm15, ymm15, ymm11
    mov     r9, rcx
    test    r9, r9
    jz      .Lstore16
.p2align 6
.Lstep16:
    HALVE
    PROBE   ymm0, ymm4, ymm10
    PROBE   ymm1, ymm5, ymm11
    PROBE   ymm2, ymm6, ymm12
    PROBE   ymm3, ymm7, ymm13
    jnz     .Lstep16
.Lstore16:
    vmovdqu [rdx + r10], ymm0
    vmovdqu [rdx + r10 + 32], ymm1
    vmovdqu [rdx + r10 + 64], ymm2
    vmovdqu [rdx + r10 + 96], ymm3
    cmp     r10, r11
    jae     .Lnantest
    add     r10, 128
    cmp     r10, r11
    cmova   r10, r11                    # the last group ends with the last needle
    jmp     .Lgroup16

    # ---- Four in step.
.Lquads:
    lea     r11, [r8 * 8 - 32]
.Lgroup4:
    vmovupd ymm4, [rsi + r10]
    vpxor   xmm0, xmm0, xmm0
    vcmpunordpd ymm10, ymm4, ymm4
    vorpd   ymm15, ymm15, ymm10
    mov     r9, rcx
    test    r9, r9
    jz      .Lstore4
.Lstep4:
    HALVE
    PROBE   ymm0, ymm4, ymm10
    jnz     .Lstep4
.Lstore4:
    vmovdqu [rdx + r10], ymm0
    cmp     r10, r11
    jae     .Lnantest
    add     r10, 32
    cmp     r10, r11
    cmova   r10, r11
    jmp     .Lgroup4

.Lnantest:
    vmovmskpd eax, ymm15
    vzeroupper
    test    eax, eax
    jnz     .Lfixnan
.Lreturn:
    ret

    # ---- One to three needles, one search at a time. No 256-bit register is
    #      written here.
.Lfew:
    xor     r10d, r10d                  # needle index
    xor     r11d, r11d                  # set when a needle is a NaN
.Lone:
    vmovsd  xmm0, qword ptr [rsi + r10 * 8]
    xor     eax, eax                    # lo
    mov     r9, rcx
    test    r9, r9
    jz      .Lonestore
.Lonestep:
    lea     r11, [r9 + 1]
    shr     r11, 1                      # half
    sub     r9, r11
    add     r11, rax                    # lo + half
    vucomisd xmm0, qword ptr [rdi + r11 * 8 - 8]
    cmova   rax, r11
    test    r9, r9
    jnz     .Lonestep
.Lonestore:
    mov     [rdx + r10 * 8], rax
    inc     r10
    cmp     r10, r8
    jb      .Lone
    # Any NaN among them? One more pass; three needles at most.
    xor     r10d, r10d
.Lfewnan:
    vmovsd  xmm0, qword ptr [rsi + r10 * 8]
    vucomisd xmm0, xmm0
    jp      .Lfixnan
    inc     r10
    cmp     r10, r8
    jb      .Lfewnan
    ret

    # ---- At least one needle is a NaN: find its answer and write it over
    #      the answer of every NaN needle.
.Lfixnan:
    # The index of the first NaN of a: the same search with "is not NaN" as
    # the test.
    xor     eax, eax
    mov     r9, rcx
    test    r9, r9
    jz      .Lfill
.Lfirstnan:
    lea     r10, [r9 + 1]
    shr     r10, 1                      # half
    sub     r9, r10
    lea     r11, [rax + r10]
    vmovsd  xmm0, qword ptr [rdi + r11 * 8 - 8]
    vucomisd xmm0, xmm0
    cmovnp  rax, r11                    # not a NaN: the first NaN is further on
    test    r9, r9
    jnz     .Lfirstnan
.Lfill:
    xor     r10d, r10d
.Lfillone:
    vmovsd  xmm0, qword ptr [rsi + r10 * 8]
    vucomisd xmm0, xmm0
    jnp     .Lfillnext
    mov     [rdx + r10 * 8], rax
.Lfillnext:
    inc     r10
    cmp     r10, r8
    jb      .Lfillone
    ret
.size searchsorted_f64, .-searchsorted_f64

.section .note.GNU-stack,"",@progbits
