.intel_syntax noprefix
# int64_t argmax_f64(const double *x, size_t n);
#
# rdi = x, rsi = n (at least 1). Returns the index of the maximum, with
# NumPy's rules: the first maximum wins, and the first NaN wins over
# everything.
#
# One pass that does not track positions, and a short second look at the one
# place that matters. M is the maximum seen so far, in every lane of a
# register. A block of 32 elements is reduced with `vmaxpd` into four
# accumulators that are never reset, so a lane of an accumulator is larger
# than M exactly when the block holds a new maximum. That, and "the block
# holds a NaN", are one branch per block, taken a handful of times in a whole
# array of random data (about the logarithm of the number of blocks). When it
# is taken (.Levent), the block is looked at again: the first NaN in it ends
# the call; otherwise M becomes the maximum of the block and the block is
# remembered. At the end the first element equal to M in the remembered block
# is the answer (.Lfinal).
#
# NaN. `vmaxpd` returns its second operand when either is a NaN, so a NaN
# among the data can be lost from an accumulator and needs a test of its own.
# `vcmpunordpd` of two vectors is true where either holds a NaN, so one
# compare covers two vectors. A NaN that does reach an accumulator is
# harmless: the block that brought it takes the branch for its NaN, and the
# call returns from there.
#
# Zeros. +0.0 and -0.0 compare equal, in the test for a new maximum and in
# the final search, so the first of them wins, as in the reference.
#
# The leftover (n mod 32) is one more block that ends exactly at the end
# of the array and overlaps the one before it. Elements seen twice cannot
# start a new maximum the second time, and a NaN among them would have ended
# the call the first time.
#
# Paths, by n:
#   1..3        one element at a time, as the reference does it
#   4..31     one vector per step, compared with M directly
#   32..        blocks of 32
#
# Registers: rdi x, rsi the index of the last step's first element, r10 the
# index of the current step's, r8 and r9 the first element and the number of
# vectors of the remembered range, r11 where to go on after an event, rax,
# rcx and rdx scratch. ymm15 M, ymm8-ymm11 the accumulators.

.section .rodata
.p2align 3
# minus infinity: no element is beyond it, so the first one equal to it starts.
.Linit:
    .quad 0xfff0000000000000

.text
.globl argmax_f64
.type argmax_f64, @function
.p2align 4
argmax_f64:
    cmp     rsi, 4
    jb      .Lscalar
    vbroadcastsd ymm15, qword ptr [rip + .Linit]
    xor     r8d, r8d                    # the remembered range: the first vector
    mov     r9d, 1
    xor     r10d, r10d
    cmp     rsi, 32
    jae     .Lblocks

    # ---- One vector per step.
    sub     rsi, 4                # index of the last step
.Lvector:
    vmovupd ymm0, [rdi + r10 * 8]
    vcmppd ymm1, ymm15, ymm0, 0x11      # a new maximum
    vcmpunordpd ymm2, ymm0, ymm0
    vorpd  ymm1, ymm1, ymm2
    vmovmskpd eax, ymm1
    test    eax, eax
    jnz     .Lvectorevent
.Lvectornext:
    cmp     r10, rsi
    jae     .Lfinal
    add     r10, 4
    cmp     r10, rsi
    cmova   r10, rsi                    # the last step ends with the last element
    jmp     .Lvector
.Lvectorevent:
    mov     r8, r10
    mov     r9d, 1
    lea     r11, [rip + .Lvectornext]
    jmp     .Levent

    # ---- Blocks of 32.
.Lblocks:
    sub     rsi, 32                # index of the last block
    vmovapd ymm8, ymm15
    vmovapd ymm9, ymm15
    vmovapd ymm10, ymm15
    vmovapd ymm11, ymm15
.p2align 6
.Lblock:
    lea     rdx, [rdi + r10 * 8]
    vmovupd ymm0, [rdx]
    vmaxpd ymm8, ymm8, ymm0
    vcmpunordpd ymm4, ymm0, [rdx + 32]
    vmaxpd ymm8, ymm8, [rdx + 32]
    vmovupd ymm1, [rdx + 64]
    vmaxpd ymm9, ymm9, ymm1
    vcmpunordpd ymm5, ymm1, [rdx + 96]
    vmaxpd ymm9, ymm9, [rdx + 96]
    vmovupd ymm2, [rdx + 128]
    vmaxpd ymm10, ymm10, ymm2
    vcmpunordpd ymm6, ymm2, [rdx + 160]
    vmaxpd ymm10, ymm10, [rdx + 160]
    vmovupd ymm3, [rdx + 192]
    vmaxpd ymm11, ymm11, ymm3
    vcmpunordpd ymm7, ymm3, [rdx + 224]
    vmaxpd ymm11, ymm11, [rdx + 224]
    vorpd  ymm4, ymm4, ymm5
    vorpd  ymm6, ymm6, ymm7
    vmaxpd ymm0, ymm8, ymm9
    vmaxpd ymm1, ymm10, ymm11
    vorpd  ymm4, ymm4, ymm6            # a NaN in the block
    vmaxpd ymm0, ymm0, ymm1
    vcmppd ymm0, ymm15, ymm0, 0x11      # a new maximum in the block
    vorpd  ymm0, ymm0, ymm4
    vmovmskpd eax, ymm0
    test    eax, eax
    jnz     .Lblockevent
.Lblocknext:
    cmp     r10, rsi
    jae     .Lfinal
    add     r10, 32
    cmp     r10, rsi
    cmova   r10, rsi                    # the last block ends with the last element
    jmp     .Lblock
.Lblockevent:
    mov     r8, r10
    mov     r9d, 8
    lea     r11, [rip + .Lblocknext]

    # ---- The r9 vectors from element r8 hold a NaN or a new maximum. The
    #      first NaN ends the call. Otherwise M becomes their maximum, they
    #      stay remembered, and the loop goes on at r11.
.Levent:
    lea     rdx, [rdi + r8 * 8]
    mov     rcx, r9
    vmovupd ymm1, [rdx]
.Leventscan:
    vmovupd ymm0, [rdx]
    vcmpunordpd ymm2, ymm0, ymm0
    vmovmskpd eax, ymm2
    test    eax, eax
    jnz     .Lfound                     # the first NaN of the array
    vmaxpd ymm1, ymm1, ymm0
    add     rdx, 32
    dec     rcx
    jnz     .Leventscan
    vextractf128 xmm0, ymm1, 1
    vmaxpd xmm1, xmm1, xmm0
    vpermilpd xmm0, xmm1, 1
    vmaxsd xmm1, xmm1, xmm0
    vbroadcastsd ymm15, xmm1
    jmp     r11

    # ---- The first element equal to M in the remembered range.
.Lfinal:
    lea     rdx, [rdi + r8 * 8]
    mov     rcx, r9
.Lfinalscan:
    vcmpeqpd ymm0, ymm15, [rdx]
    vmovmskpd eax, ymm0
    test    eax, eax
    jnz     .Lfound
    add     rdx, 32
    dec     rcx
    jnz     .Lfinalscan
    # Not reached unless the array changed under the kernel: M is an element
    # of the remembered range. Both looks at a range are bounded by its
    # length, so nothing outside the array is read even then, and the answer
    # given is an index inside it.
    mov     rax, r8
    vzeroupper
    ret
.Lfound:
    # eax: one bit per lane of the vector at rdx, the lowest set bit the answer.
    tzcnt   eax, eax
    sub     rdx, rdi
    shr     rdx, 3
    add     rax, rdx
    vzeroupper
    ret

    # ---- Fewer than 4 elements: the reference, one element at a time.
.Lscalar:
    xor     eax, eax                    # index of the best so far
    vmovsd  xmm0, qword ptr [rdi]
    vucomisd xmm0, xmm0
    jp      .Lreturn                    # x[0] is a NaN
    mov     ecx, 1
.Lscalarnext:
    cmp     rcx, rsi
    jae     .Lreturn
    vmovsd  xmm1, qword ptr [rdi + rcx * 8]
    vucomisd xmm1, xmm1
    jp      .Lscalarnan
    vucomisd xmm1, xmm0
    jbe     .Lscalarkeep
    vmovapd xmm0, xmm1
    mov     rax, rcx
.Lscalarkeep:
    inc     rcx
    jmp     .Lscalarnext
.Lscalarnan:
    mov     rax, rcx
.Lreturn:
    ret
.size argmax_f64, .-argmax_f64

.section .note.GNU-stack,"",@progbits
