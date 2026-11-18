.intel_syntax noprefix
# int64_t argmin_f32(const float *x, size_t n);
#
# rdi = x, rsi = n (at least 1). Returns the index of the minimum, with
# NumPy's rules: the first minimum wins, and the first NaN wins over
# everything.
#
# One pass that does not track positions, and a short second look at the one
# place that matters. M is the minimum seen so far, in every lane of a
# register. A block of 64 elements is reduced with `vminps` into four
# accumulators that are never reset, so a lane of an accumulator is smaller
# than M exactly when the block holds a new minimum. That, and "the block
# holds a NaN", are one branch per block, taken a handful of times in a whole
# array of random data (about the logarithm of the number of blocks). When it
# is taken (.Levent), the block is looked at again: the first NaN in it ends
# the call; otherwise M becomes the minimum of the block and the block is
# remembered. At the end the first element equal to M in the remembered block
# is the answer (.Lfinal).
#
# NaN. `vminps` returns its second operand when either is a NaN, so a NaN
# among the data can be lost from an accumulator and needs a test of its own.
# `vcmpunordps` of two vectors is true where either holds a NaN, so one
# compare covers two vectors. A NaN that does reach an accumulator is
# harmless: the block that brought it takes the branch for its NaN, and the
# call returns from there.
#
# Zeros. +0.0 and -0.0 compare equal, in the test for a new minimum and in
# the final search, so the first of them wins, as in the reference.
#
# The leftover (n mod 64) is one more block that ends exactly at the end
# of the array and overlaps the one before it. Elements seen twice cannot
# start a new minimum the second time, and a NaN among them would have ended
# the call the first time.
#
# Paths, by n:
#   1..7        one element at a time, as the reference does it
#   8..63     one vector per step, compared with M directly
#   64..        blocks of 64
#
# Registers: rdi x, rsi the index of the last step's first element, r10 the
# index of the current step's, r8 and r9 the first element and the number of
# vectors of the remembered range, r11 where to go on after an event, rax,
# rcx and rdx scratch. ymm15 M, ymm8-ymm11 the accumulators.

.section .rodata
.p2align 3
# plus infinity: no element is beyond it, so the first one equal to it starts.
.Linit:
    .long 0x7f800000

.text
.globl argmin_f32
.type argmin_f32, @function
.p2align 4
argmin_f32:
    cmp     rsi, 8
    jb      .Lscalar
    vbroadcastss ymm15, dword ptr [rip + .Linit]
    xor     r8d, r8d                    # the remembered range: the first vector
    mov     r9d, 1
    xor     r10d, r10d
    cmp     rsi, 64
    jae     .Lblocks

    # ---- One vector per step.
    sub     rsi, 8                # index of the last step
.Lvector:
    vmovups ymm0, [rdi + r10 * 4]
    vcmpps ymm1, ymm15, ymm0, 0x1e      # a new minimum
    vcmpunordps ymm2, ymm0, ymm0
    vorps  ymm1, ymm1, ymm2
    vmovmskps eax, ymm1
    test    eax, eax
    jnz     .Lvectorevent
.Lvectornext:
    cmp     r10, rsi
    jae     .Lfinal
    add     r10, 8
    cmp     r10, rsi
    cmova   r10, rsi                    # the last step ends with the last element
    jmp     .Lvector
.Lvectorevent:
    mov     r8, r10
    mov     r9d, 1
    lea     r11, [rip + .Lvectornext]
    jmp     .Levent

    # ---- Blocks of 64.
.Lblocks:
    sub     rsi, 64                # index of the last block
    vmovaps ymm8, ymm15
    vmovaps ymm9, ymm15
    vmovaps ymm10, ymm15
    vmovaps ymm11, ymm15
.p2align 6
.Lblock:
    lea     rdx, [rdi + r10 * 4]
    vmovups ymm0, [rdx]
    vminps ymm8, ymm8, ymm0
    vcmpunordps ymm4, ymm0, [rdx + 32]
    vminps ymm8, ymm8, [rdx + 32]
    vmovups ymm1, [rdx + 64]
    vminps ymm9, ymm9, ymm1
    vcmpunordps ymm5, ymm1, [rdx + 96]
    vminps ymm9, ymm9, [rdx + 96]
    vmovups ymm2, [rdx + 128]
    vminps ymm10, ymm10, ymm2
    vcmpunordps ymm6, ymm2, [rdx + 160]
    vminps ymm10, ymm10, [rdx + 160]
    vmovups ymm3, [rdx + 192]
    vminps ymm11, ymm11, ymm3
    vcmpunordps ymm7, ymm3, [rdx + 224]
    vminps ymm11, ymm11, [rdx + 224]
    vorps  ymm4, ymm4, ymm5
    vorps  ymm6, ymm6, ymm7
    vminps ymm0, ymm8, ymm9
    vminps ymm1, ymm10, ymm11
    vorps  ymm4, ymm4, ymm6            # a NaN in the block
    vminps ymm0, ymm0, ymm1
    vcmpps ymm0, ymm15, ymm0, 0x1e      # a new minimum in the block
    vorps  ymm0, ymm0, ymm4
    vmovmskps eax, ymm0
    test    eax, eax
    jnz     .Lblockevent
.Lblocknext:
    cmp     r10, rsi
    jae     .Lfinal
    add     r10, 64
    cmp     r10, rsi
    cmova   r10, rsi                    # the last block ends with the last element
    jmp     .Lblock
.Lblockevent:
    mov     r8, r10
    mov     r9d, 8
    lea     r11, [rip + .Lblocknext]

    # ---- The r9 vectors from element r8 hold a NaN or a new minimum. The
    #      first NaN ends the call. Otherwise M becomes their minimum, they
    #      stay remembered, and the loop goes on at r11.
.Levent:
    lea     rdx, [rdi + r8 * 4]
    mov     rcx, r9
    vmovups ymm1, [rdx]
.Leventscan:
    vmovups ymm0, [rdx]
    vcmpunordps ymm2, ymm0, ymm0
    vmovmskps eax, ymm2
    test    eax, eax
    jnz     .Lfound                     # the first NaN of the array
    vminps ymm1, ymm1, ymm0
    add     rdx, 32
    dec     rcx
    jnz     .Leventscan
    vextractf128 xmm0, ymm1, 1
    vminps xmm1, xmm1, xmm0
    vpermilps xmm0, xmm1, 0x4e
    vminps xmm1, xmm1, xmm0
    vpermilps xmm0, xmm1, 0x55
    vminss xmm1, xmm1, xmm0
    vbroadcastss ymm15, xmm1
    jmp     r11

    # ---- The first element equal to M in the remembered range.
.Lfinal:
    lea     rdx, [rdi + r8 * 4]
    mov     rcx, r9
.Lfinalscan:
    vcmpeqps ymm0, ymm15, [rdx]
    vmovmskps eax, ymm0
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
    shr     rdx, 2
    add     rax, rdx
    vzeroupper
    ret

    # ---- Fewer than 8 elements: the reference, one element at a time.
.Lscalar:
    xor     eax, eax                    # index of the best so far
    vmovss  xmm0, dword ptr [rdi]
    vucomiss xmm0, xmm0
    jp      .Lreturn                    # x[0] is a NaN
    mov     ecx, 1
.Lscalarnext:
    cmp     rcx, rsi
    jae     .Lreturn
    vmovss  xmm1, dword ptr [rdi + rcx * 4]
    vucomiss xmm1, xmm1
    jp      .Lscalarnan
    vucomiss xmm1, xmm0
    jae     .Lscalarkeep
    vmovaps xmm0, xmm1
    mov     rax, rcx
.Lscalarkeep:
    inc     rcx
    jmp     .Lscalarnext
.Lscalarnan:
    mov     rax, rcx
.Lreturn:
    ret
.size argmin_f32, .-argmin_f32

.section .note.GNU-stack,"",@progbits
