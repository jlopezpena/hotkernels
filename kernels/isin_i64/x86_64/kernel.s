.intel_syntax noprefix
# void isin_i64(const int64_t *x, const int64_t *test, bool *out, size_t n);
#
# rdi = x (n values), rsi = test (exactly 8 values), rdx = out (n bytes),
# rcx = n. out[i] = 1 when x[i] equals any test value, else 0.
#
# Sixteen elements per block. How a block is done is chosen once per call
# from the test values.
#
# Narrow keys, when every test value is from 0 to 32,767, which a set of
# small non-negative integers is. If all sixteen elements of a block are in
# that range too, they are narrowed from 64 to 16 bits with three
# `vpackusdw`, exactly, and the sixteen keys in one register are compared
# with the eight test keys: eight `vpcmpeqw` for sixteen elements, where
# 64-bit compares need eight for four. An element outside the range cannot be
# in the set, but it cannot be narrowed either, so a block that holds one is
# done the wide way, with the test values read from memory (.Lwidememory).
#
# Wide, for any other test values (.Lwide): each vector of four elements is
# compared with the eight test values at 64 bits, the values kept broadcast
# in registers.
#
# Both leave sixteen 16-bit lanes, all ones for a member, in an order that
# the packing instructions produce (they work inside each 128-bit half); one
# `vpacksswb`, one `vpunpcklwd` and one `vpabsb` turn them into sixteen bytes
# of 0 and 1 in element order.
#
# The leftover (n mod 16) is one more block that ends exactly at the end of
# the arrays and overlaps the one before it. Fewer than 16 elements are done
# one at a time, each against the eight test values in two compares.
#
# Registers: r9 index of the block's first element, r10 index of the last
# block, r11 the address of the loop in use. ymm8-ymm15 the eight test
# values, as 16-bit keys on the narrow path and 64-bit values on the wide
# one; ymm6 the bits that must be zero in an element for it to be narrowed.

# Bits that are set in a value outside 0 .. 32,767.
.set NARROW_SHIFT, 15

.section .rodata
.p2align 5
.Lhighbits:
    .quad 0xffffffffffff8000, 0xffffffffffff8000, 0xffffffffffff8000, 0xffffffffffff8000

.text
.globl isin_i64
.type isin_i64, @function
.p2align 4
isin_i64:
    cmp     rcx, 16
    jb      .Lsmall
    xor     r9d, r9d
    lea     r10, [rcx - 16]
    mov     rax, [rsi]
    or      rax, [rsi + 8]
    or      rax, [rsi + 16]
    or      rax, [rsi + 24]
    or      rax, [rsi + 32]
    or      rax, [rsi + 40]
    or      rax, [rsi + 48]
    or      rax, [rsi + 56]
    shr     rax, NARROW_SHIFT
    jnz     .Lwidesetup                 # a test value outside 0 .. 32,767
    vpbroadcastw ymm8, word ptr [rsi + 0]
    vpbroadcastw ymm9, word ptr [rsi + 8]
    vpbroadcastw ymm10, word ptr [rsi + 16]
    vpbroadcastw ymm11, word ptr [rsi + 24]
    vpbroadcastw ymm12, word ptr [rsi + 32]
    vpbroadcastw ymm13, word ptr [rsi + 40]
    vpbroadcastw ymm14, word ptr [rsi + 48]
    vpbroadcastw ymm15, word ptr [rsi + 56]
    vmovdqa ymm6, [rip + .Lhighbits]
    lea     r11, [rip + .Lnarrow]

.p2align 6
.Lnarrow:
    vmovdqu ymm0, [rdi + r9 * 8]
    vmovdqu ymm1, [rdi + r9 * 8 + 32]
    vmovdqu ymm2, [rdi + r9 * 8 + 64]
    vmovdqu ymm3, [rdi + r9 * 8 + 96]
    vpor    ymm4, ymm0, ymm1
    vpor    ymm5, ymm2, ymm3
    vpor    ymm4, ymm4, ymm5
    vptest  ymm4, ymm6
    jnz     .Lwidememory                # an element outside 0 .. 32,767
    vpackusdw ymm0, ymm0, ymm1
    vpackusdw ymm2, ymm2, ymm3
    vpackusdw ymm0, ymm0, ymm2          # sixteen keys
    vpcmpeqw ymm1, ymm0, ymm8
    vpcmpeqw ymm3, ymm0, ymm9
    vpcmpeqw ymm2, ymm0, ymm10
    vpor    ymm1, ymm1, ymm2
    vpor    ymm1, ymm1, ymm3
    vpcmpeqw ymm3, ymm0, ymm11
    vpcmpeqw ymm2, ymm0, ymm12
    vpor    ymm1, ymm1, ymm2
    vpor    ymm1, ymm1, ymm3
    vpcmpeqw ymm3, ymm0, ymm13
    vpcmpeqw ymm2, ymm0, ymm14
    vpor    ymm1, ymm1, ymm2
    vpor    ymm1, ymm1, ymm3
    vpcmpeqw ymm0, ymm0, ymm15
    vpor    ymm0, ymm0, ymm1

.Lemit:
    # ymm0: sixteen 16-bit lanes, all ones for a member, the low half holding
    # elements 0 1 4 5 8 9 12 13 and the high half 2 3 6 7 10 11 14 15.
    vpacksswb ymm0, ymm0, ymm0
    vextracti128 xmm1, ymm0, 1
    vpunpcklwd xmm0, xmm0, xmm1         # element order
    vpabsb  xmm0, xmm0                  # 0xff to 1
    vmovdqu [rdx + r9], xmm0
    cmp     r9, r10
    jae     .Ldone
    add     r9, 16
    cmp     r9, r10
    cmova   r9, r10                     # the last block ends with the last element
    jmp     r11

.Ldone:
    vzeroupper
    ret

    # ---- A block of a narrow call with an element that cannot be narrowed.
.Lwidememory:
    vmovdqu ymm4, [rdi + r9 * 8]
    vpbroadcastq ymm5, qword ptr [rsi]
    vpcmpeqq ymm0, ymm4, ymm5
    vpbroadcastq ymm5, qword ptr [rsi + 8]
    vpcmpeqq ymm5, ymm4, ymm5
    vpor    ymm0, ymm0, ymm5
    vpbroadcastq ymm5, qword ptr [rsi + 16]
    vpcmpeqq ymm5, ymm4, ymm5
    vpor    ymm0, ymm0, ymm5
    vpbroadcastq ymm5, qword ptr [rsi + 24]
    vpcmpeqq ymm5, ymm4, ymm5
    vpor    ymm0, ymm0, ymm5
    vpbroadcastq ymm5, qword ptr [rsi + 32]
    vpcmpeqq ymm5, ymm4, ymm5
    vpor    ymm0, ymm0, ymm5
    vpbroadcastq ymm5, qword ptr [rsi + 40]
    vpcmpeqq ymm5, ymm4, ymm5
    vpor    ymm0, ymm0, ymm5
    vpbroadcastq ymm5, qword ptr [rsi + 48]
    vpcmpeqq ymm5, ymm4, ymm5
    vpor    ymm0, ymm0, ymm5
    vpbroadcastq ymm5, qword ptr [rsi + 56]
    vpcmpeqq ymm5, ymm4, ymm5
    vpor    ymm0, ymm0, ymm5
    vmovdqu ymm4, [rdi + r9 * 8 + 32]
    vpbroadcastq ymm5, qword ptr [rsi]
    vpcmpeqq ymm1, ymm4, ymm5
    vpbroadcastq ymm5, qword ptr [rsi + 8]
    vpcmpeqq ymm5, ymm4, ymm5
    vpor    ymm1, ymm1, ymm5
    vpbroadcastq ymm5, qword ptr [rsi + 16]
    vpcmpeqq ymm5, ymm4, ymm5
    vpor    ymm1, ymm1, ymm5
    vpbroadcastq ymm5, qword ptr [rsi + 24]
    vpcmpeqq ymm5, ymm4, ymm5
    vpor    ymm1, ymm1, ymm5
    vpbroadcastq ymm5, qword ptr [rsi + 32]
    vpcmpeqq ymm5, ymm4, ymm5
    vpor    ymm1, ymm1, ymm5
    vpbroadcastq ymm5, qword ptr [rsi + 40]
    vpcmpeqq ymm5, ymm4, ymm5
    vpor    ymm1, ymm1, ymm5
    vpbroadcastq ymm5, qword ptr [rsi + 48]
    vpcmpeqq ymm5, ymm4, ymm5
    vpor    ymm1, ymm1, ymm5
    vpbroadcastq ymm5, qword ptr [rsi + 56]
    vpcmpeqq ymm5, ymm4, ymm5
    vpor    ymm1, ymm1, ymm5
    vmovdqu ymm4, [rdi + r9 * 8 + 64]
    vpbroadcastq ymm5, qword ptr [rsi]
    vpcmpeqq ymm2, ymm4, ymm5
    vpbroadcastq ymm5, qword ptr [rsi + 8]
    vpcmpeqq ymm5, ymm4, ymm5
    vpor    ymm2, ymm2, ymm5
    vpbroadcastq ymm5, qword ptr [rsi + 16]
    vpcmpeqq ymm5, ymm4, ymm5
    vpor    ymm2, ymm2, ymm5
    vpbroadcastq ymm5, qword ptr [rsi + 24]
    vpcmpeqq ymm5, ymm4, ymm5
    vpor    ymm2, ymm2, ymm5
    vpbroadcastq ymm5, qword ptr [rsi + 32]
    vpcmpeqq ymm5, ymm4, ymm5
    vpor    ymm2, ymm2, ymm5
    vpbroadcastq ymm5, qword ptr [rsi + 40]
    vpcmpeqq ymm5, ymm4, ymm5
    vpor    ymm2, ymm2, ymm5
    vpbroadcastq ymm5, qword ptr [rsi + 48]
    vpcmpeqq ymm5, ymm4, ymm5
    vpor    ymm2, ymm2, ymm5
    vpbroadcastq ymm5, qword ptr [rsi + 56]
    vpcmpeqq ymm5, ymm4, ymm5
    vpor    ymm2, ymm2, ymm5
    vmovdqu ymm4, [rdi + r9 * 8 + 96]
    vpbroadcastq ymm5, qword ptr [rsi]
    vpcmpeqq ymm3, ymm4, ymm5
    vpbroadcastq ymm5, qword ptr [rsi + 8]
    vpcmpeqq ymm5, ymm4, ymm5
    vpor    ymm3, ymm3, ymm5
    vpbroadcastq ymm5, qword ptr [rsi + 16]
    vpcmpeqq ymm5, ymm4, ymm5
    vpor    ymm3, ymm3, ymm5
    vpbroadcastq ymm5, qword ptr [rsi + 24]
    vpcmpeqq ymm5, ymm4, ymm5
    vpor    ymm3, ymm3, ymm5
    vpbroadcastq ymm5, qword ptr [rsi + 32]
    vpcmpeqq ymm5, ymm4, ymm5
    vpor    ymm3, ymm3, ymm5
    vpbroadcastq ymm5, qword ptr [rsi + 40]
    vpcmpeqq ymm5, ymm4, ymm5
    vpor    ymm3, ymm3, ymm5
    vpbroadcastq ymm5, qword ptr [rsi + 48]
    vpcmpeqq ymm5, ymm4, ymm5
    vpor    ymm3, ymm3, ymm5
    vpbroadcastq ymm5, qword ptr [rsi + 56]
    vpcmpeqq ymm5, ymm4, ymm5
    vpor    ymm3, ymm3, ymm5
    vpackssdw ymm0, ymm0, ymm1
    vpackssdw ymm2, ymm2, ymm3
    vpackssdw ymm0, ymm0, ymm2
    jmp     .Lemit

    # ---- A wide call.
.Lwidesetup:
    vpbroadcastq ymm8, qword ptr [rsi + 0]
    vpbroadcastq ymm9, qword ptr [rsi + 8]
    vpbroadcastq ymm10, qword ptr [rsi + 16]
    vpbroadcastq ymm11, qword ptr [rsi + 24]
    vpbroadcastq ymm12, qword ptr [rsi + 32]
    vpbroadcastq ymm13, qword ptr [rsi + 40]
    vpbroadcastq ymm14, qword ptr [rsi + 48]
    vpbroadcastq ymm15, qword ptr [rsi + 56]
    lea     r11, [rip + .Lwide]
.p2align 6
.Lwide:
    vmovdqu ymm4, [rdi + r9 * 8]
    vpcmpeqq ymm0, ymm4, ymm8
    vpcmpeqq ymm6, ymm4, ymm9
    vpor    ymm0, ymm0, ymm6
    vpcmpeqq ymm7, ymm4, ymm10
    vpor    ymm0, ymm0, ymm7
    vpcmpeqq ymm5, ymm4, ymm11
    vpor    ymm0, ymm0, ymm5
    vpcmpeqq ymm6, ymm4, ymm12
    vpor    ymm0, ymm0, ymm6
    vpcmpeqq ymm7, ymm4, ymm13
    vpor    ymm0, ymm0, ymm7
    vpcmpeqq ymm5, ymm4, ymm14
    vpor    ymm0, ymm0, ymm5
    vpcmpeqq ymm6, ymm4, ymm15
    vpor    ymm0, ymm0, ymm6
    vmovdqu ymm4, [rdi + r9 * 8 + 32]
    vpcmpeqq ymm1, ymm4, ymm8
    vpcmpeqq ymm6, ymm4, ymm9
    vpor    ymm1, ymm1, ymm6
    vpcmpeqq ymm7, ymm4, ymm10
    vpor    ymm1, ymm1, ymm7
    vpcmpeqq ymm5, ymm4, ymm11
    vpor    ymm1, ymm1, ymm5
    vpcmpeqq ymm6, ymm4, ymm12
    vpor    ymm1, ymm1, ymm6
    vpcmpeqq ymm7, ymm4, ymm13
    vpor    ymm1, ymm1, ymm7
    vpcmpeqq ymm5, ymm4, ymm14
    vpor    ymm1, ymm1, ymm5
    vpcmpeqq ymm6, ymm4, ymm15
    vpor    ymm1, ymm1, ymm6
    vmovdqu ymm4, [rdi + r9 * 8 + 64]
    vpcmpeqq ymm2, ymm4, ymm8
    vpcmpeqq ymm6, ymm4, ymm9
    vpor    ymm2, ymm2, ymm6
    vpcmpeqq ymm7, ymm4, ymm10
    vpor    ymm2, ymm2, ymm7
    vpcmpeqq ymm5, ymm4, ymm11
    vpor    ymm2, ymm2, ymm5
    vpcmpeqq ymm6, ymm4, ymm12
    vpor    ymm2, ymm2, ymm6
    vpcmpeqq ymm7, ymm4, ymm13
    vpor    ymm2, ymm2, ymm7
    vpcmpeqq ymm5, ymm4, ymm14
    vpor    ymm2, ymm2, ymm5
    vpcmpeqq ymm6, ymm4, ymm15
    vpor    ymm2, ymm2, ymm6
    vmovdqu ymm4, [rdi + r9 * 8 + 96]
    vpcmpeqq ymm3, ymm4, ymm8
    vpcmpeqq ymm6, ymm4, ymm9
    vpor    ymm3, ymm3, ymm6
    vpcmpeqq ymm7, ymm4, ymm10
    vpor    ymm3, ymm3, ymm7
    vpcmpeqq ymm5, ymm4, ymm11
    vpor    ymm3, ymm3, ymm5
    vpcmpeqq ymm6, ymm4, ymm12
    vpor    ymm3, ymm3, ymm6
    vpcmpeqq ymm7, ymm4, ymm13
    vpor    ymm3, ymm3, ymm7
    vpcmpeqq ymm5, ymm4, ymm14
    vpor    ymm3, ymm3, ymm5
    vpcmpeqq ymm6, ymm4, ymm15
    vpor    ymm3, ymm3, ymm6
    vpackssdw ymm0, ymm0, ymm1
    vpackssdw ymm2, ymm2, ymm3
    vpackssdw ymm0, ymm0, ymm2
    jmp     .Lemit

    # ---- Fewer than 16 elements.
.Lsmall:
    test    rcx, rcx
    jz      .Lreturn
    xor     r9d, r9d
.Lone:
    vpbroadcastq ymm0, qword ptr [rdi + r9 * 8]
    vpcmpeqq ymm1, ymm0, [rsi]
    vpcmpeqq ymm0, ymm0, [rsi + 32]
    vpor    ymm0, ymm0, ymm1
    vmovmskpd eax, ymm0
    test    eax, eax
    setnz   byte ptr [rdx + r9]
    inc     r9
    cmp     r9, rcx
    jb      .Lone
    vzeroupper
.Lreturn:
    ret
.size isin_i64, .-isin_i64

.section .note.GNU-stack,"",@progbits
