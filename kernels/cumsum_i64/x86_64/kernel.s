.intel_syntax noprefix
# void cumsum_i64(const int64_t *x, int64_t *out, size_t n);
#
# rdi = x, rsi = out, rdx = n. out[i] = x[0] + ... + x[i], wrapping on
# overflow as NumPy's does. Integer addition is exact in any order, so the
# additions are regrouped.
#
# Four elements per vector. Consecutive output vectors differ by the sum of
# the four elements ending at each lane:
#
#     out[i..i+3] - out[i-4..i-1] = W = x[i..i+3] + x[i-1..i+2] + x[i-2..i+1] + x[i-3..i]
#
# W takes two additions and needs nothing but the input:
#
#     D = x[i..i+3] + x[i-1..i+2]      the input plus the same load one element back
#     W = D + (D two elements back)    the upper half of the previous D and the
#                                      lower half of this one, one `vperm2i128`
#
# So the chain from vector to vector is one `vpaddq`, and everything else is
# done ahead of it. Before the array every value counts as zero: the first D
# shifts a zero into its first lane, and the D and the output before it are
# zero registers.
#
# The loads run ahead of the stores: the D of the next vector is loaded
# before the current output is stored. That keeps the kernel right when out
# is x itself, and it keeps a store just behind the input from holding up the
# loads that follow it.
#
# Paths, by n:
#   0..3   one element at a time
#   4..    vectors, two per step while three more are in reach and one per
#          step for the rest, then the last 0..3 elements one at a time

.text
.globl cumsum_i64
.type cumsum_i64, @function
.p2align 4
cumsum_i64:
    xor     eax, eax                    # the running sum of the scalar path
    xor     ecx, ecx                    # byte offset of the next output
    shl     rdx, 3                      # n in bytes
    cmp     rdx, 32
    jb      .Ltailtest
    vpxor   xmm0, xmm0, xmm0            # V: the last output vector
    vpxor   xmm1, xmm1, xmm1            # the D before the current one
    vmovdqu ymm2, [rdi]
    vpermq  ymm3, ymm2, 0x90            # x0 x0 x1 x2
    vpblendd ymm3, ymm3, ymm0, 0x03     # 0  x0 x1 x2
    vpaddq  ymm2, ymm2, ymm3            # D of the first vector
    cmp     rdx, 96
    jb      .Lsingle
    lea     r9, [rdx - 96]              # last offset from which three vectors are in reach
.p2align 6
.Lpair:
    # ymm0 = V, ymm1 = previous D, ymm2 = D of the vector at rcx.
    vperm2i128 ymm3, ymm1, ymm2, 0x21
    vpaddq  ymm3, ymm3, ymm2            # W
    vmovdqu ymm1, [rdi + rcx + 32]
    vpaddq  ymm1, ymm1, [rdi + rcx + 24]    # D of the next vector
    vpaddq  ymm0, ymm0, ymm3
    vperm2i128 ymm4, ymm2, ymm1, 0x21
    vpaddq  ymm4, ymm4, ymm1            # W of the next vector
    vmovdqu ymm2, [rdi + rcx + 64]
    vpaddq  ymm2, ymm2, [rdi + rcx + 56]    # D of the one after
    vmovdqu [rsi + rcx], ymm0
    vpaddq  ymm0, ymm0, ymm4
    vmovdqu [rsi + rcx + 32], ymm0
    add     rcx, 64
    cmp     rcx, r9
    jbe     .Lpair

.Lsingle:
    # The same, one vector per step. The vector at rcx is complete and its D
    # is in ymm2; the next D is loaded only if that vector is complete too.
    vperm2i128 ymm3, ymm1, ymm2, 0x21
    vpaddq  ymm3, ymm3, ymm2
    vpaddq  ymm0, ymm0, ymm3
    lea     rax, [rcx + 64]
    cmp     rax, rdx
    ja      .Llast
    vmovdqa ymm1, ymm2
    vmovdqu ymm2, [rdi + rcx + 32]
    vpaddq  ymm2, ymm2, [rdi + rcx + 24]
    vmovdqu [rsi + rcx], ymm0
    add     rcx, 32
    jmp     .Lsingle
.Llast:
    vmovdqu [rsi + rcx], ymm0
    add     rcx, 32
    vextracti128 xmm0, ymm0, 1
    vpextrq rax, xmm0, 1                # the last output
    vzeroupper

.Ltailtest:
    cmp     rcx, rdx
    jae     .Ldone
.Ltail:
    add     rax, [rdi + rcx]
    mov     [rsi + rcx], rax
    add     rcx, 8
    cmp     rcx, rdx
    jb      .Ltail
.Ldone:
    ret
.size cumsum_i64, .-cumsum_i64

.section .note.GNU-stack,"",@progbits
