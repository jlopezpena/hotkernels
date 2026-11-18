.intel_syntax noprefix
# void cumsum_i64(const int64_t *x, int64_t *out, size_t n);
#
# rdi = x, rsi = out, rdx = n. out[i] = x[0] + ... + x[i], wrapping on
# overflow as NumPy's does. Integer addition is exact in any order, so the
# additions are regrouped: what one output vector adds to the one before it
# is built from the input alone, and the chain from vector to vector, one add
# per element in the reference, is one add per vector.
#
# Two versions of that, chosen by n.
#
# Wide, four elements per 32-byte vector:
#
#     out[i..i+3] - out[i-4..i-1] = W = x[i..i+3] + x[i-1..i+2] + x[i-2..i+1] + x[i-3..i]
#     D = x[i..i+3] + x[i-1..i+2]      the input plus the same load one element back
#     W = D + (D two elements back)    the upper half of the previous D and the
#                                      lower half of this one, one `vperm2i128`
#
# Narrow, two elements per 16-byte vector:
#
#     out[i..i+1] - out[i-2..i-1] = D = x[i..i+1] + x[i-1..i]
#
# Before the array every value counts as zero.
#
# Why two. A 32-byte load or store that straddles two cache lines costs a
# cycle or two on this processor, and the overlapping load of the wide
# version does so at every second vector whatever the alignment. The narrow
# version's 16-byte accesses rarely straddle, and it is the faster one while
# the data lies in the second-level cache. The wide version has half the
# steps: it is faster for short arrays, where the steps are what a call costs,
# and for arrays beyond the second-level cache, where memory sets the speed.
#
# In both, the loads run ahead of the stores: the D of a later vector is
# loaded before the current output is stored. That keeps the kernel right
# when out is x itself, and in the narrow version, where they run two vectors
# ahead, it keeps a store just behind the input from holding up the loads
# that follow it.
#
# From 24 MB of output, the size of this processor's third-level cache, the
# wide version stores with `vmovntdq`. An ordinary store first reads the
# cache line it is about to overwrite; this one does not, which takes a third
# of the memory traffic away. The output is then not in any cache when the
# kernel returns, which costs a caller nothing at a size no cache holds.
#
# Paths, by n:
#   0..4               one element at a time
#   5..511             wide
#   512..131,071       narrow
#   131,072..3,145,727 wide
#   3,145,728..        wide, 0..3 elements first to align out, stores past
#                      the caches

# From this many bytes of input the narrow version is used. Below it the
# array is short enough that the number of steps decides.
.set NARROW_MIN_BYTES, 512 * 8
# From this many bytes the wide version is used again: input and output
# together no longer fit the 2 MB second-level cache of this processor.
.set NARROW_END_BYTES, 131072 * 8
# From this many bytes of output the stores go past the caches: 24 MB, the
# third-level cache of this processor.
.set STREAM_MIN_BYTES, 24 * 1024 * 1024

# Two vectors of the wide version. ymm0 = V, the last output vector; ymm1 =
# the previous D; ymm2 = the D of the vector at rcx. \store is the instruction
# that stores an output vector.
.macro WIDEPAIR store
    vperm2i128 ymm3, ymm1, ymm2, 0x21
    vpaddq  ymm3, ymm3, ymm2            # W
    vmovdqu ymm1, [rdi + rcx + 32]
    vpaddq  ymm1, ymm1, [rdi + rcx + 24]    # D of the next vector
    vpaddq  ymm0, ymm0, ymm3
    vperm2i128 ymm4, ymm2, ymm1, 0x21
    vpaddq  ymm4, ymm4, ymm1            # W of the next vector
    vmovdqu ymm2, [rdi + rcx + 64]
    vpaddq  ymm2, ymm2, [rdi + rcx + 56]    # D of the one after
    \store [rsi + rcx], ymm0
    vpaddq  ymm0, ymm0, ymm4
    \store [rsi + rcx + 32], ymm0
.endm

.text
.globl cumsum_i64
.type cumsum_i64, @function
.p2align 4
cumsum_i64:
    xor     eax, eax                    # the running sum of the scalar path
    xor     ecx, ecx                    # byte offset of the next output
    shl     rdx, 3                      # n in bytes
    cmp     rdx, 40
    jb      .Ltailtest
    cmp     rdx, NARROW_MIN_BYTES
    jae     .Lchoose

    # ---- Wide. A short array reaches it without a taken branch.
.Lwide:
    vpxor   xmm0, xmm0, xmm0            # V: the last output vector
    vpxor   xmm1, xmm1, xmm1            # the D before the current one
    vmovdqu ymm2, [rdi]
    vpermq  ymm3, ymm2, 0x90            # x0 x0 x1 x2
    vpblendd ymm3, ymm3, ymm0, 0x03     # 0  x0 x1 x2
    vpaddq  ymm2, ymm2, ymm3            # D of the first vector
    cmp     rdx, 96
    jb      .Lwidesingle
    lea     r9, [rdx - 96]              # last offset from which three vectors are in reach
.p2align 6
.Lwidepair:
    WIDEPAIR vmovdqu
    add     rcx, 64
    cmp     rcx, r9
    jbe     .Lwidepair

.Lwidesingle:
    # The same, one vector per step. The vector at rcx is complete and its D
    # is in ymm2; the next D is loaded only if that vector is complete too.
    vperm2i128 ymm3, ymm1, ymm2, 0x21
    vpaddq  ymm3, ymm3, ymm2
    vpaddq  ymm0, ymm0, ymm3
    lea     rax, [rcx + 64]
    cmp     rax, rdx
    ja      .Lwidelast
    vmovdqa ymm1, ymm2
    vmovdqu ymm2, [rdi + rcx + 32]
    vpaddq  ymm2, ymm2, [rdi + rcx + 24]
    vmovdqu [rsi + rcx], ymm0
    add     rcx, 32
    jmp     .Lwidesingle
.Lwidelast:
    vmovdqu [rsi + rcx], ymm0
    add     rcx, 32
    vextracti128 xmm0, ymm0, 1
    vpextrq rax, xmm0, 1                # the last output
    vzeroupper
    jmp     .Ltailtest

.Lchoose:
    cmp     rdx, NARROW_END_BYTES
    jae     .Lbeyond

    # ---- Narrow. Element 0 is stored on its own and the vectors start at
    #      element 1, with [0, x[0]] as the output vector before them. No
    #      256-bit register is written on this path.
.Lnarrow:
    vmovq   xmm0, qword ptr [rdi]
    vmovdqu xmm1, [rdi + 8]
    vpaddq  xmm1, xmm1, [rdi]           # D of elements 1 and 2
    vmovdqu xmm2, [rdi + 24]
    vpaddq  xmm2, xmm2, [rdi + 16]      # D of elements 3 and 4
    vmovq   qword ptr [rsi], xmm0
    vpslldq xmm0, xmm0, 8               # 0, x[0]
    mov     ecx, 8
    lea     r9, [rdx - 96]              # last offset from which six vectors are in reach
.p2align 6
.Lnarrowquad:
    # xmm0 = the last output vector, xmm1 and xmm2 = D of the vectors at rcx
    # and rcx + 16. Each step adds a D, loads the D two vectors on, stores.
    vpaddq  xmm0, xmm0, xmm1
    vmovdqu xmm3, [rdi + rcx + 32]
    vpaddq  xmm3, xmm3, [rdi + rcx + 24]
    vmovdqu [rsi + rcx], xmm0
    vpaddq  xmm0, xmm0, xmm2
    vmovdqu xmm4, [rdi + rcx + 48]
    vpaddq  xmm4, xmm4, [rdi + rcx + 40]
    vmovdqu [rsi + rcx + 16], xmm0
    vpaddq  xmm0, xmm0, xmm3
    vmovdqu xmm1, [rdi + rcx + 64]
    vpaddq  xmm1, xmm1, [rdi + rcx + 56]
    vmovdqu [rsi + rcx + 32], xmm0
    vpaddq  xmm0, xmm0, xmm4
    vmovdqu xmm2, [rdi + rcx + 80]
    vpaddq  xmm2, xmm2, [rdi + rcx + 72]
    vmovdqu [rsi + rcx + 48], xmm0
    add     rcx, 64
    cmp     rcx, r9
    jbe     .Lnarrowquad

.Lnarrowsingle:
    # One vector per step. The vectors at rcx and rcx + 16 are complete and
    # their D are in xmm1 and xmm2; a further D is loaded only if its vector
    # is complete too.
    vpaddq  xmm0, xmm0, xmm1
    lea     rax, [rcx + 48]
    cmp     rax, rdx
    ja      .Lnarrowlast
    vmovdqa xmm1, xmm2
    vmovdqu xmm2, [rdi + rcx + 32]
    vpaddq  xmm2, xmm2, [rdi + rcx + 24]
    vmovdqu [rsi + rcx], xmm0
    add     rcx, 16
    jmp     .Lnarrowsingle
.Lnarrowlast:
    vmovdqu [rsi + rcx], xmm0
    vpaddq  xmm0, xmm0, xmm2
    vmovdqu [rsi + rcx + 16], xmm0
    add     rcx, 32
    vpextrq rax, xmm0, 1                # the last output
    jmp     .Ltailtest

    # ---- Beyond the second-level cache: wide, and from STREAM_MIN_BYTES with
    #      stores that go past the caches.
.Lbeyond:
    cmp     rdx, STREAM_MIN_BYTES
    jb      .Lwide
    # Such a store needs an address that is a multiple of 32: 0..3 elements
    # one at a time first. If out is not a multiple of 8 there is no such
    # address, and the ordinary wide path is used.
    mov     r8d, esi
    neg     r8d
    and     r8d, 31
    test    r8d, 7
    jnz     .Lwide
    test    r8d, r8d
    jz      .Lstreamstart
.Lstreamhead:
    add     rax, [rdi + rcx]
    mov     [rsi + rcx], rax
    add     rcx, 8
    cmp     rcx, r8
    jb      .Lstreamhead
.Lstreamstart:
    # The elements before rcx count as zero from here on, and their sum, in
    # rax, as the output before it in every lane.
    vmovq   xmm0, rax
    vpbroadcastq ymm0, xmm0
    vpxor   xmm1, xmm1, xmm1
    vmovdqu ymm2, [rdi + rcx]
    vpermq  ymm3, ymm2, 0x90            # x0 x0 x1 x2
    vpblendd ymm3, ymm3, ymm1, 0x03     # 0  x0 x1 x2
    vpaddq  ymm2, ymm2, ymm3            # D of the first vector
    lea     r9, [rdx - 96]
.p2align 6
.Lstreampair:
    WIDEPAIR vmovntdq
    add     rcx, 64
    cmp     rcx, r9
    jbe     .Lstreampair
    sfence                              # the stores above are ordered before what follows
    jmp     .Lwidesingle

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
