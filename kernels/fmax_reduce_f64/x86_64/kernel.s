.intel_syntax noprefix
# double fmax_reduce_f64(const double *x, size_t n);
#
# rdi = x, rsi = n (at least 1). Returns the largest element that is not a
# NaN, and a NaN only if every element is one, as np.fmax.reduce gives it bit
# for bit. No element is a signaling NaN.
#
# The value of the maximum does not depend on the order in which the
# elements are combined. Two things do, and NumPy's order decides them: the
# sign of a zero maximum when zeros of both signs are present, and which NaN
# an array of NaNs returns. So there are two ways through the array.
#
# The quick way (from 5 elements) takes the elements in any order, some of
# them twice. `vmaxpd v, m` is "v if v > m, otherwise m": m when v is a NaN,
# so with the maximum so far as m, starting at minus infinity, a NaN is
# ignored at one instruction per vector. The bits of every element are or-ed
# together on the side. Its result is NumPy's unless
#   - it is a zero and some element has its sign bit set. Then a negative
#     zero may be present, and NumPy's way is taken (.Lexact). A zero
#     maximum with no sign bit anywhere is the positive zero.
#   - it is minus infinity, which may stand for an array of NaNs (.Lneginf).
# Paths of the quick way, none of which branches on n inside its range:
#   5..16       four vectors: the first four elements, the last four, and
#               two between them.
#   17..127     blocks of four vectors into four maxima; the last block ends
#               with the array and overlaps the block before it.
#   128..       blocks of eight vectors into eight maxima, the same way; the
#               blocks between the first and the last start at multiples of
#               32 bytes.
#
# NumPy's way: four running maxima, one per lane of a vector, all starting
# at x[0]; the elements from x[1] on in groups of four, each element into
# its own lane; the lanes combined, 0 with 2 and 1 with 3 and then the two
# results; the (n - 1) mod 4 leftover elements one at a time. In a lane and
# between lanes the later of two equal values is kept, among the leftover
# elements the earlier.
#
# `vmaxpd a, b` with the earlier value as a and the later as b keeps the
# later of two equal values; the other way round, the earlier. Every element
# is first put through a `vmaxpd` against minus infinity, which turns a NaN
# into minus infinity and changes nothing else. Keeping the later of two
# equal values is associative, so the groups may be combined with one another
# before they meet the running maxima, and a group may be taken twice as long
# as nothing earlier follows it:
#   5..16       one to three groups, as three loads: the first group, the
#               second or the first again, and the last.
#   17..        blocks of four groups; the last block is the last four
#               groups, and overlaps the block before it.
# The leftover elements are the tail of the last four elements of the array,
# which are loaded as one vector; the lanes before them are set to minus
# infinity, and the vector is reduced keeping the earlier of two equal
# values, as is its result against the lanes' result.
#
# Up to four elements NumPy's way has no group, and its result is the first
# of the equal maxima of the array: one masked load, reduced keeping the
# earlier of two equal values (.Lsmall).

# From this many elements the quick way keeps eight maxima. Setting them up
# and merging them is a dozen instructions, which costs more than it gains
# at 64 elements.
.set WIDE_MIN_N, 128

.section .rodata
.p2align 5
# Minus infinity and plus infinity, read as vectors at an offset that
# depends on a count. With k the number of lanes wanted:
#   [.Llimits + 8 * k]         the last k lanes plus infinity, the others
#                              minus: `vminpd` with it keeps the last k lanes
#                              and sets the rest to minus infinity.
#   [.Llimits + 64 - 8 * k]    the same for the first k lanes.
#   [.Llimits + 32 - 8 * k]    the sign bit set in the first k lanes: the mask
#                              of a load of the first k elements.
.Llimits:
    .quad 0xfff0000000000000, 0xfff0000000000000, 0xfff0000000000000, 0xfff0000000000000
    .quad 0x7ff0000000000000, 0x7ff0000000000000, 0x7ff0000000000000, 0x7ff0000000000000
    .quad 0xfff0000000000000, 0xfff0000000000000, 0xfff0000000000000, 0xfff0000000000000
.Lzero:
    .quad 0

# Four vectors at \p into the four maxima ymm0, ymm7, ymm8 and ymm9, which are
# never a NaN, and their bits into ymm6 and ymm10. `vmaxpd v, m` is m when v
# is a NaN.
.macro QUICK p
    vmovupd ymm1, [\p]
    vmovupd ymm2, [\p + 32]
    vmovupd ymm3, [\p + 64]
    vmovupd ymm5, [\p + 96]
    vmaxpd  ymm0, ymm1, ymm0
    vmaxpd  ymm7, ymm2, ymm7
    vmaxpd  ymm8, ymm3, ymm8
    vmaxpd  ymm9, ymm5, ymm9
    vorpd   ymm1, ymm1, ymm2
    vorpd   ymm3, ymm3, ymm5
    vorpd   ymm6, ymm6, ymm1
    vorpd   ymm10, ymm10, ymm3
.endm

# Eight vectors at \p: as QUICK, the second four into the maxima ymm11 to
# ymm14.
.macro WIDE p
    QUICK \p
    vmovupd ymm1, [\p + 128]
    vmovupd ymm2, [\p + 160]
    vmovupd ymm3, [\p + 192]
    vmovupd ymm5, [\p + 224]
    vmaxpd  ymm11, ymm1, ymm11
    vmaxpd  ymm12, ymm2, ymm12
    vmaxpd  ymm13, ymm3, ymm13
    vmaxpd  ymm14, ymm5, ymm14
    vorpd   ymm1, ymm1, ymm2
    vorpd   ymm3, ymm3, ymm5
    vorpd   ymm6, ymm6, ymm1
    vorpd   ymm10, ymm10, ymm3
.endm

# Four groups at \p, cleaned of NaN, combined with one another and then with
# the running maxima in ymm0. ymm4 = minus infinity in every lane.
.macro BLOCK p
    vmovupd ymm1, [\p]
    vmovupd ymm2, [\p + 32]
    vmovupd ymm3, [\p + 64]
    vmovupd ymm5, [\p + 96]
    vmaxpd  ymm1, ymm1, ymm4            # NaN to minus infinity
    vmaxpd  ymm2, ymm2, ymm4
    vmaxpd  ymm3, ymm3, ymm4
    vmaxpd  ymm5, ymm5, ymm4
    vmaxpd  ymm1, ymm1, ymm2
    vmaxpd  ymm3, ymm3, ymm5
    vmaxpd  ymm1, ymm1, ymm3
    vmaxpd  ymm0, ymm0, ymm1
.endm

# The vector ymm1, its lanes in the order of the array, reduced to its low
# lane keeping the earlier of two equal values. ymm2 is scratch.
.macro EARLIEST
    vpermilpd ymm2, ymm1, 5             # the lanes of each half swapped
    vmaxpd  ymm1, ymm2, ymm1            # lane 0 against 1, lane 2 against 3
    vextractf128 xmm2, ymm1, 1
    vmaxsd  xmm1, xmm2, xmm1            # the upper half against the lower
.endm

.text
.globl fmax_reduce_f64
.type fmax_reduce_f64, @function
.p2align 4
fmax_reduce_f64:
    cmp     rsi, 4
    jbe     .Lsmall
    vmovapd ymm4, [rip + .Llimits]      # minus infinity in every lane
    cmp     rsi, 16
    ja      .Lquickblocks
    # ---- 5 to 16 elements: four vectors, the first four elements, the last
    #      four, and between them the second four and the four before the
    #      last, or the nearest of the two outer ones again.
    lea     rax, [rsi * 8 - 32]         # where the last vector starts
    mov     edx, 32
    cmp     rax, rdx
    cmovb   rdx, rax                    # the second vector: 32 bytes in, or the last
    mov     rcx, rax
    sub     rcx, rdx                    # the third: as far before the last
    vmovupd ymm1, [rdi]
    vmovupd ymm2, [rdi + rdx]
    vmovupd ymm3, [rdi + rcx]
    vmovupd ymm5, [rdi + rax]
    vmaxpd  ymm0, ymm1, ymm4            # minus infinity for a NaN
    vmaxpd  ymm0, ymm2, ymm0            # the maximum so far for a NaN
    vmaxpd  ymm0, ymm3, ymm0
    vmaxpd  ymm0, ymm5, ymm0
    vorpd   ymm1, ymm1, ymm2
    vorpd   ymm3, ymm3, ymm5
    vorpd   ymm6, ymm1, ymm3
    # ---- ymm0 = four maxima, ymm6 = the bits of every element, or-ed.
.Lquickfinish:
    vextractf128 xmm1, ymm0, 1
    vmaxpd  xmm0, xmm0, xmm1
    vunpckhpd xmm1, xmm0, xmm0
    vmaxsd  xmm0, xmm0, xmm1
    vucomisd xmm0, qword ptr [rip + .Lzero]
    je      .Lquickzero
    vucomisd xmm0, xmm4
    je      .Lneginf                    # perhaps every element is a NaN
    vzeroupper
    ret
    # ---- The maximum is a zero. If no element has its sign bit set, there
    #      is no negative zero and the result is the positive one, which is
    #      what xmm0 holds. Otherwise the sign is NumPy's to decide.
.Lquickzero:
    vmovmskpd eax, ymm6
    test    eax, eax
    jnz     .Lexact
    vzeroupper
    ret

    # ---- From 17 elements: blocks of four vectors into four maxima, the
    #      last block ending with the array.
.Lquickblocks:
    cmp     rsi, WIDE_MIN_N
    jae     .Lwide
    lea     rdx, [rdi + rsi * 8 - 128]  # the last block
    vmovupd ymm1, [rdi]
    vmovupd ymm2, [rdi + 32]
    vmovupd ymm3, [rdi + 64]
    vmovupd ymm5, [rdi + 96]
    vmaxpd  ymm0, ymm1, ymm4            # minus infinity for a NaN
    vmaxpd  ymm7, ymm2, ymm4
    vmaxpd  ymm8, ymm3, ymm4
    vmaxpd  ymm9, ymm5, ymm4
    vorpd   ymm6, ymm1, ymm2
    vorpd   ymm10, ymm3, ymm5
    lea     rax, [rdi + 128]
    cmp     rax, rdx
    jae     .Lquicklast
.p2align 4
.Lquickblock:
    QUICK rax
    add     rax, 128
    cmp     rax, rdx
    jb      .Lquickblock
.Lquicklast:
    QUICK rdx
.Lquickmerge:
    vmaxpd  ymm0, ymm0, ymm7
    vmaxpd  ymm8, ymm8, ymm9
    vmaxpd  ymm0, ymm0, ymm8
    vorpd   ymm6, ymm6, ymm10
    jmp     .Lquickfinish

    # ---- From WIDE_MIN_N elements: blocks of eight vectors into eight
    #      maxima, so that the chain through a maximum is four cycles per 64
    #      elements. The last block ends with the array.
.Lwide:
    lea     rdx, [rdi + rsi * 8 - 256]  # the last block
    # The first block, at x, starts the eight maxima.
    vmovupd ymm1, [rdi]
    vmovupd ymm2, [rdi + 32]
    vmovupd ymm3, [rdi + 64]
    vmovupd ymm5, [rdi + 96]
    vmaxpd  ymm0, ymm1, ymm4            # minus infinity for a NaN
    vmaxpd  ymm7, ymm2, ymm4
    vmaxpd  ymm8, ymm3, ymm4
    vmaxpd  ymm9, ymm5, ymm4
    vorpd   ymm6, ymm1, ymm2
    vorpd   ymm10, ymm3, ymm5
    vmovupd ymm1, [rdi + 128]
    vmovupd ymm2, [rdi + 160]
    vmovupd ymm3, [rdi + 192]
    vmovupd ymm5, [rdi + 224]
    vmaxpd  ymm11, ymm1, ymm4
    vmaxpd  ymm12, ymm2, ymm4
    vmaxpd  ymm13, ymm3, ymm4
    vmaxpd  ymm14, ymm5, ymm4
    vorpd   ymm1, ymm1, ymm2
    vorpd   ymm3, ymm3, ymm5
    vorpd   ymm6, ymm6, ymm1
    vorpd   ymm10, ymm10, ymm3
    # The blocks after it start at a multiple of 32 bytes, the first of them
    # inside the first block, so that no load of the loop straddles two
    # cache lines. An x that is not at a multiple of 8 has no such boundary
    # between elements, and its blocks follow the first one directly.
    lea     rax, [rdi + 256]
    mov     ecx, eax
    and     ecx, 31
    xor     r8d, r8d
    test    cl, 7
    cmovnz  ecx, r8d
    sub     rax, rcx
    cmp     rax, rdx
    jae     .Lwidelast
.p2align 4
.Lwideblock:
    WIDE rax
    add     rax, 256
    cmp     rax, rdx
    jb      .Lwideblock
.Lwidelast:
    WIDE rdx
    vmaxpd  ymm0, ymm0, ymm11
    vmaxpd  ymm7, ymm7, ymm12
    vmaxpd  ymm8, ymm8, ymm13
    vmaxpd  ymm9, ymm9, ymm14
    jmp     .Lquickmerge

    # ---- NumPy's loop, for the sign of a zero. At least 5 elements.
.Lexact:
    lea     r10, [rip + .Llimits]
    vbroadcastsd ymm0, qword ptr [rdi]
    vmaxpd  ymm0, ymm0, ymm4            # x[0] in every lane, NaN to minus infinity
    lea     rcx, [rsi - 1]
    mov     rax, rcx
    shr     rax, 2                      # the number of groups, at least 1
    shl     rax, 5                      # their size in bytes
    cmp     rax, 128
    jae     .Lblocks
    # ---- One to three groups: the first, the second or the first again,
    #      and the last.
    lea     r8, [rax - 32]              # where the last group starts: 0, 32 or 64
    mov     r9d, 32
    cmp     r8, r9
    cmovb   r9, r8                      # the second group, or the first
    vmovupd ymm1, [rdi + 8]
    vmovupd ymm2, [rdi + r9 + 8]
    vmovupd ymm3, [rdi + r8 + 8]
    vmaxpd  ymm1, ymm1, ymm4            # NaN to minus infinity
    vmaxpd  ymm2, ymm2, ymm4
    vmaxpd  ymm3, ymm3, ymm4
    vmaxpd  ymm0, ymm0, ymm1
    vmaxpd  ymm2, ymm2, ymm3
    vmaxpd  ymm0, ymm0, ymm2
    # ---- The lanes combined: 0 with 2, 1 with 3, then the two results, the
    #      second of two equal values each time.
.Lfinish:
    vextractf128 xmm1, ymm0, 1
    vmaxpd  xmm0, xmm0, xmm1
    vunpckhpd xmm1, xmm0, xmm0
    vmaxsd  xmm0, xmm0, xmm1
    # ---- The leftover elements, in the last four of the array.
    and     ecx, 3
    vmovupd ymm1, [rdi + rsi * 8 - 32]
    vmaxpd  ymm1, ymm1, ymm4            # NaN to minus infinity
    vminpd  ymm1, ymm1, [r10 + rcx * 8] # all but the leftover elements to minus infinity
    EARLIEST
    vmaxsd  xmm0, xmm1, xmm0            # the lanes' result stays if it is equal
    vucomisd xmm0, xmm4
    je      .Lneginf
    vzeroupper
    ret

    # ---- Blocks of four groups. rax = the size of the groups in bytes.
.Lblocks:
    lea     rdx, [rdi + rax + 8 - 128]  # the last four groups
    lea     r11, [rdi + 8]
    cmp     r11, rdx
    jae     .Llastblock
.p2align 4
.Lblock:
    BLOCK r11
    add     r11, 128
    cmp     r11, rdx
    jb      .Lblock
.Llastblock:
    BLOCK rdx
    jmp     .Lfinish

    # ---- Up to four elements: the first of the equal maxima. The masked
    #      load reads the first n elements and nothing else.
.Lsmall:
    lea     r10, [rip + .Llimits]
    lea     rax, [rsi * 8]
    neg     rax
    vmovupd ymm2, [r10 + rax + 32]      # the sign bit in the first n lanes
    vmaskmovpd ymm1, ymm2, [rdi]
    vmaxpd  ymm1, ymm1, [r10]           # NaN to minus infinity
    vminpd  ymm1, ymm1, [r10 + rax + 64]    # the lanes past the array to minus infinity
    vpermilpd ymm2, ymm1, 5             # as EARLIEST, the result in xmm0
    vmaxpd  ymm1, ymm2, ymm1
    vextractf128 xmm2, ymm1, 1
    vmaxsd  xmm0, xmm2, xmm1
    vucomisd xmm0, qword ptr [r10]
    je      .Lneginf
    vzeroupper
    ret

    # ---- The result is minus infinity: right if one element is a number.
    #      If every element is a NaN, NumPy's result is x[0] when there are
    #      no leftover elements ((n - 1) mod 4 is 0) and the last element
    #      when there are. From 5 elements the array is gone through again
    #      a vector at a time, the last vector ending with the array.
.Lneginf:
    cmp     rsi, 4
    jbe     .Lscansmall
    lea     rdx, [rdi + rsi * 8 - 32]
    vcmpordpd ymm1, ymm4, [rdx]         # set in the lanes that hold a number
    mov     rax, rdi
.Lscan:
    vcmpordpd ymm2, ymm4, [rax]
    vorpd   ymm1, ymm1, ymm2
    add     rax, 32
    cmp     rax, rdx
    jb      .Lscan
    vmovmskpd eax, ymm1
    vzeroupper
    test    eax, eax
    jnz     .Lreturn                    # a number: minus infinity stands
    jmp     .Lallnan
.Lscansmall:
    vzeroupper
    xor     eax, eax
.Lscanone:
    vmovsd  xmm1, qword ptr [rdi + rax * 8]
    vucomisd xmm1, xmm1
    jnp     .Lreturn                    # a number: minus infinity stands
    inc     rax
    cmp     rax, rsi
    jb      .Lscanone
.Lallnan:
    vmovsd  xmm0, qword ptr [rdi]
    lea     rax, [rsi - 1]
    test    al, 3
    jz      .Lreturn                    # no leftover elements
    vmovsd  xmm0, qword ptr [rdi + rsi * 8 - 8]   # the last element
.Lreturn:
    ret
.size fmax_reduce_f64, .-fmax_reduce_f64

.section .note.GNU-stack,"",@progbits
