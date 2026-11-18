.intel_syntax noprefix
# double fmax_reduce_f64(const double *x, size_t n);
#
# rdi = x, rsi = n (at least 1). Returns the largest element that is not a
# NaN, and a NaN only if every element is one, as np.fmax.reduce gives it bit
# for bit. No element is a signaling NaN.
#
# NumPy's loop, which decides the sign of a zero maximum and the NaN of an
# array of NaNs: four running maxima, one per lane of a vector, all starting
# at x[0]; the elements from x[1] on in groups of four, each element into
# its own lane; the lanes combined, 0 with 2 and 1 with 3 and then the two
# results; the (n - 1) mod 4 leftover elements one at a time. In a lane and
# between lanes the later of two equal values is kept, among the leftover
# elements the earlier. This computes the same thing.
#
# `vmaxpd a, b` is "a if a > b, otherwise b": b when the two are equal and b
# when either is a NaN. With the earlier value as a and the later as b it
# keeps the later of two equal values; the other way round, the earlier. It
# does not ignore a NaN, so every element is first put through a `vmaxpd`
# against minus infinity, which turns a NaN into minus infinity and changes
# nothing else. The result is then NumPy's whenever one element is a number.
# Minus infinity as a result may stand for an array of NaNs, so that result
# is looked at again (.Lneginf).
#
# Keeping the later of two equal values is associative, so the groups may be
# combined with one another before they meet the running maxima, and a group
# may be taken twice as long as nothing earlier follows it. That gives the
# three paths, none of which branches on the data or on n mod 4:
#
#   n <= 4      no group: the first of the equal maxima of the array, from
#               one masked load.
#   5..16       one to three groups, as three loads: the first group, the
#               second or the first again, and the last.
#   17..        blocks of four groups; the last block is the last four
#               groups, and overlaps the block before it.
#
# The leftover elements are the tail of the last four elements of the array,
# which are loaded as one vector; the lanes before them are set to minus
# infinity, and the vector is reduced keeping the earlier of two equal
# values, as is its result against the lanes' result.

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
    lea     r10, [rip + .Llimits]
    cmp     rsi, 4
    jbe     .Lsmall
    vmovapd ymm4, [r10]                 # minus infinity in every lane
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
    #      when there are.
.Lneginf:
    vzeroupper
    xor     eax, eax
.Lscan:
    vmovsd  xmm1, qword ptr [rdi + rax * 8]
    vucomisd xmm1, xmm1
    jnp     .Lreturn                    # a number: minus infinity stands
    inc     rax
    cmp     rax, rsi
    jb      .Lscan
    vmovsd  xmm0, qword ptr [rdi]
    lea     rax, [rsi - 1]
    test    al, 3
    jz      .Lreturn                    # no leftover elements
    vmovsd  xmm0, qword ptr [rdi + rsi * 8 - 8]   # the last element
.Lreturn:
    ret
.size fmax_reduce_f64, .-fmax_reduce_f64

.section .note.GNU-stack,"",@progbits
