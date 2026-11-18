.intel_syntax noprefix
# void where_f64(const bool *c, const double *a, const double *b, double *out, size_t n);
#
# rdi = c, rsi = a, rdx = b, rcx = out, r8 = n. out[i] = c[i] ? a[i] : b[i].
#
# The elements are moved and never computed with: no floating-point
# arithmetic, compare or conversion touches the data of a or b, only loads,
# stores and bitwise and, and-not and or with a lane mask. A NaN keeps its
# payload and its signalling bit, a zero its sign, and 64-bit integers sent
# through as bit patterns arrive unchanged.
#
# A byte of c is true when it is not zero, whatever its value.
#
# Sixteen elements are one block. Their sixteen bytes of c are loaded once,
# into both halves of a register, and compared with zero: 0xff where c is
# false. Four `vpshufb` then spread the bytes of each group of four over the
# eight bytes of a 64-bit lane each, which gives the four lane masks, all
# ones where c is false. A group's output is (mask and b) or (a and not
# mask).
#
# The elements left after the whole blocks are done by one more block that
# ends exactly at the end of the four buffers. It writes again some elements
# the block before it wrote, with the same values: out never overlaps an
# input. Under sixteen elements the same is done with groups of four, whose
# masks come from `vpmovsxbq`.
#
# From ALIGN_MIN elements out is brought to a multiple of 32 bytes, so that
# no store straddles two cache lines: the first four elements are done as a
# group where out is, and the blocks start up to three elements in, where out
# is at such a multiple. An out that is not a multiple of 8 cannot be brought
# there.
#
# From STREAM_MIN_N elements the blocks store with `vmovntdq`. An ordinary
# store first reads the cache line it is about to overwrite; this one does
# not, which takes a quarter of the memory traffic away. The output is then in
# no cache when the kernel returns, which costs a caller nothing at a size no
# cache holds. These stores need out at a multiple of 32 bytes; an out that
# is not a multiple of 8 takes the ordinary stores.
#
# Paths, by n:
#   STREAM_MIN_N or more  the same blocks, stored past the caches
#   16 or more  blocks of 16, the last one ending at the end of the buffers
#   4..15       groups of 4, the last one ending at the end of the buffers
#   1..3        one element at a time, with a conditional move

.section .rodata
.p2align 6
# The `vpshufb` controls of the four groups of a block: group g takes bytes
# 4g and 4g+1 of the low half and bytes 4g+2 and 4g+3 of the high half, each
# eight times.
.Lspread:
    .irp g, 0, 1, 2, 3
    .fill 8, 1, 4 * \g
    .fill 8, 1, 4 * \g + 1
    .fill 8, 1, 4 * \g + 2
    .fill 8, 1, 4 * \g + 3
    .endr

# From this many elements out is brought to a multiple of 32 bytes: enough
# for the three elements that may take and one block after them.
.set ALIGN_MIN, 32
# From this many elements the stores go past the caches: 24 MB of output, the
# third-level cache of this processor.
.set STREAM_MIN_N, 3 * 1024 * 1024

# Select one group. \m = its lane mask, all ones where c is false, replaced
# by the output; \t = a scratch register. \a, \b, \o = the registers that
# point at the block in a, b and out; \k = the group's number in the block.
# \store is the instruction that stores the output.
.macro SELECT k, m, t, a, b, o, store
    vpand   \t, \m, [\b + 32 * (\k)]
    vpandn  \m, \m, [\a + 32 * (\k)]
    vpor    \m, \m, \t
    \store [\o + 32 * (\k)], \m
.endm

# One block of 16 elements. \c, \a, \b, \o = the registers that point at it
# in c, a, b and out. ymm0 = zeros, ymm12 to ymm15 = the controls.
.macro BLOCK c, a, b, o, store=vmovdqu
    vbroadcasti128 ymm1, [\c]           # 16 bytes of c in both halves
    vpcmpeqb ymm1, ymm1, ymm0           # 0xff where c is false
    vpshufb ymm2, ymm1, ymm12
    vpshufb ymm3, ymm1, ymm13
    vpshufb ymm4, ymm1, ymm14
    vpshufb ymm5, ymm1, ymm15
    SELECT 0, ymm2, ymm6, \a, \b, \o, \store
    SELECT 1, ymm3, ymm7, \a, \b, \o, \store
    SELECT 2, ymm4, ymm8, \a, \b, \o, \store
    SELECT 3, ymm5, ymm9, \a, \b, \o, \store
.endm

# Move the four pointers on by the up to three elements that bring out to a
# multiple of 32 bytes, after a first group has been done where they were.
# For an out that is not a multiple of 8 the count is as arbitrary as the
# place it leads to, and as harmless. r8 is free by now.
.macro ALIGN
    mov     r8d, ecx
    neg     r8d
    and     r8d, 24                     # bytes until out is a multiple of 32
    add     rsi, r8
    add     rdx, r8
    add     rcx, r8
    shr     r8d, 3                      # the same in elements
    add     rdi, r8
.endm

# One group of 4 elements at index \i: the arrays under 16 elements, and the
# first four elements of the others. ymm0 = zeros.
.macro GROUP i
    vpmovsxbq ymm1, dword ptr [rdi + \i]
    vpcmpeqq ymm1, ymm1, ymm0           # all ones where c is false
    vpand   ymm2, ymm1, [rdx + \i * 8]
    vpandn  ymm1, ymm1, [rsi + \i * 8]
    vpor    ymm1, ymm1, ymm2
    vmovdqu [rcx + \i * 8], ymm1
.endm

.text
.globl where_f64
.type where_f64, @function
.p2align 4
where_f64:
    cmp     r8, 16
    jb      .Lshort
    vpxor   xmm0, xmm0, xmm0
    vmovdqa ymm12, [rip + .Lspread]
    vmovdqa ymm13, [rip + .Lspread + 32]
    vmovdqa ymm14, [rip + .Lspread + 64]
    vmovdqa ymm15, [rip + .Lspread + 96]
    lea     r10, [rdi + r8 - 16]        # the last block, in each buffer
    lea     r11, [rsi + r8 * 8 - 128]
    lea     rax, [rdx + r8 * 8 - 128]
    lea     r9, [rcx + r8 * 8 - 128]
    cmp     r8, ALIGN_MIN
    jb      .Lfew
    GROUP   0                           # the first group, wherever out is
    cmp     r8, STREAM_MIN_N
    jae     .Lstream
    ALIGN
.p2align 6
.Lblock:
    BLOCK   rdi, rsi, rdx, rcx
    add     rdi, 16
    add     rsi, 128
    add     rdx, 128
    add     rcx, 128
    cmp     rdi, r10
    jb      .Lblock
.Llast:
    BLOCK   r10, r11, rax, r9
    vzeroupper
    ret

.Lfew:
    cmp     rdi, r10
    jb      .Lblock
    jmp     .Llast

    # Stores past the caches. An out that is not a multiple of 8 was not
    # brought to a multiple of 32 and takes the ordinary stores.
.Lstream:
    ALIGN
    test    cl, 7
    jnz     .Lblock
.p2align 6
.Lstreamblock:
    BLOCK   rdi, rsi, rdx, rcx, vmovntdq
    add     rdi, 16
    add     rsi, 128
    add     rdx, 128
    add     rcx, 128
    cmp     rdi, r10
    jb      .Lstreamblock
    sfence
    jmp     .Llast

.Lshort:
    xor     eax, eax
    cmp     r8, 4
    jb      .Ltail
    vpxor   xmm0, xmm0, xmm0
    lea     r9, [r8 - 4]                # the last group
    test    r9, r9
    jz      .Llastgroup
.Lgroup:
    GROUP   rax
    add     rax, 4
    cmp     rax, r9
    jb      .Lgroup
.Llastgroup:
    GROUP   r9
    vzeroupper
    ret

.Ltail:
    test    r8, r8
    jz      .Ldone
.Lone:
    mov     r9, [rdx + rax * 8]
    cmp     byte ptr [rdi + rax], 0
    cmovne  r9, [rsi + rax * 8]
    mov     [rcx + rax * 8], r9
    inc     rax
    cmp     rax, r8
    jb      .Lone
.Ldone:
    ret
.size where_f64, .-where_f64

.section .note.GNU-stack,"",@progbits
