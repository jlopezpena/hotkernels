.intel_syntax noprefix
# void cummax_i64(const int64_t *x, int64_t *out, size_t n);
#
# rdi = x, rsi = out, rdx = n. out[i] = the largest of x[0..i], compared as
# signed numbers. out does not overlap x.
#
# AVX2 has no 64-bit integer maximum: one costs a `vpcmpgtq`, which has a
# single port on this processor, and a blend of three operations. A running
# maximum computed in the register needs three of them per four elements and
# is slower than the reference's compare and branch. So the kernel does not
# compute the maximum where it can see that there is nothing to compute. It
# works in blocks of 16 elements and has three ways through a block, with M
# the maximum before it:
#
#   flat      no element of the block is above M: one compare per vector, and
#             the block is M sixteen times.
#   rising    no element of the block is below the element before it, and the
#             element before the block is M: one compare per vector against
#             the same load one element back, and the block is a copy of x.
#   general   anything else: four elements at a time in the integer registers
#             without a branch. The maxima of x0, x0..x1, x0..x2 and x0..x3
#             do not depend on M, so the chain from group to group is one
#             compare and one conditional move.
#
# The flat and rising loops run until a block fails their test; that block is
# done the general way. After a general block the next one is tried flat if M
# did not change in it, rising if its last element is M, and is done the
# general way without a test otherwise. So data that is neither pays for no
# test that fails, and no path has a branch that depends on a single element.
#
# From STREAM_MIN_BYTES of output the flat and rising loops store with
# `vmovntdq`. An ordinary store first reads the cache line it is about to
# overwrite; this one does not, which takes a third of the memory traffic
# away. The output is then in no cache when the kernel returns, which costs a
# caller nothing at a size no cache holds.
#
# rdi and rsi point at the ends of the arrays and rcx, negative, counts bytes
# up to zero.

# Bytes in a block: four vectors of four elements.
.set BLOCK, 128
# Bytes in a group of the general path: four elements.
.set GROUP_BYTES, 32
# From this many bytes the output is first brought to a multiple of 32: enough
# for the three elements that may take and one block after them.
.set ALIGN_MIN_BYTES, BLOCK + 24
# From this many bytes of output the flat and rising loops store past the
# caches: 24 MB, the third-level cache of this processor.
.set STREAM_MIN_BYTES, 24 * 1024 * 1024

# Four elements at rcx + \o the general way. rax = M, before and after.
.macro GROUP o
    mov     r8, [rdi + rcx + \o]
    mov     r9, [rdi + rcx + \o + 8]
    mov     r10, [rdi + rcx + \o + 16]
    mov     r11, [rdi + rcx + \o + 24]
    cmp     r9, r8
    cmovl   r9, r8                      # max of x0..x1
    cmp     r10, r9
    cmovl   r10, r9                     # max of x0..x2
    cmp     r11, r10
    cmovl   r11, r10                    # max of x0..x3
    cmp     r8, rax
    cmovl   r8, rax
    cmp     r9, rax
    cmovl   r9, rax
    cmp     r10, rax
    cmovl   r10, rax
    cmp     r11, rax
    cmovl   r11, rax
    mov     [rsi + rcx + \o], r8
    mov     [rsi + rcx + \o + 8], r9
    mov     [rsi + rcx + \o + 16], r10
    mov     [rsi + rcx + \o + 24], r11
    mov     rax, r11
.endm

# A block the general way. rax = M, before and after.
.macro GENERAL_BLOCK
    GROUP 0
    GROUP 32
    GROUP 64
    GROUP 96
    add     rcx, BLOCK
.endm

# The test of the flat loop: jumps to \fail if an element of the block at rcx
# is above M, which ymm0 holds in every lane.
.macro FLAT_TEST fail
    vmovdqu ymm1, [rdi + rcx]
    vmovdqu ymm2, [rdi + rcx + 32]
    vmovdqu ymm3, [rdi + rcx + 64]
    vmovdqu ymm4, [rdi + rcx + 96]
    vpcmpgtq ymm1, ymm1, ymm0
    vpcmpgtq ymm2, ymm2, ymm0
    vpcmpgtq ymm3, ymm3, ymm0
    vpcmpgtq ymm4, ymm4, ymm0
    vpor    ymm1, ymm1, ymm2
    vpor    ymm3, ymm3, ymm4
    vpor    ymm1, ymm1, ymm3
    vmovmskpd r8d, ymm1
    test    r8d, r8d
    jnz     \fail
.endm

# A flat block: M sixteen times. \store is the instruction that stores a vector.
.macro FLAT_STORE store=vmovdqu
    \store [rsi + rcx], ymm0
    \store [rsi + rcx + 32], ymm0
    \store [rsi + rcx + 64], ymm0
    \store [rsi + rcx + 96], ymm0
    add     rcx, BLOCK
.endm

# The rising loop: blocks are copied for as long as no element is below the
# one before it. The element before the first block is M. Leaves through
# \fail at a block that is not rising and through \end when fewer than 16
# elements are left; M has to be loaded again at both. \store is the
# instruction that stores a vector.
.macro RISING_LOOP fail, end, store=vmovdqu
1:
    vmovdqu ymm1, [rdi + rcx]
    vmovdqu ymm5, [rdi + rcx - 8]
    vmovdqu ymm2, [rdi + rcx + 32]
    vmovdqu ymm6, [rdi + rcx + 24]
    vmovdqu ymm3, [rdi + rcx + 64]
    vmovdqu ymm7, [rdi + rcx + 56]
    vmovdqu ymm4, [rdi + rcx + 96]
    vmovdqu ymm8, [rdi + rcx + 88]
    vpcmpgtq ymm5, ymm5, ymm1           # an element below the one before it
    vpcmpgtq ymm6, ymm6, ymm2
    vpcmpgtq ymm7, ymm7, ymm3
    vpcmpgtq ymm8, ymm8, ymm4
    vpor    ymm5, ymm5, ymm6
    vpor    ymm7, ymm7, ymm8
    vpor    ymm5, ymm5, ymm7
    vmovmskpd r8d, ymm5
    test    r8d, r8d
    jnz     \fail
    \store [rsi + rcx], ymm1
    \store [rsi + rcx + 32], ymm2
    \store [rsi + rcx + 64], ymm3
    \store [rsi + rcx + 96], ymm4
    add     rcx, BLOCK
    cmp     rcx, -BLOCK
    jle     1b
    jmp     \end
.endm

.text
.globl cummax_i64
.type cummax_i64, @function
.p2align 4
cummax_i64:
    test    rdx, rdx
    jz      .Ldone
    mov     rax, [rdi]                  # M
    shl     rdx, 3
    add     rdi, rdx
    add     rsi, rdx
    mov     rcx, rdx
    neg     rcx
    cmp     rcx, -BLOCK
    jg      .Ltail
    xor     r9d, r9d                    # 1 if the loops store past the caches
    cmp     rcx, -ALIGN_MIN_BYTES
    jg      .Lflagged
    # Up to three elements one at a time until out is a multiple of 32, so
    # that no store of a block straddles two cache lines. An out that is not
    # a multiple of 8 cannot be brought there and is left as it is; the
    # stores past the caches need the multiple of 32, so it does without.
    lea     r8d, [rsi + rcx]
    neg     r8d
    and     r8d, 31                     # bytes until out is a multiple of 32
    test    r8d, 7
    jnz     .Lflagged
    cmp     rcx, -STREAM_MIN_BYTES
    setle   r9b
    test    r8d, r8d
    jz      .Lflagged
    add     r8, rcx                     # where the head ends
.Lhead:
    mov     r10, [rdi + rcx]
    cmp     rax, r10
    cmovl   rax, r10
    mov     [rsi + rcx], rax
    add     rcx, 8
    cmp     rcx, r8
    jne     .Lhead
.Lflagged:
    mov     [rsp - 8], r9d              # kept below the stack pointer: every register is in use

    # ---- General: one block, then choose how to try the next.
.p2align 4
.Lgeneral:
    mov     rdx, rax                    # M before the block
    GENERAL_BLOCK
    cmp     rcx, -BLOCK
    jg      .Ltail
    cmp     rax, rdx
    je      .Lflatenter
    cmp     rax, [rdi + rcx - 8]
    jne     .Lgeneral

    # ---- Rising. The element before the block is M.
.Lrising:
    cmp     dword ptr [rsp - 8], 0
    jne     .Lrisingstream
.p2align 4
    RISING_LOOP .Lrisingfail, .Lrisingend
.Lrisingstream:
    RISING_LOOP .Lrisingstreamfail, .Lrisingstreamend, vmovntdq
    # Leaving a loop that stores past the caches: `sfence` orders those
    # stores before what follows.
.Lrisingstreamend:
    sfence
.Lrisingend:
    mov     rax, [rdi + rcx - 8]
    jmp     .Ltail
.Lrisingstreamfail:
    sfence
.Lrisingfail:
    mov     rax, [rdi + rcx - 8]
    jmp     .Lgeneral

.Lflatstreamfail:
    sfence
    # ---- A block of the flat loop that holds a new maximum: the general way,
    #      then back to the flat loop, since in data whose maximum rarely
    #      changes the next block is flat again. If it is not, the blocks
    #      after it are left to the general loop.
.Lflatfail:
    GENERAL_BLOCK
    cmp     rcx, -BLOCK
    jg      .Ltail
    cmp     rax, [rdi + rcx - 8]
    je      .Lrising
    vmovq   xmm0, rax
    vpbroadcastq ymm0, xmm0
    FLAT_TEST .Lgeneral
    FLAT_STORE
    cmp     rcx, -BLOCK
    jle     .Lflatchoose
    jmp     .Ltail

    # ---- Flat. ymm0 = M in every lane.
.Lflatenter:
    vmovq   xmm0, rax
    vpbroadcastq ymm0, xmm0
.Lflatchoose:
    cmp     dword ptr [rsp - 8], 0
    jne     .Lflatstream
.p2align 4
.Lflat:
    FLAT_TEST .Lflatfail
    FLAT_STORE
    cmp     rcx, -BLOCK
    jle     .Lflat
    jmp     .Ltail
.p2align 4
.Lflatstream:
    FLAT_TEST .Lflatstreamfail
    FLAT_STORE vmovntdq
    cmp     rcx, -BLOCK
    jle     .Lflatstream
    sfence

    # ---- Fewer than 16 elements left: groups of four, then one at a time.
.Ltail:
    cmp     rcx, -GROUP_BYTES
    jg      .Lsingles
.Ltailgroup:
    GROUP 0
    add     rcx, GROUP_BYTES
    cmp     rcx, -GROUP_BYTES
    jle     .Ltailgroup
.Lsingles:
    test    rcx, rcx
    jz      .Ldone
.Lsingle:
    mov     r8, [rdi + rcx]
    cmp     rax, r8
    cmovl   rax, r8
    mov     [rsi + rcx], rax
    add     rcx, 8
    jnz     .Lsingle
.Ldone:
    vzeroupper
    ret
.size cummax_i64, .-cummax_i64

.section .note.GNU-stack,"",@progbits
