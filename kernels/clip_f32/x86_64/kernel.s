.intel_syntax noprefix
# void clip_f32(const float *x, float *out, size_t n, float lo, float hi);
#
# rdi = x, rsi = out, rdx = n, xmm0 = lo, xmm1 = hi. out[i] = x[i] clamped to
# [lo, hi], as the reference computes it: v < lo ? lo : (v > hi ? hi : v).
# Valid for lo <= hi with neither bound a NaN.
#
# Two instructions per vector, and the result is the reference's bit for bit
# for every such pair of bounds. `vmaxps a, b` computes a > b ? a : b: it
# returns its second operand when the two are unordered and when they are
# equal, zeros of either sign included. So
#     t   = vmaxps lo, v      is  v < lo ? lo : v     (a NaN v stays; a zero v keeps
#                                                       its sign against a zero lo)
#     out = vminps hi, t      is  t > hi ? hi : t
# and since lo <= hi the second line leaves a t that came from lo alone. No
# path depends on the bounds or on the data, no status register is read, and
# a negative zero costs nothing. (The ARM64 kernel's `fmax` and `fmin` treat
# zeros and NaN differently, which is where its three caveats come from.)
#
# The stores are aligned to 32 bytes and the loads run a block ahead of them.
# One element can be clamped twice without harm, so alignment costs one
# vector that overlaps the next, and the leftover is a block that ends
# exactly at the end of the arrays. Loading ahead keeps a store just behind
# the input, 16 bytes after it within a page, from holding up the loads that
# follow it; the processor would take it for a store to the elements about to
# be loaded.
#
# From 24 MB of output, the size of this processor's third-level cache, the
# blocks are stored with `vmovntps`, which does not read the line it is
# about to overwrite.
#
# Paths, by n:
#   0..7       one element at a time
#   8..31    one vector per step, the last one ending with the last element
#   32..       one vector, then blocks of 32 from the next 32-byte boundary
#           of out, then one block ending with the last element
#
# Registers: r8 byte offset of the block, r9 n in bytes, r10 the last offset
# from which another whole block follows. ymm14 lo, ymm15 hi, ymm0-ymm3 the
# block waiting to be stored, ymm4-ymm7 the one after.

# From this many bytes of output the stores go past the caches: 24 MB, the
# third-level cache of this processor.
.set STREAM_MIN_BYTES, 24 * 1024 * 1024

.text
.globl clip_f32
.type clip_f32, @function
.p2align 4
clip_f32:
    cmp     rdx, 8
    jb      .Lscalar
    vbroadcastss ymm14, xmm0
    vbroadcastss ymm15, xmm1
    lea     r9, [rdx * 4]            # n in bytes
    cmp     rdx, 32
    jb      .Lvectors

    # The first vector, wherever out starts.
    vmaxps  ymm0, ymm14, [rdi]
    vminps  ymm0, ymm15, ymm0
    vmovups [rsi], ymm0
    # Where the blocks start: the next 32-byte boundary of out, 1..8 elements
    # on. If out is not a multiple of 4 there is none, and they start 8 on.
    mov     r8d, esi
    neg     r8d
    and     r8d, 31
    mov     eax, 32
    cmovz   r8d, eax
    test    r8d, 3
    cmovnz  r8d, eax
    lea     rax, [r8 + 128]
    cmp     rax, r9
    ja      .Llast                      # no whole block fits after it
    vmaxps  ymm0, ymm14, [rdi + r8]
    vminps  ymm0, ymm15, ymm0
    vmaxps  ymm1, ymm14, [rdi + r8 + 32]
    vminps  ymm1, ymm15, ymm1
    vmaxps  ymm2, ymm14, [rdi + r8 + 64]
    vminps  ymm2, ymm15, ymm2
    vmaxps  ymm3, ymm14, [rdi + r8 + 96]
    vminps  ymm3, ymm15, ymm3
    lea     rax, [r8 + 256]
    cmp     rax, r9
    ja      .Lpending
    lea     r10, [r9 - 256]             # last offset with a whole block after its own
    cmp     r9, STREAM_MIN_BYTES
    jb      .Lblock
    test    sil, 3
    jz      .Lstream

.p2align 6
.Lblock:
    vmaxps  ymm4, ymm14, [rdi + r8 + 128]
    vminps  ymm4, ymm15, ymm4
    vmaxps  ymm5, ymm14, [rdi + r8 + 128 + 32]
    vminps  ymm5, ymm15, ymm5
    vmaxps  ymm6, ymm14, [rdi + r8 + 128 + 64]
    vminps  ymm6, ymm15, ymm6
    vmaxps  ymm7, ymm14, [rdi + r8 + 128 + 96]
    vminps  ymm7, ymm15, ymm7
    vmovups [rsi + r8], ymm0
    vmovups [rsi + r8 + 32], ymm1
    vmovups [rsi + r8 + 64], ymm2
    vmovups [rsi + r8 + 96], ymm3
    vmovaps ymm0, ymm4
    vmovaps ymm1, ymm5
    vmovaps ymm2, ymm6
    vmovaps ymm3, ymm7
    add     r8, 128
    cmp     r8, r10
    jbe     .Lblock
    jmp     .Lpending

.p2align 6
.Lstream:
    vmaxps  ymm4, ymm14, [rdi + r8 + 128]
    vminps  ymm4, ymm15, ymm4
    vmaxps  ymm5, ymm14, [rdi + r8 + 128 + 32]
    vminps  ymm5, ymm15, ymm5
    vmaxps  ymm6, ymm14, [rdi + r8 + 128 + 64]
    vminps  ymm6, ymm15, ymm6
    vmaxps  ymm7, ymm14, [rdi + r8 + 128 + 96]
    vminps  ymm7, ymm15, ymm7
    vmovntps [rsi + r8], ymm0
    vmovntps [rsi + r8 + 32], ymm1
    vmovntps [rsi + r8 + 64], ymm2
    vmovntps [rsi + r8 + 96], ymm3
    vmovaps ymm0, ymm4
    vmovaps ymm1, ymm5
    vmovaps ymm2, ymm6
    vmovaps ymm3, ymm7
    add     r8, 128
    cmp     r8, r10
    jbe     .Lstream
    sfence                              # the stores above are ordered before what follows

.Lpending:
    vmovups [rsi + r8], ymm0
    vmovups [rsi + r8 + 32], ymm1
    vmovups [rsi + r8 + 64], ymm2
    vmovups [rsi + r8 + 96], ymm3

.Llast:
    # The block that ends with the last element.
    vmaxps  ymm0, ymm14, [rdi + r9 - 128]
    vminps  ymm0, ymm15, ymm0
    vmaxps  ymm1, ymm14, [rdi + r9 - 128 + 32]
    vminps  ymm1, ymm15, ymm1
    vmaxps  ymm2, ymm14, [rdi + r9 - 128 + 64]
    vminps  ymm2, ymm15, ymm2
    vmaxps  ymm3, ymm14, [rdi + r9 - 128 + 96]
    vminps  ymm3, ymm15, ymm3
    vmovups [rsi + r9 - 128], ymm0
    vmovups [rsi + r9 - 128 + 32], ymm1
    vmovups [rsi + r9 - 128 + 64], ymm2
    vmovups [rsi + r9 - 128 + 96], ymm3
    vzeroupper
    ret

    # ---- 8 to 31 elements: one vector per step.
.Lvectors:
    xor     r8d, r8d
    lea     r10, [r9 - 32]              # offset of the last vector
.Lvector:
    vmaxps  ymm0, ymm14, [rdi + r8]
    vminps  ymm0, ymm15, ymm0
    vmovups [rsi + r8], ymm0
    cmp     r8, r10
    jae     .Lvectorsdone
    add     r8, 32
    cmp     r8, r10
    cmova   r8, r10                     # the last vector ends with the last element
    jmp     .Lvector
.Lvectorsdone:
    vzeroupper
    ret

    # ---- Fewer than 8 elements.
.Lscalar:
    test    rdx, rdx
    jz      .Lreturn
    xor     r8d, r8d
.Lone:
    vmaxss  xmm2, xmm0, dword ptr [rdi + r8 * 4]
    vminss  xmm2, xmm1, xmm2
    vmovss  dword ptr [rsi + r8 * 4], xmm2
    inc     r8
    cmp     r8, rdx
    jb      .Lone
.Lreturn:
    ret
.size clip_f32, .-clip_f32

.section .note.GNU-stack,"",@progbits
