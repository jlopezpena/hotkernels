.intel_syntax noprefix
# void cast_u8_f32(const uint8_t *x, float *out, size_t n);
#
# rdi = x, rsi = out, rdx = n. out[i] = (float)x[i].
#
# Eight bytes per vector: `vpmovzxbd` widens them to 32-bit lanes straight
# from memory and `vcvtdq2ps` converts. Every value from 0 to 255 is exact.
#
# The stores are aligned to 32 bytes. An element can be converted twice
# without harm, so alignment costs one vector that overlaps the next, and the
# leftover is a block that ends exactly at the end of the arrays. From 24 MB
# of output, the size of this processor's third-level cache, the blocks are
# stored with `vmovntps`, which does not read the line it is about to
# overwrite.
#
# Paths, by n:
#   0..7    one element at a time
#   8..31   one vector per step, the last one ending with the last element
#   32..    one vector, then blocks of 32 from the next 32-byte boundary of
#           out, then one block ending with the last element

# From this many elements the stores go past the caches: 24 MB of output.
.set STREAM_MIN_N, 24 * 1024 * 1024 / 4

.text
.globl cast_u8_f32
.type cast_u8_f32, @function
.p2align 4
cast_u8_f32:
    cmp     rdx, 8
    jb      .Lscalar
    cmp     rdx, 32
    jb      .Lvectors
    vpmovzxbd ymm0, qword ptr [rdi]
    vcvtdq2ps ymm0, ymm0
    vmovups [rsi], ymm0
    # Where the blocks start: 1..8 elements on, at the next 32-byte boundary
    # of out, or 8 on if out is not a multiple of 4.
    mov     r8d, esi
    neg     r8d
    and     r8d, 31
    mov     eax, 32
    cmovz   r8d, eax
    test    r8d, 3
    cmovnz  r8d, eax
    shr     r8d, 2
    lea     r10, [rdx - 32]             # index of the last block
    cmp     r8, r10
    ja      .Llast
    cmp     rdx, STREAM_MIN_N
    jb      .Lblock
    test    sil, 3
    jz      .Lstream
.p2align 6
.Lblock:
    vpmovzxbd ymm0, qword ptr [rdi + r8]
    vcvtdq2ps ymm0, ymm0
    vpmovzxbd ymm1, qword ptr [rdi + r8 + 8]
    vcvtdq2ps ymm1, ymm1
    vpmovzxbd ymm2, qword ptr [rdi + r8 + 16]
    vcvtdq2ps ymm2, ymm2
    vpmovzxbd ymm3, qword ptr [rdi + r8 + 24]
    vcvtdq2ps ymm3, ymm3
    vmovups [rsi + r8 * 4], ymm0
    vmovups [rsi + r8 * 4 + 32], ymm1
    vmovups [rsi + r8 * 4 + 64], ymm2
    vmovups [rsi + r8 * 4 + 96], ymm3
    add     r8, 32
    cmp     r8, r10
    jbe     .Lblock
    jmp     .Llast
.p2align 6
.Lstream:
    vpmovzxbd ymm0, qword ptr [rdi + r8]
    vcvtdq2ps ymm0, ymm0
    vpmovzxbd ymm1, qword ptr [rdi + r8 + 8]
    vcvtdq2ps ymm1, ymm1
    vpmovzxbd ymm2, qword ptr [rdi + r8 + 16]
    vcvtdq2ps ymm2, ymm2
    vpmovzxbd ymm3, qword ptr [rdi + r8 + 24]
    vcvtdq2ps ymm3, ymm3
    vmovntps [rsi + r8 * 4], ymm0
    vmovntps [rsi + r8 * 4 + 32], ymm1
    vmovntps [rsi + r8 * 4 + 64], ymm2
    vmovntps [rsi + r8 * 4 + 96], ymm3
    add     r8, 32
    cmp     r8, r10
    jbe     .Lstream
    sfence
.Llast:
    vpmovzxbd ymm0, qword ptr [rdi + r10]
    vcvtdq2ps ymm0, ymm0
    vpmovzxbd ymm1, qword ptr [rdi + r10 + 8]
    vcvtdq2ps ymm1, ymm1
    vpmovzxbd ymm2, qword ptr [rdi + r10 + 16]
    vcvtdq2ps ymm2, ymm2
    vpmovzxbd ymm3, qword ptr [rdi + r10 + 24]
    vcvtdq2ps ymm3, ymm3
    vmovups [rsi + r10 * 4], ymm0
    vmovups [rsi + r10 * 4 + 32], ymm1
    vmovups [rsi + r10 * 4 + 64], ymm2
    vmovups [rsi + r10 * 4 + 96], ymm3
    vzeroupper
    ret

.Lvectors:
    xor     r8d, r8d
    lea     r10, [rdx - 8]
.Lvector:
    vpmovzxbd ymm0, qword ptr [rdi + r8]
    vcvtdq2ps ymm0, ymm0
    vmovups [rsi + r8 * 4], ymm0
    cmp     r8, r10
    jae     .Lvectorsdone
    add     r8, 8
    cmp     r8, r10
    cmova   r8, r10
    jmp     .Lvector
.Lvectorsdone:
    vzeroupper
    ret

.Lscalar:
    test    rdx, rdx
    jz      .Lreturn
    xor     r8d, r8d
.Lone:
    movzx   eax, byte ptr [rdi + r8]
    vcvtsi2ss xmm0, xmm0, eax
    vmovss  dword ptr [rsi + r8 * 4], xmm0
    inc     r8
    cmp     r8, rdx
    jb      .Lone
.Lreturn:
    ret
.size cast_u8_f32, .-cast_u8_f32

.section .note.GNU-stack,"",@progbits
