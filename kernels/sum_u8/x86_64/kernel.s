.intel_syntax noprefix
# uint64_t sum_u8(const uint8_t *x, size_t n);
#
# rdi = x, rsi = n. Returns the sum of the n bytes at x.
#
# The work is done by `vpsadbw`: against a register of zeros it adds the 32
# bytes of a vector into four 64-bit lanes, eight bytes each. A lane grows by
# at most 8 * 255 per step, so nothing can overflow.
#
# Paths, by n:
#   0..15    general registers and one 16-byte register (two overlapping
#            loads, or a byte loop)
#   16..31   one whole 16-byte vector plus the masked last 16 bytes
#   32..127  64, 32 and a masked last vector, as the bits of n say
#   128..    the bytes up to the next 32-byte boundary as one masked vector,
#            then four accumulators, 128 bytes per step, every load aligned,
#            then the path above
#
# The loads are aligned because a buffer from NumPy usually starts 16 bytes
# into a 64-byte cache line. Every second 32-byte load then straddles two
# lines, which costs nothing while the data is in the first-level cache and a
# third of the speed once it is not.

.section .rodata
.p2align 6
# Loaded at offset 32 - h (h = 0..31): h bytes of ones, then 32 - h zero
# bytes. Applied to the first 32 bytes of the array it keeps the h bytes in
# front of the first 32-byte boundary.
.Lhead:
    .fill 32, 1, 0xff
# Loaded at offset r (1..31): 32 - r zero bytes, then r bytes of ones. Applied
# to the last 32 bytes of the array it keeps only the final r bytes, the ones
# no earlier vector covered. Loaded at offset 16 + r it is the same for the
# last 16 bytes.
.Lmask:
    .fill 32, 1, 0x00
    .fill 32, 1, 0xff

.text
.globl sum_u8
.type sum_u8, @function
.p2align 4
sum_u8:
    cmp     rsi, 32
    jb      .Lunder32
    vpxor   xmm0, xmm0, xmm0            # zeros to compare against
    vpxor   xmm1, xmm1, xmm1            # four 64-bit sums
    cmp     rsi, 128
    jb      .Lunder128

    # ---- The h = 0..31 bytes in front of the next 32-byte boundary: the
    #      first 32 bytes of the array, masked to its first h.
    mov     edx, edi
    neg     edx
    and     edx, 31                     # h
    lea     rcx, [rip + .Lmask]
    sub     rcx, rdx
    vmovdqu ymm2, [rdi]
    vpand   ymm2, ymm2, [rcx]           # .Lhead + 32 - h
    vpsadbw ymm1, ymm2, ymm0
    add     rdi, rdx
    sub     rsi, rdx
    cmp     rsi, 128
    jb      .Lunder128

    # ---- 128 bytes per step into four accumulators, loads aligned.
    vpxor   xmm2, xmm2, xmm2
    vpxor   xmm3, xmm3, xmm3
    vpxor   xmm4, xmm4, xmm4
    mov     rcx, rsi
    and     rcx, -128
    add     rcx, rdi                    # where the whole steps end
.Lstep:
    vpsadbw ymm5, ymm0, [rdi]
    vpsadbw ymm6, ymm0, [rdi + 32]
    vpsadbw ymm7, ymm0, [rdi + 64]
    vpsadbw ymm8, ymm0, [rdi + 96]
    sub     rdi, -128
    vpaddq  ymm1, ymm1, ymm5
    vpaddq  ymm2, ymm2, ymm6
    vpaddq  ymm3, ymm3, ymm7
    vpaddq  ymm4, ymm4, ymm8
    cmp     rdi, rcx
    jne     .Lstep
    vpaddq  ymm1, ymm1, ymm2
    vpaddq  ymm3, ymm3, ymm4
    vpaddq  ymm1, ymm1, ymm3

    # ---- Fewer than 128 bytes left: rdi = cursor, the low seven bits of rsi
    #      = bytes left, ymm1 = sums so far, ymm0 = zeros.
.Lunder128:
    test    sil, 64
    jnz     .Ldo64
.Lafter64:
    test    sil, 32
    jnz     .Ldo32
.Lafter32:
    test    sil, 31
    jnz     .Ldopart
.Lreduce:
    vextracti128 xmm2, ymm1, 1
    vpaddq  xmm1, xmm1, xmm2
    vpshufd xmm2, xmm1, 0xee
    vpaddq  xmm1, xmm1, xmm2
    vmovq   rax, xmm1
    vzeroupper
    ret

.Ldo64:
    vpsadbw ymm2, ymm0, [rdi]
    vpsadbw ymm3, ymm0, [rdi + 32]
    add     rdi, 64
    vpaddq  ymm1, ymm1, ymm2
    vpaddq  ymm1, ymm1, ymm3
    jmp     .Lafter64
.Ldo32:
    vpsadbw ymm2, ymm0, [rdi]
    add     rdi, 32
    vpaddq  ymm1, ymm1, ymm2
    jmp     .Lafter32
.Ldopart:
    # 1..31 bytes left, starting at rdi: load the last 32 bytes of the array
    # (it has at least 32) and keep only the ones not yet counted.
    and     esi, 31
    lea     rcx, [rip + .Lmask]
    vmovdqu ymm2, [rdi + rsi - 32]
    vpand   ymm2, ymm2, [rcx + rsi]
    vpsadbw ymm2, ymm2, ymm0
    vpaddq  ymm1, ymm1, ymm2
    jmp     .Lreduce

    # ---- Fewer than 32 bytes. No 256-bit register is written here.
.Lunder32:
    cmp     esi, 16
    jb      .Lunder16
    # 16..31: the first 16 bytes whole, then the last 16 masked to the
    # n - 16 bytes the first did not cover.
    vpxor   xmm0, xmm0, xmm0
    vmovdqu xmm1, [rdi]
    vpsadbw xmm1, xmm1, xmm0
    and     esi, 15
    jnz     .Ltwo
.Lreduce16:
    vpshufd xmm2, xmm1, 0xee
    vpaddq  xmm1, xmm1, xmm2
    vmovq   rax, xmm1
    ret
.Ltwo:
    lea     rcx, [rip + .Lmask + 16]
    vmovdqu xmm2, [rdi + rsi]           # the last 16 bytes: x + n - 16
    vpand   xmm2, xmm2, [rcx + rsi]
    vpsadbw xmm2, xmm2, xmm0
    vpaddq  xmm1, xmm1, xmm2
    jmp     .Lreduce16

.Lunder16:
    cmp     esi, 8
    jae     .L8to15
    cmp     esi, 4
    jae     .L4to7
    # 0..3: one byte at a time.
    xor     eax, eax
    test    esi, esi
    jz      .Ldone
.Lbyte:
    movzx   ecx, byte ptr [rdi]
    add     rax, rcx
    inc     rdi
    dec     esi
    jnz     .Lbyte
.Ldone:
    ret

.L8to15:
    # The first eight bytes and the last eight overlap by 16 - n bytes. The
    # overlap is the low-order end of the second load, so shifting it right
    # by 8 * (16 - n) bits removes it. That is 8..64 bits and a shift takes
    # 0..63, so it is done as a shift by 8 followed by one of 8 * (15 - n).
    # `shrx` uses its count modulo 64, and 8 * (15 - n) = -8 * (n + 1) there.
    mov     rax, [rdi]
    mov     rdx, [rdi + rsi - 8]
    not     esi                         # -(n + 1)
    shl     esi, 3
    shr     rdx, 8
    shrx    rdx, rdx, rsi
    vpxor   xmm0, xmm0, xmm0
    vmovq   xmm1, rax
    vmovq   xmm2, rdx
    vpsadbw xmm1, xmm1, xmm0
    vpsadbw xmm2, xmm2, xmm0
    vpaddq  xmm1, xmm1, xmm2
    vmovq   rax, xmm1
    ret

.L4to7:
    # The same with two 4-byte loads: the overlap is 8 * (8 - n) bits, done as
    # 8 and then 8 * (7 - n) = -8 * (n + 1) modulo 32.
    mov     eax, [rdi]
    mov     edx, [rdi + rsi - 4]
    not     esi
    shl     esi, 3
    shr     edx, 8
    shrx    edx, edx, esi
    shl     rdx, 32
    or      rax, rdx
    vpxor   xmm0, xmm0, xmm0
    vmovq   xmm1, rax
    vpsadbw xmm1, xmm1, xmm0
    vmovq   rax, xmm1
    ret
.size sum_u8, .-sum_u8

.section .note.GNU-stack,"",@progbits
