.intel_syntax noprefix
# void cumsum_bool(const bool *mask, int64_t *out, size_t n);
#
# rdi = mask, rsi = out, rdx = n. out[i] = how many of mask[0..i] are true.
# Every mask byte is 0 or 1.
#
# The reference adds one byte to a running count and stores it: a chain of
# one add per element. Here four elements are one step. Their four mask bits
# index a table of 16 entries, and an entry holds the two vectors a step
# needs: the running count of the four elements within the group (P), and
# their total in every lane (C). With T the count before the group in every
# lane, the output vector is T + P and the next T is T + C. The chain is one
# add per four elements, and the output is one add and one store.
#
# The mask bits of 32 elements come from one `vpmovmskb`. `pdep` spreads the
# 32 bits over eight bytes, four bits to the upper half of each, so that each
# byte is the group's table index times 16 and a scale of 4 in the address
# makes the 64 bytes of an entry.
#
# Paths, by what is left:
#   32 or more  blocks of 32: one mask word, eight steps
#   4..31       one step per four, the bits from a 4-byte load and `pext`
#   1..3        one element at a time, as the reference does it

# One table entry for the four mask bits \m: the count within the group, then
# the group's total four times.
.macro ENTRY m
    .set .Lb0, (\m) & 1
    .set .Lb1, ((\m) >> 1) & 1
    .set .Lb2, ((\m) >> 2) & 1
    .set .Lb3, ((\m) >> 3) & 1
    .quad .Lb0
    .quad .Lb0 + .Lb1
    .quad .Lb0 + .Lb1 + .Lb2
    .quad .Lb0 + .Lb1 + .Lb2 + .Lb3
    .fill 4, 8, .Lb0 + .Lb1 + .Lb2 + .Lb3
.endm

.section .rodata
.p2align 6
.Ltable:
    .set .Lbits, 0
    .rept 16
    ENTRY .Lbits
    .set .Lbits, .Lbits + 1
    .endr

# Where `pdep` puts the mask bits of a block: the upper half of each byte.
.set SPREAD, 0xF0F0F0F0F0F0F0F0
# The bits `pext` takes from four mask bytes: the lowest of each.
.set LOW_BITS, 0x01010101

# One step of a block. al = the group's table index times 16, ymm0 = T.
# \offset is where the group's output goes, in bytes from rsi.
.macro STEP offset
    movzx   ecx, al
    shr     rax, 8
    vpaddq  ymm1, ymm0, [r10 + rcx * 4]
    vpaddq  ymm0, ymm0, [r10 + rcx * 4 + 32]
    vmovdqu [rsi + \offset], ymm1
.endm

.text
.globl cumsum_bool
.type cumsum_bool, @function
.p2align 4
cumsum_bool:
    vpxor   xmm0, xmm0, xmm0            # T: the count so far, in every lane
    lea     r10, [rip + .Ltable]
    mov     r8, rdx
    and     r8, -32                     # elements in whole blocks
    jz      .Lfours
    movabs  r11, SPREAD
    lea     r9, [rdi + r8]              # where the blocks end
.p2align 6
.Lblock:
    vmovdqu ymm1, [rdi]
    vpsllw  ymm1, ymm1, 7               # the low bit of each byte to its top
    vpmovmskb eax, ymm1
    pdep    rax, rax, r11
    STEP 0
    STEP 32
    STEP 64
    STEP 96
    STEP 128
    STEP 160
    STEP 192
    STEP 224
    add     rdi, 32
    add     rsi, 256
    cmp     rdi, r9
    jne     .Lblock

.Lfours:
    and     edx, 31                     # elements left
    cmp     edx, 4
    jb      .Ltail
    mov     r11d, LOW_BITS
.Lfour:
    mov     eax, [rdi]
    pext    eax, eax, r11d
    shl     eax, 6
    vpaddq  ymm1, ymm0, [r10 + rax]
    vpaddq  ymm0, ymm0, [r10 + rax + 32]
    vmovdqu [rsi], ymm1
    add     rdi, 4
    add     rsi, 32
    sub     edx, 4
    cmp     edx, 4
    jae     .Lfour

.Ltail:
    test    edx, edx
    jz      .Ldone
    vmovq   rax, xmm0                   # the count so far
.Lone:
    movzx   ecx, byte ptr [rdi]
    add     rax, rcx
    mov     [rsi], rax
    inc     rdi
    add     rsi, 8
    dec     edx
    jnz     .Lone
.Ldone:
    vzeroupper
    ret
.size cumsum_bool, .-cumsum_bool

.section .note.GNU-stack,"",@progbits
