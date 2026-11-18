.intel_syntax noprefix
# size_t compress_f32(const float *x, const bool *mask, float *out, size_t n);
#
# rdi = x, rsi = mask, rdx = out, rcx = n. Copies x[i] to out wherever mask[i]
# is not zero, in order, and returns how many were copied. `out` is a scratch
# buffer of n elements: stores may land anywhere inside it, and only the
# returned count of elements has to be right.
#
# Eight elements at a time. The eight mask bytes become eight bits, the bits
# index a table of 256 permutations, and `vpermps` moves the selected elements
# to the front of the register in one instruction. All eight lanes are stored
# at the write position, which then advances by the number selected, so the
# next store overwrites the lanes that were not. A store of eight lanes at
# position k covers out[k .. k+8), and k is never ahead of the read position,
# so it stays inside the n elements as long as the eight elements read do.
#
# The mask bits for 32 elements come from one compare and one `vpmovmskb`. A
# mask byte counts as true when it is not zero, whatever its value.
#
# Paths, by what is left:
#   32 or more  blocks of 32: one mask word, four table steps
#   8..31       one table step per 8, the mask bits from an 8-byte load
#   1..7        one element at a time, as the reference does it

# Offset of a table entry is 32 times the mask byte. The mask word is shifted
# left by this once per block, so that a group's offset is one `and` away.
.set ENTRY_SHIFT, 5
# The bits of the shifted mask word that hold one group's offset.
.set ENTRY_BITS, 0xff << ENTRY_SHIFT

# One entry of the permutation table: for mask byte \m, the indices of its set
# bits in ascending order, as eight 32-bit lanes. The lanes past the count are
# zero: they select element 0 of the group, which lands beyond the write
# position's advance and is overwritten or ignored.
.macro PERM_ENTRY m
    .set .Lcount, 0
    .irp bit, 0, 1, 2, 3, 4, 5, 6, 7
    .if ((\m) >> \bit) & 1
    .long \bit
    .set .Lcount, .Lcount + 1
    .endif
    .endr
    .fill 8 - .Lcount, 4, 0
.endm

.section .rodata
.p2align 6
.Lperm:
    .set .Lbyte, 0
    .rept 256
    PERM_ENTRY .Lbyte
    .set .Lbyte, .Lbyte + 1
    .endr

# One table step: compact the eight elements at \src by the group whose table
# offset is in the low bits of rax (already masked into r8 by the caller),
# store them at r9 and advance r9 by four bytes per selected element.
.macro STEP src
    vmovdqa ymm2, [r10 + r8]
    vpermps ymm2, ymm2, \src
    vmovups [r9], ymm2
    popcnt  r8d, r8d
    lea     r9, [r9 + r8 * 4]
.endm

.text
.globl compress_f32
.type compress_f32, @function
.p2align 4
compress_f32:
    mov     r9, rdx                     # write position
    lea     r10, [rip + .Lperm]
    vpxor   xmm0, xmm0, xmm0
    mov     r11, rcx
    shr     r11, 5                      # blocks of 32
    jz      .Lgroups
.p2align 6
.Lblock:
    vpcmpeqb ymm1, ymm0, [rsi]          # 0xff where the mask byte is zero
    vpmovmskb eax, ymm1
    not     eax                         # one bit per selected element
    shl     rax, ENTRY_SHIFT
    mov     r8d, eax
    and     r8d, ENTRY_BITS
    STEP    [rdi]
    rorx    r8, rax, 8
    and     r8d, ENTRY_BITS
    STEP    [rdi + 32]
    rorx    r8, rax, 16
    and     r8d, ENTRY_BITS
    STEP    [rdi + 64]
    shr     rax, 24
    and     eax, ENTRY_BITS
    vmovdqa ymm2, [r10 + rax]
    vpermps ymm2, ymm2, [rdi + 96]
    vmovups [r9], ymm2
    popcnt  eax, eax
    lea     r9, [r9 + rax * 4]
    add     rsi, 32
    sub     rdi, -128
    dec     r11
    jnz     .Lblock

.Lgroups:
    and     ecx, 31
    cmp     ecx, 8
    jb      .Ltail
.Lgroup:
    vmovq   xmm1, [rsi]
    vpcmpeqb xmm1, xmm1, xmm0
    vpmovmskb eax, xmm1
    not     eax
    movzx   r8d, al
    shl     r8d, ENTRY_SHIFT
    STEP    [rdi]
    add     rsi, 8
    add     rdi, 32
    sub     ecx, 8
    cmp     ecx, 8
    jae     .Lgroup

.Ltail:
    test    ecx, ecx
    jz      .Ldone
.Lone:
    mov     eax, [rdi]
    mov     [r9], eax
    xor     eax, eax
    cmp     byte ptr [rsi], 0
    setne   al
    lea     r9, [r9 + rax * 4]
    add     rdi, 4
    inc     rsi
    dec     ecx
    jnz     .Lone
.Ldone:
    mov     rax, r9
    sub     rax, rdx
    shr     rax, 2
    vzeroupper
    ret
.size compress_f32, .-compress_f32

.section .note.GNU-stack,"",@progbits
