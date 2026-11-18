.intel_syntax noprefix
# size_t compress_f64(const double *x, const bool *mask, double *out, size_t n);
#
# rdi = x, rsi = mask, rdx = out, rcx = n. Copies x[i] to out wherever mask[i]
# is not zero, in order, and returns how many were copied. `out` is a scratch
# buffer of n elements: stores may land anywhere inside it, and only the
# returned count of elements has to be right.
#
# Four elements at a time. Four mask bytes become four bits, the bits index a
# table of 16 permutations, and `vpermd` moves the selected elements to the
# front of the register (an element is two of its 32-bit lanes; AVX2 has no
# permute of 64-bit lanes by a variable index). All four are stored at the
# write position, which then advances by the number selected, so the next
# store overwrites the ones that were not. A store of four elements at
# position k covers out[k .. k+4), and k is never ahead of the read position,
# so it stays inside the n elements as long as the four elements read do.
#
# The mask bits for 32 elements come from one compare and one `vpmovmskb`. A
# mask byte counts as true when it is not zero, whatever its value.
#
# Paths, by what is left:
#   32 or more  blocks of 32: one mask word, eight table steps
#   4..31       one table step per 4, the mask bits from a 4-byte load
#   1..3        one element at a time, as the reference does it

# Offset of a table entry is 32 times the four mask bits. The mask word is
# shifted left by this once per block, so that a group's offset is one rotate
# and one `and` away.
.set ENTRY_SHIFT, 5
# The bits of the shifted mask word that hold one group's offset.
.set ENTRY_BITS, 0xf << ENTRY_SHIFT

# One entry of the permutation table: for mask bits \m, the 32-bit lanes of
# the selected elements in ascending order. The lanes past the count are zero:
# they select the low half of element 0, which lands beyond the write
# position's advance and is overwritten or ignored.
.macro PERM_ENTRY m
    .set .Lcount, 0
    .irp bit, 0, 1, 2, 3
    .if ((\m) >> \bit) & 1
    .long 2 * \bit, 2 * \bit + 1
    .set .Lcount, .Lcount + 1
    .endif
    .endr
    .fill 2 * (4 - .Lcount), 4, 0
.endm

.section .rodata
.p2align 6
.Lperm:
    .set .Lbits, 0
    .rept 16
    PERM_ENTRY .Lbits
    .set .Lbits, .Lbits + 1
    .endr

# One table step: compact the four elements at \src by the group whose table
# offset is in r8, store them at element r9 of the output and advance r9 by
# the number selected.
#
# r9 counts elements, and the store scales it. Counting bytes would put a
# `lea` with a scaled index on the one chain that runs from step to step, and
# on this processor that `lea` takes two cycles where the `add` takes one.
.macro STEP src
    vmovdqa ymm2, [r10 + r8]
    vpermd  ymm2, ymm2, \src
    vmovdqu [rdx + r9 * 8], ymm2
    popcnt  r8d, r8d
    add     r9, r8
.endm

.text
.globl compress_f64
.type compress_f64, @function
.p2align 4
compress_f64:
    xor     r9d, r9d                    # elements written
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
    rorx    r8, rax, 4
    and     r8d, ENTRY_BITS
    STEP    [rdi + 32]
    rorx    r8, rax, 8
    and     r8d, ENTRY_BITS
    STEP    [rdi + 64]
    rorx    r8, rax, 12
    and     r8d, ENTRY_BITS
    STEP    [rdi + 96]
    rorx    r8, rax, 16
    and     r8d, ENTRY_BITS
    STEP    [rdi + 128]
    rorx    r8, rax, 20
    and     r8d, ENTRY_BITS
    STEP    [rdi + 160]
    rorx    r8, rax, 24
    and     r8d, ENTRY_BITS
    STEP    [rdi + 192]
    rorx    r8, rax, 28
    and     r8d, ENTRY_BITS
    STEP    [rdi + 224]
    add     rsi, 32
    add     rdi, 256
    dec     r11
    jnz     .Lblock

.Lgroups:
    and     ecx, 31
    cmp     ecx, 4
    jb      .Ltail
.Lgroup:
    vmovd   xmm1, dword ptr [rsi]
    vpcmpeqb xmm1, xmm1, xmm0
    vpmovmskb r8d, xmm1
    not     r8d
    and     r8d, 0xf
    shl     r8d, ENTRY_SHIFT
    STEP    [rdi]
    add     rsi, 4
    add     rdi, 32
    sub     ecx, 4
    cmp     ecx, 4
    jae     .Lgroup

.Ltail:
    test    ecx, ecx
    jz      .Ldone
.Lone:
    mov     rax, [rdi]
    mov     [rdx + r9 * 8], rax
    xor     eax, eax
    cmp     byte ptr [rsi], 0
    setne   al
    add     r9, rax
    add     rdi, 8
    inc     rsi
    dec     ecx
    jnz     .Lone
.Ldone:
    mov     rax, r9
    vzeroupper
    ret
.size compress_f64, .-compress_f64

.section .note.GNU-stack,"",@progbits
