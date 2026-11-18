.intel_syntax noprefix
# size_t nonzero_bool(const bool *mask, int64_t *out, size_t n);
#
# rdi = mask, rsi = out, rdx = n. Writes the index of every mask byte that is
# not zero to out, ascending, and returns how many. `out` is a scratch buffer
# of n elements: stores may land anywhere inside it, and only the returned
# count of indices has to be right.
#
# Eight mask bytes at a time. They become eight bits, and the bits index a
# table of 256 entries, each the positions of the set bits in ascending order
# as eight bytes. `vpmovzxbq` widens the eight bytes to two registers of four
# 64-bit lanes, the index of the group's first element is added, and both
# registers are stored at the write position, which then advances by the
# number of set bits. The next store overwrites the lanes that were not
# indices. A store of eight lanes at position k covers out[k .. k+8), and k is
# never ahead of the read position, so it stays inside the n elements as long
# as the eight mask bytes read do.
#
# The mask bits for 32 bytes come from one compare and one `vpmovmskb`. A mask
# byte counts as true when it is not zero, whatever its value.
#
# Paths, by what is left:
#   32 or more  blocks of 32: one mask word, four groups
#   8..31       one group per 8, the mask bits from an 8-byte load
#   1..7        one element at a time, as the reference does it

# One table entry: for mask byte \m, the positions of its set bits in
# ascending order, one byte each. The bytes past the count are zero; they
# become lanes that the write position does not advance over.
.macro OFFSETS_ENTRY m
    .set .Lcount, 0
    .irp bit, 0, 1, 2, 3, 4, 5, 6, 7
    .if ((\m) >> \bit) & 1
    .byte \bit
    .set .Lcount, .Lcount + 1
    .endif
    .endr
    .fill 8 - .Lcount, 1, 0
.endm

.section .rodata
.p2align 6
.Loffsets:
    .set .Lbyte, 0
    .rept 256
    OFFSETS_ENTRY .Lbyte
    .set .Lbyte, .Lbyte + 1
    .endr
# Added to the index of a group's first element to give the next group's.
.p2align 5
.Leight:
    .quad 8, 8, 8, 8

# One group: r8 = its eight mask bits, ymm4 = the index of its first element
# in every lane, r9 = indices written so far. r9 counts elements and the
# stores scale it: an `add` on the chain from group to group takes one cycle
# on this processor, where a `lea` with a scaled index takes two.
.macro GROUP
    vpmovzxbq ymm2, dword ptr [r10 + r8 * 8]
    vpmovzxbq ymm3, dword ptr [r10 + r8 * 8 + 4]
    vpaddq  ymm2, ymm2, ymm4
    vpaddq  ymm3, ymm3, ymm4
    vmovdqu [rsi + r9 * 8], ymm2
    vmovdqu [rsi + r9 * 8 + 32], ymm3
    popcnt  r8d, r8d
    add     r9, r8
    vpaddq  ymm4, ymm4, ymm5
.endm

.text
.globl nonzero_bool
.type nonzero_bool, @function
.p2align 4
nonzero_bool:
    xor     r9d, r9d                    # indices written
    mov     rcx, rdx                    # elements left
    lea     r10, [rip + .Loffsets]
    vpxor   xmm0, xmm0, xmm0
    vpxor   xmm4, xmm4, xmm4            # index of the next group's first element
    vmovdqa ymm5, [rip + .Leight]
    mov     r11, rdx
    shr     r11, 5                      # blocks of 32
    jz      .Lgroups
.p2align 6
.Lblock:
    vpcmpeqb ymm1, ymm0, [rdi]          # 0xff where the mask byte is zero
    vpmovmskb eax, ymm1
    not     eax                         # one bit per true element
    movzx   r8d, al
    shr     eax, 8
    GROUP
    movzx   r8d, al
    shr     eax, 8
    GROUP
    movzx   r8d, al
    shr     eax, 8
    GROUP
    movzx   r8d, al
    GROUP
    add     rdi, 32
    dec     r11
    jnz     .Lblock

.Lgroups:
    and     ecx, 31
    cmp     ecx, 8
    jb      .Ltail
.Lgroup:
    vmovq   xmm1, [rdi]
    vpcmpeqb xmm1, xmm1, xmm0
    vpmovmskb eax, xmm1
    not     eax
    movzx   r8d, al
    GROUP
    add     rdi, 8
    sub     ecx, 8
    cmp     ecx, 8
    jae     .Lgroup

.Ltail:
    test    ecx, ecx
    jz      .Ldone
    mov     r8, rdx
    sub     r8, rcx                     # index of the next element
.Lone:
    mov     [rsi + r9 * 8], r8
    xor     eax, eax
    cmp     byte ptr [rdi], 0
    setne   al
    add     r9, rax
    inc     r8
    inc     rdi
    dec     ecx
    jnz     .Lone
.Ldone:
    mov     rax, r9
    vzeroupper
    ret
.size nonzero_bool, .-nonzero_bool

.section .note.GNU-stack,"",@progbits
