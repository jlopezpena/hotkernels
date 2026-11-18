.intel_syntax noprefix
# void cumsum_bool(const bool *mask, int64_t *out, size_t n);
#
# rdi = mask, rsi = out, rdx = n. out[i] = how many of mask[0..i] are true.
# Every mask byte is 0 or 1.
#
# The reference adds one byte to a running count and stores it: a chain of
# one add per element. Here eight elements are one step. Their eight mask
# bits index two tables of 256 entries. One entry is the running count within
# the group, a byte per element; `vpmovzxbq` widens its two halves to two
# registers of four 64-bit lanes. The other is the group's total, which
# `vpbroadcastq` puts in every lane. With T the count before the group in
# every lane, the two output vectors are the widened halves plus T, and the
# next T is T plus the total: a chain of one add per eight elements.
#
# The mask bits of 32 elements come from one `vpmovmskb`, and those of a
# group of eight outside a block from an 8-byte load and `pext`.
#
# From ALIGN_MIN elements the output is first brought to a multiple of 32
# bytes, up to three elements one at a time, so that no store of a block
# straddles two cache lines. The mask is a byte per element, so any number of
# elements can go first. An out that is not a multiple of 8 cannot be brought
# there and is left as it is.
#
# From STREAM_MIN_N elements the blocks store with `vmovntdq`. An ordinary
# store first reads the cache line it is about to overwrite; this one does
# not, which takes that traffic away. The output is then in no cache when the
# kernel returns, which costs a caller nothing at a size no cache holds.
#
# Paths, by what is left:
#   32 or more  blocks of 32: one mask word, four groups
#   8..31       one group per 8
#   1..7        one element at a time, as the reference does it

# One entry of the prefix table for the eight mask bits \m: the count of set
# bits up to and including each position.
.macro PREFIX_ENTRY m
    .set .Lcount, 0
    .irp bit, 0, 1, 2, 3, 4, 5, 6, 7
    .set .Lcount, .Lcount + (((\m) >> \bit) & 1)
    .byte .Lcount
    .endr
.endm

# One entry of the total table: the number of set bits of \m, as 64 bits.
.macro TOTAL_ENTRY m
    .set .Lcount, 0
    .irp bit, 0, 1, 2, 3, 4, 5, 6, 7
    .set .Lcount, .Lcount + (((\m) >> \bit) & 1)
    .endr
    .quad .Lcount
.endm

.section .rodata
.p2align 6
.Lprefix:
    .set .Lbyte, 0
    .rept 256
    PREFIX_ENTRY .Lbyte
    .set .Lbyte, .Lbyte + 1
    .endr
.Ltotal:
    .set .Lbyte, 0
    .rept 256
    TOTAL_ENTRY .Lbyte
    .set .Lbyte, .Lbyte + 1
    .endr

# From this many elements the output is first brought to a multiple of 32
# bytes: enough for the three elements that may take and one block after them.
.set ALIGN_MIN, 35
# From this many elements the stores go past the caches: 24 MB of output, the
# third-level cache of this processor.
.set STREAM_MIN_N, 3 * 1024 * 1024
# The bits `pext` takes from eight mask bytes: the lowest of each.
.set LOW_BITS, 0x0101010101010101

# One group. rcx = its eight mask bits, ymm0 = T. \offset is where the
# group's output goes, in bytes from rsi.
# \store is the instruction that stores an output vector.
.macro GROUP offset, store=vmovdqu
    vpmovzxbq ymm1, dword ptr [r10 + rcx * 8]
    vpmovzxbq ymm2, dword ptr [r10 + rcx * 8 + 4]
    vpaddq  ymm1, ymm1, ymm0
    vpaddq  ymm2, ymm2, ymm0
    vpbroadcastq ymm3, [r11 + rcx * 8]
    vpaddq  ymm0, ymm0, ymm3
    \store [rsi + \offset], ymm1
    \store [rsi + \offset + 32], ymm2
.endm

# One block of 32 elements at rdi: the mask word, then four groups.
.macro BLOCK store=vmovdqu
    vmovdqu ymm1, [rdi]
    vpsllw  ymm1, ymm1, 7               # the low bit of each byte to its top
    vpmovmskb eax, ymm1
    movzx   ecx, al
    shr     eax, 8
    GROUP 0, \store
    movzx   ecx, al
    shr     eax, 8
    GROUP 64, \store
    movzx   ecx, al
    shr     eax, 8
    GROUP 128, \store
    mov     ecx, eax
    GROUP 192, \store
    add     rdi, 32
    add     rsi, 256
.endm

.text
.globl cumsum_bool
.type cumsum_bool, @function
.p2align 4
cumsum_bool:
    vpxor   xmm0, xmm0, xmm0            # T: the count so far, in every lane
    lea     r10, [rip + .Lprefix]
    lea     r11, [rip + .Ltotal]
    cmp     rdx, ALIGN_MIN
    jb      .Lblocks
    mov     ecx, esi
    neg     ecx
    and     ecx, 31                     # bytes until out is a multiple of 32
    test    cl, 7
    jnz     .Lblocks
    shr     ecx, 3
    jz      .Lblocks
    sub     rdx, rcx
    xor     eax, eax
.Lhead:
    movzx   r8d, byte ptr [rdi]
    add     rax, r8
    mov     [rsi], rax
    inc     rdi
    add     rsi, 8
    dec     ecx
    jnz     .Lhead
    vmovq   xmm0, rax
    vpbroadcastq ymm0, xmm0
.Lblocks:
    mov     r8, rdx
    and     r8, -32                     # elements in whole blocks
    jz      .Lgroups
    lea     r9, [rdi + r8]              # where the blocks end
    cmp     rdx, STREAM_MIN_N
    jae     .Lstream
.p2align 6
.Lblock:
    BLOCK
    cmp     rdi, r9
    jne     .Lblock
    jmp     .Lgroups

    # The same with stores that go past the caches, which need out to be a
    # multiple of 32. An out that could not be brought there takes the
    # ordinary stores.
.Lstream:
    test    sil, 31
    jnz     .Lblock
.p2align 6
.Lstreamblock:
    BLOCK vmovntdq
    cmp     rdi, r9
    jne     .Lstreamblock
    sfence

.Lgroups:
    and     edx, 31                     # elements left
    cmp     edx, 8
    jb      .Ltail
    movabs  r8, LOW_BITS
.Lgroup:
    mov     rcx, [rdi]
    pext    rcx, rcx, r8
    GROUP 0
    add     rdi, 8
    add     rsi, 64
    sub     edx, 8
    cmp     edx, 8
    jae     .Lgroup

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
