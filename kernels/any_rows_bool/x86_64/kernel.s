.intel_syntax noprefix
# size_t any_rows_bool(const bool *m, bool *out, size_t n, size_t cols);
#
# rdi = m, rsi = out, rdx = n, rcx = cols. m holds n / cols whole rows of
# cols elements; out[r] = 1 if any element of row r is true, else 0. Returns
# the number of rows and writes exactly that many bytes. cols >= 1. Every
# byte of m is 0 or 1.
#
# No division: a row is taken while cols elements are left, and the row count
# is how far out has advanced.
#
# Paths, by cols:
#   1..32    one 32-byte load at the row's start and `vptest` against a
#            vector whose first cols bytes are set. A row that starts in the
#            last 31 bytes of m cannot be loaded that way: the bits of the
#            last 32 bytes of m go to a register with `vpmovmskb`, and each
#            of those rows is cols bits of it. With n under 32 the bits of
#            all of m are gathered a byte at a time.
#   33..64   the row's first 32 bytes or'ed with its last 32, which overlap.
#   65..     32-byte chunks or'ed together, the last one ending at the row's
#            end and overlapping the one before.

.section .rodata
.p2align 6
# 32 bytes of ones, then 32 of zeros. The 32 bytes that start cols before
# .Lrow_end have their first cols bytes set.
    .fill 32, 1, 0xff
.Lrow_end:
    .fill 32, 1, 0

# Bytes in one vector.
.set VEC, 32
# Rows of four in one block, which is four vectors of m and one of out, and
# in one step of the rows left under a block.
.set FOUR_BLOCK, 32
.set FOUR_STEP, 8

# 32 rows of four at \src to the 32 bytes at \dst. ymm5 = 1 in every byte.
# The rows are packed as they are, with no compare: a 32-bit lane of bytes
# that are 0 or 1 is zero or a positive number, signed saturation narrows a
# positive number to a positive number, and the minimum with 1 leaves 0 or 1.
# Each vector is put together from two 16-byte loads 64 bytes apart, so that
# the packs, which work within each half of a register, leave the rows in
# order and no permute is needed.
.macro FOUR_ROWS_32 src, dst
    vmovdqu xmm1, [\src]
    vmovdqu xmm2, [\src + 16]
    vmovdqu xmm3, [\src + 32]
    vmovdqu xmm4, [\src + 48]
    vinserti128 ymm1, ymm1, [\src + 64], 1
    vinserti128 ymm2, ymm2, [\src + 80], 1
    vinserti128 ymm3, ymm3, [\src + 96], 1
    vinserti128 ymm4, ymm4, [\src + 112], 1
    vpackssdw ymm1, ymm1, ymm2
    vpackssdw ymm3, ymm3, ymm4
    vpacksswb ymm1, ymm1, ymm3
    vpminub ymm1, ymm1, ymm5
    vmovdqu [\dst], ymm1
.endm

# Eight rows of four at \src to the eight bytes at \dst, the same way.
.macro FOUR_ROWS_8 src, dst
    vmovdqu xmm1, [\src]
    vpackssdw xmm1, xmm1, [\src + 16]
    vpacksswb xmm1, xmm1, xmm1
    vpminub xmm1, xmm1, xmm5
    vmovq   [\dst], xmm1
.endm

.text
.globl any_rows_bool
.type any_rows_bool, @function
.p2align 4
any_rows_bool:
    mov     r8, rsi                     # out, to count the rows at the end
    cmp     rdx, rcx
    jb      .Lno_rows
    cmp     rcx, 4
    je      .Lfour
    lea     r9, [rdi + rdx]             # end of m
    cmp     rcx, VEC
    ja      .Lwide
    cmp     rdx, VEC
    jb      .Lsmall
    lea     rax, [rip + .Lrow_end]
    sub     rax, rcx
    vmovdqu ymm0, [rax]                 # the first cols bytes set
    lea     r10, [r9 - VEC]             # the last place a vector can be loaded
    lea     r11, [rcx + rcx * 2]
    mov     rax, r10
    sub     rax, r11                    # the last place four rows can start
    jb      .Lrow
    cmp     rdi, rax
    ja      .Lrow
    lea     rdx, [rcx * 4]
.p2align 4
.Lrow4:
    vptest  ymm0, [rdi]
    setnz   byte ptr [rsi]
    vptest  ymm0, [rdi + rcx]
    setnz   byte ptr [rsi + 1]
    vptest  ymm0, [rdi + rcx * 2]
    setnz   byte ptr [rsi + 2]
    vptest  ymm0, [rdi + r11]
    setnz   byte ptr [rsi + 3]
    add     rdi, rdx
    add     rsi, 4
    cmp     rdi, rax
    jbe     .Lrow4
    cmp     rdi, r10
    ja      .Lrows_done
.Lrow:
    vptest  ymm0, [rdi]
    setnz   byte ptr [rsi]
    add     rdi, rcx
    inc     rsi
    cmp     rdi, r10
    jbe     .Lrow
.Lrows_done:
    mov     rdx, r9
    sub     rdx, rdi                    # elements left
    cmp     rdx, rcx
    jb      .Ldone
    # The rows left start inside the last vector of m. Bit i of eax becomes
    # the element i places after rdi.
    vpxor   xmm1, xmm1, xmm1
    vpcmpeqb ymm1, ymm1, [r10]
    vpmovmskb eax, ymm1
    not     eax
    sub     rdi, r10
    shrx    eax, eax, edi
    jmp     .Lbit_rows

.Lsmall:
    # 1 to 31 elements: gather their bits from the last to the first, so that
    # bit i of eax is element i.
    xor     eax, eax
    mov     r11, rdx
.Lgather:
    movzx   r10d, byte ptr [rdi + r11 - 1]
    add     eax, eax
    or      eax, r10d
    dec     r11
    jnz     .Lgather

.Lbit_rows:
    # eax = the bits of the rdx elements left, rdx >= cols, cols <= 32.
    mov     r10d, 1
    shlx    r10, r10, rcx
    dec     r10                         # cols low bits
.Lbit_row:
    test    rax, r10
    setnz   byte ptr [rsi]
    inc     rsi
    shrx    rax, rax, rcx
    sub     rdx, rcx
    cmp     rdx, rcx
    jae     .Lbit_row
.Ldone:
    mov     rax, rsi
    sub     rax, r8
    vzeroupper
    ret

.Lno_rows:
    xor     eax, eax
    ret

.Lfour:
    # Rows of four: a row is a 32-bit lane, and it has a true element when
    # the lane is not zero.
    shr     rdx, 2                      # rows
    mov     rax, rdx
    vpcmpeqd ymm5, ymm5, ymm5
    vpabsb  ymm5, ymm5                  # 1 in every byte
    cmp     rdx, FOUR_BLOCK
    jb      .Lfour_small
    lea     r9, [rdi + rdx * 4 - 4 * FOUR_BLOCK]    # the last block of m
    lea     r10, [rsi + rdx - FOUR_BLOCK]           # and where its rows go
.p2align 4
.Lfour_block:
    FOUR_ROWS_32 rdi, rsi
    add     rdi, 4 * FOUR_BLOCK
    add     rsi, FOUR_BLOCK
    sub     rdx, FOUR_BLOCK
    cmp     rdx, FOUR_BLOCK
    jae     .Lfour_block
    test    rdx, rdx
    jz      .Lfour_done
    # 1 to 31 rows left: the block that ends at the last row, which stores
    # some rows a second time.
    FOUR_ROWS_32 r9, r10
.Lfour_done:
    vzeroupper
    ret
.Lfour_small:
    cmp     rdx, FOUR_STEP
    jb      .Lfour_single
    lea     r9, [rdi + rdx * 4 - 4 * FOUR_STEP]
    lea     r10, [rsi + rdx - FOUR_STEP]
.p2align 4
.Lfour_step:
    FOUR_ROWS_8 rdi, rsi
    add     rdi, 4 * FOUR_STEP
    add     rsi, FOUR_STEP
    sub     rdx, FOUR_STEP
    cmp     rdx, FOUR_STEP
    jae     .Lfour_step
    test    rdx, rdx
    jz      .Lfour_done
    FOUR_ROWS_8 r9, r10
    vzeroupper
    ret
.Lfour_single:
    # 1 to 7 rows.
    cmp     dword ptr [rdi], 0
    setne   byte ptr [rsi]
    add     rdi, 4
    inc     rsi
    dec     rdx
    jnz     .Lfour_single
    vzeroupper
    ret

.Lwide:
    lea     r10, [rcx - VEC]            # where a row's last vector starts
    mov     rdx, r9
    sub     rdx, rcx                    # the last place a row can start
    cmp     rcx, 2 * VEC
    ja      .Llong_row
    lea     r11, [rcx + rcx * 2]
    mov     r9, rdx
    sub     r9, r11                     # the last place four rows can start
    jb      .Lpair_row
    cmp     rdi, r9
    ja      .Lpair_row
    lea     rax, [rcx * 4]
.p2align 4
.Lpair_row4:
    vmovdqu ymm0, [rdi]
    vpor    ymm0, ymm0, [rdi + rcx - VEC]
    vptest  ymm0, ymm0
    setnz   byte ptr [rsi]
    vmovdqu ymm1, [rdi + rcx]
    vpor    ymm1, ymm1, [rdi + rcx * 2 - VEC]
    vptest  ymm1, ymm1
    setnz   byte ptr [rsi + 1]
    vmovdqu ymm2, [rdi + rcx * 2]
    vpor    ymm2, ymm2, [rdi + r11 - VEC]
    vptest  ymm2, ymm2
    setnz   byte ptr [rsi + 2]
    vmovdqu ymm3, [rdi + r11]
    vpor    ymm3, ymm3, [rdi + rcx * 4 - VEC]
    vptest  ymm3, ymm3
    setnz   byte ptr [rsi + 3]
    add     rdi, rax
    add     rsi, 4
    cmp     rdi, r9
    jbe     .Lpair_row4
    cmp     rdi, rdx
    ja      .Ldone
.Lpair_row:
    vmovdqu ymm0, [rdi]
    vpor    ymm0, ymm0, [rdi + r10]
    vptest  ymm0, ymm0
    setnz   byte ptr [rsi]
    add     rdi, rcx
    inc     rsi
    cmp     rdi, rdx
    jbe     .Lpair_row
    jmp     .Ldone

.Llong_row:
    vmovdqu ymm0, [rdi + r10]           # the last vector of the row
    xor     eax, eax
.Lchunk:
    vpor    ymm0, ymm0, [rdi + rax]
    add     rax, VEC
    cmp     rax, r10
    jb      .Lchunk
    vptest  ymm0, ymm0
    setnz   byte ptr [rsi]
    add     rdi, rcx
    inc     rsi
    cmp     rdi, rdx
    jbe     .Llong_row
    jmp     .Ldone

.section .note.GNU-stack,"",@progbits
