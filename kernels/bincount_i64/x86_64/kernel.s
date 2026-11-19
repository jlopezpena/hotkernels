.intel_syntax noprefix
# void bincount_i64(const int64_t *x, int64_t *counts, size_t n, size_t bins);
#
# rdi = x, rsi = counts, rdx = n, rcx = bins. counts[v] = how many of the n
# elements at x equal v, for v = 0..bins-1. An element outside that range,
# negative or at or above bins, is ignored.
#
# The element is compared with bins as an unsigned number, so a negative
# element is a huge one and fails the same test as one that is too large.
# Every element takes the test, so no counter outside counts is ever
# touched, whatever the data.
#
# The counters are cleared with 32-byte stores, the last ones ending exactly
# at the end of counts and overlapping the ones before; from CLEAR_REP_MIN
# counters with `rep stosq`.
#
# One counter is loaded, incremented and stored per element, about a cycle,
# and an element that repeats the one before it has to wait for that store.
# With at most PAIR_CODES counters and from PAIRS_MIN_N elements, two
# elements are counted with one increment: of the counter of the pair, in a
# table of PAIR_CODES * PAIR_CODES 32-bit counters below the stack pointer.
# At the end counts[v] gets the total of row v and of column v, the pairs
# with v second and with v first. Four elements are tested together for
# being pair codes, 0..PAIR_CODES-1: the four or'ed together are below
# PAIR_CODES, as an unsigned number, only if each of them is. Four that are
# not all pair codes are counted one at a time into counts, each with its
# own test against bins. A pair code at or above bins is counted in the
# table and left out when the totals are added, so bins is tested there.
# The table's counters cannot overflow: the path is for n below 2^32, so
# there are fewer than 2^31 pairs, and no counter and no total reaches 2^31.
#
# With more counters than that, up to SECOND_MAX_BINS, and from SECOND_MIN_N
# elements, the kernel looks at the first SAMPLE elements. If one in eight
# equals its successor, every other element is counted in a second table of
# bins counters below the stack pointer, added to counts at the end, so that
# an element that repeats the one before it does not wait for its store.
# The second table is placed half a page from counts within a page, because
# a load also waits for a store to the same place in another page. On data
# that does not repeat, the second table costs more than it saves in
# bincount_i16, so it is not used there.
#
# Paths:
#   bins <= PAIR_CODES and PAIRS_MIN_N <= n < 2^32               the pair table
#   else bins <= SECOND_MAX_BINS, n >= SECOND_MIN_N, repeats     two tables
#   otherwise                                                    one counter per element

# From this many counters `rep stosq` clears them; below it, a loop of
# 32-byte stores, which has no start-up cost.
.set CLEAR_REP_MIN, 512

# The pair path is for at most this many counters: codes of six bits.
.set PAIR_CODES, 64
# Below this many elements clearing and adding up the pair table costs more
# than it saves.
.set PAIRS_MIN_N, 8192
# The pair table: PAIR_CODES rows of PAIR_CODES 32-bit counters. Row a1,
# column a0 counts the pairs (a0, a1).
.set ROW_BYTES, PAIR_CODES * 4
.set TABLE_BYTES, PAIR_CODES * ROW_BYTES
# Bytes of stack: the pair table, the PAIR_CODES 32-bit column totals above
# it, and eight so that the stack pointer stays a multiple of 16.
.set FRAME, TABLE_BYTES + ROW_BYTES + 8

# The second table is for at most this many counters, 8 KB of them, and from
# this many elements.
.set SECOND_MAX_BINS, 1024
.set SECOND_MIN_N, 8192
# The second table is used when at least SAMPLE_REPEATS of the first SAMPLE
# elements equal their successor, one in eight: uniform codes over eight
# values or fewer, or runs.
.set SAMPLE, 256
.set SAMPLE_REPEATS, 32
# A page of memory. A load is held up by an earlier store to the same place
# in another page, so the second table is put half a page from counts.
.set PAGE, 4096
# Bytes of stack: the second table, a page to place it in, and eight so that
# the stack pointer stays a multiple of 16.
.set SECOND_FRAME, SECOND_MAX_BINS * 8 + PAGE + 8

# Count the element at [rdi + \offset] in the table at \table: test against
# bins, increment its counter.
.macro COUNT_ELEMENT offset, table=rsi
    mov     rax, [rdi + \offset]
    cmp     rax, r9
    jae     1f
    inc     qword ptr [\table + rax * 8]
1:
.endm

.text
.globl bincount_i64
.type bincount_i64, @function
.p2align 4
bincount_i64:
    mov     r9, rcx                     # bins
    # Clear the bins counters.
    cmp     rcx, CLEAR_REP_MIN
    jae     .Lclearrep
    cmp     rcx, 4
    jb      .Lclearfew
    vpxor   xmm0, xmm0, xmm0
    cmp     rcx, 8
    jb      .Lclearfour
    lea     r8, [rsi + rcx * 8 - 64]    # the last eight counters
    mov     rax, rsi
.Lclearstep:
    vmovdqu [rax], ymm0
    vmovdqu [rax + 32], ymm0
    add     rax, 64
    cmp     rax, r8
    jb      .Lclearstep
    vmovdqu [r8], ymm0
    vmovdqu [r8 + 32], ymm0
    vzeroupper
    jmp     .Lcount
.Lclearfour:                            # 4 to 7 counters: the first four and the last four
    vmovdqu [rsi], ymm0
    vmovdqu [rsi + rcx * 8 - 32], ymm0
    vzeroupper
    jmp     .Lcount
.Lclearfew:                             # 1 to 3 counters
    mov     qword ptr [rsi + rcx * 8 - 8], 0
    dec     rcx
    jnz     .Lclearfew
    jmp     .Lcount
.Lclearrep:
    mov     r8, rdi
    mov     rdi, rsi
    xor     eax, eax
    rep stosq
    mov     rdi, r8

.Lcount:
    cmp     r9, PAIR_CODES
    ja      .Lsingle
    cmp     rdx, PAIRS_MIN_N
    jb      .Lsingle
    mov     rax, rdx
    shr     rax, 32
    jz      .Lpairs
.Lsingle:
    cmp     r9, SECOND_MAX_BINS
    ja      .Lonetable
    cmp     rdx, SECOND_MIN_N
    jb      .Lonetable
    # How many of the first SAMPLE elements equal the one after them.
    xor     eax, eax
    xor     ecx, ecx
.Lsample:
    vmovdqu ymm0, [rdi + rcx]
    vpcmpeqq ymm0, ymm0, [rdi + rcx + 8]
    vmovmskpd r8d, ymm0                 # a bit for each equal pair
    popcnt  r8d, r8d
    add     eax, r8d
    add     ecx, 32
    cmp     ecx, SAMPLE * 8
    jb      .Lsample
    vzeroupper
    cmp     eax, SAMPLE_REPEATS
    jae     .Ltwotables
.Lonetable:
    mov     r8, rdx
    shr     r8, 2                       # steps of four elements
    jz      .Llast
.p2align 4
.Lfour:
    COUNT_ELEMENT 0
    COUNT_ELEMENT 8
    COUNT_ELEMENT 16
    COUNT_ELEMENT 24
    add     rdi, 32
    dec     r8
    jnz     .Lfour
.Llast:
    and     edx, 3                      # 0 to 3 elements left
    jz      .Ldone
.Lone:
    COUNT_ELEMENT 0
    add     rdi, 8
    dec     edx
    jnz     .Lone
.Ldone:
    ret

.Ltwotables:
    sub     rsp, SECOND_FRAME
    lea     r10, [rsi + PAGE / 2]
    sub     r10, rsp
    and     r10, PAGE - 1
    add     r10, rsp                    # the second table, half a page from counts within a page
    vpxor   xmm0, xmm0, xmm0
    lea     rax, [r9 * 8 + 63]
    and     rax, -64                    # its bytes, in whole steps of 64
.Lclearsecond:
    vmovdqu [r10 + rax - 32], ymm0
    vmovdqu [r10 + rax - 64], ymm0
    sub     rax, 64
    jnz     .Lclearsecond
    mov     r8, rdx
    shr     r8, 2                       # steps of four elements, at least one
.p2align 4
.Ltwostep:
    COUNT_ELEMENT 0, rsi
    COUNT_ELEMENT 8, r10
    COUNT_ELEMENT 16, rsi
    COUNT_ELEMENT 24, r10
    add     rdi, 32
    dec     r8
    jnz     .Ltwostep
    # Add the second table to counts: four counters at a time, then one.
    xor     eax, eax
    mov     rcx, r9
    and     rcx, -4
    jz      .Laddone
.Laddfour:
    vmovdqu ymm0, [r10 + rax * 8]
    vpaddq  ymm0, ymm0, [rsi + rax * 8]
    vmovdqu [rsi + rax * 8], ymm0
    add     rax, 4
    cmp     rax, rcx
    jb      .Laddfour
.Laddone:
    cmp     rax, r9
    jae     .Ladded
    mov     rcx, [r10 + rax * 8]
    add     [rsi + rax * 8], rcx
    inc     rax
    jmp     .Laddone
.Ladded:
    vzeroupper
    add     rsp, SECOND_FRAME
    jmp     .Llast                      # 0 to 3 elements left, straight into counts

.Lpairs:
    sub     rsp, FRAME
    # Clear the pair table from the top down, so that the stack is touched a
    # page at a time in the order it grows.
    vpxor   xmm0, xmm0, xmm0
    mov     eax, TABLE_BYTES
.Lcleartable:
    vmovdqu [rsp + rax - 32], ymm0
    vmovdqu [rsp + rax - 64], ymm0
    sub     eax, 64
    jnz     .Lcleartable
    mov     r8, rdx
    shr     r8, 2                       # steps of four elements, at least one
.p2align 4
.Lpairstep:
    mov     r10, [rdi]
    or      r10, [rdi + 8]
    or      r10, [rdi + 16]
    or      r10, [rdi + 24]
    cmp     r10, PAIR_CODES
    jae     .Loutside
    mov     rcx, [rdi + 8]
    shl     rcx, 6
    or      rcx, [rdi]                  # a0 | a1 << 6
    mov     rax, [rdi + 24]
    shl     rax, 6
    or      rax, [rdi + 16]             # a2 | a3 << 6
    inc     dword ptr [rsp + rcx * 4]
    inc     dword ptr [rsp + rax * 4]
.Lpairnext:
    add     rdi, 32
    dec     r8
    jnz     .Lpairstep

    # Column totals: for each eight columns, the sum down the 64 rows.
    xor     eax, eax
.Lcolumns:
    lea     r10, [rsp + rax]
    vpxor   xmm0, xmm0, xmm0
    vpxor   xmm1, xmm1, xmm1
    vpxor   xmm2, xmm2, xmm2
    vpxor   xmm3, xmm3, xmm3
    mov     ecx, PAIR_CODES / 4
.Lcolumnrows:
    vpaddd  ymm0, ymm0, [r10]
    vpaddd  ymm1, ymm1, [r10 + ROW_BYTES]
    vpaddd  ymm2, ymm2, [r10 + 2 * ROW_BYTES]
    vpaddd  ymm3, ymm3, [r10 + 3 * ROW_BYTES]
    add     r10, 4 * ROW_BYTES
    dec     ecx
    jnz     .Lcolumnrows
    vpaddd  ymm0, ymm0, ymm1
    vpaddd  ymm2, ymm2, ymm3
    vpaddd  ymm0, ymm0, ymm2
    vmovdqu [rsp + TABLE_BYTES + rax], ymm0
    add     eax, 32
    cmp     eax, ROW_BYTES
    jb      .Lcolumns
    # counts[v] += the total of column v and the total of row v, for v < bins.
    xor     ecx, ecx
    mov     r10, rsp
.Lrows:
    vmovdqu ymm0, [r10]
    vmovdqu ymm1, [r10 + 32]
    vpaddd  ymm0, ymm0, [r10 + 64]
    vpaddd  ymm1, ymm1, [r10 + 96]
    vpaddd  ymm0, ymm0, [r10 + 128]
    vpaddd  ymm1, ymm1, [r10 + 160]
    vpaddd  ymm0, ymm0, [r10 + 192]
    vpaddd  ymm1, ymm1, [r10 + 224]
    vpaddd  ymm0, ymm0, ymm1
    vextracti128 xmm1, ymm0, 1
    vpaddd  xmm0, xmm0, xmm1
    vpshufd xmm1, xmm0, 0x4E            # the upper two of four, down
    vpaddd  xmm0, xmm0, xmm1
    vpshufd xmm1, xmm0, 0x55            # the second of four, down
    vpaddd  xmm0, xmm0, xmm1
    vmovd   eax, xmm0
    mov     r11d, [rsp + TABLE_BYTES + rcx * 4]
    add     rax, r11
    add     [rsi + rcx * 8], rax
    add     r10, ROW_BYTES
    inc     ecx
    cmp     rcx, r9
    jb      .Lrows
    vzeroupper
    add     rsp, FRAME
    jmp     .Llast                      # 0 to 3 elements left, straight into counts

.Loutside:                              # one of the four is not a pair code
    COUNT_ELEMENT 0
    COUNT_ELEMENT 8
    COUNT_ELEMENT 16
    COUNT_ELEMENT 24
    jmp     .Lpairnext
.size bincount_i64, .-bincount_i64

.section .note.GNU-stack,"",@progbits
