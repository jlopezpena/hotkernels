.intel_syntax noprefix
# void searchsorted_f64(const double *a, const double *v, int64_t *out, size_t n, size_t needles);
#
# rdi = a, rsi = v, rdx = out, rcx = n, r8 = needles. For each needle in v,
# writes the first index i such that a[i] is not less than the needle (side='left').
# a holds n values sorted ascending, NaN last.
#
# One search is a fixed number of steps that depends only on n, with no
# branch on the data:
#
#     lo = 0, len = n
#     while len > 0:
#         half = (len + 1) / 2
#         if a[lo + half - 1] < needle: lo += half
#         len -= half
#
# The answer is always in lo .. lo + len, and lo + len never grows, so every
# probe is inside a whatever the data, sorted or not. A NaN in a (they are last) compares false and is never passed, which is right for
# every needle that is not NaN.
#
# Up to 64 searches run in step. Their positions live in out, which is
# exactly one 64-bit slot per needle, and their needles stay in v. A step is,
# per four searches, a load of the positions, one `vgatherqpd` for the four
# probes, one compare with the needles, one add of `half` under the compare's
# mask, and a store. Nothing in a step waits for another search. Measured on
# this processor, the chain from one step of a search to its next (store,
# load, gather, compare, add) is about thirty cycles long, while a gather can
# start every two or three; with 64 searches in step the sixteen gathers of a
# step fill that time, where four registers of positions leave most of it
# idle. And when a is too large for the caches, the probes of a step are
# cache misses in flight together, where a search done alone has one.
#
# The first step needs no gather: every search probes the same element.
#
# A NaN needle. NumPy orders NaN after everything and equal to itself, so its
# answer is the index of the first NaN of a, or n. The compare above is false
# for it at every step and leaves 0. So the needles are also tested for NaN as
# they are read, and only if one was seen (.Lfixnan) is that index found, with
# one more search of a for its first NaN, and written over the answer of every
# NaN needle.
#
# Paths, by the number of needles:
#   0        nothing
#   1..3     one search at a time, in general registers
#   4..      chunks of up to 64 in step, four per vector; then, if the count
#            is not a multiple of four, one more vector that ends with the
#            last needle and overlaps the one before it
#
# `vgatherqpd` is fast on this processor and slow on some others (older AMD,
# and Intel processors that carry the gather mitigation).
#
# Registers: r9 len, rax the address of a[half - 1], r10 and r11 the byte
# offsets where the chunk starts and ends, rbx the offset inside it (saved
# below the stack pointer). ymm8 half in every lane, ymm9 the gather's mask,
# ymm15 collects "this needle is a NaN".

# Needles per chunk, in bytes of v: 64.
.set CHUNK_BYTES, 512

# One step for the four searches in \lo against the needles in \key. A gather
# clears its mask register, and it merges into its destination, which is
# therefore one more input: each of the four uses of this macro has a
# destination of its own, so that none waits for another.
.macro PROBE lo, key, tmp
    vpcmpeqd ymm9, ymm9, ymm9
    vgatherqpd \tmp, [rax + \lo * 8], ymm9
    vcmpltpd \tmp, \tmp, \key
    vpand   \tmp, \tmp, ymm8
    vpaddq  \lo, \lo, \tmp
.endm

# The start of a step: half into rax and ymm8, len reduced, and rax turned
# into the address of a[half - 1]. Leaves the zero flag set when this is the
# last step; nothing that follows in a step changes the flags.
.macro HALVE
    lea     rax, [r9 + 1]
    shr     rax, 1
    vmovq   xmm8, rax
    vpbroadcastq ymm8, xmm8
    sub     r9, rax
    lea     rax, [rdi + rax * 8 - 8]
.endm

.text
.globl searchsorted_f64
.type searchsorted_f64, @function
.p2align 4
searchsorted_f64:
    test    r8, r8
    jz      .Lreturn
    cmp     r8, 4
    jb      .Lfew
    mov     [rsp - 8], rbx
    vpxor   xmm15, xmm15, xmm15
    xor     r10d, r10d
    test    rcx, rcx
    jz      .Lempty

.Lchunk:
    mov     rax, r8
    shr     rax, 2
    shl     rax, 5                      # where the whole vectors end
    cmp     r10, rax
    jae     .Lremainder
    lea     r11, [r10 + CHUNK_BYTES]
    cmp     r11, rax
    cmova   r11, rax                    # where this chunk ends
    # The first step: every search probes a[half - 1].
    mov     r9, rcx
    HALVE
    vbroadcastsd ymm10, qword ptr [rax]
    mov     rbx, r10
.Lfirst:
    vmovupd ymm4, [rsi + rbx]
    vcmpunordpd ymm11, ymm4, ymm4
    vorpd   ymm15, ymm15, ymm11
    vcmpltpd ymm11, ymm10, ymm4
    vpand   ymm11, ymm11, ymm8
    vmovdqu [rdx + rbx], ymm11
    add     rbx, 32
    cmp     rbx, r11
    jb      .Lfirst
    test    r9, r9
    jz      .Lchunkdone
.Lstep:
    HALVE
    mov     rbx, r10
.p2align 5
.Lprobe:
    vmovdqu ymm0, [rdx + rbx]
    vpcmpeqd ymm9, ymm9, ymm9
    vpxor   xmm10, xmm10, xmm10         # a gather merges into its destination: start it afresh
    vgatherqpd ymm10, [rax + ymm0 * 8], ymm9
    vcmpltpd ymm10, ymm10, [rsi + rbx]
    vpand   ymm10, ymm10, ymm8
    vpaddq  ymm0, ymm0, ymm10
    vmovdqu [rdx + rbx], ymm0
    add     rbx, 32
    cmp     rbx, r11
    jb      .Lprobe
    test    r9, r9
    jnz     .Lstep
.Lchunkdone:
    mov     r10, r11
    jmp     .Lchunk

    # ---- The last one to three needles of a count that is not a multiple
    #      of four: the last four needles as one vector, in registers.
.Lremainder:
    test    r8b, 3
    jz      .Lnantest
    lea     r10, [r8 * 8 - 32]
    vmovupd ymm4, [rsi + r10]
    vcmpunordpd ymm11, ymm4, ymm4
    vorpd   ymm15, ymm15, ymm11
    vpxor   xmm0, xmm0, xmm0
    mov     r9, rcx
.Lstep4:
    HALVE
    PROBE   ymm0, ymm4, ymm10
    test    r9, r9
    jnz     .Lstep4
    vmovdqu [rdx + r10], ymm0
    jmp     .Lnantest

    # ---- a is empty: every answer is 0.
.Lempty:
    xor     eax, eax
.Lzero:
    mov     qword ptr [rdx + r10 * 8], rax
    inc     r10
    cmp     r10, r8
    jb      .Lzero
    mov     rbx, [rsp - 8]
    ret

.Lnantest:
    mov     rbx, [rsp - 8]
    vmovmskpd eax, ymm15
    vzeroupper
    test    eax, eax
    jnz     .Lfixnan
.Lreturn:
    ret

    # ---- One to three needles, one search at a time. No 256-bit register is
    #      written here.
.Lfew:
    xor     r10d, r10d                  # needle index
    xor     r11d, r11d                  # set when a needle is a NaN
.Lone:
    vmovsd  xmm0, qword ptr [rsi + r10 * 8]
    xor     eax, eax                    # lo
    mov     r9, rcx
    test    r9, r9
    jz      .Lonestore
.Lonestep:
    lea     r11, [r9 + 1]
    shr     r11, 1                      # half
    sub     r9, r11
    add     r11, rax                    # lo + half
    vucomisd xmm0, qword ptr [rdi + r11 * 8 - 8]
    cmova   rax, r11
    test    r9, r9
    jnz     .Lonestep
.Lonestore:
    mov     [rdx + r10 * 8], rax
    inc     r10
    cmp     r10, r8
    jb      .Lone
    # Any NaN among them? One more pass; three needles at most.
    xor     r10d, r10d
.Lfewnan:
    vmovsd  xmm0, qword ptr [rsi + r10 * 8]
    vucomisd xmm0, xmm0
    jp      .Lfixnan
    inc     r10
    cmp     r10, r8
    jb      .Lfewnan
    ret

    # ---- At least one needle is a NaN: find its answer and write it over
    #      the answer of every NaN needle.
.Lfixnan:
    # The index of the first NaN of a: the same search with "is not NaN" as
    # the test.
    xor     eax, eax
    mov     r9, rcx
    test    r9, r9
    jz      .Lfill
.Lfirstnan:
    lea     r10, [r9 + 1]
    shr     r10, 1                      # half
    sub     r9, r10
    lea     r11, [rax + r10]
    vmovsd  xmm0, qword ptr [rdi + r11 * 8 - 8]
    vucomisd xmm0, xmm0
    cmovnp  rax, r11                    # not a NaN: the first NaN is further on
    test    r9, r9
    jnz     .Lfirstnan
.Lfill:
    xor     r10d, r10d
.Lfillone:
    vmovsd  xmm0, qword ptr [rsi + r10 * 8]
    vucomisd xmm0, xmm0
    jnp     .Lfillnext
    mov     [rdx + r10 * 8], rax
.Lfillnext:
    inc     r10
    cmp     r10, r8
    jb      .Lfillone
    ret
.size searchsorted_f64, .-searchsorted_f64

.section .note.GNU-stack,"",@progbits
