.intel_syntax noprefix
# void cumsum_f32(const float *x, float *out, size_t n);
#
# rdi = x, rsi = out, rdx = n. out[i] = x[0] + ... + x[i] with the additions
# regrouped, so finite results differ from the left-to-right sum by rounding
# only. NaN and infinity appear exactly where the left-to-right sum has them.
# out[0] is a copy of x[0].
#
# Fast path. Eight elements per vector. Consecutive output vectors differ by
# the sum of the eight elements ending at each lane, W. It is built from the
# input alone:
#
#     P  = x[i..i+7] + x[i-1..i+6]     the input plus the same load one element back
#     P' = x[i-2..i+5] + x[i-3..i+4]   the same two elements further back
#     D  = P + P'                      the four elements ending at each lane
#     W  = D + (D four elements back)  the upper half of the previous D and the
#                                      lower half of this one, one `vperm2f128`
#     V  = (previous V) + W            the output vector
#
# So the chain from vector to vector is one `vaddps`: two cycles per eight
# elements where the reference takes two per element. Before the array every
# value is -0.0, which changes nothing it is added to. The eight lanes are
# eight separate running sums; each out[i] is still the sum of x[0..i] in
# some order.
#
# Why non-finite positions match the left-to-right sum, for every input.
# The array is done in chunks of CHUNK_BYTES. In a chunk the largest
# magnitude of any output is tracked (`vpmaxud` over the outputs with the
# sign bit cleared: the bits of IEEE floats order as their magnitudes do, and
# a NaN or an infinity is above every finite value). The chunk is accepted
# only if every output is below T = 2^64 in magnitude. If so:
#   1. Every output of the chunk is finite and below T.
#   2. Every input and every intermediate of the chunk is finite: each P, D
#      and W is an operand, directly or through further additions, of an
#      output of its own vector or the next (the D of the last vector of a
#      chunk is an operand of its own W), and an addition with a NaN or an
#      infinite operand never gives a finite result.
#   3. The left-to-right sum is finite on the chunk too. With S_j the exact
#      sum of x[0..j]: every accepted output is below T and is a rounded sum
#      whose intermediate values are outputs or differences of partial sums,
#      so |S_j| <= T * (1 + j * 2^-23). The left-to-right sum s_j picks up at
#      most 2^-24 * |s_k| of rounding error at each step k, so
#      |s_j| <= |S_j| * e^(j * 2^-24). For j below 2^29 that is under
#      T * 65 * e^32 < 2^117, far from overflow.
# Both results are finite on an accepted chunk, so no position differs.
# If a chunk is not accepted it is done again one element at a time from the
# last output of the chunk before it (.Lredo), watching every element:
#   - x[i] finite and |sum| < T: store it. Point 3 applies to it as well.
#   - x[i] is NaN or infinite: the left-to-right sum was finite up to here
#     (previous case), so from this element on it is not finite, and which of
#     NaN, +inf and -inf it is depends only on the inputs from here on, never
#     on the finite value before. The rest of the array is finished with the
#     plain left-to-right loop (.Lplain), which gives the same kinds.
#   - x[i] finite but |sum| >= T or overflowed: point 3 no longer covers the
#     left-to-right sum, so the whole array is redone from element 0 with the
#     plain loop (.Lrestart), which is the reference computation itself.
# A redone chunk that stayed in the first case is followed by the fast path
# again. The first chunk has no output before it and is redone as a restart.
# The last 0..7 elements, after the last whole vector, are done by the same
# watched loop. Point 3 needs fewer than 2^29 elements: over a longer array
# the rounding of the left-to-right float sum could in principle carry it to
# overflow from any starting size. Arrays of 2^29 elements or more are
# therefore done entirely with the plain loop.
#
# The proof and its limit T are the ARM64 kernel's. That kernel accepts a
# chunk when the sum of the squares of its outputs is finite, which is the
# same T. Comparing magnitudes costs two integer instructions per vector and
# raises no floating-point flag. The price of T is that running sums of
# about 1.8e19 and above, though finite, take the slow path.
#
# The loads run ahead of the stores: the D of the next vector is loaded
# before the current output is stored. A store just behind the input, 16
# bytes after it within a page, would otherwise hold up the loads that follow
# it: the processor compares only the low twelve address bits at first and
# takes the store for one to the elements about to be loaded.
#
# From 24 MB of output, the size of this processor's third-level cache, the
# vectors are stored with `vmovntps`. An ordinary store first reads the cache
# line it is about to overwrite; this one does not, which takes a third of
# the memory traffic away. The output is then in no cache when the kernel
# returns, which costs a caller nothing at a size no cache holds. Such a
# store needs an aligned address, so 0..7 elements are first added in the
# reference order, and their sum takes the place of -0.0 as the output
# before the first vector.
#
# Short arrays. Fewer than SMALL elements are added strictly left to right,
# which is the reference order, so nothing needs checking.
#
# The caller must not pass an out that overlaps x: the fast path reads x[i-1]
# after out[i-1] is written, and the restart reads all of x again.
#
# Registers: rdi x, rsi out, rdx n in bytes, rcx byte offset of the next
# output, r9 end of the chunk, r10 start of the chunk, r11 end of the whole
# vectors, r8 end of the chunk's pairs and limit of the watched loop. ymm0 V,
# ymm1 and ymm2 the last two D, ymm5 the chunk's largest magnitude, ymm6 the
# sign-clearing mask, ymm7 the threshold, ymm10 and ymm11 the lane indices
# of the first vector.

# Bytes of output per chunk: 1,024 elements.
.set CHUNK_BYTES, 4096
# Below this many elements the plain chain is used.
.set SMALL, 16
# Below this many bytes of input the short path is used: 64 elements.
.set SHORT_BYTES, 256
# From this many bytes of input the plain loop is used throughout: 2^29
# elements (point 3 above).
.set LONG_BYTES, 1 << 31
# From this many bytes of output the stores go past the caches: 24 MB, the
# third-level cache of this processor.
.set STREAM_MIN_BYTES, 24 * 1024 * 1024

.section .rodata
.p2align 5
.Labs:
    .fill 8, 4, 0x7fffffff
# A magnitude is acceptable when its bits are at most this: below 2^64.
.Llimit32:
    .fill 8, 4, 0x5f7fffff
.Lnegzero:
    .fill 8, 4, 0x80000000
# Lanes shifted up by one and by two, the lowest repeated.
.Lshift1:
    .long 0, 0, 1, 2, 3, 4, 5, 6
.Lshift2:
    .long 0, 0, 0, 1, 2, 3, 4, 5
# 2^64, the same limit as a float for the watched loop.
.Llimit:
    .long 0x5f800000
# Bits of +infinity: a magnitude at or above it is not finite.
.set INF_BITS, 0x7f800000

# Track the magnitude of the output vector in ymm0.
.macro TRACK
    vpand   ymm8, ymm0, ymm6
    vpmaxud ymm5, ymm5, ymm8
.endm

# Two vectors of the fast path. ymm0 = V, the last output vector; ymm1 = the
# previous D; ymm2 = the D of the vector at rcx. The D of each later vector
# is loaded before the current output is stored. \store is the instruction
# that stores an output vector.
.macro PAIR store
    vperm2f128 ymm3, ymm1, ymm2, 0x21
    vaddps  ymm3, ymm3, ymm2            # W
    vmovups ymm1, [rdi + rcx + 32]
    vaddps  ymm1, ymm1, [rdi + rcx + 28]
    vmovups ymm9, [rdi + rcx + 24]
    vaddps  ymm9, ymm9, [rdi + rcx + 20]
    vaddps  ymm1, ymm1, ymm9
    vaddps  ymm0, ymm0, ymm3            # V
    vperm2f128 ymm4, ymm2, ymm1, 0x21
    vaddps  ymm4, ymm4, ymm1            # W of the next vector
    vmovups ymm2, [rdi + rcx + 64]
    vaddps  ymm2, ymm2, [rdi + rcx + 60]
    vmovups ymm9, [rdi + rcx + 56]
    vaddps  ymm9, ymm9, [rdi + rcx + 52]
    vaddps  ymm2, ymm2, ymm9
    \store [rsi + rcx], ymm0
    TRACK
    vaddps  ymm0, ymm0, ymm4
    \store [rsi + rcx + 32], ymm0
    TRACK
.endm

# One vector of the short path at byte offset rcx. \dprev holds the D before
# it and \d receives its own. Nothing is loaded ahead.
.macro SHORTVECTOR dprev, d
    vmovups \d, [rdi + rcx]
    vaddps  \d, \d, [rdi + rcx - 4]
    vmovups ymm9, [rdi + rcx - 8]
    vaddps  ymm9, ymm9, [rdi + rcx - 12]
    vaddps  \d, \d, ymm9
    vperm2f128 ymm3, \dprev, \d, 0x21
    vaddps  ymm3, ymm3, \d
    vaddps  ymm0, ymm0, ymm3
    vmovups [rsi + rcx], ymm0
    TRACK
    add     rcx, 32
.endm

.text
.globl cumsum_f32
.type cumsum_f32, @function
.p2align 4
cumsum_f32:
    shl     rdx, 2                      # n in bytes
    cmp     rdx, SMALL * 4
    jb      .Lrestart
    mov     eax, LONG_BYTES
    cmp     rdx, rax
    jae     .Lrestart
    vmovaps ymm6, [rip + .Labs]
    vmovdqa ymm7, [rip + .Llimit32]
    vmovaps ymm1, [rip + .Lnegzero]     # the D before the first vector
    vmovdqa ymm10, [rip + .Lshift1]
    vmovdqa ymm11, [rip + .Lshift2]
    xor     ecx, ecx
    cmp     rdx, SHORT_BYTES
    jb      .Lshort
    vmovaps ymm0, ymm1                  # the output before the first vector
    cmp     rdx, STREAM_MIN_BYTES
    jb      .Lfirst

    # From 24 MB of output the stores go past the caches, which needs an
    # output address that is a multiple of 32: 0..7 elements first, in the
    # reference order. If out is not a multiple of 4 there is no such
    # address and the ordinary stores are used.
    mov     r8d, esi
    neg     r8d
    and     r8d, 31
    test    r8d, 3
    jnz     .Lfirst
    test    r8d, r8d
    jz      .Lfirst
    vmovss  xmm0, dword ptr [rdi]
    vmovss  dword ptr [rsi], xmm0
    mov     ecx, 4
    cmp     rcx, r8
    jae     .Lheaddone
.Lhead:
    vaddss  xmm0, xmm0, dword ptr [rdi + rcx]
    vmovss  dword ptr [rsi + rcx], xmm0
    add     rcx, 4
    cmp     rcx, r8
    jb      .Lhead
.Lheaddone:
    # The elements before rcx count as -0.0 from here on, and their sum as
    # the output before it in every lane.
    vbroadcastss ymm0, xmm0

.Lfirst:
    # The D of the first vector: the elements before it are -0.0.
    vmovups ymm2, [rdi + rcx]
    vpermps ymm3, ymm10, ymm2           # x0 x0 x1 .. x6
    vblendps ymm3, ymm3, ymm1, 0x01     # -0 x0 x1 .. x6
    vaddps  ymm2, ymm2, ymm3            # P: each element plus the one before
    vpermps ymm3, ymm11, ymm2           # P0 P0 P0 P1 .. P5
    vblendps ymm3, ymm3, ymm1, 0x03     # -0 -0 P0 P1 .. P5
    vaddps  ymm2, ymm2, ymm3            # D: each element plus the three before
    mov     r11, rdx
    sub     r11, rcx
    and     r11, -32
    add     r11, rcx                    # where the whole vectors end

.Lchunk:
    # ymm0 = V and ymm1 = the last D, both of the vector before rcx; ymm2 =
    # the D of the vector at rcx, if there is one.
    cmp     rcx, r11
    jae     .Lrest
    mov     r10, rcx
    lea     r9, [rcx + CHUNK_BYTES]
    cmp     r9, r11
    cmova   r9, r11                     # where this chunk ends
    lea     r8, [rdx - 32]
    cmp     r8, r9
    cmova   r8, r9                      # a pair needs a third vector in reach
    vpxor   xmm5, xmm5, xmm5
    cmp     rdx, STREAM_MIN_BYTES
    jb      .Lpairtest
    test    sil, 3                      # as at the start: no aligned address, no such stores
    jz      .Lstreamtest
    jmp     .Lpairtest
.p2align 6
.Lpair:
    PAIR    vmovups
    add     rcx, 64
.Lpairtest:
    lea     rax, [rcx + 64]
    cmp     rax, r8
    jbe     .Lpair
    jmp     .Lsingletest

.p2align 6
.Lstream:
    PAIR    vmovntps
    add     rcx, 64
.Lstreamtest:
    lea     rax, [rcx + 64]
    cmp     rax, r8
    jbe     .Lstream
    sfence                              # the stores above are ordered before what follows
    jmp     .Lsingletest

.Llastpair:
    # The last two whole vectors of the array: no D after them to load.
    vperm2f128 ymm3, ymm1, ymm2, 0x21
    vaddps  ymm3, ymm3, ymm2
    vmovups ymm1, [rdi + rcx + 32]
    vaddps  ymm1, ymm1, [rdi + rcx + 28]
    vmovups ymm9, [rdi + rcx + 24]
    vaddps  ymm9, ymm9, [rdi + rcx + 20]
    vaddps  ymm1, ymm1, ymm9
    vaddps  ymm0, ymm0, ymm3
    vperm2f128 ymm4, ymm2, ymm1, 0x21
    vaddps  ymm4, ymm4, ymm1
    vmovups [rsi + rcx], ymm0
    TRACK
    vaddps  ymm0, ymm0, ymm4
    vmovups [rsi + rcx + 32], ymm0
    TRACK
    add     rcx, 64
    jmp     .Lsingletest

.Lsingle:
    # One vector. The next D is loaded only if its vector is complete.
    vperm2f128 ymm3, ymm1, ymm2, 0x21
    vaddps  ymm3, ymm3, ymm2
    vaddps  ymm0, ymm0, ymm3
    vmovaps ymm1, ymm2
    lea     rax, [rcx + 64]
    cmp     rax, rdx
    ja      .Lsinglestore
    vmovups ymm2, [rdi + rcx + 32]
    vaddps  ymm2, ymm2, [rdi + rcx + 28]
    vmovups ymm9, [rdi + rcx + 24]
    vaddps  ymm9, ymm9, [rdi + rcx + 20]
    vaddps  ymm2, ymm2, ymm9
.Lsinglestore:
    vmovups [rsi + rcx], ymm0
    TRACK
    add     rcx, 32
.Lsingletest:
    lea     rax, [rcx + 64]
    cmp     rax, r9
    je      .Llastpair                  # two left: only at the end of the array
    cmp     rcx, r9
    jb      .Lsingle
    vpcmpgtd ymm4, ymm5, ymm7           # magnitudes above the limit
    vmovmskps eax, ymm4
    test    eax, eax
    jz      .Lchunk
    jmp     .Lredo

    # ---- Short arrays: one chunk, the same vectors with nothing loaded
    #      ahead and no choice of store, which is fewer instructions when
    #      there are only a few vectors.
.Lshort:
    mov     r11, rdx
    and     r11, -32                    # where the whole vectors end
    vmovups ymm2, [rdi + rcx]
    vpermps ymm3, ymm10, ymm2           # x0 x0 x1 .. x6
    vblendps ymm3, ymm3, ymm1, 0x01     # -0 x0 x1 .. x6
    vaddps  ymm2, ymm2, ymm3            # P: each element plus the one before
    vpermps ymm3, ymm11, ymm2           # P0 P0 P0 P1 .. P5
    vblendps ymm3, ymm3, ymm1, 0x03     # -0 -0 P0 P1 .. P5
    vaddps  ymm2, ymm2, ymm3            # D: each element plus the three before
    vperm2f128 ymm3, ymm1, ymm2, 0x21
    vaddps  ymm0, ymm3, ymm2            # W, and V: the output before it is -0.0
    vmovups [rsi], ymm0
    vpand   ymm5, ymm0, ymm6
    mov     ecx, 32
.Lshortloop:
    cmp     rcx, r11
    jae     .Lshortcheck
    SHORTVECTOR ymm2, ymm1
    cmp     rcx, r11
    jae     .Lshortcheck
    SHORTVECTOR ymm1, ymm2
    jmp     .Lshortloop
.Lshortcheck:
    vpcmpgtd ymm4, ymm5, ymm7
    vmovmskps eax, ymm4
    test    eax, eax
    jz      .Lrest
    jmp     .Lrestart

    # ---- The chunk from r10 to r9 was not accepted.
.Lredo:
    cmp     r10, 32                     # the first chunk has no accepted output before it
    jb      .Lrestart
    mov     rcx, r10
    mov     r8, r9
    jmp     .Lwatchstart

    # ---- The elements after the last whole vector.
.Lrest:
    cmp     rcx, rdx
    jae     .Ldone
    mov     r8, rdx
.Lwatchstart:
    vmovss  xmm0, dword ptr [rsi + rcx - 4]     # the last accepted output
    vmovss  xmm3, dword ptr [rip + .Llimit]
.Lwatch:
    vaddss  xmm1, xmm0, dword ptr [rdi + rcx]
    vandps  xmm2, xmm1, xmm6
    vucomiss xmm3, xmm2
    jbe     .Ltrip                      # not below the limit, or NaN
    vmovss  dword ptr [rsi + rcx], xmm1
    vmovaps xmm0, xmm1
    add     rcx, 4
    cmp     rcx, r8
    jb      .Lwatch
    cmp     rcx, rdx
    jae     .Ldone
    # Back to the fast path: V and the last D of the vector before rcx, and
    # the D of the vector at rcx if it is complete.
    vmovups ymm0, [rsi + rcx - 32]
    vmovups ymm1, [rdi + rcx - 32]
    vaddps  ymm1, ymm1, [rdi + rcx - 36]
    vmovups ymm9, [rdi + rcx - 40]
    vaddps  ymm9, ymm9, [rdi + rcx - 44]
    vaddps  ymm1, ymm1, ymm9
    cmp     rcx, r11
    jae     .Lchunk
    vmovups ymm2, [rdi + rcx]
    vaddps  ymm2, ymm2, [rdi + rcx - 4]
    vmovups ymm9, [rdi + rcx - 8]
    vaddps  ymm9, ymm9, [rdi + rcx - 12]
    vaddps  ymm2, ymm2, ymm9
    jmp     .Lchunk

.Ltrip:
    mov     eax, [rdi + rcx]
    and     eax, 0x7fffffff
    cmp     eax, INF_BITS
    jb      .Lrestart                   # x[i] is finite: only the reference order is safe
    # x[i] is not finite: the rest in the plain order, from the sum so far.
.Lplain:
    vaddss  xmm0, xmm0, dword ptr [rdi + rcx]
    vmovss  dword ptr [rsi + rcx], xmm0
    add     rcx, 4
    cmp     rcx, rdx
    jb      .Lplain
    jmp     .Ldone

    # ---- The reference computation, from element 0.
.Lrestart:
    test    rdx, rdx
    jz      .Lreturn
    vmovss  xmm0, dword ptr [rdi]
    mov     ecx, 4
    cmp     rcx, rdx
    jb      .Lplain
.Ldone:
    mov     eax, [rdi]                  # out[0] is a copy of x[0], bit for bit
    mov     [rsi], eax
.Lreturn:
    vzeroupper
    ret
.size cumsum_f32, .-cumsum_f32

.section .note.GNU-stack,"",@progbits
