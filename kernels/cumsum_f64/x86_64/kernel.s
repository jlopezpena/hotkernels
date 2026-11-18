.intel_syntax noprefix
# void cumsum_f64(const double *x, double *out, size_t n);
#
# rdi = x, rsi = out, rdx = n. out[i] = x[0] + ... + x[i] with the additions
# regrouped, so finite results differ from the left-to-right sum by rounding
# only. NaN and infinity appear exactly where the left-to-right sum has them.
# out[0] is a copy of x[0].
#
# Fast path. Four elements per vector. Consecutive output vectors differ by
# the sum of the four elements ending at each lane:
#
#     out[i..i+3] - out[i-4..i-1] = W = x[i..i+3] + x[i-1..i+2] + x[i-2..i+1] + x[i-3..i]
#     D = x[i..i+3] + x[i-1..i+2]      the input plus the same load one element back
#     W = D + (D two elements back)    the upper half of the previous D and the
#                                      lower half of this one, one `vperm2f128`
#     V = (previous V) + W             the output vector
#
# D and W need nothing but the input, so the chain from vector to vector is
# one `vaddpd`: two cycles per four elements where the reference takes two
# per element. Before the array every value is -0.0, which changes nothing
# it is added to. The four lanes are four separate running sums; each out[i]
# is still the sum of x[0..i] in some order.
#
# Why non-finite positions match the left-to-right sum, for every input.
# The array is done in chunks of CHUNK_BYTES. In a chunk the largest
# magnitude of any output is tracked (`vpmaxud` over the outputs with the
# sign bit cleared: for IEEE doubles the upper 32 bits order as the
# magnitudes do, and a NaN or an infinity is above every finite value). The
# chunk is accepted only if every output is below 2^1023 in magnitude. If so:
#   1. Every output of the chunk is finite.
#   2. Every input and every intermediate of the chunk is finite: each D and
#      each W is an operand, directly or through one more addition, of an
#      output of its own vector or the next (the D of the last vector of a
#      chunk is an operand of its own W), and an addition with a NaN or an
#      infinite operand never gives a finite result.
#   3. So each output is a rounded sum of x[0..i] in some order with only
#      finite values on the way, and so is the left-to-right sum for as long
#      as it has stayed finite. Each addition is off by at most 2^-53 of its
#      result, so two such sums of i+1 terms differ by less than
#      (i+1) * 2^-52 * DBL_MAX. The left-to-right sum is therefore within
#      that distance of a value below 2^1023, cannot overflow at this
#      element, and by induction is finite over the whole chunk.
# Both results are finite on an accepted chunk, so no position differs.
# If a chunk is not accepted it is done again one element at a time from the
# last output of the chunk before it (.Lredo), watching every element:
#   - x[i] finite and |sum| < 2^1023: store it. By the argument of point 3
#     the left-to-right sum is finite here too.
#   - x[i] is NaN or infinite: the left-to-right sum was finite up to here
#     (previous case), so from this element on it is not finite, and which of
#     NaN, +inf and -inf it is depends only on the inputs from here on, never
#     on the finite value before. The rest of the array is finished with the
#     plain left-to-right loop (.Lplain), which gives the same kinds.
#   - x[i] finite but |sum| >= 2^1023 or overflowed: finite data close to the
#     largest double. Whether the left-to-right sum overflows here can depend
#     on its exact rounding, so the whole array is redone from element 0 with
#     the plain loop (.Lrestart), which is the reference computation itself.
# A redone chunk that stayed in the first case is followed by the fast path
# again. The first chunk has no output before it and is redone as a restart.
# The last 0..3 elements, after the last whole vector, are done by the same
# watched loop. The rounding bound of point 3 needs fewer than 2^50 elements,
# more than any array can hold.
#
# The proof is the ARM64 kernel's with one change. That kernel accepts a
# chunk when the sum of the squares of its outputs is finite, which limits
# the fast path to running sums below about 1e154. Comparing magnitudes costs
# two integer instructions per vector, raises no floating-point flag, and
# keeps the fast path up to 2^1023, about 9e307.
#
# Short arrays. Fewer than SMALL elements are added strictly left to right,
# which is the reference order, so nothing needs checking.
#
# The caller must not pass an out that overlaps x: the fast path reads x[i-1]
# after out[i-1] is written, and the restart reads all of x again.
#
# Registers: rdi x, rsi out, rdx n in bytes, rcx byte offset of the next
# output, r9 end of the chunk, r10 start of the chunk, r11 end of the whole
# vectors, r8 limit of the watched loop. ymm0 V, ymm1 and ymm2 the last two
# D, ymm5 the chunk's largest magnitude, ymm6 the sign-clearing mask, ymm7
# the threshold.

# Bytes of output per chunk: 1,024 elements.
.set CHUNK_BYTES, 8192
# Below this many elements the plain chain is used.
.set SMALL, 8

.section .rodata
.p2align 5
.Labs:
    .quad 0x7fffffffffffffff, 0x7fffffffffffffff, 0x7fffffffffffffff, 0x7fffffffffffffff
# A magnitude is acceptable when its upper 32 bits are at most this: below 2^1023.
.Llimit32:
    .long 0x7fdfffff, 0x7fdfffff, 0x7fdfffff, 0x7fdfffff, 0x7fdfffff, 0x7fdfffff, 0x7fdfffff, 0x7fdfffff
.Lnegzero:
    .quad 0x8000000000000000, 0x8000000000000000, 0x8000000000000000, 0x8000000000000000
# 2^1023, the same limit as a double for the watched loop.
.Llimit:
    .quad 0x7fe0000000000000
# Bits of +infinity: a magnitude at or above it is not finite.
.Linf:
    .quad 0x7ff0000000000000

# One vector of the fast path at byte offset rcx + \off. \dprev holds the D
# before it and \d receives its own.
.macro VECTOR off, dprev, d
    vmovupd \d, [rdi + rcx + \off]
    vaddpd  \d, \d, [rdi + rcx + \off - 8]
    vperm2f128 ymm3, \dprev, \d, 0x21
    vaddpd  ymm3, ymm3, \d              # W
    vaddpd  ymm0, ymm0, ymm3            # V
    vmovupd [rsi + rcx + \off], ymm0
    vpand   ymm4, ymm0, ymm6
    vpmaxud ymm5, ymm5, ymm4
.endm

.text
.globl cumsum_f64
.type cumsum_f64, @function
.p2align 4
cumsum_f64:
    shl     rdx, 3                      # n in bytes
    cmp     rdx, SMALL * 8
    jb      .Lrestart
    vmovapd ymm6, [rip + .Labs]
    vmovdqa ymm7, [rip + .Llimit32]
    vmovapd ymm1, [rip + .Lnegzero]     # the D before the array
    mov     r11, rdx
    and     r11, -32                    # where the whole vectors end

    # The first vector: the element before it is -0.0.
    vmovupd ymm2, [rdi]
    vpermpd ymm3, ymm2, 0x90            # x0 x0 x1 x2
    vblendpd ymm3, ymm3, ymm1, 0x01     # -0 x0 x1 x2
    vaddpd  ymm2, ymm2, ymm3            # D
    vperm2f128 ymm3, ymm1, ymm2, 0x21
    vaddpd  ymm0, ymm3, ymm2            # W, and V: the output before it is -0.0
    vmovupd [rsi], ymm0
    vpand   ymm5, ymm0, ymm6            # the first chunk's largest magnitude so far
    vmovapd ymm1, ymm2
    mov     ecx, 32
    xor     r10d, r10d                  # the first chunk starts at element 0
    lea     r9, [r10 + CHUNK_BYTES]
    cmp     r9, r11
    cmova   r9, r11
    jmp     .Lpairtest

.Lchunk:
    # ymm0 = V and ymm1 = the last D, both of the vector before rcx.
    cmp     rcx, r11
    jae     .Lrest
    mov     r10, rcx
    lea     r9, [rcx + CHUNK_BYTES]
    cmp     r9, r11
    cmova   r9, r11                     # where this chunk ends
    vpxor   xmm5, xmm5, xmm5
    jmp     .Lpairtest
.p2align 6
.Lpair:
    VECTOR  0, ymm1, ymm2
    VECTOR  32, ymm2, ymm1
    add     rcx, 64
.Lpairtest:
    lea     rax, [rcx + 64]
    cmp     rax, r9
    jbe     .Lpair
    cmp     rcx, r9
    jae     .Lcheck
    VECTOR  0, ymm1, ymm2
    vmovapd ymm1, ymm2
    add     rcx, 32
.Lcheck:
    vpcmpgtd ymm4, ymm5, ymm7           # upper halves above the limit
    vmovmskpd eax, ymm4
    test    eax, eax
    jz      .Lchunk

    # ---- The chunk from r10 to r9 was not accepted.
.Lredo:
    test    r10, r10
    jz      .Lrestart
    mov     rcx, r10
    mov     r8, r9
    jmp     .Lwatchstart

    # ---- The elements after the last whole vector.
.Lrest:
    cmp     rcx, rdx
    jae     .Ldone
    mov     r8, rdx
.Lwatchstart:
    vmovsd  xmm0, qword ptr [rsi + rcx - 8]     # the last accepted output
    vmovsd  xmm3, qword ptr [rip + .Llimit]
.Lwatch:
    vaddsd  xmm1, xmm0, qword ptr [rdi + rcx]
    vandpd  xmm2, xmm1, xmm6
    vucomisd xmm3, xmm2
    jbe     .Ltrip                      # not below the limit, or NaN
    vmovsd  qword ptr [rsi + rcx], xmm1
    vmovapd xmm0, xmm1
    add     rcx, 8
    cmp     rcx, r8
    jb      .Lwatch
    cmp     rcx, rdx
    jae     .Ldone
    # Back to the fast path: V and the last D of the vector before rcx.
    vmovupd ymm0, [rsi + rcx - 32]
    vmovupd ymm1, [rdi + rcx - 32]
    vaddpd  ymm1, ymm1, [rdi + rcx - 40]
    jmp     .Lchunk

.Ltrip:
    mov     rax, [rdi + rcx]
    btr     rax, 63
    cmp     rax, [rip + .Linf]
    jb      .Lrestart                   # x[i] is finite: only the reference order is safe
    # x[i] is not finite: the rest in the plain order, from the sum so far.
.Lplain:
    vaddsd  xmm0, xmm0, qword ptr [rdi + rcx]
    vmovsd  qword ptr [rsi + rcx], xmm0
    add     rcx, 8
    cmp     rcx, rdx
    jb      .Lplain
    jmp     .Ldone

    # ---- The reference computation, from element 0.
.Lrestart:
    test    rdx, rdx
    jz      .Lreturn
    vmovsd  xmm0, qword ptr [rdi]
    mov     ecx, 8
    cmp     rcx, rdx
    jb      .Lplain
.Ldone:
    mov     rax, [rdi]                  # out[0] is a copy of x[0], bit for bit
    mov     [rsi], rax
.Lreturn:
    vzeroupper
    ret
.size cumsum_f64, .-cumsum_f64

.section .note.GNU-stack,"",@progbits
