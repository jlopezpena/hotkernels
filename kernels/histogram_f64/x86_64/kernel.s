.intel_syntax noprefix
# void histogram_f64(const double *x, int64_t *counts, size_t n, size_t nbins, double lo, double hi);
#
# rdi = x, rsi = counts, rdx = n, rcx = nbins, xmm0 = lo, xmm1 = hi. counts
# has exactly nbins slots and arrives uninitialised.
#
# naive.c, for each element v with lo <= v <= hi:
#     b = (size_t)(((v - lo) / width) * nbins)        width = hi - lo
#     if (b == nbins) b--
#     if (v < edge(b)) b--                            edge(i) = (double)i * step + lo
#     if (v >= edge(b + 1) && b != nbins - 1) b++     step = width / nbins, edge(nbins) = hi
#     counts[b]++
# Every floating-point operation is rounded on its own; an edge is a separate
# multiply and add.
#
# ---- The scalar route (.Lscalar) --------------------------------------------
# naive.c verbatim, one element at a time. It is right for every input and is
# used for whatever the vector path is not sure about: single elements, the
# last 0..3 elements, and whole calls whose bin count and range fail the check
# below.
#
# ---- The vector path --------------------------------------------------------
# NOT a transcription of naive.c. Four elements per vector. It estimates the
# position of v among the bins,
#     t = (v - lo) * S,    S = nbins / width, computed once,
# so there is no divide per element, which on this processor would cost two
# cycles per element by itself. It takes the estimate as 20.12 fixed point,
#     z = trunc(t * 4096)          (`vcvttpd2dq`; the factor is folded into S)
# and accepts the element into bin b = z >> 12 when
#     z <= nbins * 4096 - 1  as unsigned numbers,   and
#     the low 12 bits of z are neither 0 nor 0xfff.
# That is: b <= nbins - 1 and DELTA <= fraction(t) < 1 - DELTA, DELTA = 2^-12.
# About one in-range element in 2,048 is not accepted, and neither is anything
# out of range. If any of the four is not accepted, all four take the scalar
# route.
#
# Why an accepted element is counted where naive.c counts it. Write u = 2^-53,
# N = nbins, w = width, M = max(|lo|, |hi|), and tau = (v - lo) * N / w as a
# real number.
#   - NaN, and any t too large for 32 bits, convert to 0x80000000, which is
#     above the limit as an unsigned number. A negative t gives a negative z
#     (above the limit as an unsigned number) or, between -1/4096 and 0, z = 0,
#     whose low bits are 0. t >= N gives z >= N * 4096. None is accepted.
#   - For the rest, t is tau after three rounded operations (the subtraction,
#     S, and the product; the scaling by 4096 is exact), so |t - tau| <= 3.01 u N.
#     Acceptance means b + DELTA <= t < b + 1 - DELTA with b <= N - 1.
#   - edge(k) = fl(fl(k * step) + lo) with step = fl(w / N), a normal number
#     (checked). Expanding the three rounding errors, v >= edge(k) is certain
#     when tau >= k + D and v < edge(k) is certain when tau < k - D, where
#     D = 2.01 u N + 1.01 u M N / w. This is the ARM64 kernel's bound; the
#     operations it is about are naive.c's, the same on every architecture.
#   - The check below gives 4 u N + D < DELTA. So an accepted element has
#     tau >= b + D, hence v >= edge(b) and v > lo; and tau < b + 1 - D, hence,
#     for b <= N - 2, v < edge(b + 1).
#   - naive.c's own estimate, fl(fl(fl(v - lo) / w) * N), is also within
#     4 u N of tau, so it truncates to the same b: its first step does
#     nothing (b < N), "v < edge(b)" is false, and for b <= N - 2
#     "v >= edge(b + 1)" is false. Bin b.
#   - For b = N - 1 naive.c never takes its last step, so only v >= edge(N - 1)
#     is needed, and v <= hi: tau < N - DELTA + 3.01 u N puts v below
#     lo + w - w (DELTA - 3.01 u N) / N, and lo + w is at most hi + u w, so
#     v < hi as soon as DELTA >= 4.01 u N. (The ARM64 kernel sends the whole
#     last bin to its scalar route; with few bins that is a large share of the
#     elements.)
# The check, made once per call, in doubles:
#     N < 2^19                       so that N * 4096 fits the conversion
#     step is a normal number        so w > 0 and step's rounding is relative
#     w is finite
#     fl(M * N) is finite, and fl(M * N) <= fl(w * 2^39)
#     S * 4096 is finite
# With N < 2^19 and M N <= w 2^39 (an infinite right side means it really is
# the larger; two finite products are each within u of their real values,
# which the slack absorbs): 4 u N + D <= 6.01 * 2^-34 + 1.01 * 2^-14 < 2^-12.
# The left side has to be finite: near the top of the range both products
# overflow, compare as equal, and would pass for a range where the bound is
# false.
#
# Registers: rdi x, rsi counts, rcx nbins, r8 blocks left, r10 elements after
# the blocks, r9 and r11 the count and the return address of the scalar
# route, rax and rdx scratch. xmm0 lo, xmm1 hi, xmm2 width, xmm3 nbins as a
# double, xmm4 step (the scalar route's constants); ymm5-ymm7 scratch; ymm8
# -lo, ymm9 S * 4096; xmm10 ones, xmm11 0xffe, xmm12 the limit of z, xmm13
# zeros (32-bit lanes).

.section .rodata
.p2align 4
.Labs:
    .quad 0x7fffffffffffffff, 0
.Lsign:
    .quad 0x8000000000000000, 0
# The smallest normal double, the largest finite one, 2^39 and 4096.
.Ldblmin:
    .quad 0x0010000000000000
.Ldblmax:
    .quad 0x7fefffffffffffff
.Ltwo39:
    .quad 0x4260000000000000
.L4096:
    .quad 0x40b0000000000000

# The largest bin count the vector path takes, plus one.
.set VECTOR_MAX_BINS, 1 << 19
# The low bits of z that hold the fraction, without the lowest.
.set FRACTION_TEST, 0xffe

.text
.globl histogram_f64
.type histogram_f64, @function
.p2align 4
histogram_f64:
    # Clear the counts.
    mov     r8, rdi
    mov     r9, rcx
    mov     rdi, rsi
    xor     eax, eax
    rep stosq
    mov     rdi, r8
    mov     rcx, r9
    test    rdx, rdx
    jz      .Lreturn

    vsubsd  xmm2, xmm1, xmm0            # width
    vcvtsi2sd xmm3, xmm3, rcx           # nbins as a double
    vdivsd  xmm4, xmm2, xmm3            # step

    # The check. Any NaN fails the first compare that sees it.
    mov     r9, rdx                     # for the scalar route, if the check fails
    lea     r11, [rip + .Ldone]
    cmp     rcx, VECTOR_MAX_BINS
    jae     .Lscalar
    vucomisd xmm4, qword ptr [rip + .Ldblmin]
    jb      .Lscalar                    # step not normal, or NaN
    vucomisd xmm2, qword ptr [rip + .Ldblmax]
    ja      .Lscalar                    # width infinite
    vandpd  xmm5, xmm0, [rip + .Labs]
    vandpd  xmm6, xmm1, [rip + .Labs]
    vmaxsd  xmm5, xmm5, xmm6            # M
    vmulsd  xmm5, xmm5, xmm3            # M * N
    vucomisd xmm5, qword ptr [rip + .Ldblmax]
    ja      .Lscalar                    # not finite
    vmulsd  xmm6, xmm2, qword ptr [rip + .Ltwo39]
    vucomisd xmm5, xmm6
    ja      .Lscalar                    # M * N > w * 2^39
    vdivsd  xmm7, xmm3, xmm2
    vmulsd  xmm7, xmm7, qword ptr [rip + .L4096]    # S * 4096
    vucomisd xmm7, qword ptr [rip + .Ldblmax]
    ja      .Lscalar

    mov     r10, rdx
    and     r10d, 3                     # elements after the blocks
    mov     r8, rdx
    shr     r8, 2                       # blocks of four
    jz      .Ltail
    vxorpd  xmm5, xmm0, [rip + .Lsign]
    vbroadcastsd ymm8, xmm5             # -lo
    vbroadcastsd ymm9, xmm7
    vpcmpeqd xmm10, xmm10, xmm10
    vpsrld  xmm10, xmm10, 31            # ones
    mov     eax, FRACTION_TEST
    vmovd   xmm11, eax
    vpbroadcastd xmm11, xmm11
    mov     eax, ecx
    shl     eax, 12
    dec     eax                         # nbins * 4096 - 1
    vmovd   xmm12, eax
    vpbroadcastd xmm12, xmm12
    vpxor   xmm13, xmm13, xmm13
.p2align 6
.Lblock:
    vaddpd  ymm5, ymm8, [rdi]           # v - lo
    vmulpd  ymm5, ymm5, ymm9            # t * 4096
    vcvttpd2dq xmm5, ymm5               # z
    vpaddd  xmm6, xmm5, xmm10
    vpand   xmm6, xmm6, xmm11
    vpcmpeqd xmm6, xmm6, xmm13          # fraction bits all 0 or all 1
    vpminud xmm7, xmm5, xmm12
    vpcmpeqd xmm7, xmm7, xmm5           # z within the limit
    vpandn  xmm6, xmm6, xmm7            # accepted
    vmovmskps eax, xmm6
    cmp     eax, 15
    jne     .Lslowblock
    vpsrld  xmm5, xmm5, 12              # the four bins
    vmovq   rax, xmm5
    vpextrq rdx, xmm5, 1
    mov     r9d, eax
    shr     rax, 32
    inc     qword ptr [rsi + r9 * 8]
    inc     qword ptr [rsi + rax * 8]
    mov     r9d, edx
    shr     rdx, 32
    inc     qword ptr [rsi + r9 * 8]
    inc     qword ptr [rsi + rdx * 8]
    add     rdi, 32
.Lnextblock:
    dec     r8
    jnz     .Lblock
.Ltail:
    test    r10, r10
    jz      .Ldone
    mov     r9, r10
    lea     r11, [rip + .Ldone]
    jmp     .Lscalar

.Lslowblock:
    mov     r9d, 4
    lea     r11, [rip + .Lnextblock]

    # ---- The scalar route: r9 elements at rdi, as naive.c counts them, then
    #      on to the address in r11. rdi is left after the last of them.
.Lscalar:
    vmovsd  xmm5, qword ptr [rdi]
    add     rdi, 8
    vucomisd xmm5, xmm0
    jb      .Lscalarnext                # v < lo, or NaN
    vucomisd xmm1, xmm5
    jb      .Lscalarnext                # v > hi
    vsubsd  xmm6, xmm5, xmm0
    vdivsd  xmm6, xmm6, xmm2
    vmulsd  xmm6, xmm6, xmm3
    vcvttsd2si rax, xmm6                # b
    lea     rdx, [rax - 1]
    cmp     rax, rcx
    cmove   rax, rdx                    # b == nbins: b--
    vcvtsi2sd xmm6, xmm6, rax
    vmulsd  xmm6, xmm6, xmm4
    vaddsd  xmm6, xmm6, xmm0            # edge(b)
    lea     rdx, [rax - 1]
    vucomisd xmm5, xmm6
    cmovb   rax, rdx                    # v < edge(b): b--
    lea     rdx, [rax + 1]
    cmp     rdx, rcx
    je      .Lcount                     # the last bin takes everything up to hi
    vcvtsi2sd xmm6, xmm6, rdx
    vmulsd  xmm6, xmm6, xmm4
    vaddsd  xmm6, xmm6, xmm0            # edge(b + 1)
    vucomisd xmm5, xmm6
    cmovae  rax, rdx                    # v >= edge(b + 1): b++
.Lcount:
    inc     qword ptr [rsi + rax * 8]
.Lscalarnext:
    dec     r9
    jnz     .Lscalar
    jmp     r11

.Ldone:
    vzeroupper
.Lreturn:
    ret
.size histogram_f64, .-histogram_f64

.section .note.GNU-stack,"",@progbits
