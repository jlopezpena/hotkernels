.intel_syntax noprefix
# void interp_f64(const double *x, const double *xp, const double *fp, double *out,
#                 size_t n, size_t queries);
#
# rdi = x (the queries), rsi = xp and rdx = fp (n sample points each, n >= 1),
# rcx = out, r8 = n, r9 = queries. out[i] = np.interp(x[i], xp, fp), bit for
# bit, with the default left and right values. xp is ascending with no NaN.
#
# naive.c, for each query v, with n >= 2:
#     v is a NaN            -> v
#     v > xp[n-1]           -> fp[n-1]
#     v < xp[0]             -> fp[0]
#     j = (number of sample points <= v) - 1
#     j == n-1 or xp[j] == v -> fp[j]
#     slope = (fp[j+1] - fp[j]) / (xp[j+1] - xp[j])
#     r = slope * (v - xp[j]) + fp[j]
#     if r is a NaN: r = slope * (v - xp[j+1]) + fp[j+1]
#                    if r is a NaN and fp[j] == fp[j+1]: r = fp[j]
# With one sample point every answer is fp[0].
#
# The multiply and the add are two instructions here, each rounded: NumPy's
# x86-64 build computes them that way. Its ARM64 build fuses them, and so
# does the ARM64 kernel.
#
# Two phases per chunk of queries, four queries per vector.
#
# The search is the `searchsorted_right_f64` kernel's: a fixed number of
# steps that depends only on n,
#     half = (len + 1) / 2;  if xp[lo + half - 1] <= v: lo += half;  len -= half
# with one `vgatherqpd` per four queries for the probes, up to 64 queries in
# step (512 when xp does not fit the second-level cache), and lo kept in out,
# which is one 64-bit slot per query. It leaves lo = the number of sample
# points <= v, and 0 for a NaN query. Every probe is inside xp whatever the
# data.
#
# The values then take lo, as naive.c's cases, without a branch:
#   - j = lo - 1, moved into 0 .. n-2 when lo is 0 or n, so that xp[j],
#     xp[j+1], fp[j] and fp[j+1] can all be gathered;
#   - both of naive.c's expressions are computed, with a real `vdivpd`, and
#     the second replaces the first where the first is a NaN, then fp[j]
#     replaces that where it is still a NaN and fp[j] == fp[j+1];
#   - fp[j] where v == xp[j] or lo == 0 (v < xp[0], with j moved to 0);
#   - fp[j+1] where lo == n (v >= xp[n-1], with j moved to n-2);
#   - v where v is a NaN.
# Each later line overrides the earlier ones, which is naive.c's order of
# tests read backwards.
#
# A count that is not a multiple of four: the last four queries are done
# first, as one vector in registers, and their values are kept below the
# stack pointer until every chunk is done, because the chunks overwrite lo
# with values as they go. One to three queries take a scalar copy of naive.c.
#
# `vgatherqpd` is fast on this processor and slow on some others (older AMD,
# and Intel processors that carry the gather mitigation).
#
# Registers: r12 len, rax the address of xp[half - 1], r10 and r11 the byte
# offsets where the chunk starts and ends, rbx the offset inside it, r13 the
# chunk size, r14 where the whole vectors end (rbx and r12-r14 are saved
# below the stack pointer). ymm8 half, ymm9 the gather's mask, ymm13 zeros,
# ymm14 minus ones, ymm15 n (64-bit lanes).

# Queries per chunk, in bytes of x: 64 while xp fits the second-level cache.
.set CHUNK_BYTES, 512
# From this many sample points xp no longer does (2 MB), and the chunk is 512
# queries: more cache misses in flight at once.
.set LARGE_N, 262144
.set LARGE_CHUNK_BYTES, 4096
# Compare predicates that do not signal for a quiet NaN: <=.
.set LE_OQ, 0x12

.text
.globl interp_f64
.type interp_f64, @function
.p2align 4
interp_f64:
    test    r9, r9
    jz      .Lreturn
    cmp     r8, 1
    je      .Lsingle
    cmp     r9, 4
    jb      .Lfew
    mov     [rsp - 8], rbx
    mov     [rsp - 16], r12
    mov     [rsp - 24], r13
    mov     [rsp - 32], r14
    vpxor   xmm13, xmm13, xmm13
    vpcmpeqd ymm14, ymm14, ymm14
    vmovq   xmm15, r8
    vpbroadcastq ymm15, xmm15
    mov     r13d, CHUNK_BYTES
    mov     eax, LARGE_CHUNK_BYTES
    cmp     r8, LARGE_N
    cmovae  r13d, eax
    mov     r14, r9
    shr     r14, 2
    shl     r14, 5                      # where the whole vectors end

    # ---- The last four queries first, if the count is not a multiple of four.
    test    r9b, 3
    jz      .Lchunks
    vmovupd ymm4, [rdi + r9 * 8 - 32]
    vpxor   xmm0, xmm0, xmm0
    mov     r12, r8
.Llaststep:
    lea     rax, [r12 + 1]
    shr     rax, 1
    vmovq   xmm8, rax
    vpbroadcastq ymm8, xmm8
    sub     r12, rax
    lea     rax, [rsi + rax * 8 - 8]
    vpcmpeqd ymm9, ymm9, ymm9
    vpxor   xmm10, xmm10, xmm10
    vgatherqpd ymm10, [rax + ymm0 * 8], ymm9
    vcmppd  ymm10, ymm10, ymm4, LE_OQ
    vpand   ymm10, ymm10, ymm8
    vpaddq  ymm0, ymm0, ymm10
    test    r12, r12
    jnz     .Llaststep
    vpcmpeqq ymm1, ymm0, ymm13          # lo == 0: v < xp[0], or v is a NaN
    vpcmpeqq ymm2, ymm0, ymm15          # lo == n: v >= xp[n-1]
    vpaddq  ymm3, ymm0, ymm14
    vpsubq  ymm3, ymm3, ymm1
    vpaddq  ymm3, ymm3, ymm2            # j = lo - 1, kept inside 0 .. n-2
    vpcmpeqd ymm9, ymm9, ymm9
    vpxor   xmm5, xmm5, xmm5
    vgatherqpd ymm5, [rsi + ymm3 * 8], ymm9
    vpcmpeqd ymm9, ymm9, ymm9
    vpxor   xmm6, xmm6, xmm6
    vgatherqpd ymm6, [rsi + ymm3 * 8 + 8], ymm9
    vpcmpeqd ymm9, ymm9, ymm9
    vpxor   xmm7, xmm7, xmm7
    vgatherqpd ymm7, [rdx + ymm3 * 8], ymm9
    vpcmpeqd ymm9, ymm9, ymm9
    vpxor   xmm10, xmm10, xmm10
    vgatherqpd ymm10, [rdx + ymm3 * 8 + 8], ymm9
    vsubpd  ymm11, ymm10, ymm7
    vsubpd  ymm12, ymm6, ymm5
    vdivpd  ymm11, ymm11, ymm12         # slope
    vsubpd  ymm12, ymm4, ymm5
    vmulpd  ymm12, ymm12, ymm11
    vaddpd  ymm12, ymm12, ymm7          # slope * (v - xp[j]) + fp[j], rounded twice
    vsubpd  ymm0, ymm4, ymm6
    vmulpd  ymm0, ymm0, ymm11
    vaddpd  ymm0, ymm0, ymm10           # slope * (v - xp[j+1]) + fp[j+1]
    vcmpunordpd ymm3, ymm12, ymm12
    vblendvpd ymm0, ymm12, ymm0, ymm3   # the second form where the first is a NaN
    vcmpunordpd ymm3, ymm0, ymm0
    vcmpeqpd ymm11, ymm7, ymm10
    vandpd  ymm3, ymm3, ymm11
    vblendvpd ymm0, ymm0, ymm7, ymm3    # still a NaN and fp[j] == fp[j+1]: fp[j]
    vcmpeqpd ymm3, ymm5, ymm4
    vorpd   ymm3, ymm3, ymm1
    vblendvpd ymm0, ymm0, ymm7, ymm3    # v == xp[j], or v < xp[0]: fp[j]
    vblendvpd ymm0, ymm0, ymm10, ymm2   # v >= xp[n-1]: fp[n-1]
    vcmpunordpd ymm3, ymm4, ymm4
    vblendvpd ymm0, ymm0, ymm4, ymm3    # a NaN query gives that NaN
    vmovupd [rsp - 72], ymm0

.Lchunks:
    xor     r10d, r10d
.Lchunk:
    cmp     r10, r14
    jae     .Lfinish
    lea     r11, [r10 + r13]
    cmp     r11, r14
    cmova   r11, r14                    # where this chunk ends
    # The first step: every search probes xp[half - 1].
    mov     r12, r8
    lea     rax, [r12 + 1]
    shr     rax, 1
    vmovq   xmm8, rax
    vpbroadcastq ymm8, xmm8
    sub     r12, rax
    lea     rax, [rsi + rax * 8 - 8]
    vbroadcastsd ymm10, qword ptr [rax]
    mov     rbx, r10
.Lfirst:
    vcmppd  ymm11, ymm10, [rdi + rbx], LE_OQ
    vpand   ymm11, ymm11, ymm8
    vmovdqu [rcx + rbx], ymm11
    add     rbx, 32
    cmp     rbx, r11
    jb      .Lfirst
.Lstep:
    lea     rax, [r12 + 1]
    shr     rax, 1
    vmovq   xmm8, rax
    vpbroadcastq ymm8, xmm8
    sub     r12, rax
    lea     rax, [rsi + rax * 8 - 8]
    mov     rbx, r10
.p2align 5
.Lprobe:
    vmovdqu ymm0, [rcx + rbx]
    vpcmpeqd ymm9, ymm9, ymm9
    vpxor   xmm10, xmm10, xmm10         # a gather merges into its destination: start it afresh
    vgatherqpd ymm10, [rax + ymm0 * 8], ymm9
    vcmppd  ymm10, ymm10, [rdi + rbx], LE_OQ
    vpand   ymm10, ymm10, ymm8
    vpaddq  ymm0, ymm0, ymm10
    vmovdqu [rcx + rbx], ymm0
    add     rbx, 32
    cmp     rbx, r11
    jb      .Lprobe
    test    r12, r12
    jnz     .Lstep
    # The values of the chunk.
    mov     rbx, r10
.Lvalue:
    vmovdqu ymm0, [rcx + rbx]
    vmovupd ymm4, [rdi + rbx]
    vpcmpeqq ymm1, ymm0, ymm13          # lo == 0: v < xp[0], or v is a NaN
    vpcmpeqq ymm2, ymm0, ymm15          # lo == n: v >= xp[n-1]
    vpaddq  ymm3, ymm0, ymm14
    vpsubq  ymm3, ymm3, ymm1
    vpaddq  ymm3, ymm3, ymm2            # j = lo - 1, kept inside 0 .. n-2
    vpcmpeqd ymm9, ymm9, ymm9
    vpxor   xmm5, xmm5, xmm5
    vgatherqpd ymm5, [rsi + ymm3 * 8], ymm9
    vpcmpeqd ymm9, ymm9, ymm9
    vpxor   xmm6, xmm6, xmm6
    vgatherqpd ymm6, [rsi + ymm3 * 8 + 8], ymm9
    vpcmpeqd ymm9, ymm9, ymm9
    vpxor   xmm7, xmm7, xmm7
    vgatherqpd ymm7, [rdx + ymm3 * 8], ymm9
    vpcmpeqd ymm9, ymm9, ymm9
    vpxor   xmm10, xmm10, xmm10
    vgatherqpd ymm10, [rdx + ymm3 * 8 + 8], ymm9
    vsubpd  ymm11, ymm10, ymm7
    vsubpd  ymm12, ymm6, ymm5
    vdivpd  ymm11, ymm11, ymm12         # slope
    vsubpd  ymm12, ymm4, ymm5
    vmulpd  ymm12, ymm12, ymm11
    vaddpd  ymm12, ymm12, ymm7          # slope * (v - xp[j]) + fp[j], rounded twice
    vsubpd  ymm0, ymm4, ymm6
    vmulpd  ymm0, ymm0, ymm11
    vaddpd  ymm0, ymm0, ymm10           # slope * (v - xp[j+1]) + fp[j+1]
    vcmpunordpd ymm3, ymm12, ymm12
    vblendvpd ymm0, ymm12, ymm0, ymm3   # the second form where the first is a NaN
    vcmpunordpd ymm3, ymm0, ymm0
    vcmpeqpd ymm11, ymm7, ymm10
    vandpd  ymm3, ymm3, ymm11
    vblendvpd ymm0, ymm0, ymm7, ymm3    # still a NaN and fp[j] == fp[j+1]: fp[j]
    vcmpeqpd ymm3, ymm5, ymm4
    vorpd   ymm3, ymm3, ymm1
    vblendvpd ymm0, ymm0, ymm7, ymm3    # v == xp[j], or v < xp[0]: fp[j]
    vblendvpd ymm0, ymm0, ymm10, ymm2   # v >= xp[n-1]: fp[n-1]
    vcmpunordpd ymm3, ymm4, ymm4
    vblendvpd ymm0, ymm0, ymm4, ymm3    # a NaN query gives that NaN
    vmovupd [rcx + rbx], ymm0
    add     rbx, 32
    cmp     rbx, r11
    jb      .Lvalue
    mov     r10, r11
    jmp     .Lchunk

.Lfinish:
    test    r9b, 3
    jz      .Lrestore
    vmovupd ymm0, [rsp - 72]
    vmovupd [rcx + r9 * 8 - 32], ymm0
.Lrestore:
    mov     rbx, [rsp - 8]
    mov     r12, [rsp - 16]
    mov     r13, [rsp - 24]
    mov     r14, [rsp - 32]
    vzeroupper
.Lreturn:
    ret

    # ---- One sample point: every answer is fp[0].
.Lsingle:
    mov     rax, [rdx]
    xor     r10d, r10d
.Lsingleone:
    mov     [rcx + r10 * 8], rax
    inc     r10
    cmp     r10, r9
    jb      .Lsingleone
    ret

    # ---- One to three queries: naive.c, one query at a time. No 256-bit
    #      register is written here.
.Lfew:
    mov     [rsp - 40], rdi             # x: rdi is borrowed as scratch below
    xor     r10d, r10d                  # query index
.Lfewone:
    vmovsd  xmm0, qword ptr [rdi + r10 * 8]     # v
    vucomisd xmm0, xmm0
    jp      .Lfewstore                  # a NaN gives that NaN
    xor     eax, eax                    # lo
    mov     r11, r8                     # len
.Lfewstep:
    lea     rdi, [r11 + 1]
    shr     rdi, 1                      # half (rdi is free: x is addressed through r10 below)
    sub     r11, rdi
    add     rdi, rax                    # lo + half
    vucomisd xmm0, qword ptr [rsi + rdi * 8 - 8]
    cmovae  rax, rdi                    # xp[lo + half - 1] <= v
    test    r11, r11
    jnz     .Lfewstep
    vmovsd  xmm0, qword ptr [rdx]       # fp[0], for v < xp[0]
    test    rax, rax
    jz      .Lfewstore
    vmovsd  xmm0, qword ptr [rdx + r8 * 8 - 8]  # fp[n-1], for v >= xp[n-1]
    cmp     rax, r8
    je      .Lfewstore
    # 1 <= lo <= n-1: j = lo - 1, and rax indexes j + 1.
    mov     rdi, [rsp - 40]             # x again
    vmovsd  xmm1, qword ptr [rdi + r10 * 8]     # v
    vmovsd  xmm0, qword ptr [rdx + rax * 8 - 8] # fp[j]
    vmovsd  xmm2, qword ptr [rsi + rax * 8 - 8] # xp[j]
    vucomisd xmm1, xmm2
    je      .Lfewstore                  # v == xp[j]
    vmovsd  xmm3, qword ptr [rdx + rax * 8]     # fp[j+1]
    vmovsd  xmm4, qword ptr [rsi + rax * 8]     # xp[j+1]
    vsubsd  xmm5, xmm3, xmm0
    vsubsd  xmm6, xmm4, xmm2
    vdivsd  xmm5, xmm5, xmm6            # slope
    vsubsd  xmm6, xmm1, xmm2
    vmulsd  xmm6, xmm6, xmm5
    vaddsd  xmm6, xmm6, xmm0            # slope * (v - xp[j]) + fp[j]
    vucomisd xmm6, xmm6
    jnp     .Lfewvalue
    vsubsd  xmm6, xmm1, xmm4
    vmulsd  xmm6, xmm6, xmm5
    vaddsd  xmm6, xmm6, xmm3            # slope * (v - xp[j+1]) + fp[j+1]
    vucomisd xmm6, xmm6
    jnp     .Lfewvalue
    vucomisd xmm0, xmm3
    jne     .Lfewvalue
    jp      .Lfewvalue
    jmp     .Lfewstore                  # fp[j] == fp[j+1]: fp[j]
.Lfewvalue:
    vmovapd xmm0, xmm6
.Lfewstore:
    mov     rdi, [rsp - 40]
    vmovsd  qword ptr [rcx + r10 * 8], xmm0
    inc     r10
    cmp     r10, r9
    jb      .Lfewone
    ret
.size interp_f64, .-interp_f64

.section .note.GNU-stack,"",@progbits
