.intel_syntax noprefix
# void cumfmax_f64(const double *x, double *out, size_t n);
#
# rdi = x, rsi = out, rdx = n. out[i] = the largest of x[0..i] that is not
# NaN, as np.fmax.accumulate gives it bit for bit: a NaN element changes
# nothing, the output is x[0] for as long as every element so far is NaN, and
# of two equal values the later one is kept, which decides the sign of a
# zero. For n = 2 NumPy has it the other way round: the earlier of two equal
# values and the later of two NaNs. No element is a signaling NaN, and out
# does not overlap x.
#
# `vmaxpd a, b` is "a if a > b, otherwise b": b when the two are equal and b
# when either is NaN. With the older value as a and the newer as b it keeps
# the later of two equal values, which is the rule wanted. It does not ignore
# a NaN, so every NaN is first replaced by minus infinity, with a `vmaxpd`
# of the element against minus infinity. That changes no output once the
# running value M is a number: M is at least minus infinity, and a tie
# between the two is minus infinity either way. The elements before the first
# number are done one at a time.
#
# From there, four elements per vector. Two steps make the running maximum
# inside the vector: each lane against the lane before it, then against the
# lane two before. The output is that against M in every lane, and the next M
# is M against the vector's last lane, broadcast: the chain from vector to
# vector is one `vmaxpd`.
#
# rdi and rsi point at the ends of the arrays and rcx, negative, counts bytes
# up to zero.

.section .rodata
.p2align 3
.Lneginf:
    .quad 0xfff0000000000000            # minus infinity

.text
.globl cumfmax_f64
.type cumfmax_f64, @function
.p2align 4
cumfmax_f64:
    test    rdx, rdx
    jz      .Lreturn
    vmovsd  xmm0, qword ptr [rdi]       # M
    vmovsd  qword ptr [rsi], xmm0
    cmp     rdx, 2
    je      .Ltwo
    shl     rdx, 3
    add     rdi, rdx
    add     rsi, rdx
    mov     ecx, 8
    sub     rcx, rdx                    # element 1
    jz      .Lreturn
    vucomisd xmm0, xmm0
    jnp     .Lnumber
    # ---- Every element so far is NaN: the output is x[0] until the first
    #      element that is a number.
.Llead:
    vmovsd  xmm1, qword ptr [rdi + rcx]
    vucomisd xmm1, xmm1
    jnp     .Lleadend
    vmovsd  qword ptr [rsi + rcx], xmm0
    add     rcx, 8
    jnz     .Llead
.Lreturn:
    ret
.Lleadend:
    vmovapd xmm0, xmm1                  # M is that number; its own output comes below

    # ---- M is a number.
.Lnumber:
    vbroadcastsd ymm4, qword ptr [rip + .Lneginf]
    cmp     rcx, -32
    jg      .Ltail
    vbroadcastsd ymm0, xmm0             # M in every lane
.p2align 4
.Lvector:
    vmovupd ymm1, [rdi + rcx]
    vmaxpd  ymm1, ymm1, ymm4            # NaN to minus infinity
    vpermpd ymm2, ymm1, 0x90            # x0 x0 x1 x2
    vmaxpd  ymm1, ymm2, ymm1            # each lane against the one before it
    vpermpd ymm2, ymm1, 0x40            # a0 a0 a0 a1
    vmaxpd  ymm1, ymm2, ymm1            # and against the one two before
    vpermpd ymm3, ymm1, 0xff            # the vector's own maximum in every lane
    vmaxpd  ymm2, ymm0, ymm1
    vmaxpd  ymm0, ymm0, ymm3            # the next M
    vmovupd [rsi + rcx], ymm2
    add     rcx, 32
    cmp     rcx, -32
    jle     .Lvector
    test    rcx, rcx
    jz      .Ldone

    # ---- Up to three elements one at a time.
.Ltail:
    vmovsd  xmm1, qword ptr [rdi + rcx]
    vmaxsd  xmm1, xmm1, xmm4            # NaN to minus infinity
    vmaxsd  xmm0, xmm0, xmm1
    vmovsd  qword ptr [rsi + rcx], xmm0
    add     rcx, 8
    jnz     .Ltail
.Ldone:
    vzeroupper
    ret

    # ---- Two elements. xmm0 = x[0], already stored.
.Ltwo:
    vmovsd  xmm1, qword ptr [rdi + 8]
    vmaxsd  xmm2, xmm1, xmm0            # x[1] if it is the larger, otherwise x[0]
    vcmpunordsd xmm3, xmm0, xmm0
    vblendvpd xmm2, xmm2, xmm1, xmm3    # x[1] if x[0] is NaN
    vmovsd  qword ptr [rsi + 8], xmm2
    ret
.size cumfmax_f64, .-cumfmax_f64

.section .note.GNU-stack,"",@progbits
