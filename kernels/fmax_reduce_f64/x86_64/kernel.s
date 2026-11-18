.intel_syntax noprefix
# double fmax_reduce_f64(const double *x, size_t n);
#
# rdi = x, rsi = n (at least 1). Returns the largest element that is not a
# NaN, and a NaN only if every element is one, as np.fmax.reduce gives it bit
# for bit. No element is a signaling NaN.
#
# NumPy's loop, which decides the sign of a zero maximum and the NaN of an
# array of NaNs: four running maxima, one per lane of a vector, all starting
# at x[0]; the elements from x[1] on, four at a time; the lanes combined, 0
# with 2 and 1 with 3 and then the two results; the leftover elements one at
# a time. In a lane and between lanes the later of two equal values is kept,
# among the leftover elements the earlier. This follows it step for step.
#
# `vmaxpd a, b` is "a if a > b, otherwise b": b when the two are equal and b
# when either is a NaN. With the running value as a and the element as b it
# keeps the later of two equal values. It does not ignore a NaN, so every
# element is first put through a `vmaxpd` against minus infinity, which turns
# a NaN into minus infinity and changes nothing else. The result is then
# NumPy's whenever one element is a number. Minus infinity as a result may
# stand for an array of NaNs, so that result is looked at again (.Lneginf):
# NumPy returns x[0] for an array of NaNs with no leftover elements and the
# last element otherwise.

.section .rodata
.p2align 3
.Lminusinf:
    .quad 0xfff0000000000000            # minus infinity

.text
.globl fmax_reduce_f64
.type fmax_reduce_f64, @function
.p2align 4
fmax_reduce_f64:
    vmovsd  xmm0, qword ptr [rdi]
    cmp     rsi, 1
    je      .Lreturn                    # one element is its own maximum, NaN or not
    mov     r8, rdi                     # x, for the second look
    vbroadcastsd ymm4, qword ptr [rip + .Lminusinf]
    vbroadcastsd ymm0, xmm0
    vmaxpd  ymm0, ymm0, ymm4            # x[0] in every lane, NaN to minus infinity
    lea     rdx, [rdi + rsi * 8]        # the end of the array
    add     rdi, 8
    lea     rax, [rdi + 32]
    cmp     rax, rdx
    ja      .Lreduce
    # ---- Four elements per step, each into its own lane.
.Lvector:
    vmovupd ymm1, [rdi]
    vmaxpd  ymm1, ymm1, ymm4            # NaN to minus infinity
    vmaxpd  ymm0, ymm0, ymm1            # the later of two equal values
    mov     rdi, rax
    add     rax, 32
    cmp     rax, rdx
    jbe     .Lvector
    # ---- The lanes combined: 0 with 2, 1 with 3, then the two results, the
    #      second of two equal values each time.
.Lreduce:
    vextractf128 xmm1, ymm0, 1
    vmaxpd  xmm0, xmm0, xmm1
    vunpckhpd xmm1, xmm0, xmm0
    vmaxsd  xmm0, xmm0, xmm1
    # ---- Up to three leftover elements. `vmaxsd v, r` is r for a NaN
    #      element and for an equal one: the earlier of two equal values.
    cmp     rdi, rdx
    jae     .Lcheck
.Ltail:
    vmovsd  xmm1, qword ptr [rdi]
    vmaxsd  xmm0, xmm1, xmm0
    add     rdi, 8
    cmp     rdi, rdx
    jb      .Ltail
.Lcheck:
    vucomisd xmm0, xmm4
    je      .Lneginf
    vzeroupper
.Lreturn:
    ret

    # ---- The result is minus infinity: right if one element is a number.
    #      If every element is a NaN, NumPy's result is x[0] when there are
    #      no leftover elements ((n - 1) mod 4 is 0) and the last element
    #      when there are.
.Lneginf:
    vzeroupper
    mov     rdi, r8
.Lscan:
    vmovsd  xmm1, qword ptr [rdi]
    vucomisd xmm1, xmm1
    jnp     .Lreturn                    # a number: minus infinity stands
    add     rdi, 8
    cmp     rdi, rdx
    jb      .Lscan
    vmovsd  xmm0, qword ptr [r8]        # x[0]
    lea     rax, [rsi - 1]
    test    al, 3
    jz      .Lreturn                    # no leftover elements
    vmovsd  xmm0, qword ptr [rdx - 8]   # the last element
    ret
.size fmax_reduce_f64, .-fmax_reduce_f64

.section .note.GNU-stack,"",@progbits
