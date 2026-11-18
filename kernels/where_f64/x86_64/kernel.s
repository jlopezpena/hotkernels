.intel_syntax noprefix
# void where_f64(const bool *c, const double *a, const double *b, double *out, size_t n);
#
# rdi = c, rsi = a, rdx = b, rcx = out, r8 = n. out[i] = c[i] ? a[i] : b[i].
#
# The elements are moved and never computed with: no floating-point
# arithmetic, compare or conversion touches the data of a or b, only loads,
# stores and bitwise and, and-not and or with a lane mask. A NaN keeps its
# payload and its signalling bit, a zero its sign, and 64-bit integers sent
# through as bit patterns arrive unchanged.
#
# Four elements at a time: `vpmovsxbq` widens four bytes of c to four 64-bit
# lanes and a compare with zero makes the lane mask, all ones where c is
# false. The output is (mask and b) or (a and not mask). A byte of c is true
# when it is not zero, whatever its value.
#
# Paths, by what is left:
#   4 or more   one select per 4
#   1..3        one element at a time, with a conditional move

.text
.globl where_f64
.type where_f64, @function
.p2align 4
where_f64:
    xor     eax, eax                    # elements done
    mov     r9, r8
    and     r9, -4                      # elements in whole groups
    jz      .Ltail
    vpxor   xmm0, xmm0, xmm0
.p2align 4
.Lgroup:
    vpmovsxbq ymm1, dword ptr [rdi + rax]
    vpcmpeqq ymm1, ymm1, ymm0           # all ones where c is false
    vpand   ymm2, ymm1, [rdx + rax * 8]
    vpandn  ymm1, ymm1, [rsi + rax * 8]
    vpor    ymm1, ymm1, ymm2
    vmovdqu [rcx + rax * 8], ymm1
    add     rax, 4
    cmp     rax, r9
    jb      .Lgroup
    vzeroupper
.Ltail:
    cmp     rax, r8
    jae     .Ldone
.Lone:
    mov     r9, [rdx + rax * 8]
    cmp     byte ptr [rdi + rax], 0
    cmovne  r9, [rsi + rax * 8]
    mov     [rcx + rax * 8], r9
    inc     rax
    cmp     rax, r8
    jb      .Lone
.Ldone:
    ret
.size where_f64, .-where_f64

.section .note.GNU-stack,"",@progbits
