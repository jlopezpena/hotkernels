# Calls a kernel and verifies it obeyed the x86-64 System V calling convention.
#
# void hk_abi_call(void *fn, const uint64_t iargs[8], const uint64_t fargs[8], uint64_t out[3]);
#
#   iargs[0..5] -> rdi, rsi, rdx, rcx, r8, r9
#   iargs[6..7] -> the stack, where the convention passes a seventh and eighth integer
#   fargs[0..7] -> xmm0..xmm7 (raw bits)
#   out[0] = rax after the call, out[1] = xmm0 bits, out[2] = violation mask
#
# Violation mask: bits 0-5 = rbx, rbp, r12, r13, r14, r15 changed,
#                 bit 6 = stack pointer changed,
#                 bit 7 = direction flag left set,
#                 bit 8 = MXCSR control bits changed (rounding mode, exception masks),
#                 bit 9 = upper halves of the vector registers left in use (no vzeroupper).
#
# Every register a kernel must preserve is filled with a known value before the
# call and compared afterwards. The worker names the bits in judge/worker.py.

    .intel_syntax noprefix

    .bss
    .p2align 3
# The stack pointer at the call, kept where the kernel cannot reach it by accident.
hk_saved_sp:
    .skip   8

    .text
# Set by harness.c when the processor can report whether the upper halves of
# the vector registers are in use (XGETBV with ECX = 1).
    .hidden hk_can_see_upper_halves
    .globl  hk_abi_call
    .type   hk_abi_call, @function
    .p2align 4
hk_abi_call:
    push    rbp
    push    rbx
    push    r12
    push    r13
    push    r14
    push    r15
    # Locals: [rsp] = out, [rsp + 8] = MXCSR before the call, [rsp + 12] = MXCSR after.
    # Six pushes and these 24 bytes leave the stack 16-byte aligned.
    sub     rsp, 24
    mov     qword ptr [rsp], rcx
    stmxcsr dword ptr [rsp + 8]

    mov     r10, rdi                        # kernel address
    mov     r11, rsi                        # iargs
    movq    xmm0, qword ptr [rdx]
    movq    xmm1, qword ptr [rdx + 8]
    movq    xmm2, qword ptr [rdx + 16]
    movq    xmm3, qword ptr [rdx + 24]
    movq    xmm4, qword ptr [rdx + 32]
    movq    xmm5, qword ptr [rdx + 40]
    movq    xmm6, qword ptr [rdx + 48]
    movq    xmm7, qword ptr [rdx + 56]

    # The seventh and eighth integer arguments, pushed in reverse order. A
    # kernel with six or fewer never looks at them. Two pushes keep the alignment.
    push    qword ptr [r11 + 56]
    push    qword ptr [r11 + 48]
    mov     qword ptr [rip + hk_saved_sp], rsp

    # Fill callee-saved registers with sentinels.
    movabs  rax, 0x5EED5EED5EED0000
    lea     rbx, [rax + 1]
    lea     rbp, [rax + 2]
    lea     r12, [rax + 3]
    lea     r13, [rax + 4]
    lea     r14, [rax + 5]
    lea     r15, [rax + 6]

    mov     rdi, qword ptr [r11]
    mov     rsi, qword ptr [r11 + 8]
    mov     rdx, qword ptr [r11 + 16]
    mov     rcx, qword ptr [r11 + 24]
    mov     r8,  qword ptr [r11 + 32]
    mov     r9,  qword ptr [r11 + 40]
    vzeroupper                              # the caller's side of the rule checked below
    cld
    call    r10

    # rax and xmm0 hold the result; only rcx, rdx and r8-r11 are used below.
    xor     r8d, r8d                        # violation mask
    mov     r9, qword ptr [rip + hk_saved_sp]
    cmp     rsp, r9
    je      .Lsp_kept
    or      r8d, 0x40                       # bit 6: stack pointer
.Lsp_kept:
    mov     rsp, r9

    pushfq
    pop     r10
    test    r10d, 0x400                     # the direction flag
    jz      .Ldf_clear
    or      r8d, 0x80                       # bit 7: direction flag
    cld
.Ldf_clear:

    movabs  r11, 0x5EED5EED5EED0000
    lea     r10, [r11 + 1]
    cmp     rbx, r10
    je      .Lrbx_kept
    or      r8d, 0x1
.Lrbx_kept:
    lea     r10, [r11 + 2]
    cmp     rbp, r10
    je      .Lrbp_kept
    or      r8d, 0x2
.Lrbp_kept:
    lea     r10, [r11 + 3]
    cmp     r12, r10
    je      .Lr12_kept
    or      r8d, 0x4
.Lr12_kept:
    lea     r10, [r11 + 4]
    cmp     r13, r10
    je      .Lr13_kept
    or      r8d, 0x8
.Lr13_kept:
    lea     r10, [r11 + 5]
    cmp     r14, r10
    je      .Lr14_kept
    or      r8d, 0x10
.Lr14_kept:
    lea     r10, [r11 + 6]
    cmp     r15, r10
    je      .Lr15_kept
    or      r8d, 0x20
.Lr15_kept:

    # The two stack arguments are still below the locals, so the locals sit 16 higher.
    stmxcsr dword ptr [rsp + 28]
    mov     r10d, dword ptr [rsp + 24]
    xor     r10d, dword ptr [rsp + 28]
    test    r10d, 0xFFC0                    # everything but the six exception flags
    jz      .Lmxcsr_kept
    or      r8d, 0x100                      # bit 8: MXCSR control bits
    ldmxcsr dword ptr [rsp + 24]
.Lmxcsr_kept:

    cmp     dword ptr [rip + hk_can_see_upper_halves], 0
    je      .Lupper_clean
    mov     r9, rax                         # xgetbv overwrites rax and rdx
    mov     ecx, 1
    xgetbv
    mov     r10d, eax
    mov     rax, r9
    test    r10d, 0x4                       # the AVX state component is in use
    jz      .Lupper_clean
    or      r8d, 0x200                      # bit 9: upper halves left in use
    vzeroupper
.Lupper_clean:

    mov     rcx, qword ptr [rsp + 16]
    mov     qword ptr [rcx], rax
    movq    qword ptr [rcx + 8], xmm0
    mov     qword ptr [rcx + 16], r8

    add     rsp, 40
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbx
    pop     rbp
    ret
    .size   hk_abi_call, . - hk_abi_call

    .section .note.GNU-stack,"",@progbits
