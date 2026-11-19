.intel_syntax noprefix
# void bincount_u8(const uint8_t *x, int64_t *counts, size_t n);
#
# rdi = x, rsi = counts, rdx = n. counts[b] = how many of the n bytes at x
# equal b, for b = 0..255.
#
# One counter is loaded, incremented and stored per byte, and that is the
# limit whatever surrounds it. Eight bytes are read with one load, and from
# TABLES_MIN_N bytes the counts go into four tables on the stack, byte i into
# table i mod 4, which are added up at the end. Two equal bytes next to each
# other then update different counters; in one table the second update has
# to wait for the first to be stored.
#
# Paths, by n:
#   0..TABLES_MIN_N-1   counts cleared, then eight bytes per load into counts
#   TABLES_MIN_N..      four tables of 256 counters below the stack pointer

# Below this many bytes the tables cost more to clear and add up than they save.
.set TABLES_MIN_N, 32768
# Bytes of stack for the four tables, plus eight so that the stack pointer
# stays a multiple of 16 while it is moved.
.set FRAME, 4 * 256 * 8 + 8

.text
.globl bincount_u8
.type bincount_u8, @function
.p2align 4
bincount_u8:
    cmp     rdx, TABLES_MIN_N
    jae     .Ltables
    # Clear counts, then count into it directly.
    mov     r8, rdi
    mov     rdi, rsi
    xor     eax, eax
    mov     ecx, 2048
    rep stosb
    mov     rdi, r8
    mov     r8, rdx
    shr     r8, 3                       # steps of eight bytes
    jz      .Llastbytes
.Lshorteight:
    mov     rax, [rdi]
    add     rdi, 8
    movzx   ecx, al
    inc     qword ptr [rsi + rcx * 8]
    movzx   ecx, ah
    inc     qword ptr [rsi + rcx * 8]
    shr     rax, 16
    movzx   ecx, al
    inc     qword ptr [rsi + rcx * 8]
    movzx   ecx, ah
    inc     qword ptr [rsi + rcx * 8]
    shr     rax, 16
    movzx   ecx, al
    inc     qword ptr [rsi + rcx * 8]
    movzx   ecx, ah
    inc     qword ptr [rsi + rcx * 8]
    shr     rax, 16
    movzx   ecx, al
    inc     qword ptr [rsi + rcx * 8]
    movzx   ecx, ah
    inc     qword ptr [rsi + rcx * 8]
    dec     r8
    jnz     .Lshorteight
.Llastbytes:
    and     edx, 7
    jz      .Ldone
.Lbyte:
    movzx   eax, byte ptr [rdi]
    inc     qword ptr [rsi + rax * 8]
    inc     rdi
    dec     edx
    jnz     .Lbyte
.Ldone:
    ret

.Ltables:
    sub     rsp, FRAME
    # Clear the four tables.
    vpxor   xmm0, xmm0, xmm0
    xor     eax, eax
.Lcleartables:
    vmovdqu [rsp + rax], ymm0
    vmovdqu [rsp + rax + 32], ymm0
    add     eax, 64
    cmp     eax, 8192
    jb      .Lcleartables
    mov     r8, rdx
    shr     r8, 3                       # steps of eight bytes
.p2align 6
.Leight:
    mov     rax, [rdi]
    add     rdi, 8
    movzx   ecx, al
    inc     qword ptr [rsp + rcx * 8]
    movzx   ecx, ah
    inc     qword ptr [rsp + rcx * 8 + 2048]
    shr     rax, 16
    movzx   ecx, al
    inc     qword ptr [rsp + rcx * 8 + 4096]
    movzx   ecx, ah
    inc     qword ptr [rsp + rcx * 8 + 6144]
    shr     rax, 16
    movzx   ecx, al
    inc     qword ptr [rsp + rcx * 8]
    movzx   ecx, ah
    inc     qword ptr [rsp + rcx * 8 + 2048]
    shr     rax, 16
    movzx   ecx, al
    inc     qword ptr [rsp + rcx * 8 + 4096]
    movzx   ecx, ah
    inc     qword ptr [rsp + rcx * 8 + 6144]
    dec     r8
    jnz     .Leight
    and     edx, 7                      # 0..7 bytes left, into the first table
    jz      .Lsum
.Llast:
    movzx   eax, byte ptr [rdi]
    inc     qword ptr [rsp + rax * 8]
    inc     rdi
    dec     edx
    jnz     .Llast
.Lsum:
    xor     eax, eax
.Lsumstep:
    vmovdqu ymm0, [rsp + rax]
    vpaddq  ymm0, ymm0, [rsp + rax + 2048]
    vpaddq  ymm0, ymm0, [rsp + rax + 4096]
    vpaddq  ymm0, ymm0, [rsp + rax + 6144]
    vmovdqu [rsi + rax], ymm0
    add     eax, 32
    cmp     eax, 2048
    jb      .Lsumstep
    vzeroupper
    add     rsp, FRAME
    ret
.size bincount_u8, .-bincount_u8

.section .note.GNU-stack,"",@progbits
