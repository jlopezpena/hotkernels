// Calls a kernel and verifies it obeyed the ARM64 calling convention.
//
// void hk_abi_call(void *fn, const uint64_t iargs[8], const uint64_t fargs[8], uint64_t out[3]);
//
//   iargs -> x0..x7, fargs -> d0..d7 (raw bits)
//   out[0] = x0 after the call, out[1] = d0 bits, out[2] = violation mask
//
// Violation mask: bits 0-9 = x19..x28 changed, bits 10-17 = d8..d15 changed,
//                 bit 18 = stack pointer changed, bit 19 = frame pointer (x29) changed.
//
// Every register a kernel must preserve is filled with a known value before the
// call and compared afterwards. This is the only architecture-specific file in the judge.

.section __DATA,__data
.p2align 3
Lsaved_sp:
    .quad 0

.text
.globl _hk_abi_call
.p2align 2
_hk_abi_call:
    sub     sp, sp, #176
    stp     x29, x30, [sp]
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]
    stp     x23, x24, [sp, #48]
    stp     x25, x26, [sp, #64]
    stp     x27, x28, [sp, #80]
    stp     d8,  d9,  [sp, #96]
    stp     d10, d11, [sp, #112]
    stp     d12, d13, [sp, #128]
    stp     d14, d15, [sp, #144]
    str     x3, [sp, #160]
    mov     x29, sp

    // Remember the stack pointer somewhere the kernel cannot reach by accident.
    adrp    x9, Lsaved_sp@PAGE
    mov     x10, sp
    str     x10, [x9, Lsaved_sp@PAGEOFF]

    mov     x16, x0                 // kernel address
    mov     x9, x1                  // iargs
    mov     x10, x2                 // fargs
    ldp     d0, d1, [x10]
    ldp     d2, d3, [x10, #16]
    ldp     d4, d5, [x10, #32]
    ldp     d6, d7, [x10, #48]

    // Fill callee-saved registers with sentinels.
    ldr     x11, Lseed
    add     x19, x11, #19
    add     x20, x11, #20
    add     x21, x11, #21
    add     x22, x11, #22
    add     x23, x11, #23
    add     x24, x11, #24
    add     x25, x11, #25
    add     x26, x11, #26
    add     x27, x11, #27
    add     x28, x11, #28
    add     x12, x11, #8
    fmov    d8, x12
    add     x12, x11, #9
    fmov    d9, x12
    add     x12, x11, #10
    fmov    d10, x12
    add     x12, x11, #11
    fmov    d11, x12
    add     x12, x11, #12
    fmov    d12, x12
    add     x12, x11, #13
    fmov    d13, x12
    add     x12, x11, #14
    fmov    d14, x12
    add     x12, x11, #15
    fmov    d15, x12

    ldp     x0, x1, [x9]
    ldp     x2, x3, [x9, #16]
    ldp     x4, x5, [x9, #32]
    ldp     x6, x7, [x9, #48]
    blr     x16

    // x0 and d0 hold the result; only x9-x15 are used below.
    mov     x14, #0
    adrp    x9, Lsaved_sp@PAGE
    ldr     x9, [x9, Lsaved_sp@PAGEOFF]
    mov     x10, sp
    cmp     x10, x9
    b.eq    1f
    orr     x14, x14, #0x40000      // bit 18: stack pointer
1:
    mov     sp, x9
    cmp     x29, x9
    b.eq    1f
    orr     x14, x14, #0x80000      // bit 19: frame pointer
1:
    mov     x29, x9
    ldr     x11, Lseed
    add     x12, x11, #19
    cmp     x19, x12
    b.eq    1f
    orr     x14, x14, #0x1
1:
    add     x12, x11, #20
    cmp     x20, x12
    b.eq    1f
    orr     x14, x14, #0x2
1:
    add     x12, x11, #21
    cmp     x21, x12
    b.eq    1f
    orr     x14, x14, #0x4
1:
    add     x12, x11, #22
    cmp     x22, x12
    b.eq    1f
    orr     x14, x14, #0x8
1:
    add     x12, x11, #23
    cmp     x23, x12
    b.eq    1f
    orr     x14, x14, #0x10
1:
    add     x12, x11, #24
    cmp     x24, x12
    b.eq    1f
    orr     x14, x14, #0x20
1:
    add     x12, x11, #25
    cmp     x25, x12
    b.eq    1f
    orr     x14, x14, #0x40
1:
    add     x12, x11, #26
    cmp     x26, x12
    b.eq    1f
    orr     x14, x14, #0x80
1:
    add     x12, x11, #27
    cmp     x27, x12
    b.eq    1f
    orr     x14, x14, #0x100
1:
    add     x12, x11, #28
    cmp     x28, x12
    b.eq    1f
    orr     x14, x14, #0x200
1:
    add     x12, x11, #8
    fmov    x13, d8
    cmp     x13, x12
    b.eq    1f
    orr     x14, x14, #0x400
1:
    add     x12, x11, #9
    fmov    x13, d9
    cmp     x13, x12
    b.eq    1f
    orr     x14, x14, #0x800
1:
    add     x12, x11, #10
    fmov    x13, d10
    cmp     x13, x12
    b.eq    1f
    orr     x14, x14, #0x1000
1:
    add     x12, x11, #11
    fmov    x13, d11
    cmp     x13, x12
    b.eq    1f
    orr     x14, x14, #0x2000
1:
    add     x12, x11, #12
    fmov    x13, d12
    cmp     x13, x12
    b.eq    1f
    orr     x14, x14, #0x4000
1:
    add     x12, x11, #13
    fmov    x13, d13
    cmp     x13, x12
    b.eq    1f
    orr     x14, x14, #0x8000
1:
    add     x12, x11, #14
    fmov    x13, d14
    cmp     x13, x12
    b.eq    1f
    orr     x14, x14, #0x10000
1:
    add     x12, x11, #15
    fmov    x13, d15
    cmp     x13, x12
    b.eq    1f
    orr     x14, x14, #0x20000
1:
    ldr     x9, [sp, #160]
    str     x0, [x9]
    str     d0, [x9, #8]
    str     x14, [x9, #16]

    ldp     x29, x30, [sp]
    ldp     x19, x20, [sp, #16]
    ldp     x21, x22, [sp, #32]
    ldp     x23, x24, [sp, #48]
    ldp     x25, x26, [sp, #64]
    ldp     x27, x28, [sp, #80]
    ldp     d8,  d9,  [sp, #96]
    ldp     d10, d11, [sp, #112]
    ldp     d12, d13, [sp, #128]
    ldp     d14, d15, [sp, #144]
    add     sp, sp, #176
    ret

.p2align 3
Lseed:
    .quad 0x5EED5EED5EED0000
