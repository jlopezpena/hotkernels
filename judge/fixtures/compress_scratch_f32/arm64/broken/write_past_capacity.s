// BROKEN ON PURPOSE: correct result, then one store just past the capacity.
// A scratch buffer allows writes anywhere inside the capacity, never beyond it.
// size_t compress_scratch_f32(const float *x, const bool *mask, float *out, size_t n);
.text
.globl _compress_scratch_f32
.p2align 2
_compress_scratch_f32:
    mov     x4, #0                  // i
    mov     x5, #0                  // number written
Lloop:
    cmp     x4, x3
    b.hs    Ldone
    ldr     w7, [x0, x4, lsl #2]
    str     w7, [x2, x5, lsl #2]
    ldrb    w6, [x1, x4]
    add     x5, x5, x6
    add     x4, x4, #1
    b       Lloop
Ldone:
    str     wzr, [x2, x3, lsl #2]   // BUG: writes out[n], one element past the capacity
    mov     x0, x5
    ret
