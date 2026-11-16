// size_t compress_scratch_f32(const float *x, const bool *mask, float *out, size_t n);
// Simple scalar version used to test the judge. Deliberately not tuned.
//
// Stores every element, then advances only when the mask is set. This leaves a
// stray element past the returned count, which the scratch output allows. It is
// the same code that compress_f32 rejects as broken/write_past_used.s.
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
    str     w7, [x2, x5, lsl #2]    // unconditional store, inside the capacity
    ldrb    w6, [x1, x4]
    add     x5, x5, x6
    add     x4, x4, #1
    b       Lloop
Ldone:
    mov     x0, x5
    ret
