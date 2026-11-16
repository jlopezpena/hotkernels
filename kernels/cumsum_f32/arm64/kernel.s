// void cumsum_f32(const float *x, float *out, size_t n);
// out[i] = x[0] + ... + x[i], added strictly left to right. The first element
// is copied, not added to zero, so a leading -0.0 or a signalling NaN is kept.
//
// The additions are the same single fadd chain as the naive loop, in the same
// order, so the result is bit-identical and a long array still costs one fadd
// latency (3 cycles) per element. What changes is that elements are loaded and
// stored two at a time (ldp/stp), which halves the number of store operations
// waiting on the chain. That only matters for short arrays, where the core runs
// several calls at once and how many it can hold depends on how many
// operations each one has waiting.
.text
.globl _cumsum_f32
.p2align 2
_cumsum_f32:
    cmp     x2, #8
    b.lo    Lshort
    ldp     s1, s2, [x0]
    ldp     s3, s4, [x0, #8]
    ldp     s5, s6, [x0, #16]
    ldp     s7, s16, [x0, #24]
    fadd    s2, s1, s2              // x[0] + x[1]; x[0] itself is stored as loaded
    fadd    s3, s2, s3
    fadd    s4, s3, s4
    fadd    s5, s4, s5
    fadd    s6, s5, s6
    fadd    s7, s6, s7
    fadd    s0, s7, s16
    stp     s1, s2, [x1]
    stp     s3, s4, [x1, #8]
    stp     s5, s6, [x1, #16]
    stp     s7, s0, [x1, #24]
    add     x0, x0, #32
    add     x1, x1, #32
    subs    x2, x2, #16             // 8 done, 8 more for the loop; the running sum is in s0
    b.lo    Lrest
Lmain:
    ldp     s1, s2, [x0]
    ldp     s3, s4, [x0, #8]
    ldp     s5, s6, [x0, #16]
    ldp     s7, s16, [x0, #24]
    fadd    s1, s0, s1              // sum + x[i], same operand order as the C
    fadd    s2, s1, s2
    fadd    s3, s2, s3
    fadd    s4, s3, s4
    fadd    s5, s4, s5
    fadd    s6, s5, s6
    fadd    s7, s6, s7
    fadd    s0, s7, s16
    stp     s1, s2, [x1]
    stp     s3, s4, [x1, #8]
    stp     s5, s6, [x1, #16]
    stp     s7, s0, [x1, #24]
    add     x0, x0, #32
    add     x1, x1, #32
    subs    x2, x2, #8
    b.hs    Lmain
Lrest:
    tst     x2, #7                  // 0 to 7 elements left
    b.eq    Ldone
    tbz     x2, #2, Lno4
    ldp     s1, s2, [x0]
    ldp     s3, s4, [x0, #8]
    fadd    s1, s0, s1
    fadd    s2, s1, s2
    fadd    s3, s2, s3
    fadd    s0, s3, s4
    stp     s1, s2, [x1]
    stp     s3, s0, [x1, #8]
    add     x0, x0, #16
    add     x1, x1, #16
Lno4:
    tbz     x2, #1, Lno2
    ldp     s1, s2, [x0]
    fadd    s1, s0, s1
    fadd    s0, s1, s2
    stp     s1, s0, [x1]
    add     x0, x0, #8
    add     x1, x1, #8
Lno2:
    tbz     x2, #0, Ldone
    ldr     s1, [x0]
    fadd    s0, s0, s1
    str     s0, [x1]
Ldone:
    ret
Lshort:                             // fewer than 8 elements
    cbz     x2, Ldone
    ldr     s0, [x0], #4            // running sum = x[0], copied
    str     s0, [x1], #4
    subs    x2, x2, #1
    b.eq    Ldone
Lshort_loop:
    ldr     s1, [x0], #4
    fadd    s0, s0, s1
    str     s0, [x1], #4
    subs    x2, x2, #1
    b.ne    Lshort_loop
    ret
