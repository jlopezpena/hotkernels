// void cumsum_f64(const double *x, double *out, size_t n);
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
.globl _cumsum_f64
.p2align 2
_cumsum_f64:
    cmp     x2, #8
    b.lo    Lshort
    ldp     d1, d2, [x0]
    ldp     d3, d4, [x0, #16]
    ldp     d5, d6, [x0, #32]
    ldp     d7, d16, [x0, #48]
    fadd    d2, d1, d2              // x[0] + x[1]; x[0] itself is stored as loaded
    fadd    d3, d2, d3
    fadd    d4, d3, d4
    fadd    d5, d4, d5
    fadd    d6, d5, d6
    fadd    d7, d6, d7
    fadd    d0, d7, d16
    stp     d1, d2, [x1]
    stp     d3, d4, [x1, #16]
    stp     d5, d6, [x1, #32]
    stp     d7, d0, [x1, #48]
    add     x0, x0, #64
    add     x1, x1, #64
    subs    x2, x2, #16             // 8 done, 8 more for the loop; the running sum is in d0
    b.lo    Lrest
Lmain:
    ldp     d1, d2, [x0]
    ldp     d3, d4, [x0, #16]
    ldp     d5, d6, [x0, #32]
    ldp     d7, d16, [x0, #48]
    fadd    d1, d0, d1              // sum + x[i], same operand order as the C
    fadd    d2, d1, d2
    fadd    d3, d2, d3
    fadd    d4, d3, d4
    fadd    d5, d4, d5
    fadd    d6, d5, d6
    fadd    d7, d6, d7
    fadd    d0, d7, d16
    stp     d1, d2, [x1]
    stp     d3, d4, [x1, #16]
    stp     d5, d6, [x1, #32]
    stp     d7, d0, [x1, #48]
    add     x0, x0, #64
    add     x1, x1, #64
    subs    x2, x2, #8
    b.hs    Lmain
Lrest:
    tst     x2, #7                  // 0 to 7 elements left
    b.eq    Ldone
    tbz     x2, #2, Lno4
    ldp     d1, d2, [x0]
    ldp     d3, d4, [x0, #16]
    fadd    d1, d0, d1
    fadd    d2, d1, d2
    fadd    d3, d2, d3
    fadd    d0, d3, d4
    stp     d1, d2, [x1]
    stp     d3, d0, [x1, #16]
    add     x0, x0, #32
    add     x1, x1, #32
Lno4:
    tbz     x2, #1, Lno2
    ldp     d1, d2, [x0]
    fadd    d1, d0, d1
    fadd    d0, d1, d2
    stp     d1, d0, [x1]
    add     x0, x0, #16
    add     x1, x1, #16
Lno2:
    tbz     x2, #0, Ldone
    ldr     d1, [x0]
    fadd    d0, d0, d1
    str     d0, [x1]
Ldone:
    ret
Lshort:                             // fewer than 8 elements
    cbz     x2, Ldone
    ldr     d0, [x0], #8            // running sum = x[0], copied
    str     d0, [x1], #8
    subs    x2, x2, #1
    b.eq    Ldone
Lshort_loop:
    ldr     d1, [x0], #8
    fadd    d0, d0, d1
    str     d0, [x1], #8
    subs    x2, x2, #1
    b.ne    Lshort_loop
    ret
