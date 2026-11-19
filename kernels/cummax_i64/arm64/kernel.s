// void cummax_i64(const int64_t *x, int64_t *out, size_t n);
// out[i] = the largest of x[0..i], compared as signed numbers.
// x0 = x, x1 = out, x2 = n. out does not overlap x.
//
// Groups of four elements in the integer registers. The maxima of x0, x0..x1,
// x0..x2 and x0..x3 do not depend on M, the maximum before the group, so the
// chain from group to group is one cmp and one csel.

// Four elements at [x0 + \o]; x3 = M before and after.
.macro GROUP o
    ldp     x4, x5, [x0, #\o]
    ldp     x6, x7, [x0, #\o + 16]
    cmp     x5, x4
    csel    x5, x5, x4, gt          // max of x0..x1
    cmp     x6, x5
    csel    x6, x6, x5, gt          // max of x0..x2
    cmp     x7, x6
    csel    x7, x7, x6, gt          // max of x0..x3
    cmp     x4, x3
    csel    x4, x4, x3, gt
    cmp     x5, x3
    csel    x5, x5, x3, gt
    cmp     x6, x3
    csel    x6, x6, x3, gt
    cmp     x7, x3
    csel    x3, x7, x3, gt
    stp     x4, x5, [x1, #\o]
    stp     x6, x3, [x1, #\o + 16]
.endm

.text
.globl _cummax_i64
.p2align 2
_cummax_i64:
    cbz     x2, Ldone
    ldr     x3, [x0]                // M
    subs    x2, x2, #8
    b.lo    Lrest
Lloop:
    GROUP   0
    GROUP   32
    add     x0, x0, #64
    add     x1, x1, #64
    subs    x2, x2, #8
    b.hs    Lloop
Lrest:
    tbz     x2, #2, Lsingles        // bit 2 of the count: 4 or more left
    GROUP   0
    add     x0, x0, #32
    add     x1, x1, #32
Lsingles:
    ands    x2, x2, #3
    b.eq    Ldone
Lsingle:
    ldr     x4, [x0], #8
    cmp     x4, x3
    csel    x3, x4, x3, gt
    str     x3, [x1], #8
    subs    x2, x2, #1
    b.ne    Lsingle
Ldone:
    ret
