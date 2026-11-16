// void interp_f64(const double *x, const double *xp, const double *fp, double *out, size_t n);
//
// np.interp(x, xp, fp) with n queries and n sample points. Simplest correct
// version: naive.c translated line by line, one query at a time.
//
// x0 x, x1 xp, x2 fp, x3 out, x4 n, x5 n - 1, x6 i
// d16 left = fp[0], d17 right = fp[n-1], d18 xp[0], d19 xp[n-1]
.text
.globl _interp_f64
.p2align 2
_interp_f64:
    ldr     d16, [x2]
    sub     x5, x4, #1
    ldr     d17, [x2, x5, lsl #3]
    cmp     x4, #1
    b.eq    Lone
    ldr     d18, [x1]
    ldr     d19, [x1, x5, lsl #3]
    mov     x6, #0
Lquery:
    ldr     d0, [x0, x6, lsl #3]    // v
    fmov    d7, d0
    fcmp    d0, d0
    b.vs    Lstore                  // NaN query: the result is that NaN
    fmov    d7, d17
    fcmp    d0, d19
    b.gt    Lstore                  // above the range: right
    fmov    d7, d16
    fcmp    d0, d18
    b.mi    Lstore                  // below the range: left
    mov     x7, #0                  // lo
    mov     x8, x4                  // hi
Lsearch:                            // count the sample points <= v
    cmp     x7, x8
    b.hs    Lfound
    sub     x9, x8, x7
    add     x9, x7, x9, lsr #1      // mid
    ldr     d1, [x1, x9, lsl #3]
    fcmp    d0, d1
    b.ge    Lright                  // v >= xp[mid]
    mov     x8, x9
    b       Lsearch
Lright:
    add     x7, x9, #1
    b       Lsearch
Lfound:
    sub     x7, x7, #1              // j
    ldr     d1, [x1, x7, lsl #3]    // xp[j]
    ldr     d2, [x2, x7, lsl #3]    // fp[j]
    fmov    d7, d2
    cmp     x7, x5
    b.eq    Lstore                  // last sample point
    fcmp    d1, d0
    b.eq    Lstore                  // exact hit
    add     x9, x7, #1
    ldr     d3, [x1, x9, lsl #3]    // xp[j+1]
    ldr     d4, [x2, x9, lsl #3]    // fp[j+1]
    fsub    d5, d4, d2
    fsub    d6, d3, d1
    fdiv    d5, d5, d6              // slope
    fsub    d6, d0, d1
    fmadd   d7, d5, d6, d2          // slope * (v - xp[j]) + fp[j]
    fcmp    d7, d7
    b.vc    Lstore
    fsub    d6, d0, d3
    fmadd   d7, d5, d6, d4          // slope * (v - xp[j+1]) + fp[j+1]
    fcmp    d7, d7
    b.vc    Lstore
    fcmp    d2, d4
    b.ne    Lstore
    fmov    d7, d2
Lstore:
    str     d7, [x3, x6, lsl #3]
    add     x6, x6, #1
    cmp     x6, x4
    b.lo    Lquery
    ret
Lone:                               // one sample point: left = right = fp[0]
    str     d16, [x3]
    ret
