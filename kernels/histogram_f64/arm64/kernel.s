// void histogram_f64(const double *x, int64_t *counts, size_t n, size_t nbins, double lo, double hi);
// counts has exactly nbins slots; it arrives uninitialised.
//
// x0 = x, x1 = counts, x2 = n, x3 = nbins, d0 = lo, d1 = hi
//
// Follows naive.c one operation at a time. Every floating-point operation is
// rounded on its own: the edges use a separate multiply and add, never fmadd.
.text
.globl _histogram_f64
.p2align 2
_histogram_f64:
    cbz     x3, Ldone
    // Zero the counters.
    mov     x4, x1
    mov     x5, x3
Lzero:
    str     xzr, [x4], #8
    subs    x5, x5, #1
    b.ne    Lzero

    cbz     x2, Ldone
    fsub    d2, d1, d0              // width = hi - lo
    ucvtf   d3, x3                  // (double)nbins
    fdiv    d4, d2, d3              // step = width / nbins
    sub     x8, x3, #1              // nbins - 1
Lloop:
    ldr     d5, [x0], #8            // v
    // Skip unless lo <= v <= hi. NaN fails the first compare.
    fcmp    d5, d0
    fccmp   d5, d1, #2, ge
    b.hi    Lnext
    // b = (size_t)(((v - lo) / width) * nbins)
    fsub    d6, d5, d0
    fdiv    d6, d6, d2
    fmul    d6, d6, d3
    fcvtzu  x9, d6
    // if (b == nbins) b--
    cmp     x9, x3
    cset    w10, eq
    sub     x9, x9, x10
    // if (v < edge(b)) b--          (b < nbins here, so the edge is computed)
    ucvtf   d6, x9
    fmul    d6, d6, d4
    fadd    d6, d6, d0
    fcmp    d5, d6
    cset    w10, mi
    sub     x9, x9, x10
    // if (v >= edge(b + 1) && b != nbins - 1) b++
    add     x10, x9, #1
    ucvtf   d6, x10
    fmul    d6, d6, d4
    fadd    d6, d6, d0
    cmp     x10, x3
    fcsel   d6, d1, d6, eq          // edge(nbins) is exactly hi
    fcmp    d5, d6
    ccmp    x9, x8, #4, ge          // v >= edge: compare b with nbins - 1; else "equal"
    cinc    x9, x9, ne
    ldr     x10, [x1, x9, lsl #3]
    add     x10, x10, #1
    str     x10, [x1, x9, lsl #3]
Lnext:
    subs    x2, x2, #1
    b.ne    Lloop
Ldone:
    ret
