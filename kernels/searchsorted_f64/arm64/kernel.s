// void searchsorted_f64(const double *a, const double *v, int64_t *out, size_t n);
//
// For each needle v[j], out[j] = first index i with !less(a[i], v[j]), where
// less(x, y) = x < y || (y is NaN && x is not NaN).
//
// Simplest correct version: the naive branchy binary search, one needle at a time.
.text
.globl _searchsorted_f64
.p2align 2
_searchsorted_f64:
    cbz     x3, Ldone
    mov     x4, #0                  // j
Lneedle:
    ldr     d0, [x1, x4, lsl #3]    // key
    mov     x5, #0                  // lo
    mov     x6, x3                  // hi
    fcmp    d0, d0
    b.vs    Lnan_search
Lsearch:
    cmp     x5, x6
    b.hs    Lstore
    sub     x7, x6, x5
    add     x7, x5, x7, lsr #1      // mid
    ldr     d1, [x0, x7, lsl #3]
    fcmp    d1, d0
    b.mi    Lright                  // a[mid] < key (false when unordered)
    mov     x6, x7
    b       Lsearch
Lright:
    add     x5, x7, #1
    b       Lsearch
Lnan_search:                        // NaN key: less(x, key) is "x is not NaN"
    cmp     x5, x6
    b.hs    Lstore
    sub     x7, x6, x5
    add     x7, x5, x7, lsr #1
    ldr     d1, [x0, x7, lsl #3]
    fcmp    d1, d1
    b.vc    Lnan_right
    mov     x6, x7
    b       Lnan_search
Lnan_right:
    add     x5, x7, #1
    b       Lnan_search
Lstore:
    str     x5, [x2, x4, lsl #3]
    add     x4, x4, #1
    cmp     x4, x3
    b.lo    Lneedle
Ldone:
    ret
