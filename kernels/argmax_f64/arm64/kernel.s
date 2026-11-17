// int64_t argmax_f64(const double *x, size_t n);   n >= 1
//
// Index of the first maximum; the first NaN wins over everything.
//
// Method. `fmax` returns NaN if either operand is NaN, so the maximum of a
// group of elements is NaN exactly when the group contains one.
//
//   n <= 64   Straight-line code: take the maximum m of everything, then find
//             the first element equal to m from compare masks narrowed into a
//             general register (`rbit` + `clz`). The elements are covered by a
//             front half and a back half that may overlap; the back half ends
//             exactly at the end of the array. Above 16 elements the search
//             takes two steps to save vector operations: first the "quads"
//             (lane maxima of 8 consecutive elements) the maximum tree passes
//             through, then the 8 elements of the first quad that holds m.
//   n  > 64   Blocks of 32 elements. Each block is reduced to its maximum (15
//             `fmax` and one `fmaxp`). A scalar compare with the maximum so
//             far sets the flags once: unordered means the block holds a NaN,
//             strictly greater moves the "best block" pointer with `csel`. No
//             branch depends on the data unless there is a NaN. At the end
//             only the best block is rescanned. Keeping the best block in a
//             general register matters: its address is ready before the loop
//             ends, so the rescan does not wait for a vector-to-integer move.
//
// A NaN makes m NaN. A scalar loop then finds the first NaN, in the whole
// input when n <= 64 and inside the offending block otherwise.

// v\d = lane-wise maximum of v\a and v\b (NaN if either is NaN).
.macro MX d, a, b
    fmax    v\d\().2d, v\a\().2d, v\b\().2d
.endm

// v\d = the maximum of the two lanes of v\a, in both lanes.
.macro BROADCAST_MAX d, a
    fmaxp   v\d\().2d, v\a\().2d, v\a\().2d
.endm

// v\r = all-ones in each lane of v\r that equals the maximum broadcast in v\m.
.macro EQ r, m
    fcmeq   v\r\().2d, v\r\().2d, v\m\().2d
.endm

// Narrow two vectors of 64-bit lane masks (\a, \b) into one of 32-bit masks.
.macro NARROW32 a, b
    uzp1    v\a\().4s, v\a\().4s, v\b\().4s
.endm

// Narrow two vectors of 32-bit lane masks (\a, \b) into one of 16-bit masks.
.macro NARROW16 a, b
    uzp1    v\a\().8h, v\a\().8h, v\b\().8h
.endm

// Narrow two vectors of 16-bit lane masks (\a, \b) into one of byte masks.
.macro NARROW8 a, b
    uzp1    v\a\().16b, v\a\().16b, v\b\().16b
.endm

// Load one quad (8 elements at \base + \off) and leave its lane maxima in v\d.
// Clobbers v0..v3.
.macro QUAD d, base, off
    ldp     q0, q1, [\base, #\off]
    ldp     q2, q3, [\base, #\off + 32]
    MX 0, 0, 1
    MX 2, 2, 3
    MX \d, 0, 2
.endm

// x12 points at a quad that holds the maximum, x11 is the index of its first
// element, v1 holds the maximum in both lanes. Returns the index of the first
// element of the quad that equals the maximum.
.macro FINISH_IN_QUAD
    ldp     q16, q17, [x12]
    ldp     q18, q19, [x12, #32]
    EQ 16, 1
    EQ 17, 1
    EQ 18, 1
    EQ 19, 1
    NARROW32 16, 17
    NARROW32 18, 19
    NARROW16 16, 18
    xtn     v16.8b, v16.8h              // eight mask bits per element
    fmov    x10, d16
    rbit    x10, x10
    clz     x10, x10
    add     x0, x11, x10, lsr #3
    ret
.endm

.text
.globl _argmax_f64
.p2align 4
_argmax_f64:
    add     x9, x0, x1, lsl #3          // end of the array
    cmp     x1, #16
    b.hi    Lover16
    cmp     x1, #8
    b.hi    Lupto16
    cmp     x1, #4
    b.hs    Lupto8

// ---- n < 4: one element at a time -------------------------------------------
Lscalar:
    ldr     d0, [x0]                    // best value so far
    mov     x2, #0                      // its index
    fcmp    d0, d0
    b.vs    Lscalar_done                // x[0] is NaN
    mov     x3, #1
    b       Lscalar_test
Lscalar_loop:
    ldr     d1, [x0, x3, lsl #3]
    fcmp    d1, d0
    b.hi    Lscalar_new                 // greater than the best, or NaN
Lscalar_next:
    add     x3, x3, #1
Lscalar_test:
    cmp     x3, x1
    b.lo    Lscalar_loop
Lscalar_done:
    mov     x0, x2
    ret
Lscalar_new:
    mov     x2, x3
    fcmp    d1, d1
    fmov    d0, d1
    b.vc    Lscalar_next                // a real new maximum: carry on
    mov     x0, x2                      // NaN: it wins
    ret

Lover16:
    cmp     x1, #64
    b.hi    Lbig
    cmp     x1, #32
    b.hi    Lupto64
    b       Lupto32

// ---- 4 <= n <= 8: vectors at 0, 2, n-4, n-2 --------------------------------
Lupto8:
    ldp     q16, q17, [x0]
    ldp     q18, q19, [x9, #-32]
    MX 0, 16, 17
    MX 1, 18, 19
    MX 0, 0, 1
    BROADCAST_MAX 0, 0
    EQ 16, 0
    EQ 17, 0
    EQ 18, 0
    EQ 19, 0
    NARROW32 16, 17
    NARROW32 18, 19
    NARROW16 16, 18
    xtn     v16.8b, v16.8h              // eight mask bits per element
    fmov    x10, d16
    cbz     x10, Lnan_small             // nothing equals m: m is NaN
    rbit    x10, x10
    clz     x10, x10
    lsr     x10, x10, #3                // k: position among the 8 loaded
    sub     x11, x1, #8
    cmp     x10, #4
    add     x11, x11, x10               // back half: index = n - 8 + k
    csel    x0, x10, x11, lo            // front half: index = k
    ret

// ---- 8 < n <= 16: 8 elements at 0 and 8 ending at n ------------------------
// Two steps, like the larger classes: first the four pairs of vectors (4
// elements each), then the two vectors of the first pair that holds m.
Lupto16:
    ldp     q16, q17, [x0]
    ldp     q18, q19, [x0, #32]
    ldp     q20, q21, [x9, #-64]
    ldp     q22, q23, [x9, #-32]
    MX 16, 16, 17                       // pair 0: elements 0..3
    MX 18, 18, 19                       // pair 1: elements 4..7
    MX 20, 20, 21                       // pair 2: elements n-8..n-5
    MX 22, 22, 23                       // pair 3: elements n-4..n-1
    MX 0, 16, 18
    MX 1, 20, 22
    MX 0, 0, 1
    BROADCAST_MAX 1, 0                  // m in both lanes
    EQ 16, 1
    EQ 18, 1
    EQ 20, 1
    EQ 22, 1
    NARROW32 16, 18
    NARROW32 20, 22
    NARROW16 16, 20
    xtn     v16.8b, v16.8h              // eight mask bits per pair lane
    fmov    x10, d16
    cbz     x10, Lnan_small             // nothing equals m: m is NaN
    rbit    x10, x10
    clz     x10, x10
    lsr     x10, x10, #4                // p: the first pair that holds m
    sub     x11, x1, #16
    cmp     x10, #2
    csel    x11, xzr, x11, lo           // pairs 2 and 3 sit n-16 further on
    add     x11, x11, x10, lsl #2       // index of the pair's first element
    add     x12, x0, x11, lsl #3
    ldp     q16, q17, [x12]
    EQ 16, 1
    EQ 17, 1
    NARROW32 16, 17
    xtn     v16.4h, v16.4s              // sixteen mask bits per element
    fmov    x10, d16
    rbit    x10, x10
    clz     x10, x10
    add     x0, x11, x10, lsr #4
    ret

// ---- 16 < n <= 32: 16 elements at 0 and 16 ending at n ---------------------
Lupto32:
    QUAD 16, x0, 0                      // quad 0: elements 0..7
    QUAD 17, x0, 64                     // quad 1: elements 8..15
    QUAD 18, x9, -128                   // quad 2: elements n-16..n-9
    QUAD 19, x9, -64                    // quad 3: elements n-8..n-1
    MX 0, 16, 17
    MX 1, 18, 19
    MX 0, 0, 1
    BROADCAST_MAX 1, 0                  // m in both lanes
    EQ 16, 1
    EQ 17, 1
    EQ 18, 1
    EQ 19, 1
    NARROW32 16, 17
    NARROW32 18, 19
    NARROW16 16, 18
    xtn     v16.8b, v16.8h              // eight mask bits per quad lane
    fmov    x10, d16
    cbz     x10, Lnan_small             // nothing equals m: m is NaN
    rbit    x10, x10
    clz     x10, x10
    lsr     x10, x10, #4                // q: the first quad that holds m
    sub     x11, x1, #32
    cmp     x10, #2
    csel    x11, xzr, x11, lo           // quads 2 and 3 sit n-32 further on
    add     x11, x11, x10, lsl #3       // index of the quad's first element
    add     x12, x0, x11, lsl #3
    FINISH_IN_QUAD

// ---- 32 < n <= 64: 32 elements at 0 and 32 ending at n ---------------------
Lupto64:
    QUAD 16, x0, 0                      // quads 0..3: elements 0..31
    QUAD 17, x0, 64
    QUAD 18, x0, 128
    QUAD 19, x0, 192
    QUAD 20, x9, -256                   // quads 4..7: elements n-32..n-1
    QUAD 21, x9, -192
    QUAD 22, x9, -128
    QUAD 23, x9, -64
    MX 0, 16, 17
    MX 1, 18, 19
    MX 2, 20, 21
    MX 3, 22, 23
    MX 0, 0, 1
    MX 2, 2, 3
    MX 0, 0, 2
    BROADCAST_MAX 1, 0                  // m in both lanes
    EQ 16, 1
    EQ 17, 1
    EQ 18, 1
    EQ 19, 1
    EQ 20, 1
    EQ 21, 1
    EQ 22, 1
    EQ 23, 1
    NARROW32 16, 17
    NARROW32 18, 19
    NARROW32 20, 21
    NARROW32 22, 23
    NARROW16 16, 18
    NARROW16 20, 22
    NARROW8 16, 20
    shrn    v16.8b, v16.8h, #4          // four mask bits per quad lane
    fmov    x10, d16
    cbz     x10, Lnan_small
    rbit    x10, x10
    clz     x10, x10
    lsr     x10, x10, #3                // q: the first quad that holds m
    sub     x11, x1, #64
    cmp     x10, #4
    csel    x11, xzr, x11, lo           // quads 4..7 sit n-64 further on
    add     x11, x11, x10, lsl #3       // index of the quad's first element
    add     x12, x0, x11, lsl #3
    FINISH_IN_QUAD

// ---- A NaN is present ------------------------------------------------------
Lnan_small:
    mov     x8, x0
    mov     x9, x0
// x8 = start of the array, x9 = where to start looking. A NaN is known to be
// at or after x9, so the loop needs no bound.
Lnan_scan:
    ldr     d0, [x9], #8
    fcmp    d0, d0
    b.vc    Lnan_scan
    sub     x0, x9, x8
    sub     x0, x0, #8
    lsr     x0, x0, #3
    ret

// ---- n > 64 ----------------------------------------------------------------
// x8  start of the array          x0  current block
// x12 end of the whole blocks     x5  first block that reached the maximum
// x13 the block ending at n       d4  maximum so far
// x11 nonzero while that last, overlapping block is still to be done
.p2align 4
Lbig:
    mov     x8, x0
    and     x12, x1, #-32
    and     x11, x1, #31
    sub     x13, x9, #256
    add     x12, x0, x12, lsl #3
    ldr     d4, [x0]                    // start from x[0]; a NaN there is caught below
    mov     x5, x0
.p2align 4
Lblock:
    ldp     q16, q17, [x0]
    ldp     q18, q19, [x0, #32]
    ldp     q20, q21, [x0, #64]
    ldp     q22, q23, [x0, #96]
    ldp     q24, q25, [x0, #128]
    ldp     q26, q27, [x0, #160]
    ldp     q28, q29, [x0, #192]
    ldp     q30, q31, [x0, #224]
    MX 16, 16, 17
    MX 18, 18, 19
    MX 20, 20, 21
    MX 22, 22, 23
    MX 24, 24, 25
    MX 26, 26, 27
    MX 28, 28, 29
    MX 30, 30, 31
    MX 16, 16, 18
    MX 20, 20, 22
    MX 24, 24, 26
    MX 28, 28, 30
    MX 16, 16, 20
    MX 24, 24, 28
    MX 16, 16, 24
    fmaxp   d16, v16.2d                 // maximum of this block, NaN if it has one
    fcmp    d16, d4
    b.vs    Lnan_block
    csel    x5, x0, x5, gt              // strictly greater: this block takes over
    fmax    d4, d4, d16
    add     x0, x0, #256
    cmp     x0, x12
    b.lo    Lblock

    cbz     x11, Lfinish
    mov     x0, x13                     // run the block that ends at n
    mov     x11, #0
    add     x12, x13, #256
    b       Lblock

Lnan_block:
    mov     x9, x0                      // the NaN is inside this block
    b       Lnan_scan

Lfinish:
    ldp     q16, q17, [x5]
    ldp     q18, q19, [x5, #32]
    ldp     q20, q21, [x5, #64]
    ldp     q22, q23, [x5, #96]
    ldp     q24, q25, [x5, #128]
    ldp     q26, q27, [x5, #160]
    ldp     q28, q29, [x5, #192]
    ldp     q30, q31, [x5, #224]
    dup     v0.2d, v4.d[0]
    adrp    x14, Lweights@PAGE
    ldr     q7, [x14, Lweights@PAGEOFF]
    EQ 16, 0
    EQ 17, 0
    EQ 18, 0
    EQ 19, 0
    EQ 20, 0
    EQ 21, 0
    EQ 22, 0
    EQ 23, 0
    EQ 24, 0
    EQ 25, 0
    EQ 26, 0
    EQ 27, 0
    EQ 28, 0
    EQ 29, 0
    EQ 30, 0
    EQ 31, 0
    NARROW32 16, 17
    NARROW32 18, 19
    NARROW32 20, 21
    NARROW32 22, 23
    NARROW32 24, 25
    NARROW32 26, 27
    NARROW32 28, 29
    NARROW32 30, 31
    NARROW16 16, 18
    NARROW16 20, 22
    NARROW16 24, 26
    NARROW16 28, 30
    NARROW8 16, 20                      // elements  0..15, one byte each
    NARROW8 24, 28                      // elements 16..31
    and     v16.16b, v16.16b, v7.16b    // keep one distinct bit per byte
    and     v24.16b, v24.16b, v7.16b
    addp    v16.16b, v16.16b, v24.16b   // three pairwise adds fold each
    addp    v16.16b, v16.16b, v16.16b   // group of eight bytes into one byte
    addp    v16.16b, v16.16b, v16.16b
    fmov    w10, s16                    // bit k: element k of the block equals m
    rbit    w10, w10
    clz     w10, w10                    // position inside the block
    sub     x0, x5, x8
    add     x0, x10, x0, lsr #3
    ret

.section __TEXT,__const
.p2align 4
Lweights:
    .byte 1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128
