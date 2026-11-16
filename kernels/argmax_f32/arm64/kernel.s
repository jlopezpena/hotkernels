// int64_t argmax_f32(const float *x, size_t n);   n >= 1
//
// Index of the first maximum; the first NaN wins over everything.
//
// Method. `fmax` returns NaN if either operand is NaN, so the maximum of a
// group of elements is NaN exactly when the group contains one.
//
//   n <= 64   One straight-line pass: take the maximum m of everything, then
//             compare every element with m and turn the compare results into
//             a bit mask whose lowest set bit is the answer. The elements are
//             covered by a front half and a back half that may overlap; the
//             back half ends exactly at the end of the array.
//   n  > 64   Blocks of 64 elements. Each block is reduced to a vector of four
//             lane maxima. Per lane, the running maximum and the number of the
//             first block that reached it are kept without any branch. At the
//             end the winning block is the lowest block number among the lanes
//             that hold the overall maximum, and only that block is rescanned.
//
// A NaN makes m NaN, which is tested once per small input, and once per
// 65,536 elements on the large path; a scalar loop then finds the first NaN.

// Blocks per NaN test on the large path.
.set SUPER_BLOCKS, 1024

// v\d = lane-wise maximum of v\a and v\b (NaN if either is NaN).
.macro MX d, a, b
    fmax    v\d\().4s, v\a\().4s, v\b\().4s
.endm

// v\r = all-ones in each lane of v\r that equals the maximum broadcast in v0.
.macro EQ r
    fcmeq   v\r\().4s, v\r\().4s, v0.4s
.endm

// Narrow two vectors of 32-bit lane masks (\a, \b) into one of 16-bit masks.
.macro NARROW16 a, b
    uzp1    v\a\().8h, v\a\().8h, v\b\().8h
.endm

// Narrow two vectors of 16-bit lane masks (\a, \b) into one of byte masks.
.macro NARROW8 a, b
    uzp1    v\a\().16b, v\a\().16b, v\b\().16b
.endm

// Load the bit weights 1,2,4,...,128 (twice) into v7. Clobbers x14.
.macro LOAD_WEIGHTS
    adrp    x14, Lweights@PAGE
    ldr     q7, [x14, Lweights@PAGEOFF]
.endm

// Sixteen vectors v16..v31 hold 64 elements in order, v0 holds the maximum in
// every lane, v7 the bit weights. Leaves in x10 a mask with bit k set when
// element k equals the maximum. Clobbers v16..v31.
.macro MASK64
    EQ 16
    EQ 17
    EQ 18
    EQ 19
    EQ 20
    EQ 21
    EQ 22
    EQ 23
    EQ 24
    EQ 25
    EQ 26
    EQ 27
    EQ 28
    EQ 29
    EQ 30
    EQ 31
    NARROW16 16, 17
    NARROW16 18, 19
    NARROW16 20, 21
    NARROW16 22, 23
    NARROW16 24, 25
    NARROW16 26, 27
    NARROW16 28, 29
    NARROW16 30, 31
    NARROW8 16, 18                      // elements  0..15, one byte each
    NARROW8 20, 22                      // elements 16..31
    NARROW8 24, 26                      // elements 32..47
    NARROW8 28, 30                      // elements 48..63
    and     v16.16b, v16.16b, v7.16b    // keep one distinct bit per byte
    and     v20.16b, v20.16b, v7.16b
    and     v24.16b, v24.16b, v7.16b
    and     v28.16b, v28.16b, v7.16b
    addp    v16.16b, v16.16b, v20.16b   // three pairwise adds fold each
    addp    v24.16b, v24.16b, v28.16b   // group of eight bytes into one byte
    addp    v16.16b, v16.16b, v24.16b
    addp    v16.16b, v16.16b, v16.16b
    fmov    x10, d16
.endm

.text
.globl _argmax_f32
.p2align 4
_argmax_f32:
    add     x9, x0, x1, lsl #2          // end of the array
    cmp     x1, #16
    b.hi    Lover16
    cmp     x1, #8
    b.hs    Lupto16

// ---- n < 8, and absurdly large n: one element at a time --------------------
Lscalar:
    ldr     s0, [x0]                    // best value so far
    mov     x2, #0                      // its index
    fcmp    s0, s0
    b.vs    Lscalar_done                // x[0] is NaN
    mov     x3, #1
    b       Lscalar_test
Lscalar_loop:
    ldr     s1, [x0, x3, lsl #2]
    fcmp    s1, s0
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
    fcmp    s1, s1
    fmov    s0, s1
    b.vc    Lscalar_next                // a real new maximum: carry on
    mov     x0, x2                      // NaN: it wins
    ret

Lover16:
    cmp     x1, #64
    b.hi    Lbig
    cmp     x1, #32
    b.hi    Lupto64
    b       Lupto32

// ---- 8 <= n <= 16: vectors at 0, 4, n-8, n-4 -------------------------------
Lupto16:
    ldp     q16, q17, [x0]
    ldp     q18, q19, [x9, #-32]
    MX 0, 16, 17
    MX 1, 18, 19
    MX 0, 0, 1
    fmaxv   s0, v0.4s
    dup     v0.4s, v0.s[0]
    EQ 16
    EQ 17
    EQ 18
    EQ 19
    NARROW16 16, 17
    NARROW16 18, 19
    NARROW8 16, 18
    shrn    v16.8b, v16.8h, #4          // four mask bits per element
    fmov    x10, d16
    cbz     x10, Lnan_small             // nothing equals m: m is NaN
    rbit    x10, x10
    clz     x10, x10
    lsr     x10, x10, #2                // k: position among the 16 loaded
    sub     x11, x1, #16
    cmp     x10, #8
    add     x11, x11, x10               // back half: index = n - 16 + k
    csel    x0, x10, x11, lo            // front half: index = k
    ret

// ---- 16 < n <= 32: 16 elements at 0 and 16 ending at n ---------------------
Lupto32:
    ldp     q16, q17, [x0]
    ldp     q18, q19, [x0, #32]
    ldp     q20, q21, [x9, #-64]
    ldp     q22, q23, [x9, #-32]
    LOAD_WEIGHTS
    MX 0, 16, 17
    MX 1, 18, 19
    MX 2, 20, 21
    MX 3, 22, 23
    MX 0, 0, 1
    MX 2, 2, 3
    MX 0, 0, 2
    fmaxv   s0, v0.4s
    dup     v0.4s, v0.s[0]
    EQ 16
    EQ 17
    EQ 18
    EQ 19
    EQ 20
    EQ 21
    EQ 22
    EQ 23
    NARROW16 16, 17
    NARROW16 18, 19
    NARROW16 20, 21
    NARROW16 22, 23
    NARROW8 16, 18
    NARROW8 20, 22
    and     v16.16b, v16.16b, v7.16b
    and     v20.16b, v20.16b, v7.16b
    addp    v16.16b, v16.16b, v20.16b
    addp    v16.16b, v16.16b, v16.16b
    addp    v16.16b, v16.16b, v16.16b
    fmov    w10, s16                    // bit k: loaded element k equals m
    cbz     w10, Lnan_small
    rbit    w10, w10
    clz     w10, w10
    sub     x11, x1, #32
    cmp     x10, #16
    add     x11, x11, x10               // back half: index = n - 32 + k
    csel    x0, x10, x11, lo
    ret

// ---- 32 < n <= 64: 32 elements at 0 and 32 ending at n ---------------------
// Two steps, to keep the number of vector operations down: the maximum tree
// passes through four "quads" (the lane maxima of 16 consecutive elements), so
// first find the first quad that holds m, then look only at its 16 elements.
Lupto64:
    ldp     q16, q17, [x0]
    ldp     q18, q19, [x0, #32]
    ldp     q20, q21, [x0, #64]
    ldp     q22, q23, [x0, #96]
    ldp     q24, q25, [x9, #-128]
    ldp     q26, q27, [x9, #-96]
    ldp     q28, q29, [x9, #-64]
    ldp     q30, q31, [x9, #-32]
    MX 0, 16, 17
    MX 1, 18, 19
    MX 2, 20, 21
    MX 3, 22, 23
    MX 4, 24, 25
    MX 5, 26, 27
    MX 6, 28, 29
    MX 7, 30, 31
    MX 0, 0, 1                          // quad 0: elements 0..15
    MX 2, 2, 3                          // quad 1: elements 16..31
    MX 4, 4, 5                          // quad 2: elements n-32..n-17
    MX 6, 6, 7                          // quad 3: elements n-16..n-1
    MX 1, 0, 2
    MX 3, 4, 6
    MX 1, 1, 3
    fmaxv   s1, v1.4s
    dup     v1.4s, v1.s[0]              // m in every lane
    fcmeq   v0.4s, v0.4s, v1.4s
    fcmeq   v2.4s, v2.4s, v1.4s
    fcmeq   v4.4s, v4.4s, v1.4s
    fcmeq   v6.4s, v6.4s, v1.4s
    NARROW16 0, 2
    NARROW16 4, 6
    NARROW8 0, 4
    shrn    v0.8b, v0.8h, #4            // four mask bits per quad lane
    fmov    x10, d0
    cbz     x10, Lnan_small             // nothing equals m: m is NaN
    rbit    x10, x10
    clz     x10, x10
    lsr     x10, x10, #4                // q: the first quad that holds m
    sub     x11, x1, #64
    cmp     x10, #2
    csel    x11, xzr, x11, lo           // quads 2 and 3 sit n-64 further on
    add     x11, x11, x10, lsl #4       // index of the quad's first element
    add     x12, x0, x11, lsl #2
    ldp     q16, q17, [x12]
    ldp     q18, q19, [x12, #32]
    fcmeq   v16.4s, v16.4s, v1.4s
    fcmeq   v17.4s, v17.4s, v1.4s
    fcmeq   v18.4s, v18.4s, v1.4s
    fcmeq   v19.4s, v19.4s, v1.4s
    NARROW16 16, 17
    NARROW16 18, 19
    NARROW8 16, 18
    shrn    v16.8b, v16.8h, #4
    fmov    x10, d16
    rbit    x10, x10
    clz     x10, x10
    add     x0, x11, x10, lsr #2
    ret

// ---- A NaN is present ------------------------------------------------------
Lnan_small:
    mov     x8, x0
    mov     x9, x0
// x8 = start of the array, x9 = where to start looking. A NaN is known to be
// at or after x9, so the loop needs no bound.
Lnan_scan:
    ldr     s0, [x9], #4
    fcmp    s0, s0
    b.vc    Lnan_scan
    sub     x0, x9, x8
    sub     x0, x0, #4
    lsr     x0, x0, #2
    ret

// ---- n > 64 ----------------------------------------------------------------
// x8  start of the array          x0  current block
// x12 number of whole blocks      x3  whole blocks after this super-block
// x13 the block ending at n       x2  blocks left in this super-block
// x11 nonzero while that last, overlapping block is still to be done
// x9  start of the current super-block (where a NaN search would begin)
// v4  running maximum per lane    v5  block number that first reached it
// v6  current block number        v7  1 in every lane
.p2align 4
Lbig:
    lsr     x10, x1, #38                // block numbers are 32 bits wide
    cbnz    x10, Lscalar
    mov     x8, x0
    lsr     x3, x1, #6
    add     x13, x0, x1, lsl #2
    mov     x12, x3
    and     x11, x1, #63
    sub     x13, x13, #256
    mvni    v4.4s, #0x7f, msl #16       // -infinity
    movi    v5.16b, #0
    movi    v6.16b, #0
    movi    v7.4s, #1
Lsuper:
    mov     x9, x0
    mov     x2, #SUPER_BLOCKS
    cmp     x3, x2
    csel    x2, x3, x2, lo
    sub     x3, x3, x2
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
    add     x0, x0, #256
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
    MX 16, 16, 24                       // lane maxima of this block
    fcmgt   v0.4s, v16.4s, v4.4s        // lanes that beat the running maximum
    fmax    v4.4s, v4.4s, v16.4s
    bit     v5.16b, v6.16b, v0.16b      // those lanes record this block
    add     v6.4s, v6.4s, v7.4s
    subs    x2, x2, #1
    b.ne    Lblock

    fmaxv   s0, v4.4s                   // overall maximum so far
    fcmp    s0, s0
    b.vs    Lnan_scan                   // NaN somewhere in this super-block
    cbnz    x3, Lsuper
    cbz     x11, Lfinish
    mov     x0, x13                     // run the block that ends at n
    mov     x9, x13
    mov     x11, #0
    mov     x2, #1
    b       Lblock

Lfinish:
    dup     v0.4s, v0.s[0]
    fcmeq   v1.4s, v4.4s, v0.4s         // lanes holding the overall maximum
    orn     v1.16b, v5.16b, v1.16b      // other lanes become 0xffffffff
    uminv   s1, v1.4s                   // first block that contains it
    fmov    w10, s1
    cmp     x10, x12
    add     x0, x8, x10, lsl #8
    csel    x0, x13, x0, eq             // block number n/64 is the last one
    ldp     q16, q17, [x0]
    ldp     q18, q19, [x0, #32]
    ldp     q20, q21, [x0, #64]
    ldp     q22, q23, [x0, #96]
    ldp     q24, q25, [x0, #128]
    ldp     q26, q27, [x0, #160]
    ldp     q28, q29, [x0, #192]
    ldp     q30, q31, [x0, #224]
    LOAD_WEIGHTS
    MASK64
    rbit    x10, x10
    clz     x10, x10                    // position inside the block
    sub     x0, x0, x8
    add     x0, x10, x0, lsr #2
    ret

.section __TEXT,__const
.p2align 4
Lweights:
    .byte 1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128
