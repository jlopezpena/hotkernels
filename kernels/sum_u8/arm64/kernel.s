// uint64_t sum_u8(const uint8_t *x, size_t n);
//
// x0 = x, x1 = n. Returns the sum of the n bytes at x.
//
// The work is done by `uadalp acc.8h, v.16b`: one instruction adds the
// sixteen bytes of a vector, in pairs, into eight 16-bit lanes of an
// accumulator. Each step adds at most 2 * 255 = 510 to a lane, so a lane can
// take 128 steps before it could overflow; the code widens before that.
//
// Paths, by n:
//   0..15     general registers only (two overlapping loads, or a byte loop)
//   16..31    one whole vector plus the masked last vector
//   32..2047  "lean": eight accumulators, 128 bytes per step, then 64, 32, 16
//             and a masked last vector; the accumulators are added together
//             as 16-bit lanes, which cannot overflow below 2,048 bytes
//   2048..    "big": sixteen accumulators, 256 bytes per step, widened to a
//             64-bit total every 128 steps; the last 0..255 bytes go through
//             the lean path
//
// `uadalp` takes three cycles from accumulator in to accumulator out, and the
// core can load three vectors per cycle. Sixteen accumulators let a 256-byte
// step run at the load limit (16 loads in 5.33 cycles); eight would be held
// to 6 cycles by the accumulator chain.

// Steps of the big loop between widenings. 128 * 510 = 65,280 fits in 16 bits.
.set BLOCK_STEPS, 128
// From this many bytes the big path is used. Below it the lean path sees at
// most 128 vectors (127 whole ones and a masked one), and 128 * 510 = 65,280,
// so its accumulators can be added together as 16-bit lanes without overflow.
.set BIG_MIN_N, 2048

.section __TEXT,__const
.p2align 4
// Loaded at offset r (1..15): 16 - r zero bytes, then r bytes of ones. Applied
// to the last 16 bytes of the array it keeps only the final r bytes, the ones
// no earlier vector covered.
Lmask:
    .fill 16, 1, 0x00
    .fill 16, 1, 0xff

.text
.globl _sum_u8
.p2align 2
_sum_u8:
    // Short calls cost about one cycle per taken branch, so the code is laid
    // out for n = 16 and n = 64 to reach `ret` with as few as possible.
    cmp     x1, #32
    b.hs    Lge32
    tbz     x1, #4, Lsmall
    // 16..31: the first vector whole, then the last vector masked to the
    // n - 16 bytes the first did not cover.
    ldr     q0, [x0]
    ands    x4, x1, #15
    b.ne    Ltwo
    uaddlv  h0, v0.16b
    fmov    w0, s0
    ret
Ltwo:
    add     x3, x0, x1
    ldur    q1, [x3, #-16]
    adrp    x5, Lmask@PAGE
    add     x5, x5, Lmask@PAGEOFF
    ldr     q2, [x5, x4]
    and     v1.16b, v1.16b, v2.16b
    uaddlp  v0.8h, v0.16b
    uadalp  v0.8h, v1.16b
    addv    h0, v0.8h               // at most 31 * 255, fits in 16 bits
    fmov    w0, s0
    ret

Lge32:
    mov     x6, #0                  // total of everything already widened
    movi    v16.2d, #0
    movi    v17.2d, #0
    cmp     x1, #128
    b.hs    Lge128

    // ---- Lean path, fewer than 128 bytes left: x0 = cursor, x1 = a count
    //      whose low seven bits are the bytes left, x6 = total so far,
    //      v16 and v17 = 16-bit accumulators.
Llean64:
    tbz     x1, #6, Llean32
    ldp     q0, q1, [x0]
    ldp     q2, q3, [x0, #32]
    add     x0, x0, #64
    uadalp  v16.8h, v0.16b
    uadalp  v17.8h, v1.16b
    uadalp  v16.8h, v2.16b
    uadalp  v17.8h, v3.16b
Llean32:
    tbnz    x1, #5, Ldo32
Llean16:
    tbnz    x1, #4, Ldo16
Lleanpart:
    tst     x1, #15
    b.ne    Ldopart
Lleanreduce:
    add     v16.8h, v16.8h, v17.8h
    uaddlv  s0, v16.8h
    fmov    w0, s0
    add     x0, x0, x6
    ret

Ldo32:
    ldp     q0, q1, [x0], #32
    uadalp  v16.8h, v0.16b
    uadalp  v17.8h, v1.16b
    b       Llean16
Ldo16:
    ldr     q0, [x0], #16
    uadalp  v16.8h, v0.16b
    b       Lleanpart
Ldopart:
    // 1..15 bytes left, starting at x0: load the last 16 bytes of the array
    // (it has at least 32) and keep only the ones not yet counted.
    and     x4, x1, #15
    add     x5, x0, x4              // end of the array
    ldur    q0, [x5, #-16]
    adrp    x7, Lmask@PAGE
    add     x7, x7, Lmask@PAGEOFF
    ldr     q1, [x7, x4]
    and     v0.16b, v0.16b, v1.16b
    uadalp  v17.8h, v0.16b
    b       Lleanreduce

Lge128:
    cmp     x1, #BIG_MIN_N
    b.hs    Lbig
    // ---- Lean path, 128..2047 bytes left: eight accumulators, 128 bytes per
    //      step, folded into v16 and v17 afterwards.
Llean128:
    movi    v18.2d, #0
    movi    v19.2d, #0
    movi    v20.2d, #0
    movi    v21.2d, #0
    movi    v22.2d, #0
    movi    v23.2d, #0
    sub     x4, x1, #128
Lleanstep:
    ldp     q0, q1, [x0]
    ldp     q2, q3, [x0, #32]
    ldp     q4, q5, [x0, #64]
    ldp     q6, q7, [x0, #96]
    add     x0, x0, #128
    uadalp  v16.8h, v0.16b
    uadalp  v17.8h, v1.16b
    uadalp  v18.8h, v2.16b
    uadalp  v19.8h, v3.16b
    uadalp  v20.8h, v4.16b
    uadalp  v21.8h, v5.16b
    uadalp  v22.8h, v6.16b
    uadalp  v23.8h, v7.16b
    subs    x4, x4, #128
    b.hs    Lleanstep
    add     v16.8h, v16.8h, v18.8h
    add     v17.8h, v17.8h, v19.8h
    add     v20.8h, v20.8h, v22.8h
    add     v21.8h, v21.8h, v23.8h
    add     v16.8h, v16.8h, v20.8h
    add     v17.8h, v17.8h, v21.8h
    tst     x1, #127
    b.ne    Llean64
    // Nothing left: the same reduction as Lleanreduce, repeated here so a
    // multiple of 128 does not take another branch.
    add     v16.8h, v16.8h, v17.8h
    uaddlv  s0, v16.8h
    fmov    w0, s0
    add     x0, x0, x6
    ret

    // ---- Big path: n >= 2048.
Lbig:
    movi    v18.2d, #0
    movi    v19.2d, #0
    movi    v20.2d, #0
    movi    v21.2d, #0
    movi    v22.2d, #0
    movi    v23.2d, #0
    movi    v24.2d, #0
    movi    v25.2d, #0
    movi    v26.2d, #0
    movi    v27.2d, #0
    movi    v28.2d, #0
    movi    v29.2d, #0
    movi    v30.2d, #0
    movi    v31.2d, #0
    lsr     x4, x1, #8              // 256-byte steps left
Lblock:
    mov     x5, #BLOCK_STEPS
    cmp     x4, x5
    csel    x5, x4, x5, lo          // steps in this block
    sub     x4, x4, x5
    // `ldnp` is the load that hints the data will not be needed again soon.
    // It costs nothing while the array fits in the caches and is about 10%
    // faster at 16 MB, where the array does not.
Lstep:
    ldnp    q0, q1, [x0]
    ldnp    q2, q3, [x0, #32]
    ldnp    q4, q5, [x0, #64]
    ldnp    q6, q7, [x0, #96]
    uadalp  v16.8h, v0.16b
    uadalp  v17.8h, v1.16b
    uadalp  v18.8h, v2.16b
    uadalp  v19.8h, v3.16b
    uadalp  v20.8h, v4.16b
    uadalp  v21.8h, v5.16b
    uadalp  v22.8h, v6.16b
    uadalp  v23.8h, v7.16b
    ldnp    q0, q1, [x0, #128]
    ldnp    q2, q3, [x0, #160]
    ldnp    q4, q5, [x0, #192]
    ldnp    q6, q7, [x0, #224]
    add     x0, x0, #256
    uadalp  v24.8h, v0.16b
    uadalp  v25.8h, v1.16b
    uadalp  v26.8h, v2.16b
    uadalp  v27.8h, v3.16b
    uadalp  v28.8h, v4.16b
    uadalp  v29.8h, v5.16b
    uadalp  v30.8h, v6.16b
    uadalp  v31.8h, v7.16b
    subs    x5, x5, #1
    b.ne    Lstep
    // Widen: sixteen accumulators of 16-bit lanes into two of 32-bit lanes
    // (at most 16 * 65,280 per lane), then into the 64-bit total.
    uaddlp  v0.4s, v16.8h
    uaddlp  v1.4s, v24.8h
    uadalp  v0.4s, v17.8h
    uadalp  v1.4s, v25.8h
    uadalp  v0.4s, v18.8h
    uadalp  v1.4s, v26.8h
    uadalp  v0.4s, v19.8h
    uadalp  v1.4s, v27.8h
    uadalp  v0.4s, v20.8h
    uadalp  v1.4s, v28.8h
    uadalp  v0.4s, v21.8h
    uadalp  v1.4s, v29.8h
    uadalp  v0.4s, v22.8h
    uadalp  v1.4s, v30.8h
    uadalp  v0.4s, v23.8h
    uadalp  v1.4s, v31.8h
    add     v0.4s, v0.4s, v1.4s
    uaddlv  d0, v0.4s
    fmov    x7, d0
    add     x6, x6, x7
    movi    v16.2d, #0
    movi    v17.2d, #0
    movi    v18.2d, #0
    movi    v19.2d, #0
    movi    v20.2d, #0
    movi    v21.2d, #0
    movi    v22.2d, #0
    movi    v23.2d, #0
    movi    v24.2d, #0
    movi    v25.2d, #0
    movi    v26.2d, #0
    movi    v27.2d, #0
    movi    v28.2d, #0
    movi    v29.2d, #0
    movi    v30.2d, #0
    movi    v31.2d, #0
    cbnz    x4, Lblock
    // The last 0..255 bytes go through the lean path, which reads only the
    // low bits of x1 once fewer than 128 bytes are left.
    and     x1, x1, #255
    tbz     x1, #7, Llean64
    b       Llean128

Lsmall:
    tbnz    x1, #3, L8to15
    tbnz    x1, #2, L4to7
    // 0..3: one byte at a time.
    mov     x2, #0
    cbz     x1, Lsmalldone
Lbyte:
    ldrb    w4, [x0], #1
    add     x2, x2, x4
    subs    x1, x1, #1
    b.ne    Lbyte
Lsmalldone:
    mov     x0, x2
    ret

L8to15:
    // The first eight bytes and the last eight overlap by 16 - n bytes. The
    // overlap is the low-order end of the second load (little-endian), so
    // shifting it right by 8 * (16 - n) bits removes it. That is 8..64 bits,
    // and a register shift only takes 0..63, so it is done as a shift by 8
    // followed by one of 8 * (15 - n). The shift uses the count modulo 64,
    // and 8 * (15 - n) = -8 * (n + 1) modulo 64.
    add     x3, x0, x1
    ldr     x4, [x0]
    ldur    x5, [x3, #-8]
    mvn     x7, x1                  // -(n + 1)
    lsl     x7, x7, #3
    lsr     x5, x5, #8
    lsr     x5, x5, x7
    fmov    d0, x4
    mov     v0.d[1], x5
    uaddlv  h0, v0.16b
    fmov    w0, s0
    ret

L4to7:
    // The same with two 4-byte loads: the overlap is 8 * (8 - n) bits, done as
    // 8 and then 8 * (7 - n) = -8 * (n + 1) modulo 32.
    add     x3, x0, x1
    ldr     w4, [x0]
    ldur    w5, [x3, #-4]
    mvn     w7, w1
    lsl     w7, w7, #3
    lsr     w5, w5, #8
    lsr     w5, w5, w7
    orr     x4, x4, x5, lsl #32
    fmov    d0, x4
    uaddlv  h0, v0.8b
    fmov    w0, s0
    ret
