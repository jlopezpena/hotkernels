// void clip_f64(const double *x, double *out, size_t n, double lo, double hi);
// out[i] = x[i] < lo ? lo : (x[i] > hi ? hi : x[i])
//
// x0 = x, x1 = out, x2 = n, d0 = lo, d1 = hi.
//
// The same design as clip_f32, with two doubles per vector. There are two
// paths, and both give the bits the C expression gives.
//
// The exact path is compare and select, as written in C: two fcmgt and two
// bit per vector. Both compares are false for a NaN element, so it is copied
// unchanged; the same holds for a NaN bound. Nothing in it is arithmetic, so
// signed zeros and signalling NaNs come out as they went in. Four vector
// operations per vector is what limits it: about 4.3 cycles per 8 elements.
//
// The fast path is fmin(fmax(x, lo), hi): two operations per vector, which
// leaves the loop limited by its stores. fmax and fmin differ from the C
// expression in exactly these cases, and each is ruled out before or after:
//
//   1. lo > hi, or a NaN bound. C gives lo for x < lo even when lo > hi, and
//      ignores a NaN bound. Checked at entry (fcmp, unordered or greater).
//   2. Signed zeros. fmax(-0.0, +0.0) is +0.0 and fmin(+0.0, -0.0) is -0.0,
//      where C keeps the element. With lo <= hi this changes a result only
//      when lo is +0.0 or hi is -0.0 (bit patterns). Checked at entry. For
//      every other pair: x < lo gives fmin(lo, hi) = lo; x > hi gives hi;
//      lo <= x <= hi gives x, because a tie between x and a bound is then a
//      tie between identical bit patterns, or a zero pair where the element's
//      own zero is the one fmax or fmin returns.
//   3. FPCR modes. With flush-to-zero a denormal element would come out as
//      zero, and with default-NaN every NaN would become the default one.
//      The fast path is taken only when every FPCR bit except the rounding
//      mode is clear. fmax and fmin are exact, so rounding does not matter.
//   4. A quiet NaN element is returned by fmax and fmin unchanged, sign and
//      payload included. A signalling NaN would be returned quieted, where C
//      copies it. fmax and fmin raise the invalid-operation flag (FPSR.IOC)
//      for a signalling NaN and for nothing else, so the flag is read after
//      the loop; if it is set, the whole array is done again on the exact
//      path, from the input, which the fast path has not touched.
//
// For 4 to work the flag has to be clear before the loop, so FPSR is read at
// entry as well. If the caller left the flag set, it is cleared first and set
// again afterwards. Flags are only ever added: a flag that was set on entry
// is still set on return.
//
// The reads and writes of FPSR are what decide which path a short array
// takes. Measured on an M1 Pro: the two reads together cost about 3 ns, which
// the fast path earns back from about 85 elements; clearing and setting the
// flag costs about 33 ns, earned back from about 500 elements.
//
// A tail that is not a whole 8-element block is one more block that ends
// exactly at the end of the array and overlaps the previous one. It is
// computed from the input again, never from output already written.

// Below this many elements the two FPSR reads cost more than the fast path saves.
.set FAST_MIN_N, 96
// Below this many elements, clearing and restoring a flag the caller left set
// costs more than the fast path saves.
.set STALE_MIN_N, 768
// FPSR.IOC, the cumulative invalid-operation flag, is bit 0.
.set IOC_BIT, 0
// Every FPCR bit except the rounding mode (bits 22 and 23).
.set FPCR_NOT_ROUNDING, 0xff3fffff
// The sign bit of a float64: the bit pattern of -0.0.
.set SIGN_BIT, 0x8000000000000000

.text
.globl _clip_f64
.p2align 2
_clip_f64:
    dup.2d  v2, v0[0]               // lo in every lane
    dup.2d  v3, v1[0]               // hi in every lane
    cmp     x2, #FAST_MIN_N
    b.lo    Lexact
    fcmp    d0, d1
    b.hi    Lexact                  // lo > hi, or a NaN bound
    fmov    x8, d0
    fmov    x9, d1
    cbz     x8, Lexact              // lo is +0.0
    eor     x9, x9, #SIGN_BIT
    cbz     x9, Lexact              // hi is -0.0
    mrs     x10, fpcr
    tst     w10, #FPCR_NOT_ROUNDING
    b.ne    Lexact                  // flush-to-zero, default-NaN or anything else unusual
    mrs     x11, fpsr
    mov     w16, #0                 // w16 = 1 when IOC has to be set again on return
    tbnz    w11, #IOC_BIT, Lstale
Lfast:
    // x0, x1, x2, v2 and v3 are left as they are, for the exact path to start over.
    mov     x14, x0
    mov     x15, x1
    sub     x8, x2, #8              // x8 = elements left after this block
Lfblock:
    ldp     q4, q5, [x14]
    ldp     q6, q7, [x14, #32]
    fmax.2d v4, v4, v2
    fmax.2d v5, v5, v2
    fmax.2d v6, v6, v2
    fmax.2d v7, v7, v2
    fmin.2d v4, v4, v3
    fmin.2d v5, v5, v3
    fmin.2d v6, v6, v3
    fmin.2d v7, v7, v3
    stp     q4, q5, [x15]
    stp     q6, q7, [x15, #32]
    add     x14, x14, #64
    add     x15, x15, #64
    subs    x8, x8, #8
    b.hs    Lfblock
    adds    x8, x8, #8              // elements left: 0 to 7
    b.eq    Lfend
    // Step back so that one more block ends exactly at the end.
    add     x14, x14, x8, lsl #3
    add     x15, x15, x8, lsl #3
    sub     x14, x14, #64
    sub     x15, x15, #64
    mov     x8, #0
    b       Lfblock
Lfend:
    mrs     x9, fpsr
    tbnz    w9, #IOC_BIT, Lexact    // a signalling NaN went through: start over, exactly
    cbnz    w16, Lrestore
    ret
Lrestore:
    orr     x9, x9, #(1 << IOC_BIT)
    msr     fpsr, x9
    ret
Lstale:
    // The caller left IOC set. Clear it so the loop can use it, and remember to set it again.
    cmp     x2, #STALE_MIN_N
    b.lo    Lexact
    and     x9, x11, #~(1 << IOC_BIT)
    msr     fpsr, x9
    mov     w16, #1
    b       Lfast

Lexact:
    subs    x8, x2, #8              // x8 = elements left after this block
    b.lo    Lsmall
Lblock:
    ldp     q4, q5, [x0]
    ldp     q6, q7, [x0, #32]
    fcmgt.2d v16, v2, v4            // x < lo
    fcmgt.2d v17, v2, v5
    fcmgt.2d v18, v2, v6
    fcmgt.2d v19, v2, v7
    fcmgt.2d v20, v4, v3            // x > hi
    fcmgt.2d v21, v5, v3
    fcmgt.2d v22, v6, v3
    fcmgt.2d v23, v7, v3
    bit.16b v4, v3, v20             // hi where x > hi
    bit.16b v5, v3, v21
    bit.16b v6, v3, v22
    bit.16b v7, v3, v23
    bit.16b v4, v2, v16             // lo where x < lo; applied last, as in C
    bit.16b v5, v2, v17
    bit.16b v6, v2, v18
    bit.16b v7, v2, v19
    stp     q4, q5, [x1]
    stp     q6, q7, [x1, #32]
    add     x0, x0, #64
    add     x1, x1, #64
    subs    x8, x8, #8
    b.hs    Lblock
    adds    x8, x8, #8              // elements left: 0 to 7
    b.eq    Ldone
    // Step back so that one more block ends exactly at the end.
    add     x0, x0, x8, lsl #3
    add     x1, x1, x8, lsl #3
    sub     x0, x0, #64
    sub     x1, x1, #64
    mov     x8, #0
    b       Lblock

Lsmall:                             // n < 8
    cmp     x2, #2
    b.lo    Lscalar
    add     x9, x0, x2, lsl #3      // end of x
    add     x10, x1, x2, lsl #3     // end of out
    tbz     x2, #2, Ltwo
    // 4 <= n < 8: the first four elements and the last four, overlapping.
    ldp     q4, q5, [x0]
    ldp     q6, q7, [x9, #-32]
    fcmgt.2d v16, v2, v4
    fcmgt.2d v17, v2, v5
    fcmgt.2d v18, v2, v6
    fcmgt.2d v19, v2, v7
    fcmgt.2d v20, v4, v3
    fcmgt.2d v21, v5, v3
    fcmgt.2d v22, v6, v3
    fcmgt.2d v23, v7, v3
    bit.16b v4, v3, v20
    bit.16b v5, v3, v21
    bit.16b v6, v3, v22
    bit.16b v7, v3, v23
    bit.16b v4, v2, v16
    bit.16b v5, v2, v17
    bit.16b v6, v2, v18
    bit.16b v7, v2, v19
    stp     q4, q5, [x1]
    stp     q6, q7, [x10, #-32]
    ret
Ltwo:
    // n is 2 or 3: the first two elements and the last two, overlapping.
    ldr     q4, [x0]
    ldur    q5, [x9, #-16]
    fcmgt.2d v16, v2, v4
    fcmgt.2d v17, v2, v5
    fcmgt.2d v20, v4, v3
    fcmgt.2d v21, v5, v3
    bit.16b v4, v3, v20
    bit.16b v5, v3, v21
    bit.16b v4, v2, v16
    bit.16b v5, v2, v17
    str     q4, [x1]
    stur    q5, [x10, #-16]
Ldone:
    ret

Lscalar:                            // n is 0 or 1
    cbz     x2, Ldone
    ldr     d4, [x0]
    fcmp    d4, d1
    fcsel   d5, d1, d4, gt          // x > hi ? hi : x
    fcmp    d4, d0
    fcsel   d4, d0, d5, mi          // x < lo ? lo : that
    str     d4, [x1]
    ret
