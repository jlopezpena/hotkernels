// Inner-loop hook: a ufunc's loop for one element type, replaced through
// PyUFunc_ReplaceLoopBySignature, the interface NumPy provides for this.
//
// `np.clip(x, lo, hi)` and `x.clip(lo, hi)` both run the `clip` ufunc, whose
// loop receives four operands: x, lo, hi and the output. NumPy has settled
// the element type before the loop is called and describes the memory layout
// through `steps`, the distance in bytes between consecutive elements of each
// operand. A scalar bound has a step of zero.
//
// On ARM64 the loops of `add` and `subtract` on float64 are hooked too, for
// the calls that write their result over their first input: `w -= b`,
// `w += b`, and the in-place add NumPy substitutes for `t + b` when `t` is a
// temporary. See docs/gradient-step-findings.md and below.
#include "binding.h"

// Operands of the clip loop, in the order NumPy passes them.
enum { CLIP_X, CLIP_LO, CLIP_HI, CLIP_OUT, CLIP_OPERANDS };

// Operands of a binary arithmetic loop such as add's: in1, in2, out.
enum { BINARY_IN1, BINARY_IN2, BINARY_OUT, BINARY_OPERANDS };

// The most operands any hooked loop has.
#define MAX_OPERANDS CLIP_OPERANDS

// One hooked loop.
typedef struct {
    // Name of the ufunc in numpy._core.umath.
    const char *ufunc;
    // Number of operands of its loop, inputs and output: the length of the
    // signature PyUFunc_ReplaceLoopBySignature matches.
    int operands;
    // NumPy type number of all its operands.
    int type_num;
    // The binding's loop.
    PyUFuncGenericFunction wrapper;
    // NumPy's own loop, kept while the hook is applied.
    PyUFuncGenericFunction original;
} hk_loop;

// Indices of the loops in `loops`.
enum { LOOP_CLIP_F32, LOOP_CLIP_F64, LOOP_ADD_F64, LOOP_SUBTRACT_F64, LOOP_COUNT };

static hk_loop loops[LOOP_COUNT];

// The kernel is taken when x and the output are contiguous, both bounds are
// scalars with lo <= hi (which a NaN bound fails), and the output is either
// clear of the input or exactly the input. The kernels are wrong when their
// output is their input (issue 7), so an in-place call is clipped through a
// buffer of the binding's (hk_clip_in_place). Everything else is NumPy's:
// strided operands, array bounds (NumPy uses different arithmetic for them,
// issue 9), reversed or NaN bounds, and an output that overlaps the input
// only in part.
//
// The kernels leave the invalid-operation flag set on NaN data. NumPy turns
// status flags into warnings after a loop returns, and its own clip loop ends
// by clearing them, so the hook does the same.
#define HK_CLIP_LOOP(function, index, kernel, kernel_id, c_type)                              \
    static void function(char **args, npy_intp const *dimensions, npy_intp const *steps,      \
                         void *data) {                                                        \
        const npy_intp n = dimensions[0];                                                     \
        const npy_intp size = (npy_intp)sizeof(c_type);                                       \
        if (n > 0 && steps[CLIP_X] == size && steps[CLIP_OUT] == size &&                      \
            steps[CLIP_LO] == 0 && steps[CLIP_HI] == 0) {                                     \
            c_type lo, hi;                                                                    \
            memcpy(&lo, args[CLIP_LO], sizeof lo);                                            \
            memcpy(&hi, args[CLIP_HI], sizeof hi);                                            \
            const size_t bytes = (size_t)n * sizeof(c_type);                                  \
            const bool in_place = args[CLIP_X] == args[CLIP_OUT];                             \
            if (lo <= hi && (in_place || hk_disjoint(args[CLIP_X], bytes, args[CLIP_OUT], bytes))) { \
                if (in_place) hk_clip_in_place(kernel_id, args[CLIP_OUT], (size_t)n, lo, hi);  \
                else kernel((const c_type *)args[CLIP_X], (c_type *)args[CLIP_OUT], (size_t)n, \
                            lo, hi);                                                          \
                feclearexcept(FE_ALL_EXCEPT);                                                 \
                hk_take(kernel_id);                                                           \
                return;                                                                       \
            }                                                                                 \
        }                                                                                     \
        hk_pass(HK_OP_CLIP);                                                                  \
        loops[index].original(args, dimensions, steps, data);                                 \
    }

HK_CLIP_LOOP(loop_clip_f32, LOOP_CLIP_F32, clip_f32, HK_CLIP_F32, float)
HK_CLIP_LOOP(loop_clip_f64, LOOP_CLIP_F64, clip_f64, HK_CLIP_F64, double)

// --------------------------------------------- in-place add and subtract

#if defined(__aarch64__)

// After Accelerate has multiplied by a matrix on several threads, the
// matrix's cache lines sit in other cores' caches. NumPy's loop then takes
// up to 15 times as long to subtract in place from that matrix as from one
// in this core's cache: each line is fetched shared for the load and fetched
// again for the store (docs/gradient-step-findings.md). The loops below
// handle one 128-byte line of the output per step and prefetch the line
// INPLACE_AHEAD bytes ahead for a store (`prfm pstl1keep`), so that it
// arrives ready to be written. On a line in the cache already they run at
// the speed of NumPy's loop.
//
// The result is NumPy's bit for bit: the same two elements are added or
// subtracted, in the same order of operands, by the same instruction
// (`fadd`, `fsub`), so NaN payloads, signed zeros and the floating-point
// flags NumPy turns into warnings are the same. A prefetch is a hint, and no
// prefetch names an address past the end of the output.

// Doubles in one 128-byte cache line of the Apple M1: one step of the loop.
#define INPLACE_LINE 16
// Doubles per NEON register.
#define INPLACE_LANES 2
// How far ahead of its loads the loop prefetches the output for a store, in
// doubles: 4,096 bytes. Chosen from 0 to 8,192 bytes after a threaded matrix
// product had read the output; 4,096 and 8,192 were the fastest.
#define INPLACE_AHEAD 512

// Defines `function(w, b, n)`, which sets w[i] = w[i] OP b[i] for i < n.
// `vector` is the NEON operation on two lanes, `scalar` the same on one,
// both taking w's element first, as NumPy's loop does.
#define HK_INPLACE(function, vector, scalar)                                                  \
    static void function(double *w, const double *b, size_t n) {                             \
        size_t i = 0;                                                                         \
        const size_t prefetched = n > INPLACE_AHEAD ? n - INPLACE_AHEAD : 0;                  \
        for (; i + INPLACE_LINE <= n; i += INPLACE_LINE) {                                   \
            if (i < prefetched) __builtin_prefetch(w + i + INPLACE_AHEAD, 1, 3);              \
            for (size_t j = 0; j < INPLACE_LINE; j += INPLACE_LANES)                         \
                vst1q_f64(w + i + j, vector(vld1q_f64(w + i + j), vld1q_f64(b + i + j)));     \
        }                                                                                     \
        for (; i < n; i++) vst1_f64(w + i, scalar(vld1_f64(w + i), vld1_f64(b + i)));         \
    }

HK_INPLACE(add_inplace_f64, vaddq_f64, vadd_f64)
HK_INPLACE(subtract_inplace_f64, vsubq_f64, vsub_f64)

// The hook's loop for `function`: taken when the output is the first input,
// all three operands are contiguous, and the second input is the output or
// shares no byte with it. NumPy copies an operand that partly overlaps the
// output before calling a loop, so the last test only guards against a
// caller that does not. A reduction steps its accumulator by zero, and goes
// to NumPy with every strided operand. Calls left to NumPy are not counted:
// every float64 add and subtract reaches this hook.
#define HK_INPLACE_LOOP(wrapper, index, function, kernel_id)                                  \
    static void wrapper(char **args, npy_intp const *dimensions, npy_intp const *steps,       \
                        void *data) {                                                        \
        const npy_intp n = dimensions[0];                                                     \
        const npy_intp size = (npy_intp)sizeof(double);                                       \
        if (n > 0 && args[BINARY_OUT] == args[BINARY_IN1] && steps[BINARY_IN1] == size &&     \
            steps[BINARY_IN2] == size && steps[BINARY_OUT] == size) {                         \
            const size_t bytes = (size_t)n * sizeof(double);                                  \
            if (args[BINARY_IN2] == args[BINARY_OUT] ||                                       \
                hk_disjoint(args[BINARY_IN2], bytes, args[BINARY_OUT], bytes)) {              \
                function((double *)args[BINARY_OUT], (const double *)args[BINARY_IN2],        \
                         (size_t)n);                                                          \
                hk_take(kernel_id);                                                           \
                return;                                                                       \
            }                                                                                 \
        }                                                                                     \
        loops[index].original(args, dimensions, steps, data);                                 \
    }

HK_INPLACE_LOOP(loop_add_f64, LOOP_ADD_F64, add_inplace_f64, HK_ADD_INPLACE_F64)
HK_INPLACE_LOOP(loop_subtract_f64, LOOP_SUBTRACT_F64, subtract_inplace_f64, HK_SUBTRACT_INPLACE_F64)

#define LOOP_ADD_F64_WRAPPER loop_add_f64
#define LOOP_SUBTRACT_F64_WRAPPER loop_subtract_f64
#else
// Not installed on other architectures: see docs/gradient-step-findings.md.
#define LOOP_ADD_F64_WRAPPER NULL
#define LOOP_SUBTRACT_F64_WRAPPER NULL
#endif

static hk_loop loops[LOOP_COUNT] = {
    [LOOP_CLIP_F32] = {"clip", CLIP_OPERANDS, NPY_FLOAT, loop_clip_f32, NULL},
    [LOOP_CLIP_F64] = {"clip", CLIP_OPERANDS, NPY_DOUBLE, loop_clip_f64, NULL},
    [LOOP_ADD_F64] = {"add", BINARY_OPERANDS, NPY_DOUBLE, LOOP_ADD_F64_WRAPPER, NULL},
    [LOOP_SUBTRACT_F64] = {"subtract", BINARY_OPERANDS, NPY_DOUBLE, LOOP_SUBTRACT_F64_WRAPPER, NULL},
};

// Return the ufunc numpy._core.umath.<name> as a new reference, or NULL with
// an exception set.
static PyObject *get_ufunc(const char *name) {
    PyObject *umath = PyImport_ImportModule("numpy._core.umath");
    if (umath == NULL) return NULL;
    PyObject *ufunc = PyObject_GetAttrString(umath, name);
    Py_DECREF(umath);
    return ufunc;
}

// Swap the loop of entry `index` to `replacement`, storing the loop that was
// there in `previous`. Returns 0, or -1 with an exception set.
static int swap_loop(int index, PyUFuncGenericFunction replacement, PyUFuncGenericFunction *previous) {
    const hk_loop *loop = &loops[index];
    PyObject *ufunc = get_ufunc(loop->ufunc);
    if (ufunc == NULL) return -1;
    int signature[MAX_OPERANDS];
    for (int i = 0; i < loop->operands; i++) signature[i] = loop->type_num;
    int status = PyUFunc_ReplaceLoopBySignature((PyUFuncObject *)ufunc, replacement, signature, previous);
    Py_DECREF(ufunc);
    if (status != 0) {
        if (!PyErr_Occurred()) PyErr_Format(PyExc_RuntimeError, "the %s ufunc has no loop for this type", loop->ufunc);
        return -1;
    }
    return 0;
}

// Point the loop at the binding. See hk_hook.apply.
static int loop_apply(int index) {
    if (loops[index].wrapper == NULL) {
        PyErr_Format(PyExc_RuntimeError, "the %s loop is not built for this architecture", loops[index].ufunc);
        return -1;
    }
    return swap_loop(index, loops[index].wrapper, &loops[index].original);
}

// Put NumPy's loop back. See hk_hook.remove. If something else replaced the
// loop after the binding did, its loop is put back and left in place.
static int loop_remove(int index) {
    PyUFuncGenericFunction found = NULL, ignored = NULL;
    if (swap_loop(index, loops[index].original, &found) != 0) {
        PyErr_Clear();
        return 0;
    }
    if (found == loops[index].wrapper) return 1;
    if (swap_loop(index, found, &ignored) != 0) PyErr_Clear();
    return 0;
}

hk_hook hk_loop_hooks[] = {
    {"loop.clip.f32", loop_apply, loop_remove, LOOP_CLIP_F32, false},
    {"loop.clip.f64", loop_apply, loop_remove, LOOP_CLIP_F64, false},
#if defined(__aarch64__)
    {"loop.add.f64", loop_apply, loop_remove, LOOP_ADD_F64, false},
    {"loop.subtract.f64", loop_apply, loop_remove, LOOP_SUBTRACT_F64, false},
#endif
    {NULL, NULL, NULL, 0, false},
};
