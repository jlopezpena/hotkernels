// Inner-loop hook: a ufunc's loop for one element type, replaced through
// PyUFunc_ReplaceLoopBySignature, the interface NumPy provides for this.
//
// `np.clip(x, lo, hi)` and `x.clip(lo, hi)` both run the `clip` ufunc, whose
// loop receives four operands: x, lo, hi and the output. NumPy has settled
// the element type before the loop is called and describes the memory layout
// through `steps`, the distance in bytes between consecutive elements of each
// operand. A scalar bound has a step of zero.
#include "binding.h"

// Operands of the clip loop, in the order NumPy passes them.
enum { CLIP_X, CLIP_LO, CLIP_HI, CLIP_OUT, CLIP_OPERANDS };

// One hooked loop.
typedef struct {
    // NumPy type number of all four operands.
    int type_num;
    // The binding's loop.
    PyUFuncGenericFunction wrapper;
    // NumPy's own loop, kept while the hook is applied.
    PyUFuncGenericFunction original;
} hk_loop;

static hk_loop loops[2];

// The `clip` ufunc, numpy._core.umath.clip. Fetched on first use and kept.
static PyObject *clip_ufunc;

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

HK_CLIP_LOOP(loop_clip_f32, 0, clip_f32, HK_CLIP_F32, float)
HK_CLIP_LOOP(loop_clip_f64, 1, clip_f64, HK_CLIP_F64, double)

static hk_loop loops[2] = {
    {NPY_FLOAT, loop_clip_f32, NULL},
    {NPY_DOUBLE, loop_clip_f64, NULL},
};

// Return the clip ufunc as a borrowed reference, or NULL with an exception set.
static PyObject *get_clip_ufunc(void) {
    if (clip_ufunc == NULL) {
        PyObject *umath = PyImport_ImportModule("numpy._core.umath");
        if (umath == NULL) return NULL;
        clip_ufunc = PyObject_GetAttrString(umath, "clip");
        Py_DECREF(umath);
    }
    return clip_ufunc;
}

// Swap the loop for `type_num` to `replacement`, storing the loop that was
// there in `previous`. Returns 0, or -1 with an exception set.
static int swap_loop(int type_num, PyUFuncGenericFunction replacement,
                     PyUFuncGenericFunction *previous) {
    PyObject *ufunc = get_clip_ufunc();
    if (ufunc == NULL) return -1;
    int signature[CLIP_OPERANDS] = {type_num, type_num, type_num, type_num};
    if (PyUFunc_ReplaceLoopBySignature((PyUFuncObject *)ufunc, replacement, signature, previous) != 0) {
        if (!PyErr_Occurred()) PyErr_SetString(PyExc_RuntimeError, "the clip ufunc has no loop for this type");
        return -1;
    }
    return 0;
}

// Point the loop at the binding. See hk_hook.apply.
static int loop_apply(int index) {
    return swap_loop(loops[index].type_num, loops[index].wrapper, &loops[index].original);
}

// Put NumPy's loop back. See hk_hook.remove. If something else replaced the
// loop after the binding did, its loop is put back and left in place.
static int loop_remove(int index) {
    PyUFuncGenericFunction found = NULL, ignored = NULL;
    if (swap_loop(loops[index].type_num, loops[index].original, &found) != 0) {
        PyErr_Clear();
        return 0;
    }
    if (found == loops[index].wrapper) return 1;
    if (swap_loop(loops[index].type_num, found, &ignored) != 0) PyErr_Clear();
    return 0;
}

hk_hook hk_loop_hooks[] = {
    {"loop.clip.f32", loop_apply, loop_remove, 0, false},
    {"loop.clip.f64", loop_apply, loop_remove, 1, false},
    {NULL, NULL, NULL, 0, false},
};
