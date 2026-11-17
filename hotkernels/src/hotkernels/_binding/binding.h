// Shared declarations of the hotkernels binding: the code between NumPy and
// the kernels. See docs/delivery-design.md.
//
// The binding is one extension module built from several files. NumPy's C API
// is a table of pointers that each file reaches through a symbol, so every
// file must name the same symbol and only module.c may define and fill it.
#ifndef HOTKERNELS_BINDING_H
#define HOTKERNELS_BINDING_H

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_2_0_API_VERSION
#define NPY_TARGET_VERSION NPY_2_0_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL hotkernels_ARRAY_API
#define PY_UFUNC_UNIQUE_SYMBOL hotkernels_UFUNC_API
#ifndef HK_DEFINE_NUMPY_API
#define NO_IMPORT_ARRAY
#define NO_IMPORT_UFUNC
#endif
#include <numpy/arrayobject.h>
#include <numpy/arrayscalars.h>
#include <numpy/ufuncobject.h>

#include <fenv.h>
#include <math.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include "kernels.h"

// ---------------------------------------------------------------- counters

// One entry per kernel the binding can call. The order is the order of
// HK_KERNEL_NAMES in module.c.
typedef enum {
    HK_ARGMAX_F32, HK_ARGMAX_F64, HK_ARGMIN_F32, HK_ARGMIN_F64,
    HK_CLIP_F32, HK_CLIP_F64,
    HK_COMPRESS_F32, HK_COMPRESS_F64, HK_MASKFILL_F64, HK_NONZERO_BOOL,
    HK_CUMSUM_I64, HK_CUMSUM_F32, HK_CUMSUM_F64,
    HK_CAST_U8_F32, HK_SUM_U8, HK_COUNT_NONZERO_I64, HK_COUNT_NONZERO_F64,
    HK_HISTOGRAM_F64, HK_BINCOUNT_U8, HK_ISIN_I64,
    HK_KERNEL_COUNT
} hk_kernel;

// One entry per NumPy operation that has a hook. A call that reaches a hook
// and does not meet a kernel's conditions is counted here. The order is the
// order of HK_OPERATION_NAMES in module.c.
typedef enum {
    HK_OP_ARGMAX, HK_OP_ARGMIN, HK_OP_CLIP, HK_OP_COMPRESS, HK_OP_MASKFILL,
    HK_OP_NONZERO, HK_OP_CUMSUM, HK_OP_ASTYPE, HK_OP_SUM, HK_OP_COUNT_NONZERO,
    HK_OP_HISTOGRAM, HK_OP_BINCOUNT, HK_OP_ISIN,
    HK_OPERATION_COUNT
} hk_operation;

// Calls a kernel took, and calls passed on to NumPy. Atomic because NumPy
// runs inner loops with the interpreter lock released, so two threads can be
// inside a hook at once. Relaxed ordering: these are statistics.
extern _Atomic uint64_t hk_taken[HK_KERNEL_COUNT];
extern _Atomic uint64_t hk_passed[HK_OPERATION_COUNT];

// Record that kernel `k` handled a call.
static inline void hk_take(hk_kernel k) {
    atomic_fetch_add_explicit(&hk_taken[k], 1, memory_order_relaxed);
}

// Record that a call to operation `op` reached a hook and went to NumPy.
static inline void hk_pass(hk_operation op) {
    atomic_fetch_add_explicit(&hk_passed[op], 1, memory_order_relaxed);
}

// True when install(inexact=True) was given: kernels that are not
// bit-identical to NumPy may then take calls.
extern bool hk_inexact;

// -------------------------------------------------------------- the checks

// The largest integer magnitude a double represents exactly, 2^53. A Python
// int inside it converts to the double NumPy would also produce.
#define HK_EXACT_INT_LIMIT 9007199254740992.0

// Element count from which a hook releases the interpreter lock around a
// kernel. Below it, taking the lock back costs more than other threads gain.
#define HK_NOGIL_MIN_N 8192

// Largest scratch buffer, in bytes, a hook keeps on the stack. Above it the
// buffer comes from malloc.
#define HK_STACK_SCRATCH_BYTES 2048

// True when `obj` is exactly an ndarray, not a subclass, holding elements of
// NumPy type `type_num` in native byte order as one C-contiguous run with at
// least one dimension. A kernel can then treat its data as a flat buffer of
// PyArray_SIZE elements. Alignment is not required: the kernels accept
// unaligned buffers.
static inline bool hk_is_flat(PyObject *obj, int type_num) {
    if (!PyArray_CheckExact(obj)) return false;
    PyArrayObject *a = (PyArrayObject *)obj;
    return PyArray_TYPE(a) == type_num && PyArray_ISNOTSWAPPED(a) &&
           PyArray_NDIM(a) >= 1 && PyArray_IS_C_CONTIGUOUS(a);
}

// As hk_is_flat, and one-dimensional.
static inline bool hk_is_flat_1d(PyObject *obj, int type_num) {
    return hk_is_flat(obj, type_num) && PyArray_NDIM((PyArrayObject *)obj) == 1;
}

// True when the `a_bytes` bytes at `a` and the `b_bytes` bytes at `b` share
// no byte.
static inline bool hk_disjoint(const char *a, size_t a_bytes, const char *b, size_t b_bytes) {
    return a + a_bytes <= b || b + b_bytes <= a;
}

// True when every one of the `n` bytes at `mask` is 0 or 1. NumPy treats any
// non-zero byte of a boolean array as True; the mask kernels use the byte's
// value as a count and write outside their buffers when it is above 1, so a
// mask is read once here before a kernel sees it. Bytes of 0 and 1 OR
// together to at most 1.
static inline bool hk_mask_is_clean(const uint8_t *mask, size_t n) {
    uint8_t seen = 0;
    for (size_t i = 0; i < n; i++) seen |= mask[i];
    return seen <= 1;
}

// Convert a Python float, or a Python int a double holds exactly, to a
// double. Returns false, with no exception set, for anything else, including
// bool and NumPy scalars that are not float subclasses.
static inline bool hk_as_double(PyObject *obj, double *out) {
    if (PyFloat_CheckExact(obj)) {
        *out = PyFloat_AS_DOUBLE(obj);
        return true;
    }
    if (PyLong_CheckExact(obj)) {
        double v = PyLong_AsDouble(obj);
        if (v == -1.0 && PyErr_Occurred()) {
            PyErr_Clear();
            return false;
        }
        if (v < -HK_EXACT_INT_LIMIT || v > HK_EXACT_INT_LIMIT) return false;
        *out = v;
        return true;
    }
    return false;
}

// Release the interpreter lock for the block that follows when `n` elements
// are enough to be worth it. HK_END_NOGIL takes it back.
#define HK_BEGIN_NOGIL(n) \
    PyThreadState *hk_saved_thread = (n) >= HK_NOGIL_MIN_N ? PyEval_SaveThread() : NULL;
#define HK_END_NOGIL \
    if (hk_saved_thread) PyEval_RestoreThread(hk_saved_thread);

// ----------------------------------------------- operations shared by hooks

// Each returns a new reference when a kernel took the call, and NULL with no
// exception set when the call does not meet the kernel's conditions and
// belongs to NumPy. None of them raises.
PyObject *hk_try_sum(PyObject *array);
PyObject *hk_try_argext(PyObject *array, bool want_max);

// Build a NumPy scalar of type `type_num` from the value at `value`. Returns
// NULL with no exception set if it cannot be built.
PyObject *hk_scalar(int type_num, void *value);

// ------------------------------------------------------------------- hooks

// One hook: one pointer inside NumPy that can be pointed at the binding.
typedef struct {
    // Name used from Python, for example "slot.argmax.f32".
    const char *name;
    // Point NumPy at the binding. Takes `index`. Returns 0, or -1 with an
    // exception set, in which case nothing was changed.
    int (*apply)(int index);
    // Put the original pointer back. Takes `index`. Returns 1 when it did, and
    // 0 when the pointer no longer held the binding's function and was left
    // alone.
    int (*remove)(int index);
    // Which entry of its kind's own table this hook is.
    int index;
    // Whether the hook is in place now.
    bool applied;
} hk_hook;

// The hooks of each kind, each array ended by an entry with a NULL name.
extern hk_hook hk_slot_hooks[];
extern hk_hook hk_loop_hooks[];
extern hk_hook hk_function_hooks[];
extern hk_hook hk_method_hooks[];
extern hk_hook hk_index_hooks[];

#endif
