// Indexing hook: the two function pointers behind `a[i]` and `a[i] = v`.
//
// CPython runs `a[i]` by calling a pointer in the type of `a`
// (tp_as_mapping->mp_subscript), and `a[i] = v` by calling its neighbour
// (mp_ass_subscript). For ndarray both point into NumPy. They are the only
// way to reach `x[mask]` and `x[mask] = v`, which are neither functions nor
// methods.
//
// This hook sits on every index operation on every plain ndarray in the
// process, so the common case, an index that is not a boolean array, must
// leave after one comparison. Subclasses of ndarray defined in Python do not
// come through here: CPython gives them their own pointers.
#include "binding.h"

// NumPy's own functions, kept while the hooks are applied.
static binaryfunc original_get;
static objobjargproc original_set;

// True when `index` is a plain boolean ndarray. Tested first, on every index
// operation: for an integer or a slice it fails on the first comparison.
static inline bool is_mask(PyObject *index) {
    return PyArray_CheckExact(index) && PyArray_TYPE((PyArrayObject *)index) == NPY_BOOL;
}

// x[mask] for one-dimensional contiguous float32 or float64 `x` and a
// boolean mask of the same length. As with nonzero, the kernel's output is
// scratch, so it runs on a buffer of the binding's own and the selected
// elements are copied into an array of exactly their number.
static PyObject *try_compress(PyObject *self, PyObject *index) {
    hk_kernel kernel;
    int type_num;
    if (hk_is_flat_1d(self, NPY_DOUBLE)) kernel = HK_COMPRESS_F64, type_num = NPY_DOUBLE;
    else if (hk_is_flat_1d(self, NPY_FLOAT)) kernel = HK_COMPRESS_F32, type_num = NPY_FLOAT;
    else return NULL;
    if (!hk_is_flat_1d(index, NPY_BOOL)) return NULL;
    PyArrayObject *x = (PyArrayObject *)self, *mask = (PyArrayObject *)index;
    const npy_intp n = PyArray_SIZE(x);
    if (n != PyArray_SIZE(mask)) return NULL;

    const size_t element = (size_t)PyArray_ITEMSIZE(x);
    const size_t bytes = (size_t)n * element;
    double on_stack[HK_STACK_SCRATCH_BYTES / sizeof(double)];
    void *scratch = bytes <= sizeof on_stack ? (void *)on_stack : malloc(bytes);
    if (scratch == NULL) return NULL;

    npy_intp count = -1;
    {
        HK_BEGIN_NOGIL(n)
        if (hk_mask_is_clean(PyArray_DATA(mask), (size_t)n)) {
            count = kernel == HK_COMPRESS_F64
                ? (npy_intp)compress_f64(PyArray_DATA(x), PyArray_DATA(mask), scratch, (size_t)n)
                : (npy_intp)compress_f32(PyArray_DATA(x), PyArray_DATA(mask), scratch, (size_t)n);
        }
        HK_END_NOGIL
    }
    PyObject *result = count >= 0 ? PyArray_SimpleNew(1, &count, type_num) : NULL;
    if (result != NULL) memcpy(PyArray_DATA((PyArrayObject *)result), scratch, (size_t)count * element);
    if (scratch != (void *)on_stack) free(scratch);
    if (result == NULL) {
        PyErr_Clear();
        return NULL;
    }
    hk_take(kernel);
    return result;
}

// The hooked `a[i]`.
static PyObject *index_get(PyObject *self, PyObject *index) {
    if (is_mask(index) && PyArray_CheckExact(self)) {
        PyObject *result = try_compress(self, index);
        if (result != NULL) return result;
        hk_pass(HK_OP_COMPRESS);
    }
    return original_get(self, index);
}

// x[mask] = v for one-dimensional contiguous writable float64 `x`, a boolean
// mask of the same length, and a Python float or int.
//
// The kernel rewrites elements the mask does not select with their own bits.
// The interpreter lock stays held throughout, so no other Python thread can
// write one of them in between and have its write undone.
static bool try_maskfill(PyObject *self, PyObject *index, PyObject *value) {
    double fill;
    if (!hk_is_flat_1d(self, NPY_DOUBLE) || !hk_is_flat_1d(index, NPY_BOOL)) return false;
    PyArrayObject *x = (PyArrayObject *)self, *mask = (PyArrayObject *)index;
    const npy_intp n = PyArray_SIZE(x);
    if (n != PyArray_SIZE(mask) || !PyArray_ISWRITEABLE(x) || !hk_as_double(value, &fill)) return false;
    // A boolean view of x's own memory would be read while it is written.
    if (!hk_disjoint(PyArray_DATA(x), (size_t)n * sizeof(double), PyArray_DATA(mask), (size_t)n)) return false;
    if (!hk_mask_is_clean(PyArray_DATA(mask), (size_t)n)) return false;
    maskfill_f64(PyArray_DATA(x), PyArray_DATA(mask), (size_t)n, fill);
    hk_take(HK_MASKFILL_F64);
    return true;
}

// The hooked `a[i] = v`. A NULL value is `del a[i]`, which is NumPy's to refuse.
static int index_set(PyObject *self, PyObject *index, PyObject *value) {
    if (value != NULL && is_mask(index) && PyArray_CheckExact(self)) {
        if (try_maskfill(self, index, value)) return 0;
        hk_pass(HK_OP_MASKFILL);
    }
    return original_set(self, index, value);
}

// Point `a[i]` (index 0) or `a[i] = v` (index 1) at the binding. See
// hk_hook.apply. PyType_Modified tells CPython to drop anything it cached
// about the type.
static int index_apply(int index) {
    PyMappingMethods *mapping = PyArray_Type.tp_as_mapping;
    if (mapping == NULL || mapping->mp_subscript == NULL || mapping->mp_ass_subscript == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "ndarray has no indexing functions to hook");
        return -1;
    }
    if (index == 0) {
        original_get = mapping->mp_subscript;
        mapping->mp_subscript = index_get;
    } else {
        original_set = mapping->mp_ass_subscript;
        mapping->mp_ass_subscript = index_set;
    }
    PyType_Modified(&PyArray_Type);
    return 0;
}

// Put NumPy's function back. See hk_hook.remove.
static int index_remove(int index) {
    PyMappingMethods *mapping = PyArray_Type.tp_as_mapping;
    if (index == 0) {
        if (mapping->mp_subscript != index_get) return 0;
        mapping->mp_subscript = original_get;
    } else {
        if (mapping->mp_ass_subscript != index_set) return 0;
        mapping->mp_ass_subscript = original_set;
    }
    PyType_Modified(&PyArray_Type);
    return 1;
}

hk_hook hk_index_hooks[] = {
    {"index.get", index_apply, index_remove, 0, false},
    {"index.set", index_apply, index_remove, 1, false},
    {NULL, NULL, NULL, 0, false},
};
