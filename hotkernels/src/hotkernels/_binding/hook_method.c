// Method hook: the C function pointer in ndarray's method table.
//
// Each method of ndarray is described by a PyMethodDef in a table inside
// NumPy, and every way of calling the method, including a bound method saved
// earlier, reads the function pointer from that entry at the time of the
// call. Changing the pointer hooks the method and leaves `np.ndarray.astype`
// the same object. NumPy's own functions of the same name call the method, so
// `np.nonzero(x)` and `np.cumsum(x)` arrive here too. `np.sum(x)` does not:
// it has its own function hook.
//
// A method is called in one of several conventions, recorded in the entry's
// flags. A wrapper must have the shape the flags say, so a hook is applied
// only when the flags are the ones its wrapper was written for.
#include "binding.h"

// The methods that are hooked, as indices into `methods`.
enum { M_ASTYPE, M_SUM, M_NONZERO, M_CUMSUM, M_COUNT };

// A method called with an argument array and keyword names
// (METH_FASTCALL | METH_KEYWORDS). CPython names this type differently
// across versions.
typedef PyObject *(*hk_fast_keywords)(PyObject *self, PyObject *const *args, Py_ssize_t nargs,
                                      PyObject *kwnames);

// One hooked method.
typedef struct {
    // Method name on ndarray.
    const char *name;
    // Calling convention the wrapper implements, as PyMethodDef.ml_flags.
    int flags;
    // The binding's function, stored as the generic pointer type.
    PyCFunction wrapper;
    // NumPy's table entry, set while the hook is applied.
    PyMethodDef *definition;
    // NumPy's own function, kept while the hook is applied.
    PyCFunction original;
} hk_method;

static hk_method methods[M_COUNT];

// The names of cumsum's parameters, in order.
static const char *const CUMSUM_PARAMETERS[] = {"axis", "dtype", "out"};
#define CUMSUM_PARAMETER_COUNT 3

// Floating-point exceptions NumPy can turn into a warning or an error.
#define HK_REPORTED_EXCEPTIONS (FE_INVALID | FE_OVERFLOW | FE_DIVBYZERO | FE_UNDERFLOW)

// ------------------------------------------------------------------ astype

// True when `dtype`, the argument of astype, names native float32 in one of
// the usual spellings: the type `np.float32`, its dtype object, or the
// strings "float32" and "f4".
static bool names_float32(PyObject *dtype) {
    if (dtype == (PyObject *)&PyFloatArrType_Type) return true;
    if (PyArray_DescrCheck(dtype)) {
        PyArray_Descr *float32 = PyArray_DescrFromType(NPY_FLOAT);
        const bool same = dtype == (PyObject *)float32;
        Py_DECREF(float32);
        return same;
    }
    if (PyUnicode_CheckExact(dtype)) {
        return PyUnicode_CompareWithASCIIString(dtype, "float32") == 0 ||
               PyUnicode_CompareWithASCIIString(dtype, "f4") == 0;
    }
    return false;
}

// x.astype(np.float32) for a contiguous uint8 array, with every other
// argument at its default. The result has the shape of the input.
static PyObject *method_astype(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    if (nargs == 1 && kwnames == NULL && hk_is_flat(self, NPY_UINT8) && names_float32(args[0])) {
        PyArrayObject *x = (PyArrayObject *)self;
        PyObject *result = PyArray_SimpleNew(PyArray_NDIM(x), PyArray_DIMS(x), NPY_FLOAT);
        if (result != NULL) {
            const npy_intp n = PyArray_SIZE(x);
            HK_BEGIN_NOGIL(n)
            cast_u8_f32(PyArray_DATA(x), PyArray_DATA((PyArrayObject *)result), (size_t)n);
            HK_END_NOGIL
            hk_take(HK_CAST_U8_F32);
            return result;
        }
        PyErr_Clear();
    }
    hk_pass(HK_OP_ASTYPE);
    return ((hk_fast_keywords)(void (*)(void))methods[M_ASTYPE].original)(self, args, nargs, kwnames);
}

// --------------------------------------------------------------------- sum

// x.sum() with no arguments.
static PyObject *method_sum(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    if (nargs == 0 && kwnames == NULL) {
        PyObject *result = hk_try_sum(self);
        if (result != NULL) return result;
    }
    hk_pass(HK_OP_SUM);
    return ((hk_fast_keywords)(void (*)(void))methods[M_SUM].original)(self, args, nargs, kwnames);
}

// ----------------------------------------------------------------- nonzero

// mask.nonzero() for a contiguous one-dimensional boolean array: a tuple
// holding one int64 array of the positions that are True.
//
// The kernel's output is scratch: it needs room for n indices and may write
// anywhere in it, whatever it reports. So it runs on a buffer of the
// binding's own, and the reported indices are copied into an array of
// exactly that size.
static PyObject *try_nonzero(PyObject *self) {
    if (!hk_is_flat_1d(self, NPY_BOOL)) return NULL;
    PyArrayObject *mask = (PyArrayObject *)self;
    const npy_intp n = PyArray_SIZE(mask);
    const size_t bytes = (size_t)n * sizeof(int64_t);
    int64_t on_stack[HK_STACK_SCRATCH_BYTES / sizeof(int64_t)];
    int64_t *scratch = bytes <= sizeof on_stack ? on_stack : malloc(bytes);
    if (scratch == NULL) return NULL;

    npy_intp count = -1;
    {
        HK_BEGIN_NOGIL(n)
        if (hk_mask_is_clean(PyArray_DATA(mask), (size_t)n)) {
            count = (npy_intp)nonzero_bool(PyArray_DATA(mask), scratch, (size_t)n);
        }
        HK_END_NOGIL
    }
    PyObject *indices = count >= 0 ? PyArray_SimpleNew(1, &count, NPY_INTP) : NULL;
    PyObject *result = NULL;
    if (indices != NULL) {
        memcpy(PyArray_DATA((PyArrayObject *)indices), scratch, (size_t)count * sizeof(int64_t));
        result = PyTuple_Pack(1, indices);
        Py_DECREF(indices);
    }
    if (scratch != on_stack) free(scratch);
    if (result == NULL) {
        PyErr_Clear();
        return NULL;
    }
    hk_take(HK_NONZERO_BOOL);
    return result;
}

// Called with a tuple of positional arguments (METH_VARARGS).
static PyObject *method_nonzero(PyObject *self, PyObject *args) {
    if (PyTuple_GET_SIZE(args) == 0) {
        PyObject *result = try_nonzero(self);
        if (result != NULL) return result;
    }
    hk_pass(HK_OP_NONZERO);
    return methods[M_NONZERO].original(self, args);
}

// ------------------------------------------------------------------ cumsum

// True when a cumsum call leaves `axis`, `dtype` and `out` at None, however
// it spells that: `x.cumsum()`, `x.cumsum(None)` or, as np.cumsum(x) does,
// `x.cumsum(axis=None, dtype=None, out=None)`.
static bool cumsum_arguments_are_default(PyObject *args, PyObject *kwargs) {
    const Py_ssize_t positional = PyTuple_GET_SIZE(args);
    if (positional > CUMSUM_PARAMETER_COUNT) return false;
    for (Py_ssize_t i = 0; i < positional; i++) {
        if (PyTuple_GET_ITEM(args, i) != Py_None) return false;
    }
    if (kwargs == NULL) return true;
    PyObject *key, *value;
    Py_ssize_t position = 0;
    while (PyDict_Next(kwargs, &position, &key, &value)) {
        if (value != Py_None) return false;
        // Only a parameter that was not also given by position.
        Py_ssize_t i = positional;
        while (i < CUMSUM_PARAMETER_COUNT && PyUnicode_CompareWithASCIIString(key, CUMSUM_PARAMETERS[i]) != 0) i++;
        if (i == CUMSUM_PARAMETER_COUNT) return false;
    }
    return true;
}

// x.cumsum() for a contiguous int64 array, and for float32 and float64 when
// install(inexact=True) was given. The result is one-dimensional, as NumPy's
// is when no axis is given.
//
// The float kernels regroup the additions and can raise floating-point
// exceptions NumPy's sequential sum does not, or the other way round (issue
// 9). NumPy turns those into warnings. So the flags are cleared before the
// kernel, and if any that NumPy reports is set afterwards the kernel's output
// is dropped and NumPy computes the result and issues its own warnings.
static PyObject *try_cumsum(PyObject *self) {
    hk_kernel kernel;
    int type_num;
    if (hk_is_flat(self, NPY_INT64)) kernel = HK_CUMSUM_I64, type_num = NPY_INT64;
    else if (hk_inexact && hk_is_flat(self, NPY_FLOAT)) kernel = HK_CUMSUM_F32, type_num = NPY_FLOAT;
    else if (hk_inexact && hk_is_flat(self, NPY_DOUBLE)) kernel = HK_CUMSUM_F64, type_num = NPY_DOUBLE;
    else return NULL;
    PyArrayObject *x = (PyArrayObject *)self;
    npy_intp n = PyArray_SIZE(x);
    if (n < 1) return NULL;
    PyObject *result = PyArray_SimpleNew(1, &n, type_num);
    if (result == NULL) {
        PyErr_Clear();
        return NULL;
    }
    void *out = PyArray_DATA((PyArrayObject *)result);
    bool raised = false;
    {
        HK_BEGIN_NOGIL(n)
        if (kernel == HK_CUMSUM_I64) {
            cumsum_i64(PyArray_DATA(x), out, (size_t)n);
        } else {
            feclearexcept(FE_ALL_EXCEPT);
            if (kernel == HK_CUMSUM_F32) cumsum_f32(PyArray_DATA(x), out, (size_t)n);
            else cumsum_f64(PyArray_DATA(x), out, (size_t)n);
            raised = fetestexcept(HK_REPORTED_EXCEPTIONS) != 0;
            feclearexcept(FE_ALL_EXCEPT);
        }
        HK_END_NOGIL
    }
    if (raised) {
        Py_DECREF(result);
        return NULL;
    }
    hk_take(kernel);
    return result;
}

// Called with a tuple and a dictionary (METH_VARARGS | METH_KEYWORDS).
static PyObject *method_cumsum(PyObject *self, PyObject *args, PyObject *kwargs) {
    if (cumsum_arguments_are_default(args, kwargs)) {
        PyObject *result = try_cumsum(self);
        if (result != NULL) return result;
    }
    hk_pass(HK_OP_CUMSUM);
    return ((PyCFunctionWithKeywords)(void (*)(void))methods[M_CUMSUM].original)(self, args, kwargs);
}

// ----------------------------------------------------------------- the hook

static hk_method methods[M_COUNT] = {
    [M_ASTYPE] = {"astype", METH_FASTCALL | METH_KEYWORDS, (PyCFunction)(void (*)(void))method_astype, NULL, NULL},
    [M_SUM] = {"sum", METH_FASTCALL | METH_KEYWORDS, (PyCFunction)(void (*)(void))method_sum, NULL, NULL},
    [M_NONZERO] = {"nonzero", METH_VARARGS, method_nonzero, NULL, NULL},
    [M_CUMSUM] = {"cumsum", METH_VARARGS | METH_KEYWORDS, (PyCFunction)(void (*)(void))method_cumsum, NULL, NULL},
};

// Point the method at the binding. See hk_hook.apply.
static int method_apply(int index) {
    PyObject *descriptor = PyObject_GetAttrString((PyObject *)&PyArray_Type, methods[index].name);
    if (descriptor == NULL) return -1;
    if (!Py_IS_TYPE(descriptor, &PyMethodDescr_Type)) {
        PyErr_Format(PyExc_TypeError, "ndarray.%s is not a built-in method", methods[index].name);
        Py_DECREF(descriptor);
        return -1;
    }
    // The table entry belongs to NumPy and outlives the descriptor object.
    PyMethodDef *definition = ((PyMethodDescrObject *)descriptor)->d_method;
    Py_DECREF(descriptor);
    if (definition->ml_flags != methods[index].flags) {
        PyErr_Format(PyExc_TypeError, "ndarray.%s is called with convention %d, and the hook was written for %d",
                     methods[index].name, definition->ml_flags, methods[index].flags);
        return -1;
    }
    methods[index].definition = definition;
    methods[index].original = definition->ml_meth;
    definition->ml_meth = methods[index].wrapper;
    return 0;
}

// Put NumPy's function back. See hk_hook.remove.
static int method_remove(int index) {
    PyMethodDef *definition = methods[index].definition;
    if (definition == NULL || definition->ml_meth != methods[index].wrapper) return 0;
    definition->ml_meth = methods[index].original;
    methods[index].definition = NULL;
    return 1;
}

hk_hook hk_method_hooks[] = {
    {"method.astype", method_apply, method_remove, M_ASTYPE, false},
    {"method.sum", method_apply, method_remove, M_SUM, false},
    {"method.nonzero", method_apply, method_remove, M_NONZERO, false},
    {"method.cumsum", method_apply, method_remove, M_CUMSUM, false},
    {NULL, NULL, NULL, 0, false},
};
