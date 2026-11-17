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
enum { M_ASTYPE, M_SUM, M_NONZERO, M_CUMSUM, M_SEARCHSORTED, M_COUNT };

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

// The type `np.float32` and its dtype object, which NumPy keeps one of.
// Fetched when the hook is applied.
static PyObject *float32_type;
static PyObject *float32_descr;

// True when `dtype`, the argument of astype, names native float32 in one of
// the usual spellings: the type `np.float32`, its dtype object, or the
// strings "float32" and "f4". The two objects are compared by identity,
// which is the cheapest test and comes first: astype is called with a type
// object far more often than with anything else.
static bool names_float32(PyObject *dtype) {
    if (dtype == float32_type || dtype == float32_descr) return true;
    if (PyUnicode_CheckExact(dtype)) {
        return PyUnicode_CompareWithASCIIString(dtype, "float32") == 0 ||
               PyUnicode_CompareWithASCIIString(dtype, "f4") == 0;
    }
    return false;
}

// Convert a contiguous uint8 array to float32. The result has the shape of
// the input.
static HK_NOINLINE PyObject *try_astype(PyObject *self) {
    PyArrayObject *x = (PyArrayObject *)self;
    PyObject *result = PyArray_SimpleNew(PyArray_NDIM(x), PyArray_DIMS(x), NPY_FLOAT);
    if (result == NULL) {
        PyErr_Clear();
        return NULL;
    }
    const npy_intp n = PyArray_SIZE(x);
    HK_BEGIN_NOGIL(n)
    cast_u8_f32(PyArray_DATA(x), PyArray_DATA((PyArrayObject *)result), (size_t)n);
    HK_END_NOGIL
    hk_take(HK_CAST_U8_F32);
    return result;
}

// x.astype(np.float32) for a contiguous uint8 array, with every other
// argument at its default.
static PyObject *method_astype(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    if (hk_is_flat(self, NPY_UINT8) && nargs == 1 && kwnames == NULL && names_float32(args[0])) {
        PyObject *result = try_astype(self);
        if (result != NULL) return result;
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
// holding one int64 array of the positions that are True. Shared with the
// function hook for np.where(mask).
//
// The kernel's output is scratch: it needs room for n indices and may write
// anywhere in it, whatever it reports. So it runs on a buffer of the
// binding's own, and the reported indices are copied into an array of
// exactly that size.
HK_NOINLINE PyObject *hk_try_nonzero(PyObject *self) {
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
        PyObject *result = hk_try_nonzero(self);
        if (result != NULL) return result;
    }
    hk_pass(HK_OP_NONZERO);
    return methods[M_NONZERO].original(self, args);
}

// ------------------------------------------------------------------ cumsum

// True when a cumsum call leaves `axis`, `dtype` and `out` at None, however
// it spells that: `x.cumsum()`, `x.cumsum(None)` or, as np.cumsum(x) does,
// `x.cumsum(axis=None, dtype=None, out=None)`.
static HK_NOINLINE bool cumsum_arguments_are_default(PyObject *args, PyObject *kwargs) {
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

// The kernel for `x.cumsum()` on this array: int64 always, float32 and
// float64 when install(inexact=True) was given. Returns HK_KERNEL_COUNT when
// there is none. This is tested before the arguments are looked at: most
// calls are for another element type, and rejecting those must cost almost
// nothing.
static hk_kernel cumsum_kernel(PyObject *self) {
    if (hk_is_flat(self, NPY_INT64)) return HK_CUMSUM_I64;
    if (hk_inexact && hk_is_flat(self, NPY_FLOAT)) return HK_CUMSUM_F32;
    if (hk_inexact && hk_is_flat(self, NPY_DOUBLE)) return HK_CUMSUM_F64;
    return HK_KERNEL_COUNT;
}

// Run a cumsum kernel. The result is one-dimensional, as NumPy's is when no
// axis is given.
//
// The float kernels regroup the additions and can raise floating-point
// exceptions NumPy's sequential sum does not, or the other way round (issue
// 9). NumPy turns those into warnings. So the flags are cleared before the
// kernel, and if any that NumPy reports is set afterwards the kernel's output
// is dropped and NumPy computes the result and issues its own warnings.
static HK_NOINLINE PyObject *try_cumsum(PyObject *self, hk_kernel kernel) {
    PyArrayObject *x = (PyArrayObject *)self;
    npy_intp n = PyArray_SIZE(x);
    if (n < 1) return NULL;
    PyObject *result = PyArray_SimpleNew(1, &n, PyArray_TYPE(x));
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
    const hk_kernel kernel = cumsum_kernel(self);
    if (kernel != HK_KERNEL_COUNT && cumsum_arguments_are_default(args, kwargs)) {
        PyObject *result = try_cumsum(self, kernel);
        if (result != NULL) return result;
    }
    hk_pass(HK_OP_CUMSUM);
    return ((PyCFunctionWithKeywords)(void (*)(void))methods[M_CUMSUM].original)(self, args, kwargs);
}

// ------------------------------------------------------------ searchsorted

// A search kernel: searchsorted_f64 for side='left', searchsorted_right_f64
// for side='right'. The two have one signature.
typedef void (*search_kernel)(const double *a, const double *v, int64_t *out, size_t n, size_t needles);

// Run the search for one Python number, as `a.searchsorted(2.5)`: NumPy
// returns the position as an np.intp scalar. `search` is the kernel for the
// side asked for. Returns NULL with no exception set if `needle` is not a
// Python float, or a Python int a double holds exactly.
static PyObject *searchsorted_scalar(PyArrayObject *a, PyObject *needle, search_kernel search) {
    double value;
    if (!hk_as_exact_double(needle, &value)) return NULL;
    npy_intp position;
    search(PyArray_DATA(a), &value, (int64_t *)&position, (size_t)PyArray_SIZE(a), 1);
    return hk_scalar(NPY_INTP, &position);
}

// Choose the kernel for the `side` argument of a search: absent or 'left'
// gives the left kernel, 'right' the right one. Sets *kernel to the counter
// the call is recorded under. Returns NULL for anything else, including a
// subclass of str, which is left for NumPy to judge.
static search_kernel search_side(PyObject *side, hk_kernel *kernel) {
    if (side == NULL || (PyUnicode_CheckExact(side) && hk_text_is(side, "left"))) {
        *kernel = HK_SEARCHSORTED_F64;
        return searchsorted_f64;
    }
    if (PyUnicode_CheckExact(side) && hk_text_is(side, "right")) {
        *kernel = HK_SEARCHSORTED_RIGHT_F64;
        return searchsorted_right_f64;
    }
    return NULL;
}

// a.searchsorted(v) for a one-dimensional contiguous float64 array `a` and
// float64 needles: a contiguous array of any shape, whose shape the result
// takes, or one Python number. `side` must be absent, 'left' or 'right'; each
// side has its own kernel. There must be no `sorter`.
// np.searchsorted(a, v) calls the method as
// `a.searchsorted(v, side='left', sorter=None)`, and np.digitize calls
// np.searchsorted: with side='right' by default, with side='left' for
// right=True, and on a reversed view, which is not contiguous and is passed
// on, when its bins decrease.
//
// An array that is not sorted is not detected. NumPy does not detect it
// either; both give a meaningless result, not the same one, and the kernel
// stays inside its buffers whatever the data.
static HK_NOINLINE PyObject *try_searchsorted(PyObject *self, PyObject *const *args, Py_ssize_t nargs,
                                              PyObject *kwnames) {
    enum { V, SIDE, SORTER, PARAMETERS };
    static const char *const names[PARAMETERS] = {"v", "side", "sorter"};
    PyObject *bound[PARAMETERS];
    if (!hk_bind_arguments(args, nargs, kwnames, names, PARAMETERS, PARAMETERS, bound)) return NULL;
    if (bound[V] == NULL || !hk_is_absent(bound[SORTER])) return NULL;
    hk_kernel kernel;
    const search_kernel search = search_side(bound[SIDE], &kernel);
    if (search == NULL) return NULL;

    PyArrayObject *a = (PyArrayObject *)self;
    PyObject *result;
    if (hk_is_flat(bound[V], NPY_DOUBLE)) {
        PyArrayObject *v = (PyArrayObject *)bound[V];
        result = PyArray_SimpleNew(PyArray_NDIM(v), PyArray_DIMS(v), NPY_INTP);
        if (result == NULL) {
            PyErr_Clear();
            return NULL;
        }
        const npy_intp needles = PyArray_SIZE(v);
        HK_BEGIN_NOGIL(needles)
        search(PyArray_DATA(a), PyArray_DATA(v), PyArray_DATA((PyArrayObject *)result),
               (size_t)PyArray_SIZE(a), (size_t)needles);
        HK_END_NOGIL
    } else {
        result = searchsorted_scalar(a, bound[V], search);
        if (result == NULL) return NULL;
    }
    hk_take(kernel);
    return result;
}

// The array's type is tested before the arguments are looked at: most calls
// are for another element type, and rejecting those must cost almost nothing.
static PyObject *method_searchsorted(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    if (hk_is_flat_1d(self, NPY_DOUBLE)) {
        PyObject *result = try_searchsorted(self, args, nargs, kwnames);
        if (result != NULL) return result;
    }
    hk_pass(HK_OP_SEARCHSORTED);
    return ((hk_fast_keywords)(void (*)(void))methods[M_SEARCHSORTED].original)(self, args, nargs, kwnames);
}

// ----------------------------------------------------------------- the hook

static hk_method methods[M_COUNT] = {
    [M_ASTYPE] = {"astype", METH_FASTCALL | METH_KEYWORDS, (PyCFunction)(void (*)(void))method_astype, NULL, NULL},
    [M_SUM] = {"sum", METH_FASTCALL | METH_KEYWORDS, (PyCFunction)(void (*)(void))method_sum, NULL, NULL},
    [M_NONZERO] = {"nonzero", METH_VARARGS, method_nonzero, NULL, NULL},
    [M_CUMSUM] = {"cumsum", METH_VARARGS | METH_KEYWORDS, (PyCFunction)(void (*)(void))method_cumsum, NULL, NULL},
    [M_SEARCHSORTED] = {"searchsorted", METH_FASTCALL | METH_KEYWORDS,
                        (PyCFunction)(void (*)(void))method_searchsorted, NULL, NULL},
};

// Point the method at the binding. See hk_hook.apply.
static int method_apply(int index) {
    if (float32_descr == NULL) {
        float32_type = (PyObject *)&PyFloatArrType_Type;
        float32_descr = (PyObject *)PyArray_DescrFromType(NPY_FLOAT);
    }
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
    {"method.searchsorted", method_apply, method_remove, M_SEARCHSORTED, false},
    {NULL, NULL, NULL, 0, false},
};
