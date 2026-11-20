// Bottleneck hook: the call pointer inside bottleneck's own function objects.
//
// bottleneck's `nanmean`, `nanstd`, `nanmin` and `nanmax` are C functions of
// the module `bottleneck.reduce`, objects of CPython's type
// `builtin_function_or_method`. Libraries hold on to these objects: empyrical
// wraps them in closures when it is imported, pandas' nanops looks them up
// once. CPython calls such an object through a pointer inside it, at the
// offset its type publishes as `tp_vectorcall_offset`, as it calls NumPy's
// functions; this is the function hook of hook_function.c on an object of
// another library. bottleneck defines the four with METH_VARARGS |
// METH_KEYWORDS, for which the pointer is NULL and CPython calls the type's
// `tp_call` with a tuple and a dictionary. Setting the pointer sends every
// call of the object to the wrapper below, however the object was found.
//
// A call that takes no kernel goes on to the type's `tp_call` with the
// caller's arguments, which is what CPython does for it without the hook, so
// bottleneck parses them, converts them and raises its own errors.
//
// The hook is applied only when bottleneck is already imported: hotkernels
// never imports it. See docs/bottleneck-design.md.
#include "binding.h"

// The arguments of a vectorcall: the callable, the positional arguments
// followed by the keyword values, the positional count with a flag bit, and
// the keyword names.
#define HK_VECTORCALL_PARAMETERS \
    PyObject *callable, PyObject *const *args, size_t nargsf, PyObject *kwnames

// The module that defines the functions. Looked up in sys.modules, never imported.
#define BOTTLENECK_REDUCE "bottleneck.reduce"
// The calling convention bottleneck 1.3 to 1.6 define the four functions
// with. Another one means a bottleneck this hook was not written for.
#define BOTTLENECK_FLAGS (METH_VARARGS | METH_KEYWORDS)
// Bytes of one float64.
#define F64_BYTES 8

// The hooked functions, as indices into `functions`.
enum { B_NANMEAN, B_NANSTD, B_NANMIN, B_NANMAX, B_COUNT };

// A kernel of the column form: reduce each column of the matrix whose element
// (i, j) is a[i * step + j]. See kernels/nanmean_cols_f64/spec.py.
typedef size_t (*cols_kernel)(const double *a, double *out, size_t n, size_t cols, size_t step);

// One hooked function.
typedef struct {
    // Attribute of `bottleneck.reduce` that holds the function object.
    const char *attribute;
    // The binding's call function.
    vectorcallfunc wrapper;
    // The function object, kept alive while the hook is applied.
    PyObject *object;
} bn_function;

static PyObject *call_nanmean(HK_VECTORCALL_PARAMETERS);
static PyObject *call_nanstd(HK_VECTORCALL_PARAMETERS);
static PyObject *call_nanmin(HK_VECTORCALL_PARAMETERS);
static PyObject *call_nanmax(HK_VECTORCALL_PARAMETERS);

static bn_function functions[B_COUNT] = {
    [B_NANMEAN] = {"nanmean", call_nanmean, NULL},
    [B_NANSTD] = {"nanstd", call_nanstd, NULL},
    [B_NANMIN] = {"nanmin", call_nanmin, NULL},
    [B_NANMAX] = {"nanmax", call_nanmax, NULL},
};

// ------------------------------------------------------------- passing on

// Call the function as CPython calls it without the hook: its type's
// `tp_call`, with the positional arguments as a tuple and the keywords as a
// dictionary. `tp_call` does not go through the call pointer, so it does not
// come back here. Returns what bottleneck returns, or NULL with its exception.
static PyObject *pass_on(HK_VECTORCALL_PARAMETERS) {
    const Py_ssize_t nargs = PyVectorcall_NARGS(nargsf);
    PyObject *positional = PyTuple_New(nargs);
    if (positional == NULL) return NULL;
    for (Py_ssize_t i = 0; i < nargs; i++) {
        Py_INCREF(args[i]);
        PyTuple_SET_ITEM(positional, i, args[i]);
    }
    PyObject *keywords = NULL;
    const Py_ssize_t count = kwnames ? PyTuple_GET_SIZE(kwnames) : 0;
    if (count > 0) {
        keywords = PyDict_New();
        for (Py_ssize_t k = 0; keywords != NULL && k < count; k++) {
            if (PyDict_SetItem(keywords, PyTuple_GET_ITEM(kwnames, k), args[nargs + k]) != 0) Py_CLEAR(keywords);
        }
        if (keywords == NULL) {
            Py_DECREF(positional);
            return NULL;
        }
    }
    PyObject *result = Py_TYPE(callable)->tp_call(callable, positional, keywords);
    Py_DECREF(positional);
    Py_XDECREF(keywords);
    return result;
}

// --------------------------------------------------------- taking a kernel

// The matrix a column kernel reduces, read from a call's array and axis.
typedef struct {
    // First element.
    const double *data;
    // Elements the kernel may read: (rows - 1) * step + cols.
    size_t n;
    // Length of the kept axis: one result each.
    size_t cols;
    // Elements from one step of the reduced axis to the next.
    size_t step;
    // Length of the reduced axis.
    size_t rows;
} bn_matrix;

// Read `array` and `axis` as a matrix a column kernel takes. True when
// `array` is exactly an ndarray of float64 in native byte order with two
// dimensions of at least one element each; `axis` is a Python int naming one
// of them; the other, kept axis has a stride of one element; and the reduced
// axis a positive stride that is a whole number of elements. Then the kernel
// sees the kept axis as its columns and the reduced axis as its rows. A bool,
// a NumPy integer or None is bottleneck's to interpret.
static bool read_matrix(PyObject *array, PyObject *axis, bn_matrix *m) {
    if (!hk_is_plain_array(array) || axis == NULL || !PyLong_CheckExact(axis)) return false;
    PyArrayObject *a = (PyArrayObject *)array;
    if (PyArray_TYPE(a) != NPY_DOUBLE || !PyArray_ISNOTSWAPPED(a) || PyArray_NDIM(a) != 2) return false;
    int overflow = 0;
    long reduced = PyLong_AsLongAndOverflow(axis, &overflow);
    if (overflow != 0 || reduced < -2 || reduced > 1) return false;
    if (reduced < 0) reduced += 2;
    const int kept = 1 - (int)reduced;
    const npy_intp *shape = PyArray_DIMS(a), *strides = PyArray_STRIDES(a);
    if (shape[0] < 1 || shape[1] < 1) return false;
    if (strides[kept] != F64_BYTES || strides[reduced] <= 0 || strides[reduced] % F64_BYTES != 0) return false;
    m->data = (const double *)PyArray_DATA(a);
    m->rows = (size_t)shape[reduced];
    m->cols = (size_t)shape[kept];
    m->step = (size_t)(strides[reduced] / F64_BYTES);
    m->n = (m->rows - 1) * m->step + m->cols;
    return true;
}

// A new one-dimensional float64 array of `cols` elements, as bottleneck
// allocates its result, or NULL with no exception set.
static PyArrayObject *new_result(size_t cols) {
    npy_intp length = (npy_intp)cols;
    PyObject *result = PyArray_SimpleNew(1, &length, NPY_DOUBLE);
    if (result == NULL) PyErr_Clear();
    return (PyArrayObject *)result;
}

// bottleneck's parameters, in its order. `ddof` exists only for nanstd.
enum { P_A, P_AXIS, P_DDOF, PARAMETERS };
static const char *const PARAMETER_NAMES[PARAMETERS] = {"a", "axis", "ddof"};

// The result of `nanmean`, `nanmin` or `nanmax` from a column kernel, or NULL
// with no exception set when the call is bottleneck's.
static HK_NOINLINE PyObject *try_cols(PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames,
                                      cols_kernel kernel, hk_kernel counter) {
    PyObject *bound[PARAMETERS];
    // Two parameters, a and axis: a third is bottleneck's to reject.
    if (!hk_bind_arguments(args, nargs, kwnames, PARAMETER_NAMES, P_DDOF, P_DDOF, bound)) return NULL;
    bn_matrix m;
    if (bound[P_A] == NULL || !read_matrix(bound[P_A], bound[P_AXIS], &m)) return NULL;
    PyArrayObject *result = new_result(m.cols);
    if (result == NULL) return NULL;
    double *out = (double *)PyArray_DATA(result);
    HK_BEGIN_NOGIL(m.rows * m.cols)
    kernel(m.data, out, m.n, m.cols, m.step);
    HK_END_NOGIL
    hk_take(counter);
    return (PyObject *)result;
}

// The result of `nanstd` from its column kernel, or NULL with no exception
// set when the call is bottleneck's. `ddof` must be absent or a Python int
// that bottleneck's conversion to a C int accepts.
static HK_NOINLINE PyObject *try_nanstd(PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    PyObject *bound[PARAMETERS];
    if (!hk_bind_arguments(args, nargs, kwnames, PARAMETER_NAMES, PARAMETERS, PARAMETERS, bound)) return NULL;
    long ddof = 0;
    if (bound[P_DDOF] != NULL) {
        if (!PyLong_CheckExact(bound[P_DDOF])) return NULL;
        int overflow = 0;
        ddof = PyLong_AsLongAndOverflow(bound[P_DDOF], &overflow);
        if (overflow != 0 || ddof < INT_MIN || ddof > INT_MAX) return NULL;
    }
    bn_matrix m;
    if (bound[P_A] == NULL || !read_matrix(bound[P_A], bound[P_AXIS], &m)) return NULL;
    PyArrayObject *result = new_result(m.cols);
    if (result == NULL) return NULL;
    double *out = (double *)PyArray_DATA(result);
    HK_BEGIN_NOGIL(m.rows * m.cols)
    nanstd_cols_f64(m.data, out, m.n, m.cols, m.step, (int64_t)ddof);
    HK_END_NOGIL
    hk_take(HK_NANSTD_COLS_F64);
    return (PyObject *)result;
}

// The four wrappers: a kernel if the call is one, bottleneck otherwise. Every
// call not taken is counted.
static PyObject *call_nanmean(HK_VECTORCALL_PARAMETERS) {
    PyObject *result = try_cols(args, PyVectorcall_NARGS(nargsf), kwnames, nanmean_cols_f64, HK_NANMEAN_COLS_F64);
    if (result != NULL) return result;
    hk_pass(HK_OP_BN_NANMEAN);
    return pass_on(callable, args, nargsf, kwnames);
}

static PyObject *call_nanstd(HK_VECTORCALL_PARAMETERS) {
    PyObject *result = try_nanstd(args, PyVectorcall_NARGS(nargsf), kwnames);
    if (result != NULL) return result;
    hk_pass(HK_OP_BN_NANSTD);
    return pass_on(callable, args, nargsf, kwnames);
}

static PyObject *call_nanmin(HK_VECTORCALL_PARAMETERS) {
    PyObject *result = try_cols(args, PyVectorcall_NARGS(nargsf), kwnames, nanmin_cols_f64, HK_NANMIN_COLS_F64);
    if (result != NULL) return result;
    hk_pass(HK_OP_BN_NANMIN);
    return pass_on(callable, args, nargsf, kwnames);
}

static PyObject *call_nanmax(HK_VECTORCALL_PARAMETERS) {
    PyObject *result = try_cols(args, PyVectorcall_NARGS(nargsf), kwnames, nanmax_cols_f64, HK_NANMAX_COLS_F64);
    if (result != NULL) return result;
    hk_pass(HK_OP_BN_NANMAX);
    return pass_on(callable, args, nargsf, kwnames);
}

// ------------------------------------------------------- apply and remove

// The address of the call pointer inside a `builtin_function_or_method`.
static vectorcallfunc *call_pointer(PyObject *object) {
    return (vectorcallfunc *)((char *)object + Py_TYPE(object)->tp_vectorcall_offset);
}

// Point one of bottleneck's functions at the binding. See hk_hook.apply.
//
// Refuses, changing nothing, unless the object is exactly what the wrapper
// was written for: a `builtin_function_or_method` (whose type keeps a call
// pointer and calls a function of these flags through `tp_call`, not through
// that pointer), defined with METH_VARARGS | METH_KEYWORDS, with its call
// pointer still NULL. A bottleneck that defines the function otherwise, or
// something else that has changed the pointer, leaves the hook off.
static int bottleneck_apply(int index) {
    PyObject *name = PyUnicode_FromString(BOTTLENECK_REDUCE);
    if (name == NULL) return -1;
    // A module in sys.modules, or NULL: this never imports one.
    PyObject *module = PyImport_GetModule(name);
    Py_DECREF(name);
    if (module == NULL) {
        if (!PyErr_Occurred()) PyErr_SetString(PyExc_RuntimeError, "bottleneck is not imported");
        return -1;
    }
    PyObject *object = PyObject_GetAttrString(module, functions[index].attribute);
    Py_DECREF(module);
    if (object == NULL) return -1;
    const char *problem = NULL;
    if (!PyCFunction_CheckExact(object)) problem = "is not a builtin function";
    else if (!PyType_HasFeature(Py_TYPE(object), Py_TPFLAGS_HAVE_VECTORCALL) ||
             Py_TYPE(object)->tp_vectorcall_offset <= 0 || Py_TYPE(object)->tp_call == NULL ||
             Py_TYPE(object)->tp_call == PyVectorcall_Call)
        problem = "has a type that is not called as expected";
    else if (PyCFunction_GET_FLAGS(object) != BOTTLENECK_FLAGS) problem = "takes its arguments another way";
    else if (*call_pointer(object) != NULL) problem = "already has a call pointer";
    if (problem != NULL) {
        PyErr_Format(PyExc_RuntimeError, "bottleneck's %s %s", functions[index].attribute, problem);
        Py_DECREF(object);
        return -1;
    }
    functions[index].object = object;
    *call_pointer(object) = functions[index].wrapper;
    return 0;
}

// Put NULL back in the function's call pointer. See hk_hook.remove.
static int bottleneck_remove(int index) {
    PyObject *object = functions[index].object;
    if (object == NULL) return 0;
    if (*call_pointer(object) != functions[index].wrapper) return 0;
    *call_pointer(object) = NULL;
    functions[index].object = NULL;
    Py_DECREF(object);
    return 1;
}

hk_hook hk_bottleneck_hooks[] = {
    {"bottleneck.nanmean", bottleneck_apply, bottleneck_remove, B_NANMEAN, false},
    {"bottleneck.nanstd", bottleneck_apply, bottleneck_remove, B_NANSTD, false},
    {"bottleneck.nanmin", bottleneck_apply, bottleneck_remove, B_NANMIN, false},
    {"bottleneck.nanmax", bottleneck_apply, bottleneck_remove, B_NANMAX, false},
    {NULL, NULL, NULL, 0, false},
};
