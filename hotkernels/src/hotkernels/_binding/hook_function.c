// Function hook: the call pointer inside a NumPy function object.
//
// `np.argmax`, `np.histogram` and the rest are objects of one NumPy type.
// CPython calls such an object through a function pointer stored inside it
// (the vectorcall protocol), at an offset the object's type publishes as
// `tp_vectorcall_offset`. Changing that one pointer leaves `np.argmax` the
// same object: names imported before install(), NumPy's own internal callers
// and code that uses the function as a dictionary key all keep working, and
// all reach the hook.
//
// A hooked call arrives with the caller's arguments exactly as written. The
// wrapper takes a kernel only when they already are what the kernel needs.
// Otherwise it calls the original pointer with the same arguments, so NumPy
// does its own conversions and raises its own errors.
#include "binding.h"

// The arguments of a vectorcall: the callable, the positional arguments
// followed by the keyword values, the positional count with a flag bit, and
// the keyword names.
#define HK_VECTORCALL_PARAMETERS \
    PyObject *callable, PyObject *const *args, size_t nargsf, PyObject *kwnames

// The functions that are hooked, as indices into `functions`.
enum {
    F_ARGMAX, F_ARGMIN, F_COUNT_NONZERO, F_SUM, F_HISTOGRAM, F_BINCOUNT, F_ISIN, F_INTERP,
    F_COMPRESS, F_EXTRACT, F_PUTMASK, F_COPYTO, F_WHERE,
    F_COUNT
};

// One hooked function.
typedef struct {
    // Attribute of the `numpy` module that holds the function object.
    const char *attribute;
    // The binding's call function.
    vectorcallfunc wrapper;
    // The function object, kept alive while the hook is applied.
    PyObject *object;
    // NumPy's own call function, kept while the hook is applied.
    vectorcallfunc original;
} hk_function;

static hk_function functions[F_COUNT];

// `np.histogram`'s default number of bins.
#define HISTOGRAM_DEFAULT_BINS 10
// Number of counters `bincount_u8` writes: one per byte value.
#define BYTE_VALUES 256
// Number of test values `isin_i64` reads. Fewer are padded by repeating one.
#define ISIN_TEST_VALUES 8
// Most sample points per query for which `np.interp` takes the kernel. The
// sample points are read once for NaN before the kernel sees them, at about
// 0.13 ns each, and NumPy spends at least 30 ns on a query. At 64 sample
// points per query the pass costs under a third of what NumPy would have
// spent; with fewer queries than that it would be most of the call.
#define INTERP_MAX_POINTS_PER_QUERY 64

// numpy.linspace and the float64 type descriptor, fetched on first use.
static PyObject *numpy_linspace;
static PyObject *linspace_keywords;

// ------------------------------------------------------------ small helpers

PyObject *hk_scalar(int type_num, void *value) {
    PyArray_Descr *descr = PyArray_DescrFromType(type_num);
    PyObject *scalar = PyArray_Scalar(value, descr, NULL);
    Py_DECREF(descr);
    if (scalar == NULL) PyErr_Clear();
    return scalar;
}

bool hk_bind_arguments(PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames,
                       const char *const *names, Py_ssize_t count,
                       Py_ssize_t max_positional, PyObject **bound) {
    if (nargs > max_positional) return false;
    for (Py_ssize_t i = 0; i < count; i++) bound[i] = i < nargs ? args[i] : NULL;
    const Py_ssize_t keywords = kwnames ? PyTuple_GET_SIZE(kwnames) : 0;
    for (Py_ssize_t k = 0; k < keywords; k++) {
        PyObject *name = PyTuple_GET_ITEM(kwnames, k);
        Py_ssize_t i = 0;
        while (i < count && !(names[i] && hk_text_is(name, names[i]))) i++;
        if (i == count || bound[i] != NULL) return false;
        bound[i] = args[nargs + k];
    }
    return true;
}

// ------------------------------------------------------ argmax and argmin

PyObject *hk_try_argext(PyObject *array, bool want_max) {
    hk_kernel kernel;
    npy_intp index;
    if (hk_is_flat(array, NPY_FLOAT)) kernel = want_max ? HK_ARGMAX_F32 : HK_ARGMIN_F32;
    else if (hk_is_flat(array, NPY_DOUBLE)) kernel = want_max ? HK_ARGMAX_F64 : HK_ARGMIN_F64;
    else return NULL;
    PyArrayObject *a = (PyArrayObject *)array;
    const npy_intp n = PyArray_SIZE(a);
    // NumPy raises on an empty array; the kernels are never called with n = 0.
    if (n < 1) return NULL;
    // The interpreter lock stays held: the kernels read the array twice and
    // are not safe against a writer (issue 9).
    switch (kernel) {
        case HK_ARGMAX_F32: index = (npy_intp)argmax_f32(PyArray_DATA(a), (size_t)n); break;
        case HK_ARGMIN_F32: index = (npy_intp)argmin_f32(PyArray_DATA(a), (size_t)n); break;
        case HK_ARGMAX_F64: index = (npy_intp)argmax_f64(PyArray_DATA(a), (size_t)n); break;
        default: index = (npy_intp)argmin_f64(PyArray_DATA(a), (size_t)n); break;
    }
    PyObject *result = hk_scalar(NPY_INTP, &index);
    if (result != NULL) hk_take(kernel);
    return result;
}

// True when `array` is a plain float32 or float64 ndarray. NumPy sends such a
// call on to the type slot, whose hook counts it, so it is not counted here.
static bool reaches_type_slot(PyObject *array) {
    if (!hk_is_plain_array(array)) return false;
    const int type_num = PyArray_TYPE((PyArrayObject *)array);
    return type_num == NPY_FLOAT || type_num == NPY_DOUBLE;
}

// np.argmax(a) and np.argmin(a) with nothing but the array: the kernel is
// called directly, which skips NumPy's Python wrapper. That wrapper is most
// of the cost of a call on a small array.
static PyObject *call_argext(HK_VECTORCALL_PARAMETERS, int function, bool want_max, hk_operation operation) {
    const Py_ssize_t nargs = PyVectorcall_NARGS(nargsf);
    if (nargs == 1 && kwnames == NULL) {
        PyObject *result = hk_try_argext(args[0], want_max);
        if (result != NULL) return result;
    }
    if (!(nargs >= 1 && reaches_type_slot(args[0]))) hk_pass(operation);
    return functions[function].original(callable, args, nargsf, kwnames);
}

static PyObject *call_argmax(HK_VECTORCALL_PARAMETERS) {
    return call_argext(callable, args, nargsf, kwnames, F_ARGMAX, true, HK_OP_ARGMAX);
}

static PyObject *call_argmin(HK_VECTORCALL_PARAMETERS) {
    return call_argext(callable, args, nargsf, kwnames, F_ARGMIN, false, HK_OP_ARGMIN);
}

// ----------------------------------------------------------- count_nonzero

// np.count_nonzero(a) for a contiguous int64 or float64 array. NumPy returns
// the count as an np.intp scalar when no axis is given.
static HK_NOINLINE PyObject *try_count_nonzero(PyObject *array) {
    hk_kernel kernel;
    if (hk_is_flat(array, NPY_INT64)) kernel = HK_COUNT_NONZERO_I64;
    else if (hk_is_flat(array, NPY_DOUBLE)) kernel = HK_COUNT_NONZERO_F64;
    else return NULL;
    PyArrayObject *a = (PyArrayObject *)array;
    const npy_intp n = PyArray_SIZE(a);
    npy_intp count;
    {
        HK_BEGIN_NOGIL(n)
        count = (npy_intp)(kernel == HK_COUNT_NONZERO_I64 ? count_nonzero_i64(PyArray_DATA(a), (size_t)n)
                                                          : count_nonzero_f64(PyArray_DATA(a), (size_t)n));
        HK_END_NOGIL
    }
    PyObject *result = hk_scalar(NPY_INTP, &count);
    if (result != NULL) hk_take(kernel);
    return result;
}

static PyObject *call_count_nonzero(HK_VECTORCALL_PARAMETERS) {
    if (PyVectorcall_NARGS(nargsf) == 1 && kwnames == NULL) {
        PyObject *result = try_count_nonzero(args[0]);
        if (result != NULL) return result;
    }
    hk_pass(HK_OP_COUNT_NONZERO);
    return functions[F_COUNT_NONZERO].original(callable, args, nargsf, kwnames);
}

// --------------------------------------------------------------------- sum

// The sum of a contiguous uint8 array, as the np.uint64 NumPy returns.
PyObject *hk_try_sum(PyObject *array) {
    if (!hk_is_flat(array, NPY_UINT8)) return NULL;
    PyArrayObject *a = (PyArrayObject *)array;
    const npy_intp n = PyArray_SIZE(a);
    uint64_t total;
    {
        HK_BEGIN_NOGIL(n)
        total = sum_u8(PyArray_DATA(a), (size_t)n);
        HK_END_NOGIL
    }
    PyObject *result = hk_scalar(NPY_UINT64, &total);
    if (result != NULL) hk_take(HK_SUM_U8);
    return result;
}

static PyObject *call_sum(HK_VECTORCALL_PARAMETERS) {
    if (PyVectorcall_NARGS(nargsf) == 1 && kwnames == NULL) {
        PyObject *result = hk_try_sum(args[0]);
        if (result != NULL) return result;
    }
    hk_pass(HK_OP_SUM);
    return functions[F_SUM].original(callable, args, nargsf, kwnames);
}

// --------------------------------------------------------------- histogram

// Fetch numpy.linspace and build the keywords it is called with. Returns
// false with no exception set if either fails.
static bool prepare_linspace(void) {
    if (numpy_linspace != NULL) return true;
    PyObject *numpy = PyImport_ImportModule("numpy");
    PyObject *linspace = numpy ? PyObject_GetAttrString(numpy, "linspace") : NULL;
    PyObject *keywords = PyDict_New();
    PyObject *float64 = (PyObject *)PyArray_DescrFromType(NPY_DOUBLE);
    Py_XDECREF(numpy);
    if (linspace == NULL || keywords == NULL || PyDict_SetItemString(keywords, "dtype", float64) != 0) {
        PyErr_Clear();
        Py_XDECREF(linspace);
        Py_XDECREF(keywords);
        Py_DECREF(float64);
        return false;
    }
    Py_DECREF(float64);
    numpy_linspace = linspace;
    linspace_keywords = keywords;
    return true;
}

// Work out the two outer bin edges of a histogram call, as new references to
// the objects NumPy itself would use: the caller's `range` values, or the
// array's minimum and maximum when no range was given. `lo` and `hi` receive
// them as doubles. Returns false, with no exception set, when the range is
// one NumPy treats specially: not a tuple of two plain numbers, not finite,
// empty, reversed, or wider than a double can hold. NumPy then widens it,
// warns or raises, and the call is its own.
static bool histogram_range(PyObject *array, PyObject *range, PyObject **first, PyObject **last,
                            double *lo, double *hi) {
    *first = *last = NULL;
    if (hk_is_absent(range)) {
        if (PyArray_SIZE((PyArrayObject *)array) == 0) return false;
        *first = PyObject_CallMethod(array, "min", NULL);
        *last = PyObject_CallMethod(array, "max", NULL);
        if (*first == NULL || *last == NULL) goto fail;
        *lo = PyFloat_AsDouble(*first);
        *hi = PyFloat_AsDouble(*last);
        if (PyErr_Occurred()) goto fail;
    } else {
        if (!PyTuple_CheckExact(range) || PyTuple_GET_SIZE(range) != 2) return false;
        if (!hk_as_exact_double(PyTuple_GET_ITEM(range, 0), lo) || !hk_as_exact_double(PyTuple_GET_ITEM(range, 1), hi)) return false;
        *first = Py_NewRef(PyTuple_GET_ITEM(range, 0));
        *last = Py_NewRef(PyTuple_GET_ITEM(range, 1));
    }
    // A width that overflows makes NumPy's own arithmetic warn. Such a call
    // is left to NumPy before anything here has computed with the range, so
    // that each warning is issued once.
    if (isfinite(*lo) && isfinite(*hi) && *lo < *hi && isfinite(*hi - *lo)) return true;
fail:
    PyErr_Clear();
    Py_CLEAR(*first);
    Py_CLEAR(*last);
    return false;
}

// Build the bin edges exactly as np.histogram does, with the same call to
// np.linspace, so they are identical by construction. Returns NULL with no
// exception set if that fails or if the edges are not strictly increasing:
// NumPy raises "Too many bins for data range" then, and the kernel would
// return counts (issue 9).
static PyObject *histogram_edges(PyObject *first, PyObject *last, Py_ssize_t bins) {
    if (!prepare_linspace()) return NULL;
    PyObject *positional = Py_BuildValue("(OOn)", first, last, bins + 1);
    if (positional == NULL) {
        PyErr_Clear();
        return NULL;
    }
    PyObject *edges = PyObject_Call(numpy_linspace, positional, linspace_keywords);
    Py_DECREF(positional);
    if (edges == NULL) {
        PyErr_Clear();
        return NULL;
    }
    if (!hk_is_flat_1d(edges, NPY_DOUBLE) || PyArray_SIZE((PyArrayObject *)edges) != bins + 1) {
        Py_DECREF(edges);
        return NULL;
    }
    const double *e = PyArray_DATA((PyArrayObject *)edges);
    for (Py_ssize_t i = 0; i < bins; i++) {
        if (!(e[i] < e[i + 1])) {
            Py_DECREF(edges);
            return NULL;
        }
    }
    return edges;
}

// np.histogram(a, bins, range) for a contiguous float64 array, an integer
// number of bins, and no density or weights. Returns the pair NumPy returns:
// the counts as int64 and the bin edges as float64.
static HK_NOINLINE PyObject *try_histogram(PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    enum { A, BINS, RANGE, DENSITY, WEIGHTS, PARAMETERS };
    static const char *const names[PARAMETERS] = {"a", "bins", "range", "density", "weights"};
    PyObject *bound[PARAMETERS];
    if (!hk_bind_arguments(args, nargs, kwnames, names, PARAMETERS, PARAMETERS, bound)) return NULL;
    if (bound[A] == NULL || !hk_is_flat(bound[A], NPY_DOUBLE)) return NULL;
    if (!hk_is_absent(bound[DENSITY]) || !hk_is_absent(bound[WEIGHTS])) return NULL;

    Py_ssize_t bins = HISTOGRAM_DEFAULT_BINS;
    if (bound[BINS] != NULL) {
        if (!PyLong_CheckExact(bound[BINS])) return NULL;
        bins = PyLong_AsSsize_t(bound[BINS]);
        if (bins == -1 && PyErr_Occurred()) {
            PyErr_Clear();
            return NULL;
        }
    }
    if (bins < 1) return NULL;

    PyObject *first, *last;
    double lo, hi;
    if (!histogram_range(bound[A], bound[RANGE], &first, &last, &lo, &hi)) return NULL;
    PyObject *edges = histogram_edges(first, last, bins);
    Py_DECREF(first);
    Py_DECREF(last);
    if (edges == NULL) return NULL;

    npy_intp shape = bins;
    PyObject *counts = PyArray_SimpleNew(1, &shape, NPY_INTP);
    PyObject *result = counts ? PyTuple_Pack(2, counts, edges) : NULL;
    Py_DECREF(edges);
    Py_XDECREF(counts);
    if (result == NULL) {
        PyErr_Clear();
        return NULL;
    }
    PyArrayObject *a = (PyArrayObject *)bound[A];
    const npy_intp n = PyArray_SIZE(a);
    {
        HK_BEGIN_NOGIL(n)
        histogram_f64(PyArray_DATA(a), PyArray_DATA((PyArrayObject *)counts), (size_t)n, (size_t)bins, lo, hi);
        HK_END_NOGIL
    }
    hk_take(HK_HISTOGRAM_F64);
    return result;
}

static PyObject *call_histogram(HK_VECTORCALL_PARAMETERS) {
    PyObject *result = try_histogram(args, PyVectorcall_NARGS(nargsf), kwnames);
    if (result != NULL) return result;
    hk_pass(HK_OP_HISTOGRAM);
    return functions[F_HISTOGRAM].original(callable, args, nargsf, kwnames);
}

// ---------------------------------------------------------------- bincount

// np.bincount(x, minlength=m) for a contiguous one-dimensional uint8 array
// without weights. The kernel always fills 256 counters. NumPy's result has
// one entry more than the largest value present, or `minlength` entries if
// that is more, so the counters are cut or padded to that length.
static HK_NOINLINE PyObject *try_bincount(PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    enum { X, WEIGHTS, MINLENGTH, PARAMETERS };
    // `x` can only be given by position.
    static const char *const names[PARAMETERS] = {NULL, "weights", "minlength"};
    PyObject *bound[PARAMETERS];
    if (!hk_bind_arguments(args, nargs, kwnames, names, PARAMETERS, PARAMETERS, bound)) return NULL;
    if (bound[X] == NULL || !hk_is_flat_1d(bound[X], NPY_UINT8) || !hk_is_absent(bound[WEIGHTS])) return NULL;

    Py_ssize_t minimum = 0;
    if (bound[MINLENGTH] != NULL) {
        if (!PyLong_CheckExact(bound[MINLENGTH])) return NULL;
        minimum = PyLong_AsSsize_t(bound[MINLENGTH]);
        if (minimum == -1 && PyErr_Occurred()) {
            PyErr_Clear();
            return NULL;
        }
        if (minimum < 0) return NULL;
    }

    PyArrayObject *x = (PyArrayObject *)bound[X];
    const npy_intp n = PyArray_SIZE(x);
    int64_t counters[BYTE_VALUES];
    {
        HK_BEGIN_NOGIL(n)
        bincount_u8(PyArray_DATA(x), counters, (size_t)n);
        HK_END_NOGIL
    }
    npy_intp used = BYTE_VALUES;
    while (used > 0 && counters[used - 1] == 0) used--;
    npy_intp length = used > minimum ? used : minimum;
    PyObject *result = PyArray_ZEROS(1, &length, NPY_INTP, 0);
    if (result == NULL) {
        PyErr_Clear();
        return NULL;
    }
    memcpy(PyArray_DATA((PyArrayObject *)result), counters, (size_t)used * sizeof counters[0]);
    hk_take(HK_BINCOUNT_U8);
    return result;
}

static PyObject *call_bincount(HK_VECTORCALL_PARAMETERS) {
    PyObject *result = try_bincount(args, PyVectorcall_NARGS(nargsf), kwnames);
    if (result != NULL) return result;
    hk_pass(HK_OP_BINCOUNT);
    return functions[F_BINCOUNT].original(callable, args, nargsf, kwnames);
}

// -------------------------------------------------------------------- isin

// Read up to eight int64 test values from a contiguous one-dimensional int64
// array, or from a list or tuple of Python ints that fit int64, which is what
// NumPy would convert such a list to. The kernel reads exactly eight; a
// shorter set is padded by repeating its first value, which does not change
// membership. Returns false for anything else.
static bool isin_test_values(PyObject *test, int64_t values[ISIN_TEST_VALUES]) {
    Py_ssize_t count;
    if (hk_is_flat_1d(test, NPY_INT64)) {
        count = PyArray_SIZE((PyArrayObject *)test);
        if (count < 1 || count > ISIN_TEST_VALUES) return false;
        memcpy(values, PyArray_DATA((PyArrayObject *)test), (size_t)count * sizeof values[0]);
    } else if (PyList_CheckExact(test) || PyTuple_CheckExact(test)) {
        count = PySequence_Fast_GET_SIZE(test);
        if (count < 1 || count > ISIN_TEST_VALUES) return false;
        for (Py_ssize_t i = 0; i < count; i++) {
            PyObject *item = PySequence_Fast_GET_ITEM(test, i);
            int overflow = 0;
            if (!PyLong_CheckExact(item)) return false;
            values[i] = PyLong_AsLongLongAndOverflow(item, &overflow);
            if (overflow != 0) return false;
        }
    } else {
        return false;
    }
    for (Py_ssize_t i = count; i < ISIN_TEST_VALUES; i++) values[i] = values[0];
    return true;
}

// np.isin(element, test_elements) for a contiguous int64 array and at most
// eight int64 test values, with every other argument left at its default.
static HK_NOINLINE PyObject *try_isin(PyObject *element, PyObject *test) {
    int64_t values[ISIN_TEST_VALUES];
    if (!hk_is_flat(element, NPY_INT64) || !isin_test_values(test, values)) return NULL;
    PyArrayObject *x = (PyArrayObject *)element;
    PyObject *result = PyArray_SimpleNew(PyArray_NDIM(x), PyArray_DIMS(x), NPY_BOOL);
    if (result == NULL) {
        PyErr_Clear();
        return NULL;
    }
    const npy_intp n = PyArray_SIZE(x);
    {
        HK_BEGIN_NOGIL(n)
        isin_i64(PyArray_DATA(x), values, PyArray_DATA((PyArrayObject *)result), (size_t)n);
        HK_END_NOGIL
    }
    hk_take(HK_ISIN_I64);
    return result;
}

static PyObject *call_isin(HK_VECTORCALL_PARAMETERS) {
    if (PyVectorcall_NARGS(nargsf) == 2 && kwnames == NULL) {
        PyObject *result = try_isin(args[0], args[1]);
        if (result != NULL) return result;
    }
    hk_pass(HK_OP_ISIN);
    return functions[F_ISIN].original(callable, args, nargsf, kwnames);
}

// ------------------------------------------------------------------ interp

// np.interp(x, xp, fp) for a contiguous float64 array of queries and two
// one-dimensional contiguous float64 arrays of sample points, with `left`,
// `right` and `period` left alone. The result has the shape of `x`.
//
// The kernel requires sample points without a NaN. With one at xp[0] it reads
// before the start of xp and fp, and with one anywhere it returns something
// other than NumPy, whose own answer then depends on which elements its
// search happens to probe (issue 9). So xp is read once first, and a NaN
// sends the call to NumPy. That pass is only worth making when there are
// enough queries to pay for it; see INTERP_MAX_POINTS_PER_QUERY.
//
// Sample points that are not increasing are not detected. NumPy does not
// detect them either; both give a meaningless result, not the same one, and
// the kernel stays inside its buffers whatever the data.
static HK_NOINLINE PyObject *try_interp(PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    enum { X, XP, FP, LEFT, RIGHT, PERIOD, PARAMETERS };
    static const char *const names[PARAMETERS] = {"x", "xp", "fp", "left", "right", "period"};
    PyObject *bound[PARAMETERS];
    if (!hk_bind_arguments(args, nargs, kwnames, names, PARAMETERS, PARAMETERS, bound)) return NULL;
    if (bound[X] == NULL || bound[XP] == NULL || bound[FP] == NULL) return NULL;
    if (!hk_is_absent(bound[LEFT]) || !hk_is_absent(bound[RIGHT]) || !hk_is_absent(bound[PERIOD])) return NULL;
    if (!hk_is_flat(bound[X], NPY_DOUBLE) || !hk_is_flat_1d(bound[XP], NPY_DOUBLE) ||
        !hk_is_flat_1d(bound[FP], NPY_DOUBLE)) return NULL;

    PyArrayObject *x = (PyArrayObject *)bound[X];
    PyArrayObject *xp = (PyArrayObject *)bound[XP];
    const npy_intp queries = PyArray_SIZE(x);
    const npy_intp points = PyArray_SIZE(xp);
    // NumPy raises on empty sample points and on lengths that differ.
    if (points < 1 || points != PyArray_SIZE((PyArrayObject *)bound[FP])) return NULL;
    if (points / INTERP_MAX_POINTS_PER_QUERY > queries) return NULL;

    PyObject *result = PyArray_SimpleNew(PyArray_NDIM(x), PyArray_DIMS(x), NPY_DOUBLE);
    if (result == NULL) {
        PyErr_Clear();
        return NULL;
    }
    bool clean;
    {
        HK_BEGIN_NOGIL(queries)
        clean = !hk_has_nan(PyArray_DATA(xp), (size_t)points);
        if (clean) {
            interp_f64(PyArray_DATA(x), PyArray_DATA(xp), PyArray_DATA((PyArrayObject *)bound[FP]),
                       PyArray_DATA((PyArrayObject *)result), (size_t)points, (size_t)queries);
        }
        HK_END_NOGIL
    }
    if (!clean) {
        Py_DECREF(result);
        return NULL;
    }
    hk_take(HK_INTERP_F64);
    return result;
}

static PyObject *call_interp(HK_VECTORCALL_PARAMETERS) {
    PyObject *result = try_interp(args, PyVectorcall_NARGS(nargsf), kwnames);
    if (result != NULL) return result;
    hk_pass(HK_OP_INTERP);
    return functions[F_INTERP].original(callable, args, nargsf, kwnames);
}

// ------------------------------------------------------ the mask operations
//
// The function forms of x[mask], x[mask] = v and mask.nonzero(). Each takes
// the kernel under the conditions of the indexing or method hook that serves
// the same kernel, and under whatever more NumPy's own function requires.

// np.compress(condition, a) for a one-dimensional boolean condition and a
// contiguous float32 or float64 array of any shape and the same size, with
// no `axis` and no `out`. Without an axis NumPy flattens `a` in C order, so
// the result is the flat array's selected elements. NumPy raises for a
// condition of more than one dimension, and cuts or rejects one of another
// length; both go to NumPy. With `out=` the result would be written into the
// caller's buffer, and a scratch kernel must only ever see a buffer of the
// binding's own: that call is NumPy's too.
static HK_NOINLINE PyObject *try_compress(PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    enum { CONDITION, A, AXIS, OUT, PARAMETERS };
    static const char *const names[PARAMETERS] = {"condition", "a", "axis", "out"};
    PyObject *bound[PARAMETERS];
    if (!hk_bind_arguments(args, nargs, kwnames, names, PARAMETERS, PARAMETERS, bound)) return NULL;
    if (bound[CONDITION] == NULL || bound[A] == NULL) return NULL;
    if (!hk_is_absent(bound[AXIS]) || !hk_is_absent(bound[OUT])) return NULL;
    if (!hk_is_plain_array(bound[CONDITION]) || PyArray_NDIM((PyArrayObject *)bound[CONDITION]) != 1) return NULL;
    return hk_try_compress(bound[A], bound[CONDITION]);
}

static PyObject *call_compress(HK_VECTORCALL_PARAMETERS) {
    PyObject *result = try_compress(args, PyVectorcall_NARGS(nargsf), kwnames);
    if (result != NULL) return result;
    hk_pass(HK_OP_COMPRESS);
    return functions[F_COMPRESS].original(callable, args, nargsf, kwnames);
}

// np.extract(condition, arr) for a boolean condition and a contiguous float32
// or float64 array of the same size, each of any shape. NumPy flattens both
// in C order and takes the elements where the flat condition is True.
static HK_NOINLINE PyObject *try_extract(PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    enum { CONDITION, ARR, PARAMETERS };
    static const char *const names[PARAMETERS] = {"condition", "arr"};
    PyObject *bound[PARAMETERS];
    if (!hk_bind_arguments(args, nargs, kwnames, names, PARAMETERS, PARAMETERS, bound)) return NULL;
    if (bound[CONDITION] == NULL || bound[ARR] == NULL) return NULL;
    return hk_try_compress(bound[ARR], bound[CONDITION]);
}

// A call left to NumPy runs np.nonzero on the condition, which the method
// hook may serve; that is counted there, under nonzero.
static PyObject *call_extract(HK_VECTORCALL_PARAMETERS) {
    PyObject *result = try_extract(args, PyVectorcall_NARGS(nargsf), kwnames);
    if (result != NULL) return result;
    hk_pass(HK_OP_COMPRESS);
    return functions[F_EXTRACT].original(callable, args, nargsf, kwnames);
}

// np.putmask(a, mask, values) for a contiguous writable float64 `a`, a
// contiguous boolean mask of the same size, each of any shape, and a Python
// float or int as the value. NumPy reads both arrays in C order and requires
// only their sizes to match. `a` can only be given by position.
static HK_NOINLINE bool try_putmask(PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    enum { A, MASK, VALUES, PARAMETERS };
    static const char *const names[PARAMETERS] = {NULL, "mask", "values"};
    PyObject *bound[PARAMETERS];
    if (!hk_bind_arguments(args, nargs, kwnames, names, PARAMETERS, PARAMETERS, bound)) return false;
    if (bound[A] == NULL || bound[MASK] == NULL || bound[VALUES] == NULL) return false;
    return hk_try_maskfill(bound[A], bound[MASK], bound[VALUES]);
}

// NumPy's putmask returns None.
static PyObject *call_putmask(HK_VECTORCALL_PARAMETERS) {
    if (try_putmask(args, PyVectorcall_NARGS(nargsf), kwnames)) Py_RETURN_NONE;
    hk_pass(HK_OP_MASKFILL);
    return functions[F_PUTMASK].original(callable, args, nargsf, kwnames);
}

// True when two arrays have the same shape.
static bool same_shape(PyArrayObject *a, PyArrayObject *b) {
    return PyArray_NDIM(a) == PyArray_NDIM(b) &&
           PyArray_CompareLists(PyArray_DIMS(a), PyArray_DIMS(b), PyArray_NDIM(a));
}

// np.copyto(dst, src, where=mask) for a contiguous writable float64 `dst`, a
// Python float or int as `src`, and a contiguous boolean mask of the same
// shape, with no `casting`. NumPy broadcasts the mask against `dst`, so a
// mask of another shape, even of the same size, is NumPy's to broadcast or
// refuse. Sets *masked when the call has a `where` argument at all: only
// then is it a masked assignment, and only then is a call left to NumPy
// counted.
static HK_NOINLINE bool try_copyto(PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames, bool *masked) {
    enum { DST, SRC, CASTING, WHERE, PARAMETERS };
    static const char *const names[PARAMETERS] = {"dst", "src", "casting", "where"};
    PyObject *bound[PARAMETERS];
    *masked = false;
    if (!hk_bind_arguments(args, nargs, kwnames, names, PARAMETERS, PARAMETERS, bound)) return false;
    *masked = bound[WHERE] != NULL;
    if (bound[DST] == NULL || bound[SRC] == NULL || bound[WHERE] == NULL || bound[CASTING] != NULL) return false;
    if (!hk_is_plain_array(bound[DST]) || !hk_is_plain_array(bound[WHERE])) return false;
    if (!same_shape((PyArrayObject *)bound[DST], (PyArrayObject *)bound[WHERE])) return false;
    return hk_try_maskfill(bound[DST], bound[WHERE], bound[SRC]);
}

// NumPy's copyto returns None. A copy without `where` is not a masked
// assignment and is passed on without being counted.
static PyObject *call_copyto(HK_VECTORCALL_PARAMETERS) {
    bool masked;
    if (try_copyto(args, PyVectorcall_NARGS(nargsf), kwnames, &masked)) Py_RETURN_NONE;
    if (masked) hk_pass(HK_OP_MASKFILL);
    return functions[F_COPYTO].original(callable, args, nargsf, kwnames);
}

// np.where(condition) with nothing but a one-dimensional contiguous boolean
// array: the positions that are True, as np.nonzero gives them. np.where
// calls NumPy's C code for nonzero directly, so the method hook does not see
// this call. With `x` and `y` it is another operation, which is passed on
// without being counted. NumPy accepts no keywords here.
static PyObject *call_where(HK_VECTORCALL_PARAMETERS) {
    if (PyVectorcall_NARGS(nargsf) == 1 && kwnames == NULL) {
        PyObject *result = hk_try_nonzero(args[0]);
        if (result != NULL) return result;
        hk_pass(HK_OP_NONZERO);
    }
    return functions[F_WHERE].original(callable, args, nargsf, kwnames);
}

// ----------------------------------------------------------------- the hook

static hk_function functions[F_COUNT] = {
    [F_ARGMAX] = {"argmax", call_argmax, NULL, NULL},
    [F_ARGMIN] = {"argmin", call_argmin, NULL, NULL},
    [F_COUNT_NONZERO] = {"count_nonzero", call_count_nonzero, NULL, NULL},
    [F_SUM] = {"sum", call_sum, NULL, NULL},
    [F_HISTOGRAM] = {"histogram", call_histogram, NULL, NULL},
    [F_BINCOUNT] = {"bincount", call_bincount, NULL, NULL},
    [F_ISIN] = {"isin", call_isin, NULL, NULL},
    [F_INTERP] = {"interp", call_interp, NULL, NULL},
    [F_COMPRESS] = {"compress", call_compress, NULL, NULL},
    [F_EXTRACT] = {"extract", call_extract, NULL, NULL},
    [F_PUTMASK] = {"putmask", call_putmask, NULL, NULL},
    [F_COPYTO] = {"copyto", call_copyto, NULL, NULL},
    [F_WHERE] = {"where", call_where, NULL, NULL},
};

// Return the address of the call pointer inside a callable object, or NULL
// with an exception set if its type does not use the vectorcall protocol.
static vectorcallfunc *call_pointer(PyObject *object) {
    PyTypeObject *type = Py_TYPE(object);
    if (!PyType_HasFeature(type, Py_TPFLAGS_HAVE_VECTORCALL) || type->tp_vectorcall_offset <= 0) {
        PyErr_Format(PyExc_TypeError, "%s objects have no call pointer to hook", type->tp_name);
        return NULL;
    }
    return (vectorcallfunc *)((char *)object + type->tp_vectorcall_offset);
}

// Point the function at the binding. See hk_hook.apply.
static int function_apply(int index) {
    PyObject *numpy = PyImport_ImportModule("numpy");
    if (numpy == NULL) return -1;
    PyObject *object = PyObject_GetAttrString(numpy, functions[index].attribute);
    Py_DECREF(numpy);
    if (object == NULL) return -1;
    vectorcallfunc *pointer = call_pointer(object);
    if (pointer == NULL || *pointer == NULL) {
        if (!PyErr_Occurred()) PyErr_SetString(PyExc_RuntimeError, "the function has no call pointer set");
        Py_DECREF(object);
        return -1;
    }
    functions[index].object = object;
    functions[index].original = *pointer;
    *pointer = functions[index].wrapper;
    return 0;
}

// Put NumPy's call pointer back. See hk_hook.remove.
static int function_remove(int index) {
    PyObject *object = functions[index].object;
    if (object == NULL) return 0;
    vectorcallfunc *pointer = call_pointer(object);
    if (pointer == NULL) {
        PyErr_Clear();
        return 0;
    }
    if (*pointer != functions[index].wrapper) return 0;
    *pointer = functions[index].original;
    functions[index].object = NULL;
    Py_DECREF(object);
    return 1;
}

hk_hook hk_function_hooks[] = {
    {"function.argmax", function_apply, function_remove, F_ARGMAX, false},
    {"function.argmin", function_apply, function_remove, F_ARGMIN, false},
    {"function.count_nonzero", function_apply, function_remove, F_COUNT_NONZERO, false},
    {"function.sum", function_apply, function_remove, F_SUM, false},
    {"function.histogram", function_apply, function_remove, F_HISTOGRAM, false},
    {"function.bincount", function_apply, function_remove, F_BINCOUNT, false},
    {"function.isin", function_apply, function_remove, F_ISIN, false},
    {"function.interp", function_apply, function_remove, F_INTERP, false},
    {"function.compress", function_apply, function_remove, F_COMPRESS, false},
    {"function.extract", function_apply, function_remove, F_EXTRACT, false},
    {"function.putmask", function_apply, function_remove, F_PUTMASK, false},
    {"function.copyto", function_apply, function_remove, F_COPYTO, false},
    {"function.where", function_apply, function_remove, F_WHERE, false},
    {NULL, NULL, NULL, 0, false},
};
