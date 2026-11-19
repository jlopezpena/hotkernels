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
    F_COMPRESS, F_EXTRACT, F_PUTMASK, F_COPYTO, F_WHERE, F_CLIP, F_ANY,
    F_COUNT
};

// One hooked function.
typedef struct {
    // Attribute of the `numpy` module that holds the function object.
    const char *attribute;
    // The binding's call function.
    vectorcallfunc wrapper;
    // Reads what the wrapper needs to know about this NumPy's function when
    // the hook is applied, or NULL. Takes the function object; raises nothing.
    void (*prepare)(PyObject *object);
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
// Counters `bincount_i16` and `bincount_i64` are always given, whatever the
// element count: 512 KB. At that size a call on 1,000 elements takes about
// as long with the kernel as NumPy's does (6.5 against 6.9 us on the Intel
// Core i5-14600KF), both being mostly the counters. Above it a call is taken
// only while it has no more counters than elements; see bincount_codes.
#define BINCOUNT_FREE_BINS 65536
// Bytes of a boolean array that `np.sum` checks and then sums at a time. Each
// block is read twice, so it is a third of the smallest first-level data
// cache the kernels run on (48 KiB on the x86-64 performance cores).
#define SUM_BOOL_BLOCK 16384
// Number of test values `isin_i64` reads. Fewer are padded by repeating one.
#define ISIN_TEST_VALUES 8
// Most sample points per query for which `np.interp` takes the kernel. The
// sample points are read once for NaN before the kernel sees them, at about
// 0.13 ns each, and NumPy spends at least 30 ns on a query. At 64 sample
// points per query the pass costs under a third of what NumPy would have
// spent; with fewer queries than that it would be most of the call.
#define INTERP_MAX_POINTS_PER_QUERY 64
// Elements of the buffer np.where spreads a scalar over, and so of each call
// of the kernel when one side is a scalar: the largest buffer kept on the
// stack.
#define WHERE_BLOCK ((npy_intp)(HK_STACK_SCRATCH_BYTES / sizeof(double)))

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

// The number of True elements of a contiguous boolean array, as the np.intp
// NumPy's sum returns for one: the sum_u8 kernel on the mask's bytes. A byte
// other than 0 or 1 counts once in NumPy and by its value in the kernel, so
// the bytes are checked first and such a mask is NumPy's. Checking and
// summing go block by block, so that the kernel reads each block from the
// first-level cache and not from wherever the whole array lies.
static HK_NOINLINE PyObject *try_sum_bool(PyArrayObject *mask) {
    const uint8_t *bytes = PyArray_DATA(mask);
    const npy_intp n = PyArray_SIZE(mask);
    npy_intp total = 0;
    bool clean = true;
    {
        HK_BEGIN_NOGIL(n)
        for (npy_intp done = 0; done < n && clean; done += SUM_BOOL_BLOCK) {
            const size_t count = (size_t)(n - done < SUM_BOOL_BLOCK ? n - done : SUM_BOOL_BLOCK);
            clean = hk_mask_is_clean(bytes + done, count);
            if (clean) total += (npy_intp)sum_u8(bytes + done, count);
        }
        HK_END_NOGIL
    }
    if (!clean) return NULL;
    PyObject *result = hk_scalar(NPY_INTP, &total);
    if (result != NULL) hk_take(HK_SUM_U8);
    return result;
}

// The sum of a contiguous uint8 array, as the np.uint64 NumPy returns, or of
// a contiguous boolean array, as the np.intp it returns for that.
PyObject *hk_try_sum(PyObject *array) {
    if (hk_is_flat(array, NPY_BOOL)) return try_sum_bool((PyArrayObject *)array);
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

// The range scans of np.bincount keep BINCOUNT_SCAN_VECTORS running maxima
// in 32-byte vectors, of these types. They are written with the compiler's
// vector types because, written as loops over elements, the compiler kept one
// running maximum for int64: a chain of a compare and a blend through every
// four elements, about 0.3 ns per element on the Intel Core i5-14600KF, as
// long as the kernel itself takes. Several independent maxima leave the loop
// bound by its loads.
typedef uint16_t bincount_u16x16 __attribute__((vector_size(32)));
typedef uint64_t bincount_u64x4 __attribute__((vector_size(32)));
#define BINCOUNT_SCAN_VECTORS 4
// Bytes of one vector, and of the array one step of a scan reads.
#define BINCOUNT_SCAN_VECTOR_BYTES 32
#define BINCOUNT_SCAN_STEP_BYTES (BINCOUNT_SCAN_VECTORS * BINCOUNT_SCAN_VECTOR_BYTES)

// The largest of the `n` int16 elements at `x`, each read as unsigned. A
// negative element reads as 0x8000 or more, so the one maximum is both the
// largest element, when none is negative, and the test for a negative one.
// No branch on the data. The vectors are loaded with memcpy, which the
// compiler turns into unaligned loads, and the maximum is a compare and a
// select, which it turns into vpmaxuw.
static HK_TARGET_KERNEL_LEVEL uint64_t bincount_top_i16(const int16_t *x, size_t n) {
    const size_t step = BINCOUNT_SCAN_STEP_BYTES / sizeof(int16_t);
    bincount_u16x16 top[BINCOUNT_SCAN_VECTORS] = {{0}};
    size_t i = 0;
    for (; i + step <= n; i += step) {
        for (size_t k = 0; k < BINCOUNT_SCAN_VECTORS; k++) {
            bincount_u16x16 v;
            memcpy(&v, (const char *)(x + i) + k * BINCOUNT_SCAN_VECTOR_BYTES, sizeof v);
            const bincount_u16x16 more = (bincount_u16x16)(v > top[k]);
            top[k] = (v & more) | (top[k] & ~more);
        }
    }
    uint16_t largest = 0;
    for (; i < n; i++) largest = (uint16_t)x[i] > largest ? (uint16_t)x[i] : largest;
    for (size_t k = 0; k < BINCOUNT_SCAN_VECTORS; k++) {
        for (size_t j = 0; j < sizeof top[k] / sizeof top[k][0]; j++) {
            largest = top[k][j] > largest ? top[k][j] : largest;
        }
    }
    return largest;
}

// As bincount_top_i16 for int64 elements: a negative one reads as 2^63 or
// more. AVX2 has no 64-bit maximum: the compiler makes a signed compare of
// the elements with their top bit flipped, and a blend.
static HK_TARGET_KERNEL_LEVEL uint64_t bincount_top_i64(const int64_t *x, size_t n) {
    const size_t step = BINCOUNT_SCAN_STEP_BYTES / sizeof(int64_t);
    bincount_u64x4 top[BINCOUNT_SCAN_VECTORS] = {{0}};
    size_t i = 0;
    for (; i + step <= n; i += step) {
        for (size_t k = 0; k < BINCOUNT_SCAN_VECTORS; k++) {
            bincount_u64x4 v;
            memcpy(&v, (const char *)(x + i) + k * BINCOUNT_SCAN_VECTOR_BYTES, sizeof v);
            const bincount_u64x4 more = (bincount_u64x4)(v > top[k]);
            top[k] = (v & more) | (top[k] & ~more);
        }
    }
    uint64_t largest = 0;
    for (; i < n; i++) largest = (uint64_t)x[i] > largest ? (uint64_t)x[i] : largest;
    for (size_t k = 0; k < BINCOUNT_SCAN_VECTORS; k++) {
        for (size_t j = 0; j < sizeof top[k] / sizeof top[k][0]; j++) {
            largest = top[k][j] > largest ? top[k][j] : largest;
        }
    }
    return largest;
}

// np.bincount(x, minlength=m) for a contiguous one-dimensional int16 or
// int64 array of at least one element: the bincount_i16 and bincount_i64
// kernels. `minimum` is m, at least 0.
//
// NumPy sizes its result by the largest element, plus one, or by m if that is
// more, and raises ValueError for a negative element. So the array is read
// once for its largest element first, as an unsigned number, which a negative
// element makes too large for its type's positive range: that call is
// NumPy's, and it raises. Otherwise every element lies in [0, bins), where
// the kernel's counts and NumPy's agree, and the kernel zeroes the counters
// itself. The read cannot be done a block at a time ahead of the kernel, as
// the guards of np.fmax.reduceat and m.any(axis=1) are: the kernel needs the
// number of counters before it reads its first element.
//
// NumPy takes its result from calloc, whose large blocks are pages the system
// zeroes only when they are first touched; the kernel stores every counter.
// A call with far more counters than elements, such as np.bincount([10**9]),
// costs NumPy almost nothing and would have the kernel write 8 GB, so it is
// left to NumPy. So is a largest element whose length overflows, and a
// result that cannot be allocated: NumPy raises its own error.
//
// The array is read twice, by the scan and by the kernel, and the lock is
// released for both all the same: the kernel ignores elements outside
// [0, bins), so a thread that writes the array between the two reads can at
// worst change the counts, never make the kernel write outside them.
static HK_NOINLINE PyObject *bincount_codes(PyArrayObject *x, npy_intp minimum) {
    const bool narrow = PyArray_TYPE(x) == NPY_INT16;
    const npy_intp n = PyArray_SIZE(x);
    // NumPy's answer for no elements is m zeros; the kernel has no largest
    // element to size them by, and the call is NumPy's.
    if (n == 0) return NULL;
    uint64_t top;
    {
        HK_BEGIN_NOGIL(n)
        top = narrow ? bincount_top_i16(PyArray_DATA(x), (size_t)n) : bincount_top_i64(PyArray_DATA(x), (size_t)n);
        HK_END_NOGIL
    }
    // The sign bit of the element type: set in `top` when some element is
    // negative. For int64 it also keeps `top + 1` below 2^63, but the
    // largest int64, whose length NumPy cannot hold, is still excluded.
    const uint64_t sign = narrow ? (uint64_t)1 << 15 : (uint64_t)1 << 63;
    if (top >= sign || top >= (uint64_t)NPY_MAX_INTP) return NULL;
    npy_intp bins = (npy_intp)top + 1;
    if (bins < minimum) bins = minimum;
    if (bins > BINCOUNT_FREE_BINS && bins > n) return NULL;
    PyObject *result = PyArray_SimpleNew(1, &bins, NPY_INTP);
    if (result == NULL) {
        PyErr_Clear();
        return NULL;
    }
    int64_t *counts = PyArray_DATA((PyArrayObject *)result);
    {
        HK_BEGIN_NOGIL(n)
        if (narrow) bincount_i16(PyArray_DATA(x), counts, (size_t)n, (size_t)bins);
        else bincount_i64(PyArray_DATA(x), counts, (size_t)n, (size_t)bins);
        HK_END_NOGIL
    }
    hk_take(narrow ? HK_BINCOUNT_I16 : HK_BINCOUNT_I64);
    return result;
}

// np.bincount(x, minlength=m) for a contiguous one-dimensional uint8, int16
// or int64 array without weights. For uint8 the kernel always fills 256
// counters. NumPy's result has one entry more than the largest value present,
// or `minlength` entries if that is more, so the counters are cut or padded
// to that length. int16 and int64 go to bincount_codes.
static HK_NOINLINE PyObject *try_bincount(PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    enum { X, WEIGHTS, MINLENGTH, PARAMETERS };
    // `x` can only be given by position.
    static const char *const names[PARAMETERS] = {NULL, "weights", "minlength"};
    PyObject *bound[PARAMETERS];
    if (!hk_bind_arguments(args, nargs, kwnames, names, PARAMETERS, PARAMETERS, bound)) return NULL;
    if (bound[X] == NULL || !hk_is_absent(bound[WEIGHTS])) return NULL;
    // The element types with a kernel. Every other one, bool and int32
    // included, which NumPy converts to intp, stays with NumPy.
    const bool bytes = hk_is_flat_1d(bound[X], NPY_UINT8);
    if (!bytes && !hk_is_flat_1d(bound[X], NPY_INT16) && !hk_is_flat_1d(bound[X], NPY_INT64)) return NULL;

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
    if (!bytes) return bincount_codes(x, minimum);
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
    if (!hk_same_shape((PyArrayObject *)bound[DST], (PyArrayObject *)bound[WHERE])) return false;
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

// -------------------------------------------------------------------- clip
//
// `np.clip(x, lo, hi)` runs a Python wrapper, then `x.clip`, then NumPy's
// `_methods._clip`, then the `clip` ufunc, whose inner loop the loop hook
// serves. On a small array that path is nearly the whole call. The function
// and method hooks call the kernel directly when the call needs nothing the
// ufunc machinery does: no output buffer, no `dtype`, `casting` or `where`,
// and bounds that leave the result in the array's element type.
//
// Which element type NumPy gives the result follows NEP 50. A Python float
// or int is a weak scalar and takes the array's type: a float32 array clipped
// to Python floats stays float32. A NumPy scalar is not: clipped to
// np.float64 bounds, a float32 array gives float64. So a bound is accepted
// when it is a Python float, a Python int, or a NumPy scalar of exactly the
// array's type, and anything else goes to NumPy.

// The smallest normal float32 and the largest finite one, as doubles. NumPy
// converts a Python float bound for a float32 array with a C cast, which
// raises the overflow flag beyond the largest (NumPy warns) and the underflow
// flag below the smallest normal. Such a bound is left to NumPy, and the
// conversion here never raises a flag NumPy reports.
#define CLIP_F32_SMALLEST_NORMAL ((double)FLT_MIN)
#define CLIP_F32_LARGEST ((double)FLT_MAX)

// True when np.clip accepts the keywords `min` and `max`, which NumPy 2.1
// added. In NumPy 2.0 they are passed on to the ufunc, which rejects them.
// Set when the hook is applied.
static bool clip_takes_min_max;

// Convert a bound for a float64 array: a Python float, a Python int a double
// holds exactly, or an np.float64. Returns false for anything else.
static bool clip_bound_f64(PyObject *bound, double *value) {
    if (Py_IS_TYPE(bound, &PyDoubleArrType_Type)) {
        *value = PyArrayScalar_VAL(bound, Double);
        return true;
    }
    return hk_as_exact_double(bound, value);
}

// Convert a bound for a float32 array as NumPy does: an np.float32 as it is,
// a Python float rounded to the nearest float32, a Python int only when a
// float32 holds it exactly. Returns false for anything else, and for a
// Python float that is a NaN (a NaN bound is NumPy's either way) or whose
// conversion would overflow or underflow. The conversion can raise only the
// inexact flag, which NumPy never reports.
static bool clip_bound_f32(PyObject *bound, float *value) {
    if (Py_IS_TYPE(bound, &PyFloatArrType_Type)) {
        *value = PyArrayScalar_VAL(bound, Float);
        return true;
    }
    double d;
    if (PyFloat_CheckExact(bound)) {
        d = PyFloat_AS_DOUBLE(bound);
        // The comparison macros, unlike the operators, raise no flag for a NaN.
        const double magnitude = fabs(d);
        const bool in_range = magnitude == 0.0 || isinf(d) ||
                              (isgreaterequal(magnitude, CLIP_F32_SMALLEST_NORMAL) &&
                               islessequal(magnitude, CLIP_F32_LARGEST));
        if (!in_range) return false;
        *value = (float)d;
        return true;
    }
    if (!PyLong_CheckExact(bound) || !hk_as_exact_double(bound, &d)) return false;
    *value = (float)d;
    return (double)*value == d;
}

// Run a clip kernel on the whole of `x` into a new array of its shape. The
// kernels leave the invalid-operation flag set on NaN data; NumPy's own loop
// ends by clearing the flags, and so does this, as the loop hook does.
static PyObject *run_clip(PyArrayObject *x, hk_kernel kernel, double lo, double hi) {
    PyObject *result = PyArray_SimpleNew(PyArray_NDIM(x), PyArray_DIMS(x), PyArray_TYPE(x));
    if (result == NULL) {
        PyErr_Clear();
        return NULL;
    }
    const npy_intp n = PyArray_SIZE(x);
    void *out = PyArray_DATA((PyArrayObject *)result);
    {
        HK_BEGIN_NOGIL(n)
        if (kernel == HK_CLIP_F32) clip_f32(PyArray_DATA(x), out, (size_t)n, (float)lo, (float)hi);
        else clip_f64(PyArray_DATA(x), out, (size_t)n, lo, hi);
        feclearexcept(FE_ALL_EXCEPT);
        HK_END_NOGIL
    }
    hk_take(kernel);
    return result;
}

// The bounds are compared with islessequal, which raises no flag for a NaN:
// a NaN bound fails the comparison, as does lo > hi, and both are NumPy's.
// The float32 bounds are compared after their conversion, as the loop sees
// them.
HK_NOINLINE PyObject *hk_try_clip(PyObject *array, PyObject *lo, PyObject *hi) {
    if (!hk_is_plain_array(array)) return NULL;
    PyArrayObject *x = (PyArrayObject *)array;
    // NumPy returns an empty array without calling the loop; the kernels are
    // never called with n = 0.
    if (PyArray_SIZE(x) < 1) return NULL;
    if (hk_is_flat(array, NPY_FLOAT)) {
        float lo32, hi32;
        if (!clip_bound_f32(lo, &lo32) || !clip_bound_f32(hi, &hi32) || !islessequal(lo32, hi32)) return NULL;
        return run_clip(x, HK_CLIP_F32, lo32, hi32);
    }
    if (hk_is_flat(array, NPY_DOUBLE)) {
        double lo64, hi64;
        if (!clip_bound_f64(lo, &lo64) || !clip_bound_f64(hi, &hi64) || !islessequal(lo64, hi64)) return NULL;
        return run_clip(x, HK_CLIP_F64, lo64, hi64);
    }
    return NULL;
}

// np.clip(a, a_min, a_max) for a contiguous float32 or float64 array, with
// the bounds given by position, as `a_min=` and `a_max=`, or, where NumPy
// has them, as `min=` and `max=`; `out` absent or None; nothing else. A
// bound left out, None, or both spellings at once are NumPy's: it raises, or
// calls np.maximum or np.minimum instead.
static HK_NOINLINE PyObject *try_clip(PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    enum { A, A_MIN, A_MAX, OUT, MIN, MAX, PARAMETERS };
    // `min` and `max` can only be given by keyword.
    static const char *const names[PARAMETERS] = {"a", "a_min", "a_max", "out", "min", "max"};
    PyObject *bound[PARAMETERS];
    if (!hk_bind_arguments(args, nargs, kwnames, names, PARAMETERS, OUT + 1, bound)) return NULL;
    if (bound[A] == NULL || !hk_is_absent(bound[OUT])) return NULL;
    PyObject *lo = bound[A_MIN], *hi = bound[A_MAX];
    if (bound[MIN] != NULL || bound[MAX] != NULL) {
        if (!clip_takes_min_max || lo != NULL || hi != NULL) return NULL;
        lo = bound[MIN];
        hi = bound[MAX];
    }
    // A bound left out or None is the commonest call left to NumPy, and the
    // cheapest to recognise.
    if (hk_is_absent(lo) || hk_is_absent(hi)) return NULL;
    return hk_try_clip(bound[A], lo, hi);
}

// A call left to NumPy reaches `x.clip` and then the clip loop, whose hooks
// count it, so it is not counted here.
static PyObject *call_clip(HK_VECTORCALL_PARAMETERS) {
    PyObject *result = try_clip(args, PyVectorcall_NARGS(nargsf), kwnames);
    if (result != NULL) return result;
    return functions[F_CLIP].original(callable, args, nargsf, kwnames);
}

// Find out whether this NumPy's np.clip takes `min` and `max`: they are
// keyword-only parameters of the Python function behind the function object,
// which NumPy keeps as its `_implementation`. If that cannot be read, the
// keywords are left to NumPy.
static void prepare_clip(PyObject *object) {
    clip_takes_min_max = false;
    PyObject *implementation = PyObject_GetAttrString(object, "_implementation");
    PyObject *defaults = implementation ? PyObject_GetAttrString(implementation, "__kwdefaults__") : NULL;
    if (defaults != NULL && PyDict_Check(defaults)) {
        clip_takes_min_max = PyDict_GetItemString(defaults, "min") != NULL &&
                             PyDict_GetItemString(defaults, "max") != NULL;
    }
    Py_XDECREF(defaults);
    Py_XDECREF(implementation);
    PyErr_Clear();
}

// ------------------------------------------------------------------- where
//
// np.where has two operations behind one name. With the condition alone it
// gives the positions that are True, as np.nonzero does: the nonzero_bool
// kernel, counted under nonzero. With `x` and `y` it selects, element by
// element, from `x` where the condition is True and from `y` where it is
// not: the where_f64 kernel, counted under where. NumPy accepts no keywords
// for either, and a call with keywords or another number of arguments is
// passed on without being counted.
//
// The selection is taken only when it needs no broadcasting and no type
// conversion: the condition and the arrays have one shape, and the arrays
// one element type. The kernel only moves 64-bit elements, so it serves
// float64 and int64 alike, and a NaN keeps its payload. Which type NumPy
// gives the result follows NEP 50: beside a float64 array a Python float, a
// Python int or an np.float64 leaves it float64, so such a scalar is taken
// on either side, converted as a float64 clip bound is (clip_bound_f64: an
// int only when a double holds it exactly). Beside an int64 array a float
// would change the result's type, and an int may not fit; both are NumPy's.

// Run where_f64 into a new array of the condition's shape and of NumPy type
// `type_num`. `a` and `b` are the data of the arrays selected from, of the
// condition's size; either may be NULL, and that side is then the scalar
// `fill` at every position. The result is the binding's own buffer and cannot
// overlap an input, which the kernel requires.
//
// The kernel reads n elements from each side, so a scalar is spread over a
// buffer on the stack, WHERE_BLOCK elements long, and the kernel runs block
// by block with that buffer as the scalar's side. A buffer of n elements
// would be written in full and read back from wherever it lies; at a million
// elements that made the call no faster than NumPy's. The blocks give up the
// kernel's stores past the caches, which it starts from millions of elements.
//
// The kernel takes any non-zero byte of the condition as True, as NumPy
// does, but its spec promises only 0 and 1 and ARM64 runs it as C; the
// condition is checked first, as for the other mask kernels, and one with
// another byte is NumPy's. Returns NULL with no exception set when the call
// goes to NumPy after all.
static HK_NOINLINE PyObject *run_where(PyArrayObject *condition, const void *a, const void *b, double fill,
                                       int type_num) {
    const npy_intp n = PyArray_SIZE(condition);
    PyObject *result = PyArray_SimpleNew(PyArray_NDIM(condition), PyArray_DIMS(condition), type_num);
    if (result == NULL) {
        PyErr_Clear();
        return NULL;
    }
    const bool *c = PyArray_DATA(condition);
    double *out = PyArray_DATA((PyArrayObject *)result);
    bool clean = true;
    {
        // The kernel reads each input once and writes only the new array, which
        // no other thread can see yet.
        HK_BEGIN_NOGIL(n)
        if (a != NULL && b != NULL) {
            clean = hk_mask_is_clean((const uint8_t *)c, (size_t)n);
            if (clean) where_f64(c, a, b, out, (size_t)n);
        } else {
            double spread[WHERE_BLOCK];
            const npy_intp filled = n < WHERE_BLOCK ? n : WHERE_BLOCK;
            for (npy_intp i = 0; i < filled; i++) spread[i] = fill;
            const double *array = a != NULL ? a : b;
            // Each block's condition is checked just before the kernel reads it,
            // from the first-level cache. A bad byte drops the whole result.
            for (npy_intp done = 0; done < n; done += WHERE_BLOCK) {
                const size_t count = (size_t)(n - done < WHERE_BLOCK ? n - done : WHERE_BLOCK);
                clean = hk_mask_is_clean((const uint8_t *)c + done, count);
                if (!clean) break;
                if (a != NULL) where_f64(c + done, array + done, spread, out + done, count);
                else where_f64(c + done, spread, array + done, out + done, count);
            }
        }
        HK_END_NOGIL
    }
    if (!clean) {
        Py_DECREF(result);
        return NULL;
    }
    hk_take(HK_WHERE_F64);
    return result;
}

// np.where(condition, x, y) for a contiguous boolean condition with at least
// one dimension and, of exactly its shape, two contiguous float64 arrays, two
// contiguous int64 arrays, or one contiguous float64 array and a scalar that
// keeps the result float64 (see above). An empty condition is taken: the
// kernel writes nothing and the result is NumPy's empty array of that shape.
static HK_NOINLINE PyObject *try_where(PyObject *condition, PyObject *x, PyObject *y) {
    if (!hk_is_flat(condition, NPY_BOOL)) return NULL;
    PyArrayObject *c = (PyArrayObject *)condition;
    double fill = 0.0;
    if (hk_is_plain_array(x) && hk_is_plain_array(y)) {
        const int type_num = PyArray_TYPE((PyArrayObject *)x);
        if (type_num != NPY_DOUBLE && type_num != NPY_INT64) return NULL;
        if (!hk_is_flat(x, type_num) || !hk_is_flat(y, type_num)) return NULL;
        if (!hk_same_shape(c, (PyArrayObject *)x) || !hk_same_shape(c, (PyArrayObject *)y)) return NULL;
        return run_where(c, PyArray_DATA((PyArrayObject *)x), PyArray_DATA((PyArrayObject *)y), fill, type_num);
    }
    if (hk_is_flat(x, NPY_DOUBLE) && hk_same_shape(c, (PyArrayObject *)x) && clip_bound_f64(y, &fill)) {
        return run_where(c, PyArray_DATA((PyArrayObject *)x), NULL, fill, NPY_DOUBLE);
    }
    if (hk_is_flat(y, NPY_DOUBLE) && hk_same_shape(c, (PyArrayObject *)y) && clip_bound_f64(x, &fill)) {
        return run_where(c, NULL, PyArray_DATA((PyArrayObject *)y), fill, NPY_DOUBLE);
    }
    return NULL;
}

// np.where(condition, x, y), and np.where(condition) with nothing but a
// one-dimensional contiguous boolean array: the positions that are True, as
// np.nonzero gives them. np.where calls NumPy's C code for nonzero directly,
// so the method hook does not see that call.
static PyObject *call_where(HK_VECTORCALL_PARAMETERS) {
    const Py_ssize_t nargs = PyVectorcall_NARGS(nargsf);
    if (kwnames == NULL && nargs == 3) {
        PyObject *result = try_where(args[0], args[1], args[2]);
        if (result != NULL) return result;
        hk_pass(HK_OP_WHERE);
    } else if (kwnames == NULL && nargs == 1) {
        PyObject *result = hk_try_nonzero(args[0]);
        if (result != NULL) return result;
        hk_pass(HK_OP_NONZERO);
    }
    return functions[F_WHERE].original(callable, args, nargsf, kwnames);
}

// --------------------------------------------------------------------- any

// np.any(m, axis=1) for a two-dimensional boolean array: the array and the
// axis, the Python int 1 or -1, by position or the axis as `axis=`, and
// nothing else. For a plain ndarray np.any calls np.logical_or.reduce itself
// and never reaches the method, so it needs this hook as well; the work is
// the method hook's (hk_try_any_rows). The parameters are a, axis, out,
// keepdims and, by keyword only, where.
static HK_NOINLINE PyObject *try_any(PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    const Py_ssize_t keywords = kwnames == NULL ? 0 : PyTuple_GET_SIZE(kwnames);
    if (nargs < 1 || nargs + keywords != 2) return NULL;
    if (keywords == 1 && !hk_text_is(PyTuple_GET_ITEM(kwnames, 0), "axis")) return NULL;
    // A keyword's value follows the positional arguments, so either way the
    // axis is args[1].
    if (!hk_is_last_axis(args[1])) return NULL;
    return hk_try_any_rows(args[0]);
}

// `np.any(m)` with the array alone is one of the commonest NumPy calls and
// has no kernel: it goes on at the first test, uncounted, as does any call
// whose first argument is not a plain boolean array of two dimensions. A call
// on such an array with more arguments that is not taken is counted.
static PyObject *call_any(HK_VECTORCALL_PARAMETERS) {
    const Py_ssize_t nargs = PyVectorcall_NARGS(nargsf);
    if ((nargs > 1 || kwnames != NULL) && nargs >= 1 && hk_is_bool_matrix(args[0])) {
        PyObject *result = try_any(args, nargs, kwnames);
        if (result != NULL) return result;
        hk_pass(HK_OP_ANY);
    }
    return functions[F_ANY].original(callable, args, nargsf, kwnames);
}

// ----------------------------------------------------------------- the hook

static hk_function functions[F_COUNT] = {
    [F_ARGMAX] = {"argmax", call_argmax, NULL, NULL, NULL},
    [F_ARGMIN] = {"argmin", call_argmin, NULL, NULL, NULL},
    [F_COUNT_NONZERO] = {"count_nonzero", call_count_nonzero, NULL, NULL, NULL},
    [F_SUM] = {"sum", call_sum, NULL, NULL, NULL},
    [F_HISTOGRAM] = {"histogram", call_histogram, NULL, NULL, NULL},
    [F_BINCOUNT] = {"bincount", call_bincount, NULL, NULL, NULL},
    [F_ISIN] = {"isin", call_isin, NULL, NULL, NULL},
    [F_INTERP] = {"interp", call_interp, NULL, NULL, NULL},
    [F_COMPRESS] = {"compress", call_compress, NULL, NULL, NULL},
    [F_EXTRACT] = {"extract", call_extract, NULL, NULL, NULL},
    [F_PUTMASK] = {"putmask", call_putmask, NULL, NULL, NULL},
    [F_COPYTO] = {"copyto", call_copyto, NULL, NULL, NULL},
    [F_WHERE] = {"where", call_where, NULL, NULL, NULL},
    [F_CLIP] = {"clip", call_clip, prepare_clip, NULL, NULL},
    [F_ANY] = {"any", call_any, NULL, NULL, NULL},
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
    if (functions[index].prepare != NULL) functions[index].prepare(object);
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
    {"function.clip", function_apply, function_remove, F_CLIP, false},
    {"function.any", function_apply, function_remove, F_ANY, false},
    {NULL, NULL, NULL, 0, false},
};
