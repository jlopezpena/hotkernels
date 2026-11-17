// Type-slot hook: the `argmax` and `argmin` functions a NumPy type holds.
//
// NumPy keeps, per element type, a table of C functions (PyArray_ArrFuncs).
// Every route to argmax ends in its `argmax` entry: `np.argmax(x)`,
// `x.argmax()`, `axis=` (one call per row or column) and `np.nanargmax`.
// NumPy calls it with a contiguous, native-order run of `n` elements, having
// copied the data first if the array was laid out any other way, so the hook
// needs no checks of its own.
#include "binding.h"

// Below this many elements the hook leaves the call to NumPy's own loop.
// The kernel has to take the interpreter lock (see below), which costs more
// than the kernel saves on a short run. Short one-dimensional calls are
// served by the function hook instead, which already holds the lock.
#define HK_SLOT_MIN_N 128

// One hooked slot.
typedef struct {
    // NumPy type number of the element type whose table is changed.
    int type_num;
    // True for the `argmax` entry, false for `argmin`.
    bool is_max;
    // The binding's function for this slot.
    PyArray_ArgFunc *wrapper;
    // NumPy's own function, kept while the hook is applied.
    PyArray_ArgFunc *original;
} hk_slot;

static hk_slot slots[4];

// NumPy releases the interpreter lock around argmax. The kernels read the
// array twice and are not safe if it changes in between (issue 9), so the
// hook takes the lock back for the kernel. That stops Python threads; code
// that writes the array without the lock is not stopped.
#define HK_SLOT_FUNCTION(function, index, kernel, kernel_id, c_type, operation)        \
    static int function(void *data, npy_intp n, npy_intp *result, void *array) {       \
        if (n < HK_SLOT_MIN_N) {                                                       \
            hk_pass(operation);                                                        \
            return slots[index].original(data, n, result, array);                      \
        }                                                                              \
        PyGILState_STATE lock = PyGILState_Ensure();                                   \
        *result = (npy_intp)kernel((const c_type *)data, (size_t)n);                   \
        PyGILState_Release(lock);                                                      \
        hk_take(kernel_id);                                                            \
        return 0;                                                                      \
    }

HK_SLOT_FUNCTION(slot_argmax_f32, 0, argmax_f32, HK_ARGMAX_F32, float, HK_OP_ARGMAX)
HK_SLOT_FUNCTION(slot_argmax_f64, 1, argmax_f64, HK_ARGMAX_F64, double, HK_OP_ARGMAX)
HK_SLOT_FUNCTION(slot_argmin_f32, 2, argmin_f32, HK_ARGMIN_F32, float, HK_OP_ARGMIN)
HK_SLOT_FUNCTION(slot_argmin_f64, 3, argmin_f64, HK_ARGMIN_F64, double, HK_OP_ARGMIN)

static hk_slot slots[4] = {
    {NPY_FLOAT, true, slot_argmax_f32, NULL},
    {NPY_DOUBLE, true, slot_argmax_f64, NULL},
    {NPY_FLOAT, false, slot_argmin_f32, NULL},
    {NPY_DOUBLE, false, slot_argmin_f64, NULL},
};

// Return the address of the table entry slot `index` replaces. The table
// belongs to the element type and lives as long as NumPy does.
static PyArray_ArgFunc **slot_entry(int index) {
    PyArray_Descr *descr = PyArray_DescrFromType(slots[index].type_num);
    PyArray_ArrFuncs *functions = PyDataType_GetArrFuncs(descr);
    Py_DECREF(descr);
    return slots[index].is_max ? &functions->argmax : &functions->argmin;
}

// Point the slot at the binding. See hk_hook.apply.
static int slot_apply(int index) {
    PyArray_ArgFunc **entry = slot_entry(index);
    if (*entry == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "the NumPy type has no function in this slot");
        return -1;
    }
    slots[index].original = *entry;
    *entry = slots[index].wrapper;
    return 0;
}

// Put NumPy's function back. See hk_hook.remove.
static int slot_remove(int index) {
    PyArray_ArgFunc **entry = slot_entry(index);
    if (*entry != slots[index].wrapper) return 0;
    *entry = slots[index].original;
    return 1;
}

hk_hook hk_slot_hooks[] = {
    {"slot.argmax.f32", slot_apply, slot_remove, 0, false},
    {"slot.argmax.f64", slot_apply, slot_remove, 1, false},
    {"slot.argmin.f32", slot_apply, slot_remove, 2, false},
    {"slot.argmin.f64", slot_apply, slot_remove, 3, false},
    {NULL, NULL, NULL, 0, false},
};
