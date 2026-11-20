// A caching data allocator for NumPy, built by the gradient-step study
// (docs/gradient-step-findings.md) to measure what reusing freed array memory
// is worth. It is a prototype for measurement, not part of the package.
//
// NumPy takes the memory of an array's elements from a handler that can be
// replaced (NEP 49: PyDataMem_SetHandler). This one keeps up to SLOTS freed
// blocks of at least SMALLEST bytes, CAPACITY bytes in all, and hands a kept
// block back to the next request of exactly its size. Everything else goes to
// the C library. One mutex guards the slots, because NumPy may allocate
// without the interpreter lock.
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_2_0_API_VERSION
#include <numpy/arrayobject.h>

#include <pthread.h>
#include <stdlib.h>
#include <string.h>

// Freed blocks kept at once.
#define SLOTS 32
// Blocks below this size go straight back to the C library, whose own caches
// already serve them.
#define SMALLEST (64 * 1024)
// Most memory the cache holds, in bytes.
#define CAPACITY (64 * 1024 * 1024)

typedef struct {
    // Start of a kept block, or NULL for an empty slot.
    void *pointer;
    // Its size in bytes, as NumPy asked for it.
    size_t size;
} slot;

static slot slots[SLOTS];
static size_t held;
static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
// Requests served from the cache, and requests passed to the C library.
static unsigned long long hits, misses;

// Take a kept block of exactly `size` bytes out of the cache, or return NULL.
static void *take(size_t size) {
    void *found = NULL;
    pthread_mutex_lock(&lock);
    for (int i = 0; i < SLOTS; i++) {
        if (slots[i].pointer != NULL && slots[i].size == size) {
            found = slots[i].pointer;
            slots[i].pointer = NULL;
            held -= size;
            break;
        }
    }
    if (found) hits++; else misses++;
    pthread_mutex_unlock(&lock);
    return found;
}

// Keep a freed block if there is room. Returns 1 if it was kept.
static int keep(void *pointer, size_t size) {
    int kept = 0;
    pthread_mutex_lock(&lock);
    if (held + size <= CAPACITY) {
        for (int i = 0; i < SLOTS; i++) {
            if (slots[i].pointer == NULL) {
                slots[i].pointer = pointer;
                slots[i].size = size;
                held += size;
                kept = 1;
                break;
            }
        }
    }
    pthread_mutex_unlock(&lock);
    return kept;
}

static void *cache_malloc(void *ctx, size_t size) {
    (void)ctx;
    if (size >= SMALLEST) {
        void *p = take(size);
        if (p) return p;
    }
    return malloc(size);
}

// calloc must return zeros, so a kept block is cleared; NumPy uses calloc for
// np.zeros, where the C library can often hand out pages that are zero already.
static void *cache_calloc(void *ctx, size_t count, size_t size) {
    (void)ctx;
    size_t bytes = count * size;
    if (size != 0 && bytes / size != count) return NULL;
    if (bytes >= SMALLEST) {
        void *p = take(bytes);
        if (p) return memset(p, 0, bytes);
    }
    return calloc(count, size);
}

// A block that is resized leaves the cache's hands: realloc goes to the C library.
static void *cache_realloc(void *ctx, void *pointer, size_t size) {
    (void)ctx;
    return realloc(pointer, size);
}

// NumPy passes the size it asked for. A block NumPy resized was taken from the
// C library with realloc, and its size is again what NumPy last asked for.
static void cache_free(void *ctx, void *pointer, size_t size) {
    (void)ctx;
    if (pointer == NULL) return;
    if (size >= SMALLEST && keep(pointer, size)) return;
    free(pointer);
}

static PyDataMem_Handler handler = {
    "gradient_step_study_cache",
    1,
    {NULL, cache_malloc, cache_calloc, cache_realloc, cache_free},
};

// The handler NumPy had before install(), put back by uninstall().
static PyObject *previous;

// install(): make the cache NumPy's data allocator for this thread's context.
static PyObject *install(PyObject *self, PyObject *args) {
    (void)self; (void)args;
    PyObject *capsule = PyCapsule_New(&handler, "mem_handler", NULL);
    if (capsule == NULL) return NULL;
    PyObject *old = PyDataMem_SetHandler(capsule);
    Py_DECREF(capsule);
    if (old == NULL) return NULL;
    Py_XSETREF(previous, old);
    Py_RETURN_NONE;
}

// uninstall(): put the previous handler back. Kept blocks stay in the cache;
// arrays made under it are freed through it whenever they die.
static PyObject *uninstall(PyObject *self, PyObject *args) {
    (void)self; (void)args;
    if (previous != NULL) {
        PyObject *ours = PyDataMem_SetHandler(previous);
        Py_XDECREF(ours);
        Py_CLEAR(previous);
    }
    Py_RETURN_NONE;
}

// counts(): (requests served from the cache, requests passed on, bytes held).
static PyObject *counts(PyObject *self, PyObject *args) {
    (void)self; (void)args;
    pthread_mutex_lock(&lock);
    PyObject *result = Py_BuildValue("KKn", hits, misses, (Py_ssize_t)held);
    pthread_mutex_unlock(&lock);
    return result;
}

static PyMethodDef methods[] = {
    {"install", install, METH_NOARGS, "Make the cache NumPy's data allocator."},
    {"uninstall", uninstall, METH_NOARGS, "Put NumPy's previous data allocator back."},
    {"counts", counts, METH_NOARGS, "(hits, misses, bytes held)."},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "study_allocator", NULL, -1, methods};

PyMODINIT_FUNC PyInit_study_allocator(void) {
    import_array();
    return PyModule_Create(&module);
}
