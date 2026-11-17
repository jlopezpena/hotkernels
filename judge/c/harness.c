// Timing helpers shared by every generated benchmark loop.
#ifndef __APPLE__
// On Linux, clock_gettime is only declared under a feature macro when the
// compiler is asked for strict C11.
#define _GNU_SOURCE
#endif

#include <stdint.h>
#include <time.h>

#ifdef __APPLE__
#include <pthread.h>
#include <pthread/qos.h>
#endif

#ifdef __x86_64__
#include <cpuid.h>

// CPUID leaf and sub-leaf that describe the processor's extended state features.
#define HK_CPUID_XSTATE_LEAF 0xD
#define HK_CPUID_XSTATE_SUBLEAF 1
// In that leaf's EAX: XGETBV with ECX = 1 is supported, which reports which
// parts of the vector state are in use.
#define HK_CPUID_XGETBV1_BIT 2

// Read by the calling-convention wrapper (abi_x86_64.s): 1 when it can ask the
// processor whether a kernel left the upper halves of the vector registers in use.
__attribute__((visibility("hidden"))) uint32_t hk_can_see_upper_halves;

// Runs when the library is loaded, so the wrapper works without hk_init.
__attribute__((constructor)) static void hk_detect(void) {
    unsigned eax = 0, ebx = 0, ecx = 0, edx = 0;
    int known = __get_cpuid_count(HK_CPUID_XSTATE_LEAF, HK_CPUID_XSTATE_SUBLEAF, &eax, &ebx, &ecx, &edx);
    hk_can_see_upper_halves = known && ((eax >> HK_CPUID_XGETBV1_BIT) & 1);
}
#endif

uint64_t hk_now(void) {
#ifdef __APPLE__
    return clock_gettime_nsec_np(CLOCK_MONOTONIC_RAW);
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
#endif
}

// Ask for the highest scheduling class, which keeps the thread on performance
// cores. That is how macOS does it; on Linux the judge pins the thread to a
// performance core from Python (judge/bench.py), so there is nothing to do here.
int hk_init(void) {
#ifdef __APPLE__
    return pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#else
    return 0;
#endif
}

// A fixed amount of work. Timed before and after a benchmark to detect a noisy machine.
uint64_t hk_canary(void) {
    volatile uint64_t sink;
    uint64_t x = 88172645463325252ull;
    uint64_t t0 = hk_now();
    for (uint64_t i = 0; i < 20000000ull; i++) {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
    }
    sink = x;
    (void)sink;
    return hk_now() - t0;
}
