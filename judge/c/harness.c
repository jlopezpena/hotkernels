// Timing helpers shared by every generated benchmark loop.
#include <stdint.h>
#include <time.h>

#ifdef __APPLE__
#include <pthread.h>
#include <pthread/qos.h>
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

// Ask for the highest scheduling class, which keeps the thread on performance cores.
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
