# Native terminal reporting: no Python callbacks or numerical buffer access.
cdef extern from *:
    r"""
    #include <stdio.h>
    #ifdef _WIN32
    #include <windows.h>
    static double fdtd_wall_time(void) {
        LARGE_INTEGER ticks, frequency;
        QueryPerformanceCounter(&ticks);
        QueryPerformanceFrequency(&frequency);
        return (double)ticks.QuadPart / (double)frequency.QuadPart;
    }
    static void *fdtd_progress_wait_create(void) {
        /* CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, available on Windows 10+. */
        return (void *)CreateWaitableTimerExW(NULL, NULL, 0x2, TIMER_ALL_ACCESS);
    }
    static void fdtd_progress_wait_destroy(void *timer) {
        if (timer) CloseHandle((HANDLE)timer);
    }
    static void fdtd_progress_sleep(void *timer) {
        LARGE_INTEGER delay;
        delay.QuadPart = -10000; /* relative 1 ms in 100 ns units */
        if (timer && SetWaitableTimer((HANDLE)timer, &delay, 0, NULL, NULL, FALSE))
            WaitForSingleObject((HANDLE)timer, INFINITE);
        else
            Sleep(1);
    }
    #else
    #include <time.h>
    static double fdtd_wall_time(void) {
        struct timespec now;
        clock_gettime(CLOCK_MONOTONIC, &now);
        return (double)now.tv_sec + 1e-9 * now.tv_nsec;
    }
    static void *fdtd_progress_wait_create(void) { return NULL; }
    static void fdtd_progress_wait_destroy(void *timer) { (void)timer; }
    static void fdtd_progress_sleep(void *timer) {
        struct timespec delay = {0, 1000000};
        nanosleep(&delay, NULL);
    }
    #endif
    static void fdtd_progress_write(long long done, long long total,
                                    double elapsed, int final) {
        char bar[31], line[192];
        double fraction = total ? (double)done / total : 1.0;
        int i, length;
        for (i = 0; i < 30; ++i) bar[i] = i < (int)(30 * fraction) ? '#' : '-';
        bar[30] = 0;
        length = snprintf(line, sizeof(line),
            "\rFDTD [%s] %6.2f%% %lld/%lld steps  %.1fs%s",
            bar, 100 * fraction, done, total, elapsed, final ? "\n" : "");
        if (length > 0 && length < (int)sizeof(line)) {
            fwrite(line, 1, (size_t)length, stderr);
            fflush(stderr);
        }
    }
    """
    double fdtd_wall_time() noexcept nogil
    void *fdtd_progress_wait_create() noexcept nogil
    void fdtd_progress_wait_destroy(void *) noexcept nogil
    void fdtd_progress_sleep(void *) noexcept nogil
    void fdtd_progress_write(long long, long long, double, int) noexcept nogil

cdef struct NativeProgress:
    double start
    double last

cdef void progress_start(NativeProgress *state, Py_ssize_t total) noexcept nogil:
    state.start = fdtd_wall_time()
    state.last = state.start
    fdtd_progress_write(0, total, 0, 0)

cdef void progress_update(NativeProgress *state, Py_ssize_t done,
                          Py_ssize_t total, bint final=False) noexcept nogil:
    cdef double now = fdtd_wall_time()
    if final or now - state.last >= 0.2:
        fdtd_progress_write(done, total, now - state.start, final)
        state.last = now
