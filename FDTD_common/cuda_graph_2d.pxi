# CUDA driver entry points are resolved at runtime; no toolkit headers or link
# libraries are required to build or import the CPU extension.
from libc.stdint cimport uintptr_t
from libc.stdlib cimport malloc, free

cdef extern from *:
    """
    #include <stdlib.h>
    #ifdef _WIN32
    #include <windows.h>
    #define FDTD_CUDA_CALL __stdcall
    static void *fdtd_cuda_library = NULL;
    static void *fdtd_cuda_symbol(const char *name) {
        return (void *)GetProcAddress((HMODULE)fdtd_cuda_library, name);
    }
    #else
    #include <dlfcn.h>
    #define FDTD_CUDA_CALL
    static void *fdtd_cuda_library = NULL;
    static void *fdtd_cuda_symbol(const char *name) {
        return dlsym(fdtd_cuda_library, name);
    }
    #endif
    typedef int (FDTD_CUDA_CALL *fdtd_create_fn)(void **, unsigned int);
    typedef int (FDTD_CUDA_CALL *fdtd_one_fn)(void *);
    typedef int (FDTD_CUDA_CALL *fdtd_begin_fn)(void *, int);
    typedef int (FDTD_CUDA_CALL *fdtd_end_fn)(void *, void **);
    typedef int (FDTD_CUDA_CALL *fdtd_instantiate_fn)(void **, void *, unsigned long long);
    typedef int (FDTD_CUDA_CALL *fdtd_launch_fn)(void *, void *);
    typedef int (FDTD_CUDA_CALL *fdtd_nodes_fn)(void *, void **, size_t *);
    typedef int (FDTD_CUDA_CALL *fdtd_node_type_fn)(void *, int *);
    static fdtd_create_fn fdtd_stream_create;
    static fdtd_one_fn fdtd_stream_destroy, fdtd_stream_sync;
    static fdtd_one_fn fdtd_graph_destroy, fdtd_exec_destroy;
    static fdtd_begin_fn fdtd_begin;
    static fdtd_end_fn fdtd_end;
    static fdtd_instantiate_fn fdtd_instantiate;
    static fdtd_launch_fn fdtd_launch;
    static fdtd_nodes_fn fdtd_nodes;
    static fdtd_node_type_fn fdtd_node_type;
    static fdtd_create_fn fdtd_event_create;
    static fdtd_one_fn fdtd_event_destroy, fdtd_event_query;
    static fdtd_launch_fn fdtd_event_record;
    static int fdtd_cuda_ready = 0;
    static int fdtd_cuda_load(void) {
        if (fdtd_cuda_ready) return 0;
        if (!fdtd_cuda_library) {
    #ifdef _WIN32
            fdtd_cuda_library = (void *)LoadLibraryA("nvcuda.dll");
    #else
            fdtd_cuda_library = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    #endif
        }
        if (!fdtd_cuda_library) return -1;
        fdtd_stream_create = (fdtd_create_fn)fdtd_cuda_symbol("cuStreamCreate");
        fdtd_stream_destroy = (fdtd_one_fn)fdtd_cuda_symbol("cuStreamDestroy_v2");
        fdtd_stream_sync = (fdtd_one_fn)fdtd_cuda_symbol("cuStreamSynchronize");
        fdtd_begin = (fdtd_begin_fn)fdtd_cuda_symbol("cuStreamBeginCapture");
        fdtd_end = (fdtd_end_fn)fdtd_cuda_symbol("cuStreamEndCapture");
        fdtd_instantiate = (fdtd_instantiate_fn)fdtd_cuda_symbol("cuGraphInstantiateWithFlags");
        fdtd_graph_destroy = (fdtd_one_fn)fdtd_cuda_symbol("cuGraphDestroy");
        fdtd_exec_destroy = (fdtd_one_fn)fdtd_cuda_symbol("cuGraphExecDestroy");
        fdtd_launch = (fdtd_launch_fn)fdtd_cuda_symbol("cuGraphLaunch");
        fdtd_nodes = (fdtd_nodes_fn)fdtd_cuda_symbol("cuGraphGetNodes");
        fdtd_node_type = (fdtd_node_type_fn)fdtd_cuda_symbol("cuGraphNodeGetType");
        fdtd_event_create = (fdtd_create_fn)fdtd_cuda_symbol("cuEventCreate");
        fdtd_event_destroy = (fdtd_one_fn)fdtd_cuda_symbol("cuEventDestroy_v2");
        fdtd_event_query = (fdtd_one_fn)fdtd_cuda_symbol("cuEventQuery");
        fdtd_event_record = (fdtd_launch_fn)fdtd_cuda_symbol("cuEventRecord");
        if (!fdtd_stream_create || !fdtd_stream_destroy || !fdtd_stream_sync ||
            !fdtd_begin || !fdtd_end || !fdtd_instantiate || !fdtd_graph_destroy ||
            !fdtd_exec_destroy || !fdtd_launch || !fdtd_nodes || !fdtd_node_type ||
            !fdtd_event_create || !fdtd_event_destroy || !fdtd_event_query ||
            !fdtd_event_record) return -2;
        fdtd_cuda_ready = 1;
        return 0;
    }
    """
    int fdtd_cuda_load()
    int fdtd_stream_create(void **, unsigned int) noexcept nogil
    int fdtd_stream_destroy(void *) noexcept nogil
    int fdtd_stream_sync(void *) noexcept nogil
    int fdtd_begin(void *, int) noexcept nogil
    int fdtd_end(void *, void **) noexcept nogil
    int fdtd_instantiate(void **, void *, unsigned long long) noexcept nogil
    int fdtd_graph_destroy(void *) noexcept nogil
    int fdtd_exec_destroy(void *) noexcept nogil
    int fdtd_launch(void *, void *) noexcept nogil
    int fdtd_nodes(void *, void **, size_t *) noexcept nogil
    int fdtd_node_type(void *, int *) noexcept nogil
    int fdtd_event_create(void **, unsigned int) noexcept nogil
    int fdtd_event_destroy(void *) noexcept nogil
    int fdtd_event_query(void *) noexcept nogil
    int fdtd_event_record(void *, void *) noexcept nogil


cdef void _cuda_check(int result, str operation) except *:
    if result != 0:
        error = RuntimeError(f"CUDA graph {operation} failed (driver error {result}).")
        error.cuda_error_code = result
        raise error


def check_cuda_driver():
    cdef int result = fdtd_cuda_load()
    if result != 0:
        raise RuntimeError('CUDA graph driver entry points are unavailable; '
                           'install a driver supporting CUDA 11.4 or newer.')


cdef class CudaGraph:
    cdef void *_stream
    cdef void *_graph
    cdef void *_executable
    cdef bint _capturing
    cdef void *_events[128]
    cdef void *_progress_timer
    cdef Py_ssize_t _event_count, _progress_steps, _progress_chunk

    def __cinit__(self):
        self._stream = NULL
        self._graph = NULL
        self._executable = NULL
        self._capturing = False
        self._event_count = 0
        self._progress_steps = -1
        self._progress_timer = NULL

    def __init__(self):
        check_cuda_driver()
        _cuda_check(fdtd_stream_create(&self._stream, 1), 'stream creation')

    @property
    def stream(self):
        return <uintptr_t>self._stream

    def begin(self):
        if self._stream == NULL or self._capturing or self._graph != NULL:
            raise RuntimeError('CUDA graph is closed or already captured.')
        _cuda_check(fdtd_begin(self._stream, 0), 'capture begin')
        self._capturing = True

    def end(self):
        cdef size_t count = 0
        cdef size_t index
        cdef int node_type
        cdef void **nodes
        if not self._capturing:
            raise RuntimeError('CUDA graph capture has not started.')
        cdef int result = fdtd_end(self._stream, &self._graph)
        self._capturing = False
        _cuda_check(result, 'capture end')
        # Reject captured transfers, allocations, or host callbacks as well as
        # direct Python-side operations. CU_GRAPH_NODE_TYPE_KERNEL is zero.
        _cuda_check(fdtd_nodes(self._graph, NULL, &count), 'node count')
        nodes = <void **>malloc(max(count, 1) * sizeof(void *))
        if nodes == NULL:
            raise MemoryError('Cannot allocate CUDA graph validation metadata.')
        try:
            _cuda_check(fdtd_nodes(self._graph, nodes, &count), 'node enumeration')
            for index in range(count):
                _cuda_check(fdtd_node_type(nodes[index], &node_type), 'node type')
                if node_type != 0:
                    raise RuntimeError('2D timestep graphs must contain only CUDA kernel nodes.')
        finally:
            free(nodes)
        _cuda_check(fdtd_instantiate(&self._executable, self._graph, 0), 'instantiation')

    def prepare_progress(self, Py_ssize_t steps):
        """Allocate completion markers before stepping; each run owns its events."""
        if self._executable == NULL or self._event_count or self._progress_steps >= 0 or steps < 0:
            raise ValueError('Progress requires a ready graph and nonnegative steps; prepare once.')
        cdef Py_ssize_t count, index
        # Event records are relatively costly under WDDM. Bound their count and
        # amortize each over at least 1024 graph launches, even for short runs.
        self._progress_chunk = max(1024, steps // 128 + (steps % 128 != 0))
        count = steps // self._progress_chunk + (steps % self._progress_chunk != 0)
        for index in range(count):
            # CU_EVENT_DISABLE_TIMING: query completion without timestamp work.
            _cuda_check(fdtd_event_create(&self._events[index], 2), 'progress event creation')
            self._event_count += 1
        self._progress_steps = steps
        self._progress_timer = fdtd_progress_wait_create()

    cdef int _completed(self, Py_ssize_t recorded, Py_ssize_t *completed) noexcept nogil:
        cdef int result
        while completed[0] < recorded:
            result = fdtd_event_query(self._events[completed[0]])
            if result == 600:  # CUDA_ERROR_NOT_READY
                return 0
            if result != 0:
                return result
            completed[0] += 1
        return 0

    def replay(self, Py_ssize_t steps):
        if self._executable == NULL or steps < 0:
            raise ValueError('CUDA graph is not ready or steps is negative.')
        cdef Py_ssize_t step
        cdef Py_ssize_t recorded = 0, completed = 0
        cdef bint progress = self._progress_steps >= 0
        cdef NativeProgress reporter
        cdef int result = 0
        if progress and steps != self._progress_steps:
            raise ValueError('Replay step count differs from prepared progress events.')
        with nogil:
            if progress:
                progress_start(&reporter, steps)
            for step in range(steps):
                result = fdtd_launch(self._executable, self._stream)
                if result != 0:
                    break
                if progress and ((step + 1) % self._progress_chunk == 0 or step + 1 == steps):
                    result = fdtd_event_record(self._events[recorded], self._stream)
                    if result != 0:
                        break
                    recorded += 1
                    if fdtd_wall_time() - reporter.last >= 0.2:
                        result = self._completed(recorded, &completed)
                        if result != 0:
                            break
                        progress_update(&reporter, min(completed * self._progress_chunk, steps), steps)
            if progress and result == 0:
                while completed < recorded:
                    result = self._completed(recorded, &completed)
                    if result != 0:
                        break
                    if completed < recorded:
                        progress_update(&reporter, min(completed * self._progress_chunk, steps), steps)
                        fdtd_progress_sleep(self._progress_timer)
            if result == 0:
                result = fdtd_stream_sync(self._stream)
            if progress:
                progress_update(&reporter, steps if result == 0 else min(completed * self._progress_chunk, steps),
                                steps, True)
        _cuda_check(result, 'replay/synchronization')

    def synchronize(self):
        cdef int result = 0
        if self._stream != NULL and not self._capturing:
            with nogil:
                result = fdtd_stream_sync(self._stream)
            _cuda_check(result, 'synchronization')

    cdef void _close(self) noexcept nogil:
        cdef Py_ssize_t index
        if self._capturing:
            fdtd_end(self._stream, &self._graph)
            self._capturing = False
        if self._stream != NULL:
            fdtd_stream_sync(self._stream)
        for index in range(self._event_count):
            fdtd_event_destroy(self._events[index])
        self._event_count = 0
        self._progress_steps = -1
        fdtd_progress_wait_destroy(self._progress_timer)
        self._progress_timer = NULL
        if self._executable != NULL:
            fdtd_exec_destroy(self._executable)
            self._executable = NULL
        if self._graph != NULL:
            fdtd_graph_destroy(self._graph)
            self._graph = NULL
        if self._stream != NULL:
            fdtd_stream_destroy(self._stream)
            self._stream = NULL

    def close(self):
        with nogil:
            self._close()

    def __dealloc__(self):
        self._close()
