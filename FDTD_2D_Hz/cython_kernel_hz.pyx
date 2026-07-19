# cython: boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True


cpdef void curl_e(double[:, ::1] ex, double[:, ::1] ey,
                  double[:, ::1] d_ex_y, double[:, ::1] d_ey_x,
                  double dx, double dy):
    cdef Py_ssize_t i, j
    cdef Py_ssize_t nx = d_ex_y.shape[0]
    cdef Py_ssize_t ny = d_ex_y.shape[1]

    with nogil:
        for i in range(nx):
            for j in range(ny):
                d_ex_y[i, j] = (ex[i, j + 1] - ex[i, j]) / dy
                d_ey_x[i, j] = (ey[i + 1, j] - ey[i, j]) / dx


cpdef void curl_h(double[:, ::1] hz, double[:, ::1] d_hz_y,
                  double[:, ::1] d_hz_x, double dx, double dy,
                  bint periodic_x, bint periodic_y):
    cdef Py_ssize_t i, j
    cdef Py_ssize_t nx = hz.shape[0]
    cdef Py_ssize_t ny = hz.shape[1]

    with nogil:
        for i in range(nx):
            d_hz_y[i, 0] = ((hz[i, 0] - hz[i, ny - 1]) if periodic_y else hz[i, 0]) / dy
            for j in range(1, ny):
                d_hz_y[i, j] = (hz[i, j] - hz[i, j - 1]) / dy
            d_hz_y[i, ny] = ((hz[i, 0] - hz[i, ny - 1]) if periodic_y else -hz[i, ny - 1]) / dy

        for j in range(ny):
            d_hz_x[0, j] = ((hz[0, j] - hz[nx - 1, j]) if periodic_x else hz[0, j]) / dx
            for i in range(1, nx):
                d_hz_x[i, j] = (hz[i, j] - hz[i - 1, j]) / dx
            d_hz_x[nx, j] = ((hz[0, j] - hz[nx - 1, j]) if periodic_x else -hz[nx - 1, j]) / dx
