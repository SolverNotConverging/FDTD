# cython: boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True


cpdef void curl_e(double[:, ::1] ez, double[:, ::1] d_ez_y,
                  double[:, ::1] d_ez_x, double dx, double dy,
                  bint periodic_x, bint periodic_y):
    cdef Py_ssize_t i, j
    cdef Py_ssize_t nx = ez.shape[0] - 1
    cdef Py_ssize_t ny = ez.shape[1] - 1

    with nogil:
        for i in range(nx + 1):
            for j in range(ny):
                d_ez_y[i, j] = (ez[i, j + 1] - ez[i, j]) / dy

        for i in range(nx):
            for j in range(ny + 1):
                d_ez_x[i, j] = (ez[i + 1, j] - ez[i, j]) / dx


cpdef void curl_h(double[:, ::1] hx, double[:, ::1] hy,
                  double[:, ::1] d_hx_y, double[:, ::1] d_hy_x,
                  double dx, double dy, bint periodic_x, bint periodic_y):
    cdef Py_ssize_t i, j
    cdef Py_ssize_t nx = d_hx_y.shape[0] - 1
    cdef Py_ssize_t ny = d_hx_y.shape[1] - 1

    with nogil:
        for i in range(nx + 1):
            d_hx_y[i, 0] = ((hx[i, 0] - hx[i, ny - 1]) if periodic_y else hx[i, 0]) / dy
            for j in range(1, ny):
                d_hx_y[i, j] = (hx[i, j] - hx[i, j - 1]) / dy
            d_hx_y[i, ny] = ((hx[i, 0] - hx[i, ny - 1]) if periodic_y else -hx[i, ny - 1]) / dy

        for j in range(ny + 1):
            d_hy_x[0, j] = ((hy[0, j] - hy[nx - 1, j]) if periodic_x else hy[0, j]) / dx
            for i in range(1, nx):
                d_hy_x[i, j] = (hy[i, j] - hy[i - 1, j]) / dx
            d_hy_x[nx, j] = ((hy[0, j] - hy[nx - 1, j]) if periodic_x else -hy[nx - 1, j]) / dx
