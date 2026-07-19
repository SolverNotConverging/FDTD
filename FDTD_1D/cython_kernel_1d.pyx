# cython: boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True


cpdef void update_h(double[::1] hx, double[::1] ey, double[::1] cah,
                    double[::1] mhx, double dz):
    """Update all cell-centred Hx values from the face-centred Ey values."""
    cdef Py_ssize_t i
    cdef Py_ssize_t nz = hx.shape[0]

    if ey.shape[0] != nz + 1 or mhx.shape[0] != nz or cah.shape[0] != nz:
        raise ValueError("Yee array shapes must be Ey=Nz+1 and Hx=CaH=mHx=Nz.")

    with nogil:
        for i in range(nz):
            hx[i] = cah[i] * hx[i] + mhx[i] * (ey[i + 1] - ey[i]) / dz


cpdef void update_e(double[::1] ey, double[::1] hx, double[::1] cae,
                    double[::1] mey, double dz):
    """Update the interior face-centred Ey values from cell-centred Hx."""
    cdef Py_ssize_t i
    cdef Py_ssize_t nz = hx.shape[0]

    if ey.shape[0] != nz + 1 or mey.shape[0] != nz + 1 or cae.shape[0] != nz + 1:
        raise ValueError("Yee array shapes must be Ey=CaE=mEy=Nz+1 and Hx=Nz.")

    with nogil:
        for i in range(1, nz):
            ey[i] = cae[i] * ey[i] + mey[i] * (hx[i] - hx[i - 1]) / dz
