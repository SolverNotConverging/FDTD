# cython: boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True

"""Cython time-step kernel for the equatorial Schwarzschild solver.

The public function deliberately accepts the already sampled metric/material
and damping arrays.  Geometry construction and policy stay in Python; only the
hot, allocation-free polar Yee loop lives here.
"""


ctypedef double complex field_complex_t


cpdef void step_fields(
    field_complex_t[:, ::1] hz,
    field_complex_t[:, ::1] er,
    field_complex_t[:, ::1] ephi,
    const double[::1] rho_centers,
    const double[::1] rho_faces,
    const double[::1] n_hz,
    const double[::1] n_er,
    const double[::1] n_ephi,
    const double[::1] damp_h_half,
    const double[::1] damp_er_half,
    const double[::1] damp_ephi_half,
    double dt,
    double drho,
    double dphi,
    int boundary_mode,
    Py_ssize_t count,
):
    """Advance complex polar-Yee fields by ``count`` complete time steps.

    ``boundary_mode=0`` applies the solver's unit-impedance one-way radial
    conditions; ``boundary_mode=1`` applies PEC radial walls.  The operation
    order intentionally mirrors :meth:`FDTD_2D_GR.step`, including both
    half-sponge applications and every boundary refresh.
    """

    cdef Py_ssize_t i, j, j_next, j_previous, iteration
    cdef Py_ssize_t nr = hz.shape[0]
    cdef Py_ssize_t nphi = hz.shape[1]
    cdef field_complex_t radial_curl, angular_curl, curl_e
    cdef field_complex_t d_hz_dphi, d_hz_drho

    if count < 1:
        raise ValueError("count must be a positive integer.")
    if boundary_mode != 0 and boundary_mode != 1:
        raise ValueError("boundary_mode must be 0 (characteristic) or 1 (PEC).")
    if drho <= 0.0 or dphi <= 0.0 or dt <= 0.0:
        raise ValueError("drho, dphi, and dt must be positive.")
    if er.shape[0] != nr or er.shape[1] != nphi:
        raise ValueError("Er must have the same shape as Hz.")
    if ephi.shape[0] != nr + 1 or ephi.shape[1] != nphi:
        raise ValueError("Ephi must have shape (Nr + 1, Nphi).")
    if rho_centers.shape[0] != nr or rho_faces.shape[0] != nr + 1:
        raise ValueError("Polar radius arrays do not match the field shapes.")
    if n_hz.shape[0] != nr or n_er.shape[0] != nr or n_ephi.shape[0] != nr + 1:
        raise ValueError("Material arrays do not match the field shapes.")
    if (
        damp_h_half.shape[0] != nr
        or damp_er_half.shape[0] != nr
        or damp_ephi_half.shape[0] != nr + 1
    ):
        raise ValueError("Damping arrays do not match the field shapes.")

    with nogil:
        for iteration in range(count):
            # First matched-loss half step on E.
            for i in range(nr):
                for j in range(nphi):
                    er[i, j] = er[i, j] * damp_er_half[i]
            for i in range(nr + 1):
                for j in range(nphi):
                    ephi[i, j] = ephi[i, j] * damp_ephi_half[i]

            # Radial boundary refresh before curl(E).
            if boundary_mode == 0:
                for j in range(nphi):
                    ephi[0, j] = -hz[0, j]
                    ephi[nr, j] = hz[nr - 1, j]
            else:
                for j in range(nphi):
                    ephi[0, j] = 0.0
                    ephi[nr, j] = 0.0

            # Magnetic half-step staggering plus matched H damping.
            for i in range(nr):
                for j in range(nphi):
                    j_next = j + 1
                    if j_next == nphi:
                        j_next = 0
                    radial_curl = (
                        rho_faces[i + 1] * ephi[i + 1, j]
                        - rho_faces[i] * ephi[i, j]
                    ) / drho
                    angular_curl = (er[i, j_next] - er[i, j]) / dphi
                    curl_e = (radial_curl - angular_curl) / rho_centers[i]
                    hz[i, j] = hz[i, j] * damp_h_half[i]
                    hz[i, j] = hz[i, j] - dt * curl_e / n_hz[i]
                    hz[i, j] = hz[i, j] * damp_h_half[i]

            # Electric update.  Azimuth is exactly periodic; radial Ephi
            # contains one extra face and is updated only in the interior.
            for i in range(nr):
                for j in range(nphi):
                    j_previous = j - 1
                    if j == 0:
                        j_previous = nphi - 1
                    d_hz_dphi = (
                        (hz[i, j] - hz[i, j_previous])
                        / (rho_centers[i] * dphi)
                    )
                    er[i, j] = er[i, j] + dt * d_hz_dphi / n_er[i]
            for i in range(1, nr):
                for j in range(nphi):
                    d_hz_drho = (hz[i, j] - hz[i - 1, j]) / drho
                    ephi[i, j] = ephi[i, j] - dt * d_hz_drho / n_ephi[i]

            # The Python path refreshes the radial condition both before and
            # after the second E damping half step; preserve that ordering.
            if boundary_mode == 0:
                for j in range(nphi):
                    ephi[0, j] = -hz[0, j]
                    ephi[nr, j] = hz[nr - 1, j]
            else:
                for j in range(nphi):
                    ephi[0, j] = 0.0
                    ephi[nr, j] = 0.0

            for i in range(nr):
                for j in range(nphi):
                    er[i, j] = er[i, j] * damp_er_half[i]
            for i in range(nr + 1):
                for j in range(nphi):
                    ephi[i, j] = ephi[i, j] * damp_ephi_half[i]

            if boundary_mode == 0:
                for j in range(nphi):
                    ephi[0, j] = -hz[0, j]
                    ephi[nr, j] = hz[nr - 1, j]
            else:
                for j in range(nphi):
                    ephi[0, j] = 0.0
                    ephi[nr, j] = 0.0
