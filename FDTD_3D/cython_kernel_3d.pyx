# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
"""Whole-run Cython kernel for the 3D Yee solver."""

import numpy as np
cimport numpy as cnp


def run_fdtd(
        double[:, :, ::1] Ex, double[:, :, ::1] Ey, double[:, :, ::1] Ez,
        double[:, :, ::1] Hx, double[:, :, ::1] Hy, double[:, :, ::1] Hz,
        double[:, :, ::1] phxy, double[:, :, ::1] phxz,
        double[:, :, ::1] phyx, double[:, :, ::1] phyz,
        double[:, :, ::1] phzx, double[:, :, ::1] phzy,
        double[:, :, ::1] pexy, double[:, :, ::1] pexz,
        double[:, :, ::1] peyx, double[:, :, ::1] peyz,
        double[:, :, ::1] pezx, double[:, :, ::1] pezy,
        double[:, :, ::1] caex, double[:, :, ::1] cbex,
        double[:, :, ::1] caey, double[:, :, ::1] cbey,
        double[:, :, ::1] caez, double[:, :, ::1] cbez,
        double[:, :, ::1] cahx, double[:, :, ::1] cbhx,
        double[:, :, ::1] cahy, double[:, :, ::1] cbhy,
        double[:, :, ::1] cahz, double[:, :, ::1] cbhz,
        double[::1] xnk, double[::1] xnb, double[::1] xnc,
        double[::1] xck, double[::1] xcb, double[::1] xcc,
        double[::1] ynk, double[::1] ynb, double[::1] ync,
        double[::1] yck, double[::1] ycb, double[::1] ycc,
        double[::1] znk, double[::1] znb, double[::1] znc,
        double[::1] zck, double[::1] zcb, double[::1] zcc,
        double dx, double dy, double dz,
        double[:, ::1] source_values, cnp.int32_t[:, ::1] source_coords,
        cnp.int32_t[::1] source_ids, cnp.int8_t[::1] source_pols,
        cnp.int32_t[:, ::1] monitor_coords, double[:, :, ::1] history,
        int record_stride):
    cdef Py_ssize_t Nx = Ex.shape[0]
    cdef Py_ssize_t Ny = Ey.shape[1]
    cdef Py_ssize_t Nz = Ez.shape[2]
    cdef Py_ssize_t steps = source_values.shape[0]
    cdef Py_ssize_t nsource_points = source_coords.shape[0]
    cdef Py_ssize_t nmonitor = monitor_coords.shape[0]
    cdef Py_ssize_t n, i, j, k, q, rec = 0
    cdef double a, b, curl, value

    for n in range(steps):
        # Hx = Hx - dt/mu (dEz/dy - dEy/dz)
        for i in range(Nx + 1):
            for j in range(Ny):
                for k in range(Nz):
                    a = (Ez[i, j + 1, k] - Ez[i, j, k]) / dy
                    b = (Ey[i, j, k + 1] - Ey[i, j, k]) / dz
                    phxy[i, j, k] = ycb[j] * phxy[i, j, k] + ycc[j] * a
                    phxz[i, j, k] = zcb[k] * phxz[i, j, k] + zcc[k] * b
                    curl = a / yck[j] + phxy[i, j, k] - b / zck[k] - phxz[i, j, k]
                    Hx[i, j, k] = cahx[i, j, k] * Hx[i, j, k] - cbhx[i, j, k] * curl

        # Hy = Hy - dt/mu (dEx/dz - dEz/dx)
        for i in range(Nx):
            for j in range(Ny + 1):
                for k in range(Nz):
                    a = (Ex[i, j, k + 1] - Ex[i, j, k]) / dz
                    b = (Ez[i + 1, j, k] - Ez[i, j, k]) / dx
                    phyz[i, j, k] = zcb[k] * phyz[i, j, k] + zcc[k] * a
                    phyx[i, j, k] = xcb[i] * phyx[i, j, k] + xcc[i] * b
                    curl = a / zck[k] + phyz[i, j, k] - b / xck[i] - phyx[i, j, k]
                    Hy[i, j, k] = cahy[i, j, k] * Hy[i, j, k] - cbhy[i, j, k] * curl

        # Hz = Hz - dt/mu (dEy/dx - dEx/dy)
        for i in range(Nx):
            for j in range(Ny):
                for k in range(Nz + 1):
                    a = (Ey[i + 1, j, k] - Ey[i, j, k]) / dx
                    b = (Ex[i, j + 1, k] - Ex[i, j, k]) / dy
                    phzx[i, j, k] = xcb[i] * phzx[i, j, k] + xcc[i] * a
                    phzy[i, j, k] = ycb[j] * phzy[i, j, k] + ycc[j] * b
                    curl = a / xck[i] + phzx[i, j, k] - b / yck[j] - phzy[i, j, k]
                    Hz[i, j, k] = cahz[i, j, k] * Hz[i, j, k] - cbhz[i, j, k] * curl

        # Ex = Ex + dt/eps (dHz/dy - dHy/dz), interior tangential nodes.
        for i in range(Nx):
            for j in range(1, Ny):
                for k in range(1, Nz):
                    a = (Hz[i, j, k] - Hz[i, j - 1, k]) / dy
                    b = (Hy[i, j, k] - Hy[i, j, k - 1]) / dz
                    pexy[i, j, k] = ynb[j] * pexy[i, j, k] + ync[j] * a
                    pexz[i, j, k] = znb[k] * pexz[i, j, k] + znc[k] * b
                    curl = a / ynk[j] + pexy[i, j, k] - b / znk[k] - pexz[i, j, k]
                    Ex[i, j, k] = caex[i, j, k] * Ex[i, j, k] + cbex[i, j, k] * curl

        # Ey = Ey + dt/eps (dHx/dz - dHz/dx)
        for i in range(1, Nx):
            for j in range(Ny):
                for k in range(1, Nz):
                    a = (Hx[i, j, k] - Hx[i, j, k - 1]) / dz
                    b = (Hz[i, j, k] - Hz[i - 1, j, k]) / dx
                    peyz[i, j, k] = znb[k] * peyz[i, j, k] + znc[k] * a
                    peyx[i, j, k] = xnb[i] * peyx[i, j, k] + xnc[i] * b
                    curl = a / znk[k] + peyz[i, j, k] - b / xnk[i] - peyx[i, j, k]
                    Ey[i, j, k] = caey[i, j, k] * Ey[i, j, k] + cbey[i, j, k] * curl

        # Ez = Ez + dt/eps (dHy/dx - dHx/dy)
        for i in range(1, Nx):
            for j in range(1, Ny):
                for k in range(Nz):
                    a = (Hy[i, j, k] - Hy[i - 1, j, k]) / dx
                    b = (Hx[i, j, k] - Hx[i, j - 1, k]) / dy
                    pezx[i, j, k] = xnb[i] * pezx[i, j, k] + xnc[i] * a
                    pezy[i, j, k] = ynb[j] * pezy[i, j, k] + ync[j] * b
                    curl = a / xnk[i] + pezx[i, j, k] - b / ynk[j] - pezy[i, j, k]
                    Ez[i, j, k] = caez[i, j, k] * Ez[i, j, k] + cbez[i, j, k] * curl

        # Add all soft electric source samples.
        for q in range(nsource_points):
            i = source_coords[q, 0]
            j = source_coords[q, 1]
            k = source_coords[q, 2]
            value = source_values[n, source_ids[q]]
            if source_pols[q] == 0:
                Ex[i, j, k] += value
            elif source_pols[q] == 1:
                Ey[i, j, k] += value
            else:
                Ez[i, j, k] += value

        # Sample all requested monitor voxels at common cell centers.
        if n % record_stride == 0:
            for q in range(nmonitor):
                i = monitor_coords[q, 0]
                j = monitor_coords[q, 1]
                k = monitor_coords[q, 2]
                history[rec, q, 0] = 0.25 * (Ex[i,j,k] + Ex[i,j+1,k] + Ex[i,j,k+1] + Ex[i,j+1,k+1])
                history[rec, q, 1] = 0.25 * (Ey[i,j,k] + Ey[i+1,j,k] + Ey[i,j,k+1] + Ey[i+1,j,k+1])
                history[rec, q, 2] = 0.25 * (Ez[i,j,k] + Ez[i+1,j,k] + Ez[i,j+1,k] + Ez[i+1,j+1,k])
                history[rec, q, 3] = 0.5 * (Hx[i,j,k] + Hx[i+1,j,k])
                history[rec, q, 4] = 0.5 * (Hy[i,j,k] + Hy[i,j+1,k])
                history[rec, q, 5] = 0.5 * (Hz[i,j,k] + Hz[i,j,k+1])
            rec += 1
