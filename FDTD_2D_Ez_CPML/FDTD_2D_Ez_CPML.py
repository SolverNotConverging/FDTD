import numpy as np

from FDTD_2D_Ez.FDTD_2D import FDTD_2D


class FDTD_2D_Ez_CPML(FDTD_2D):
    """TMz FDTD solver with convolutional PML boundaries."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cpml_active = False
        self._cpml_profiles = {}
        self._psi_CEx_y = None
        self._psi_CEy_x = None
        self._psi_CHz_x = None
        self._psi_CHz_y = None

    def add_CPML(self, pml_width, sigma_max, kappa_max=5.0, alpha=0.05, order=3, direction='xy'):
        """Configure convolutional PML regions on the simulation boundaries."""

        if isinstance(pml_width, int):
            npx = npy = max(1, int(pml_width))
        elif isinstance(pml_width, float):
            npx = max(1, int(np.ceil(pml_width / self.dx)))
            npy = max(1, int(np.ceil(pml_width / self.dy)))
        else:
            raise TypeError("pml_width must be int (cells) or float (meters).")

        dir_x = 'x' in direction.lower()
        dir_y = 'y' in direction.lower()

        self._cpml_profiles = {}

        def _build_profile(N, npml, delta, scale_const):
            sigma = np.zeros(N)
            kappa = np.ones(N)
            alpha_arr = np.zeros(N)
            if npml <= 0:
                return sigma, kappa, alpha_arr, np.ones(N), np.zeros(N)
            for n in range(npml):
                pos = (npml - n) / npml
                sig = sigma_max * pos ** order
                kap = 1.0 + (kappa_max - 1.0) * pos ** order
                alpha_val = alpha * (1.0 - pos)
                for idx in (n, N - n - 1):
                    sigma[idx] = sig
                    kappa[idx] = kap
                    alpha_arr[idx] = alpha_val
            b = np.exp(-(sigma / kappa + alpha_arr) * self.dt / scale_const)
            c = sigma * (b - 1.0) / (sigma + kappa * alpha_arr + 1e-30) / (kappa * scale_const)
            return sigma, kappa, alpha_arr, b, c

        if dir_x:
            sig_x_e, kap_x_e, alp_x_e, b_x_e, c_x_e = _build_profile(self.Nx, npx, self.dx, self.eps0)
            sig_x_h, kap_x_h, alp_x_h, b_x_h, c_x_h = _build_profile(self.Nx, npx, self.dx, self.mu0)
        else:
            b_x_e = np.ones(self.Nx)
            c_x_e = np.zeros(self.Nx)
            kap_x_e = np.ones(self.Nx)
            b_x_h = np.ones(self.Nx)
            c_x_h = np.zeros(self.Nx)
            kap_x_h = np.ones(self.Nx)

        if dir_y:
            sig_y_e, kap_y_e, alp_y_e, b_y_e, c_y_e = _build_profile(self.Ny, npy, self.dy, self.eps0)
            sig_y_h, kap_y_h, alp_y_h, b_y_h, c_y_h = _build_profile(self.Ny, npy, self.dy, self.mu0)
        else:
            b_y_e = np.ones(self.Ny)
            c_y_e = np.zeros(self.Ny)
            kap_y_e = np.ones(self.Ny)
            b_y_h = np.ones(self.Ny)
            c_y_h = np.zeros(self.Ny)
            kap_y_h = np.ones(self.Ny)

        self._cpml_profiles = dict(
            b_x_e=b_x_e, c_x_e=c_x_e, kappa_x_e=kap_x_e,
            b_y_e=b_y_e, c_y_e=c_y_e, kappa_y_e=kap_y_e,
            b_x_h=b_x_h, c_x_h=c_x_h, kappa_x_h=kap_x_h,
            b_y_h=b_y_h, c_y_h=c_y_h, kappa_y_h=kap_y_h,
        )

        self._psi_CEx_y = np.zeros((self.Nx, self.Ny))
        self._psi_CEy_x = np.zeros((self.Nx, self.Ny))
        self._psi_CHz_x = np.zeros((self.Nx, self.Ny))
        self._psi_CHz_y = np.zeros((self.Nx, self.Ny))

        self.sigma_hx.fill(0.0)
        self.sigma_hy.fill(0.0)
        self.sigma_dx.fill(0.0)
        self.sigma_dy.fill(0.0)
        self._cpml_active = True
        self._init_m()

    def calculate_CE(self):
        if not self._cpml_active:
            return super().calculate_CE()

        b_y = self._cpml_profiles['b_y_e']
        c_y = self._cpml_profiles['c_y_e']
        kappa_y = self._cpml_profiles['kappa_y_e']
        b_x = self._cpml_profiles['b_x_e']
        c_x = self._cpml_profiles['c_x_e']
        kappa_x = self._cpml_profiles['kappa_x_e']

        per_x = hasattr(self, "periodic") and ('x' in self.periodic)
        per_y = hasattr(self, "periodic") and ('y' in self.periodic)

        for nx in range(self.Nx):
            for ny in range(self.Ny - 1):
                diff = (self.Ez[nx, ny + 1] - self.Ez[nx, ny]) / self.dy
                psi = self._psi_CEx_y[nx, ny]
                psi = b_y[ny] * psi + c_y[ny] * diff
                self._psi_CEx_y[nx, ny] = psi
                self.CEx[nx, ny] = (diff / kappa_y[ny]) + psi
            if per_y:
                diff = (self.Ez[nx, 0] - self.Ez[nx, self.Ny - 1]) / self.dy
            else:
                diff = (0.0 - self.Ez[nx, self.Ny - 1]) / self.dy
            psi = self._psi_CEx_y[nx, self.Ny - 1]
            psi = b_y[self.Ny - 1] * psi + c_y[self.Ny - 1] * diff
            self._psi_CEx_y[nx, self.Ny - 1] = psi
            self.CEx[nx, self.Ny - 1] = (diff / kappa_y[self.Ny - 1]) + psi

        for ny in range(self.Ny):
            for nx in range(self.Nx - 1):
                diff = (self.Ez[nx + 1, ny] - self.Ez[nx, ny]) / self.dx
                psi = self._psi_CEy_x[nx, ny]
                psi = b_x[nx] * psi + c_x[nx] * diff
                self._psi_CEy_x[nx, ny] = psi
                self.CEy[nx, ny] = -(diff / kappa_x[nx] + psi)
            if per_x:
                diff = (self.Ez[0, ny] - self.Ez[self.Nx - 1, ny]) / self.dx
            else:
                diff = (0.0 - self.Ez[self.Nx - 1, ny]) / self.dx
            psi = self._psi_CEy_x[self.Nx - 1, ny]
            psi = b_x[self.Nx - 1] * psi + c_x[self.Nx - 1] * diff
            self._psi_CEy_x[self.Nx - 1, ny] = psi
            self.CEy[self.Nx - 1, ny] = -(diff / kappa_x[self.Nx - 1] + psi)

    def calculate_CH(self):
        if not self._cpml_active:
            return super().calculate_CH()

        b_x = self._cpml_profiles['b_x_h']
        c_x = self._cpml_profiles['c_x_h']
        kappa_x = self._cpml_profiles['kappa_x_h']
        b_y = self._cpml_profiles['b_y_h']
        c_y = self._cpml_profiles['c_y_h']
        kappa_y = self._cpml_profiles['kappa_y_h']

        per_x = hasattr(self, "periodic") and ('x' in self.periodic)
        per_y = hasattr(self, "periodic") and ('y' in self.periodic)

        for nx in range(self.Nx):
            for ny in range(self.Ny):
                if nx == 0:
                    if per_x:
                        diff_x = (self.Hy[nx, ny] - self.Hy[self.Nx - 1, ny]) / self.dx
                    else:
                        diff_x = (self.Hy[nx, ny] - 0.0) / self.dx
                else:
                    diff_x = (self.Hy[nx, ny] - self.Hy[nx - 1, ny]) / self.dx

                if ny == 0:
                    if per_y:
                        diff_y = (self.Hx[nx, ny] - self.Hx[nx, self.Ny - 1]) / self.dy
                    else:
                        diff_y = (self.Hx[nx, ny] - 0.0) / self.dy
                else:
                    diff_y = (self.Hx[nx, ny] - self.Hx[nx, ny - 1]) / self.dy

                psi_x = self._psi_CHz_x[nx, ny]
                psi_y = self._psi_CHz_y[nx, ny]
                psi_x = b_x[nx] * psi_x + c_x[nx] * diff_x
                psi_y = b_y[ny] * psi_y + c_y[ny] * diff_y
                self._psi_CHz_x[nx, ny] = psi_x
                self._psi_CHz_y[nx, ny] = psi_y
                term_x = diff_x / kappa_x[nx] + psi_x
                term_y = diff_y / kappa_y[ny] + psi_y
                self.CHz[nx, ny] = term_x - term_y

    # convenience alias
    def add_PML(self, *args, **kwargs):
        self.add_CPML(*args, **kwargs)
