import argparse
import copy
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle


def parse_metadata(path: Path):
    out = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def load_history(out_dir: Path):
    meta = parse_metadata(out_dir / "metadata.txt")
    nx, ny = int(meta["nx"]), int(meta["ny"])
    nt_rec = int(meta["nt_rec"])

    ex = np.fromfile(out_dir / "ex_history.bin", dtype=np.float32).reshape(nt_rec, nx, ny)
    ey = np.fromfile(out_dir / "ey_history.bin", dtype=np.float32).reshape(nt_rec, nx, ny)
    hz = np.fromfile(out_dir / "hz_history.bin", dtype=np.float32).reshape(nt_rec, nx, ny)
    er = np.fromfile(out_dir / "eravg.bin", dtype=np.float64).reshape(nx, ny)
    return meta, ex, ey, hz, er


def _coord_to_idx(v, step, n):
    return int(np.clip(np.round(float(v) / step), 0, n - 1))


def _draw_sources_and_monitors(ax, cfg, dx, dy):
    for s in cfg.get("sources", []):
        kind = s.get("kind", "").lower()
        x = s["x"]
        y = s["y"]
        if kind == "sftf" and isinstance(x, list) and isinstance(y, list):
            ax.plot([x[0], x[1]], [y[0], y[0]], "-", color="red", lw=2)
            ax.plot([x[0], x[1]], [y[1], y[1]], "-", color="red", lw=2)
            ax.plot([x[0], x[0]], [y[0], y[1]], "-", color="red", lw=2)
            ax.plot([x[1], x[1]], [y[0], y[1]], "-", color="red", lw=2)
            continue
        if isinstance(x, list) and not isinstance(y, list):
            x0 = _coord_to_idx(x[0], dx, int(round(cfg["x_range"] / dx))) * dx
            x1 = _coord_to_idx(x[1], dx, int(round(cfg["x_range"] / dx))) * dx
            yy = _coord_to_idx(y, dy, int(round(cfg["y_range"] / dy))) * dy
            ax.plot([x0, x1], [yy, yy], "-", color="red", lw=2)
        elif not isinstance(x, list) and isinstance(y, list):
            xx = _coord_to_idx(x, dx, int(round(cfg["x_range"] / dx))) * dx
            y0 = _coord_to_idx(y[0], dy, int(round(cfg["y_range"] / dy))) * dy
            y1 = _coord_to_idx(y[1], dy, int(round(cfg["y_range"] / dy))) * dy
            ax.plot([xx, xx], [y0, y1], "-", color="red", lw=2)
        else:
            xx = _coord_to_idx(x, dx, int(round(cfg["x_range"] / dx))) * dx
            yy = _coord_to_idx(y, dy, int(round(cfg["y_range"] / dy))) * dy
            marker = "o" if kind == "point" else "x"
            ax.plot([xx], [yy], marker, color="red", ms=5, mew=0)

    for m in cfg.get("monitors", []):
        x = m["x"]
        y = m["y"]
        if isinstance(x, list) and not isinstance(y, list):
            ax.plot([x[0], x[1]], [y, y], "--", color="green", lw=1.5)
        elif not isinstance(x, list) and isinstance(y, list):
            ax.plot([x, x], [y[0], y[1]], "--", color="green", lw=1.5)


def _draw_pml(ax, cfg):
    pml = cfg.get("pml", {}) or {}
    if not bool(pml.get("enabled", False)):
        return
    nx = int(cfg["nx"])
    ny = int(cfg["ny"])
    x_range = float(cfg["x_range"])
    y_range = float(cfg["y_range"])
    width = int(pml.get("width_cells", 20))
    width = max(1, min(width, min(nx // 2, ny // 2)))
    xw = width * (x_range / nx)
    yw = width * (y_range / ny)
    ax.add_patch(Rectangle((0.0, 0.0), xw, y_range, facecolor="black", alpha=0.3, lw=0))
    ax.add_patch(Rectangle((x_range - xw, 0.0), xw, y_range, facecolor="black", alpha=0.3, lw=0))
    ax.add_patch(Rectangle((0.0, 0.0), x_range, yw, facecolor="black", alpha=0.3, lw=0))
    ax.add_patch(Rectangle((0.0, y_range - yw), x_range, yw, facecolor="black", alpha=0.3, lw=0))


def animate(out_dir: Path, cfg: dict, fps: int, dynamic_clim: bool, clim_smooth: float, pad: float = 1e-12):
    meta, ex, ey, hz, er = load_history(out_dir)
    x_range = float(meta["x_range_m"])
    y_range = float(meta["y_range_m"])
    dt = float(meta["dt_s"])
    stride = int(meta["record_stride"])

    dx = x_range / int(meta["nx"])
    dy = y_range / int(meta["ny"])

    extent = [0, x_range, 0, y_range]
    n_map = np.sqrt(er)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    ax_n, ax_ex = axes[0]
    ax_ey, ax_hz = axes[1]

    im_n = ax_n.imshow(n_map.T, origin="lower", aspect="auto", extent=extent, cmap="viridis")
    im_n.set_clim(np.min(n_map), min(5, np.max(n_map)))
    fig.colorbar(im_n, ax=ax_n).set_label("n")
    ax_n.set_title("Refractive index")
    ax_n.set_xlabel("x (m)")
    ax_n.set_ylabel("y (m)")

    im_ex = ax_ex.imshow(ex[0].T, origin="lower", aspect="auto", extent=extent, cmap="jet")
    fig.colorbar(im_ex, ax=ax_ex).set_label("Ex")
    ax_ex.set_title("Ex")
    ax_ex.set_xlabel("x (m)")
    ax_ex.set_ylabel("y (m)")

    im_ey = ax_ey.imshow(ey[0].T, origin="lower", aspect="auto", extent=extent, cmap="jet")
    fig.colorbar(im_ey, ax=ax_ey).set_label("Ey")
    ax_ey.set_title("Ey")
    ax_ey.set_xlabel("x (m)")
    ax_ey.set_ylabel("y (m)")

    im_hz = ax_hz.imshow(hz[0].T, origin="lower", aspect="auto", extent=extent, cmap="jet")
    fig.colorbar(im_hz, ax=ax_hz).set_label("Hz")
    ax_hz.set_title("Hz")
    ax_hz.set_xlabel("x (m)")
    ax_hz.set_ylabel("y (m)")

    _draw_pml(ax_n, cfg)
    _draw_sources_and_monitors(ax_n, cfg, dx, dy)

    time_text = ax_hz.text(
        0.02,
        0.02,
        "",
        transform=ax_hz.transAxes,
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
    )

    if dynamic_clim:
        smoothed_vmax = max(np.max(np.abs(ex[0])) + pad, np.max(np.abs(ey[0])) + pad, np.max(np.abs(hz[0])) + pad)
        im_ex.set_clim(-smoothed_vmax, smoothed_vmax)
        im_ey.set_clim(-smoothed_vmax, smoothed_vmax)
        im_hz.set_clim(-smoothed_vmax, smoothed_vmax)
    else:
        vmax = max(np.max(np.abs(ex)) + pad, np.max(np.abs(ey)) + pad, np.max(np.abs(hz)) + pad)
        im_ex.set_clim(-vmax, vmax)
        im_ey.set_clim(-vmax, vmax)
        im_hz.set_clim(-vmax, vmax)
        smoothed_vmax = vmax

    def update(frame):
        nonlocal smoothed_vmax
        im_ex.set_data(ex[frame].T)
        im_ey.set_data(ey[frame].T)
        im_hz.set_data(hz[frame].T)

        if dynamic_clim:
            vmax_now = max(np.max(np.abs(ex[frame])) + pad, np.max(np.abs(ey[frame])) + pad, np.max(np.abs(hz[frame])) + pad)
            smoothed_vmax = (1.0 - clim_smooth) * vmax_now + clim_smooth * smoothed_vmax
            v = max(smoothed_vmax, pad)
            im_ex.set_clim(-v, v)
            im_ey.set_clim(-v, v)
            im_hz.set_clim(-v, v)

        time_text.set_text(f"t = {frame * stride * dt * 1e12:.3f} ps")
        return im_ex, im_ey, im_hz, time_text

    interval_ms = 1000.0 / max(1, fps)
    ani = FuncAnimation(fig, update, frames=int(meta["nt_rec"]), interval=interval_ms, blit=True, repeat=False)
    fig._ani = ani
    plt.show()


def show_fft(out_dir: Path, cfg: dict):
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    index_file = out_dir / "fft_index.txt"
    if not index_file.exists():
        return

    for line in index_file.read_text().splitlines():
        parts = line.split(",", 2)
        if len(parts) < 3:
            continue
        typ, idx, name = parts
        csv_file = out_dir / f"{typ}_fft_{idx}.csv"
        if not csv_file.exists():
            continue
        data = np.loadtxt(csv_file, delimiter=",", skiprows=1)
        f = data[:, 0]
        p = data[:, 1]
        fmin = float(cfg.get("f_min", 0.0))
        fmax = float(cfg.get("f_max", np.max(f) if len(f) else 0.0))
        mask = (f >= fmin) & (f <= fmax)
        f = f[mask]
        p = p[mask]
        if len(f) == 0:
            continue
        label = f"{typ} {idx} ({name})"
        ax.plot(f / 1e9, p, lw=1.5, label=label)

    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Power (W, signed)")
    ax.grid(True, alpha=0.3)
    ax.set_title("FFT power spectrum")
    if ax.lines:
        ax.legend()
    fig.tight_layout()
    plt.show()


def show_nf2ff(out_dir: Path):
    ff = out_dir / "nf2ff.csv"
    if not ff.exists():
        return
    data = np.loadtxt(ff, delimiter=",", skiprows=1)
    if data.ndim != 2 or data.shape[1] < 7:
        return
    freqs = np.unique(data[:, 0])
    phi = data[:, 1]
    ptheta = data[:, 6]
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="polar")
    for f0 in freqs[: min(6, len(freqs))]:
        m = data[:, 0] == f0
        p = ptheta[m]
        p = p - np.min(p)
        p = p / (np.max(p) + 1e-30)
        ax.plot(phi[m], p, lw=1.5, label=f"{f0/1e9:.2f} GHz")
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_title("NF2FF Far-Field (normalized)")
    ax.grid(True, alpha=0.3)
    if ax.lines:
        ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.05))
    fig.tight_layout()
    plt.show()


def _g_from_cfg_source(s: dict, t: np.ndarray, dt: float, global_fmax: float) -> np.ndarray:
    amp = float(s.get("amplitude", 1.0))
    fmin = s.get("f_min", None)
    fmax = float(s.get("f_max", s.get("f0", global_fmax)))
    tw = float(s.get("tw", 0.5 / max(fmax, 1e-30)))
    t0 = float(s.get("t0", 4.0 * tw))
    if fmin is None:
        return amp * np.exp(-((t - t0) / tw) ** 2)
    fmin = float(fmin)
    if abs(fmin - fmax) <= 1e-12 * max(abs(fmax), 1.0):
        tr = max(1.0 / max(fmax, 1e-30), dt)
        tau = np.maximum(t - t0, 0.0)
        ramp = 1.0 - np.exp(-((tau / tr) ** 3))
        return amp * ramp * np.sin(2.0 * np.pi * fmax * (t - t0))
    f0 = 0.5 * (fmin + fmax)
    return amp * np.sin(2.0 * np.pi * f0 * (t - t0)) * np.exp(-((t - t0) / tw) ** 2)


def _phasor_series(series: np.ndarray, dt: float, freqs: np.ndarray) -> np.ndarray:
    nt = series.shape[0]
    t = np.arange(nt, dtype=float) * dt
    k = np.exp(-1j * 2.0 * np.pi * freqs[:, None] * t[None, :]) * dt
    return k @ series


def _monitor_geom_from_cfg(cfg: dict, monitor_index: int):
    nx = int(cfg["nx"])
    ny = int(cfg["ny"])
    dx = float(cfg["x_range"]) / nx
    dy = float(cfg["y_range"]) / ny
    m = cfg.get("monitors", [])[monitor_index]
    ix0, ix1 = _span_to_idx(m["x"], dx, nx)
    iy0, iy1 = _span_to_idx(m["y"], dy, ny)
    if ix0 != ix1 and iy0 == iy1:
        x = (np.arange(ix0, ix1, dtype=float) + 0.5) * dx
        y = np.full_like(x, (iy0 + 0.5) * dy, dtype=float)
        dl = dx
    elif ix0 == ix1 and iy0 != iy1:
        y = (np.arange(iy0, iy1, dtype=float) + 0.5) * dy
        x = np.full_like(y, (ix0 + 0.5) * dx, dtype=float)
        dl = dy
    else:
        raise ValueError("monitor must be horizontal or vertical")
    return x, y, dl


def compute_nf2ff_python(out_dir: Path, cfg: dict):
    pp = cfg.get("postprocessing", {}) or {}
    nf = (pp.get("nf2ff", {}) or {})
    if not bool(nf.get("enabled", False)):
        return
    sides = {"top": nf.get("top", None), "bottom": nf.get("bottom", None), "left": nf.get("left", None), "right": nf.get("right", None)}
    if all(v is None for v in sides.values()):
        return

    meta = parse_metadata(out_dir / "metadata.txt")
    nt = int(meta["nt"])
    dt = float(meta["dt_s"])
    c0 = 1.0 / np.sqrt(float(meta["eps0"]) * float(meta["mu0"]))
    eta0 = np.sqrt(float(meta["mu0"]) / float(meta["eps0"]))

    nphi = int(nf.get("nphi", 361))
    nfreq = int(nf.get("freq_count", 10))
    fmin = float(cfg.get("f_min", 0.0))
    fmax = float(cfg.get("f_max", fmin + 1.0))
    if nfreq <= 1:
        freqs = np.array([fmin], dtype=float)
    else:
        freqs = np.linspace(fmin, fmax, nfreq)

    side_data = {}
    for side, mi in sides.items():
        if mi is None:
            side_data[side] = None
            continue
        nline = int(meta.get(f"monitor_{mi}_nline", "0"))
        if nline <= 0:
            side_data[side] = None
            continue
        hz = np.fromfile(out_dir / f"monitor_time_hz_{mi}.bin", dtype=np.float64).reshape(nt, nline)
        ex = np.fromfile(out_dir / f"monitor_time_ex_{mi}.bin", dtype=np.float64).reshape(nt, nline)
        ey = np.fromfile(out_dir / f"monitor_time_ey_{mi}.bin", dtype=np.float64).reshape(nt, nline)
        hz_f = _phasor_series(hz, dt, freqs)
        ex_f = _phasor_series(ex, dt, freqs) * eta0
        ey_f = _phasor_series(ey, dt, freqs) * eta0
        x, y, dl = _monitor_geom_from_cfg(cfg, int(mi))
        side_data[side] = {"Hz": hz_f, "Ex": ex_f, "Ey": ey_f, "x": x, "y": y, "dl": dl}

    src_index = nf.get("source_index", None)
    gsrc = np.ones(freqs.shape[0], dtype=complex)
    if src_index is not None and 0 <= int(src_index) < len(cfg.get("sources", [])):
        src = cfg["sources"][int(src_index)]
        t = np.arange(nt, dtype=float) * dt
        g = _g_from_cfg_source(src, t, dt, float(cfg.get("f_max", 1.0)))
        gsrc = _phasor_series(g[:, None], dt, freqs).ravel()

    phi = np.linspace(0.0, 2 * np.pi, nphi, endpoint=False)
    cph = np.cos(phi)[None, :]
    sph = np.sin(phi)[None, :]
    k0 = 2 * np.pi * freqs[:, None] / c0

    def int_side(side: str, key: str):
        sd = side_data.get(side, None)
        if sd is None:
            return np.zeros((freqs.size, phi.size), dtype=complex)
        phase = np.exp(1j * (k0[..., None]) * (sd["x"][None, None, :] * cph[..., None] + sd["y"][None, None, :] * sph[..., None]))
        return np.sum(sd[key][:, None, :] * phase, axis=2) * sd["dl"]

    n_phi = +sph * int_side("bottom", "Hz") - cph * int_side("right", "Hz") - sph * int_side("top", "Hz") + cph * int_side("left", "Hz")
    l_theta = +int_side("bottom", "Ex") + int_side("right", "Ey") - int_side("top", "Ex") - int_side("left", "Ey")
    e_phi = k0 * (eta0 * n_phi - l_theta)
    h_theta = k0 * (l_theta / eta0 - n_phi)

    gmax = np.max(np.abs(gsrc))
    if gmax > 0:
        good = np.abs(gsrc) >= 1e-6 * gmax
        e_phi[good, :] /= gsrc[good, None]
        h_theta[good, :] /= gsrc[good, None]
        e_phi[~good, :] = 0.0
        h_theta[~good, :] = 0.0

    p_phi = 0.5 * np.real(e_phi * h_theta)
    with (out_dir / "nf2ff.csv").open("w", encoding="utf-8") as f:
        f.write("freq_hz,phi_rad,e_phi_re,e_phi_im,h_theta_re,h_theta_im,p_phi_re\n")
        for fi, fr in enumerate(freqs):
            for pi, ph in enumerate(phi):
                f.write(f"{fr:.12e},{ph:.12e},{e_phi[fi,pi].real:.12e},{e_phi[fi,pi].imag:.12e},{h_theta[fi,pi].real:.12e},{h_theta[fi,pi].imag:.12e},{p_phi[fi,pi]:.12e}\n")


def show_source_profiles_from_config(cfg: dict):
    eps0 = 8.85e-12
    mu0 = 4e-7 * np.pi
    c0 = 1.0 / np.sqrt(eps0 * mu0)

    dx = float(cfg["x_range"]) / int(cfg["nx"])
    dy = float(cfg["y_range"]) / int(cfg["ny"])
    dt = cfg.get("dt", None)
    if dt is None:
        dt_cfl = np.sqrt(dx * dx + dy * dy) / (2.0 * c0)
        dt_fs = 1.0 / (20.0 * float(cfg["f_max"]))
        dt = min(dt_cfl, dt_fs)

    nt = int(cfg["nt"])

    for s in cfg.get("sources", []):
        if not bool(s.get("is_show", False)):
            continue
        amp = float(s.get("amplitude", 1.0))
        fmax = float(s.get("f_max", cfg["f_max"]))
        tw = float(s.get("tw", 0.5 / fmax))
        t0 = float(s.get("t0", 4 * tw))
        t = np.arange(0, nt * dt, dt)
        pulse = amp * np.exp(-((t - t0) / tw) ** 2)

        freq = np.fft.fftfreq(len(t), d=dt)
        spectrum = np.fft.fft(pulse)
        pos = freq >= 0
        fp = freq[pos]
        sp = np.abs(spectrum[pos]) + 1e-30
        favg = np.sum(fp * sp) / np.sum(sp)

        fig, axs = plt.subplots(2, 1, figsize=(6, 6))
        axs[0].plot(t * 1e9, pulse)
        axs[0].set_xlabel("Time (ns)")
        axs[0].set_ylabel("Amplitude")
        axs[0].set_title("Source (Time Domain)")

        axs[1].plot(fp / 1e9, sp)
        axs[1].set_xlim(0, 2 * fmax / 1e9)
        axs[1].set_xlabel("Frequency (GHz)")
        axs[1].set_ylabel("Magnitude")
        axs[1].set_title(f"Source Spectrum (Avg f ~ {favg / 1e9:.3f} GHz)")
        plt.tight_layout()
        plt.show()


def _span_to_idx(v, step, n):
    if isinstance(v, list):
        a0 = _coord_to_idx(v[0], step, n)
        a1 = _coord_to_idx(v[1], step, n)
        return (min(a0, a1), max(a0, a1))
    a = _coord_to_idx(v, step, n)
    return (a, a)


def _build_material_maps(cfg: dict):
    nx = int(cfg["nx"])
    ny = int(cfg["ny"])
    dx = float(cfg["x_range"]) / nx
    dy = float(cfg["y_range"]) / ny
    erxx = np.ones((nx, ny), dtype=float)
    eryy = np.ones((nx, ny), dtype=float)
    mrzz = np.ones((nx, ny), dtype=float)
    for o in cfg.get("objects", []):
        x0, x1 = _span_to_idx(o["x_range"], dx, nx)
        y0, y1 = _span_to_idx(o["y_range"], dy, ny)
        er = float(o.get("er", 1.0))
        mr = float(o.get("mr", 1.0))
        erxx[x0:x1, y0:y1] = er
        eryy[x0:x1, y0:y1] = er
        mrzz[x0:x1, y0:y1] = mr
    return erxx, eryy, mrzz


def _wg_modes_y_hz(erxx, eryy, mrzz, dx, c0, ix0, ix1, iy, f_center, num_modes):
    from scipy.sparse import diags as spdiags
    from scipy.sparse.linalg import eigs

    lo, hi = min(ix0, ix1), max(ix0, ix1)
    nxm = int(max(1, hi - lo))
    if nxm < 2:
        return None
    k0 = 2.0 * np.pi * float(f_center) / c0
    mrzz_vec = np.asarray(mrzz[lo:hi, iy], dtype=float)
    erxx_vec = np.asarray(erxx[lo:hi, iy], dtype=float)
    eryy_vec = np.asarray(eryy[lo:hi, iy], dtype=float)
    mrzz_diag = spdiags(mrzz_vec, 0, shape=(nxm, nxm))
    erxx_inv = spdiags(1.0 / erxx_vec, 0, shape=(nxm, nxm))
    eryy_inv = spdiags(1.0 / eryy_vec, 0, shape=(nxm, nxm))
    d_plus = np.ones(nxm)
    d_minus = -np.ones(nxm)
    dex = spdiags([d_plus, d_minus], [1, 0], shape=(nxm, nxm)) / (dx * k0)
    dhx = spdiags([d_plus, d_minus], [0, -1], shape=(nxm, nxm)) / (dx * k0)
    a = mrzz_diag + dex @ (erxx_inv @ dhx)
    b = eryy_inv
    n_slice = np.sqrt(mrzz_vec * 0.5 * (erxx_vec + eryy_vec))
    guess = max(float(np.max(n_slice)) ** 2, 1.0)
    evals, evecs = eigs(a, M=b, k=max(1, int(num_modes)), sigma=guess)
    n_eff = np.sqrt(np.maximum(evals.real, 0.0))
    order = np.argsort(-n_eff)
    evecs = evecs[:, order]
    n_eff = n_eff[order]
    hz_modes = []
    ex_modes = []
    for m in range(evecs.shape[1]):
        hz = evecs[:, m]
        phase = np.angle(hz[np.argmax(np.abs(hz))])
        hz = (hz * np.exp(-1j * phase)).real
        ex = np.asarray((-(n_eff[m]) * (erxx_inv @ hz))).squeeze().real
        norm = 1.0 / (np.max(np.abs(ex)) + 1e-30)
        ex_modes.append(ex * norm)
        hz_modes.append(hz * norm)
    return np.asarray(hz_modes), np.asarray(ex_modes), np.asarray(n_eff, dtype=float), lo, hi


def _wg_modes_x_hz(erxx, eryy, mrzz, dy, c0, iy0, iy1, ix, f_center, num_modes):
    from scipy.sparse import diags as spdiags
    from scipy.sparse.linalg import eigs

    lo, hi = min(iy0, iy1), max(iy0, iy1)
    nym = int(max(1, hi - lo))
    if nym < 2:
        return None
    k0 = 2.0 * np.pi * float(f_center) / c0
    mrzz_vec = np.asarray(mrzz[ix, lo:hi], dtype=float)
    erxx_vec = np.asarray(erxx[ix, lo:hi], dtype=float)
    eryy_vec = np.asarray(eryy[ix, lo:hi], dtype=float)
    mrzz_diag = spdiags(mrzz_vec, 0, shape=(nym, nym))
    erxx_inv = spdiags(1.0 / erxx_vec, 0, shape=(nym, nym))
    eryy_inv = spdiags(1.0 / eryy_vec, 0, shape=(nym, nym))
    d_plus = np.ones(nym)
    d_minus = -np.ones(nym)
    dey = spdiags([d_plus, d_minus], [1, 0], shape=(nym, nym)) / (dy * k0)
    dhy = spdiags([d_plus, d_minus], [0, -1], shape=(nym, nym)) / (dy * k0)
    a = mrzz_diag + dhy @ (eryy_inv @ dey)
    b = erxx_inv
    n_slice = np.sqrt(mrzz_vec * 0.5 * (erxx_vec + eryy_vec))
    guess = max(float(np.max(n_slice)) ** 2, 1.0)
    evals, evecs = eigs(a, M=b, k=max(1, int(num_modes)), sigma=guess)
    n_eff = np.sqrt(np.maximum(evals.real, 0.0))
    order = np.argsort(-n_eff)
    evecs = evecs[:, order]
    n_eff = n_eff[order]
    hz_modes = []
    ey_modes = []
    for m in range(evecs.shape[1]):
        hz = evecs[:, m]
        phase = np.angle(hz[np.argmax(np.abs(hz))])
        hz = (hz * np.exp(-1j * phase)).real
        ey = np.asarray((-(n_eff[m]) * (eryy_inv @ hz))).squeeze().real
        norm = 1.0 / (np.max(np.abs(ey)) + 1e-30)
        ey_modes.append(ey * norm)
        hz_modes.append(hz * norm)
    return np.asarray(hz_modes), np.asarray(ey_modes), np.asarray(n_eff, dtype=float), lo, hi


def prepare_waveguide_profiles_from_config(cfg: dict):
    out = copy.deepcopy(cfg)
    eps0 = 8.85e-12
    mu0 = 4e-7 * np.pi
    c0 = 1.0 / np.sqrt(eps0 * mu0)
    nx = int(out["nx"])
    ny = int(out["ny"])
    dx = float(out["x_range"]) / nx
    dy = float(out["y_range"]) / ny
    erxx, eryy, mrzz = _build_material_maps(out)

    for s in out.get("sources", []):
        k = str(s.get("kind", "")).lower()
        if k not in ("waveguide-x", "waveguide-y"):
            continue
        fmin = s.get("f_min", out.get("f_min", None))
        fmax = s.get("f_max", out.get("f_max", 0.0))
        f_center = 0.5 * (fmin + fmax) if fmin is not None else fmax
        mode_index = max(0, int(s.get("mode_index", 0)))
        modes_to_show = max(1, int(s.get("modes_to_show", max(4, mode_index + 1))))
        ix0, ix1 = _span_to_idx(s["x"], dx, nx)
        iy0, iy1 = _span_to_idx(s["y"], dy, ny)
        if k == "waveguide-y":
            res = _wg_modes_y_hz(erxx, eryy, mrzz, dx, c0, ix0, ix1, iy0, f_center, modes_to_show)
            if res is None:
                continue
            hz_modes, ex_modes, n_eff, lo, hi = res
            mi = min(mode_index, hz_modes.shape[0] - 1)
            s["profile"] = hz_modes[mi].astype(float).tolist()
            s["profile_e"] = ex_modes[mi].astype(float).tolist()
            s["profile_h"] = hz_modes[mi].astype(float).tolist()
            s["n_eff"] = float(n_eff[mi])
            if bool(s.get("is_show", False)):
                x_axis = (np.arange(lo, hi) + 0.5) * dx
                rows = min(hz_modes.shape[0], modes_to_show)
                fig, axs = plt.subplots(rows, 1, figsize=(8, 2.6 * rows), sharex=True)
                if rows == 1:
                    axs = [axs]
                for m in range(rows):
                    ax1 = axs[m]
                    ax2 = ax1.twinx()
                    ax1.plot(x_axis, ex_modes[m], linewidth=1.6, label="Ex")
                    ax2.plot(x_axis, hz_modes[m], linestyle="--", linewidth=1.2, label="Hz")
                    ax1.set_title(f"mode {m}: n_eff = {n_eff[m]:.6f}")
                    ax1.grid(True, alpha=0.25)
                axs[-1].set_xlabel("x (m)")
                fig.suptitle(f"waveguide-y port at y={iy0}")
                fig.tight_layout()
                plt.show()
        else:
            res = _wg_modes_x_hz(erxx, eryy, mrzz, dy, c0, iy0, iy1, ix0, f_center, modes_to_show)
            if res is None:
                continue
            hz_modes, ey_modes, n_eff, lo, hi = res
            mi = min(mode_index, hz_modes.shape[0] - 1)
            s["profile"] = hz_modes[mi].astype(float).tolist()
            s["profile_e"] = ey_modes[mi].astype(float).tolist()
            s["profile_h"] = hz_modes[mi].astype(float).tolist()
            s["n_eff"] = float(n_eff[mi])
            if bool(s.get("is_show", False)):
                y_axis = (np.arange(lo, hi) + 0.5) * dy
                rows = min(hz_modes.shape[0], modes_to_show)
                fig, axs = plt.subplots(rows, 1, figsize=(8, 2.6 * rows), sharex=True)
                if rows == 1:
                    axs = [axs]
                for m in range(rows):
                    ax1 = axs[m]
                    ax2 = ax1.twinx()
                    ax1.plot(y_axis, ey_modes[m], linewidth=1.6, label="Ey")
                    ax2.plot(y_axis, hz_modes[m], linestyle="--", linewidth=1.2, label="Hz")
                    ax1.set_title(f"mode {m}: n_eff = {n_eff[m]:.6f}")
                    ax1.grid(True, alpha=0.25)
                axs[-1].set_xlabel("y (m)")
                fig.suptitle(f"waveguide-x port at x={ix0}")
                fig.tight_layout()
                plt.show()
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/example_1_simple_source.json")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = json.loads(cfg_path.read_text())
    out_dir = Path(args.output) if args.output else Path(cfg.get("output_dir", "output"))

    meta = parse_metadata(out_dir / "metadata.txt")
    fps = int(meta.get("plot_fps", cfg.get("plot", {}).get("fps", 60)))
    dynamic = str(meta.get("dynamic_clim", cfg.get("plot", {}).get("dynamic_clim", True))).lower() == "true"
    clim_smooth = float(meta.get("clim_smooth", cfg.get("plot", {}).get("clim_smooth", 0.25)))

    if cfg.get("plot", {}).get("show_animation", True):
        animate(out_dir, cfg, fps=fps, dynamic_clim=dynamic, clim_smooth=clim_smooth)
    if cfg.get("plot", {}).get("show_fft", True):
        show_fft(out_dir, cfg)
    if cfg.get("plot", {}).get("show_nf2ff", False):
        compute_nf2ff_python(out_dir, cfg)
        show_nf2ff(out_dir)


if __name__ == "__main__":
    main()

