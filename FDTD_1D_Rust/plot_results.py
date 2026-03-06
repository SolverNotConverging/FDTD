import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


def parse_metadata(path: Path) -> dict:
    data = {}
    for line in path.read_text().splitlines():
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        data[k.strip()] = v.strip()
    return data


def load_data(output_dir: Path):
    ey = np.loadtxt(output_dir / "ey_history.csv", delimiter=",")
    hx = np.loadtxt(output_dir / "hx_history.csv", delimiter=",")
    ref_power = np.loadtxt(output_dir / "ref_power_history.csv", delimiter=",")
    trn_power = np.loadtxt(output_dir / "trn_power_history.csv", delimiter=",")
    freq_hz = np.loadtxt(output_dir / "frequency_hz.csv")
    z_m = np.loadtxt(output_dir / "z_m.csv")
    er = np.loadtxt(output_dir / "er_profile.csv")
    mr = np.loadtxt(output_dir / "mr_profile.csv")
    meta = parse_metadata(output_dir / "metadata.txt")
    return ey, hx, ref_power, trn_power, freq_hz, z_m, er, mr, meta


def show_source_profile(dt, nt, f_max, amplitude, t0, tw):
    t = np.arange(0.0, nt * dt, dt)
    pulse = amplitude * np.exp(-((t - t0) / tw) ** 2)

    freq = np.fft.fftfreq(len(t), d=dt)
    spectrum = np.fft.fft(pulse)

    pos_mask = freq >= 0
    freq_pos = freq[pos_mask]
    spec_pos = np.abs(spectrum[pos_mask]) + 1e-30
    avg_freq = float(np.sum(freq_pos * spec_pos) / np.sum(spec_pos))

    fig, axs = plt.subplots(2, 1, figsize=(6, 6))
    axs[0].plot(t * 1e9, pulse)
    axs[0].set_xlabel("Time (ns)")
    axs[0].set_ylabel("Amplitude")
    axs[0].set_title("Source (Time Domain)")

    axs[1].plot(freq_pos / 1e9, spec_pos)
    axs[1].set_xlim(0, 2 * f_max / 1e9)
    axs[1].set_xlabel("Frequency (GHz)")
    axs[1].set_ylabel("Magnitude")
    axs[1].set_title(f"Source Spectrum (Avg f ~ {avg_freq / 1e9:.3f} GHz)")

    plt.tight_layout()
    plt.show()


def animate(output_dir: Path, fps: int):
    ey, hx, ref_power, trn_power, freq_hz, z_m, er, mr, meta = load_data(output_dir)
    dt = float(meta["dt_s"])
    src_idx = int(meta.get("source_index", "0"))

    x_e_mm = z_m * 1e3
    dz = z_m[1] - z_m[0]
    x_h_mm = (z_m + dz / 2.0) * 1e3

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(9, 6), gridspec_kw={"height_ratios": [3, 1, 2]})

    line_e, = ax1.plot(x_e_mm, ey[0], label="E field", color="red")
    line_h, = ax1.plot(x_h_mm, hx[0], label="H field", color="blue")
    ax1.axvline(x=x_e_mm[src_idx], color="green", linestyle="--")
    ax1.set_title("1D FDTD Simulation")
    ax1.set_xlim(x_e_mm[0], x_e_mm[-1])
    lim = max(1e-6, np.max(np.abs(ey)), np.max(np.abs(hx)))
    ax1.set_ylim(-1.1 * lim, 1.1 * lim)
    ax1.set_ylabel("Amplitude")
    ax1.legend()

    time_text = ax1.text(0.02, 0.8, "", transform=ax1.transAxes, bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))

    ax2.plot(x_e_mm, er, label="εr (real)")
    ax2.plot(x_e_mm, mr, label="μr (real)")
    ax2.set_xlim(x_e_mm[0], x_e_mm[-1])
    ax2.set_xlabel("Position (mm)")
    ax2.set_ylabel("Value")
    ax2.legend(loc="upper right")
    ax2.set_title("Material Profiles")

    line_ref, = ax3.plot(freq_hz / 1e9, ref_power[0], label="Reflection", color="red")
    line_trn, = ax3.plot(freq_hz / 1e9, trn_power[0], label="Transmission", color="blue")
    ax3.set_xlim(0, freq_hz[-1] / 1e9)
    ax3.set_ylim(-0.1, 1.1)
    ax3.set_xlabel("Frequency (GHz)")
    ax3.set_ylabel("Magnitude")
    ax3.legend(loc="upper right")
    ax3.set_title("Transmission and Reflection")
    ax3.grid()

    def update(frame):
        line_e.set_ydata(ey[frame])
        line_h.set_ydata(hx[frame])
        line_ref.set_ydata(ref_power[frame])
        line_trn.set_ydata(trn_power[frame])

        t_e = frame * dt
        t_h = (frame + 0.5) * dt
        time_text.set_text(f"t(E) = {t_e * 1e12:.5f} ps\\nt(H) = {t_h * 1e12:.5f} ps")
        return line_e, line_h, time_text, line_ref, line_trn

    interval_ms = max(1, int(1000 / max(1, fps)))
    ani = FuncAnimation(fig, update, frames=ey.shape[0], interval=interval_ms, blit=True, repeat=False)
    fig._ani = ani
    plt.tight_layout()
    plt.show()


def spectrum(output_dir: Path):
    _, _, ref_power, trn_power, freq_hz, _, _, _, _ = load_data(output_dir)
    plt.figure(figsize=(8, 4))
    plt.plot(freq_hz / 1e9, ref_power[-1], label="Reflection", color="red")
    plt.plot(freq_hz / 1e9, trn_power[-1], label="Transmission", color="blue")
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Power")
    plt.ylim(-0.1, 1.1)
    plt.grid(True)
    plt.legend()
    plt.title("Final Reflection/Transmission")
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot Rust FDTD 1D outputs")
    parser.add_argument("--output", default="output", help="Directory with CSV outputs")
    parser.add_argument("--mode", choices=["animate", "spectrum"], default="animate")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    output_dir = Path(args.output)
    if args.mode == "animate":
        animate(output_dir, args.fps)
    else:
        spectrum(output_dir)


if __name__ == "__main__":
    main()
