"""Million-cell 3D CPU FDTD scattering example.

The example uses the Cython whole-run backend when it is built.  Its physical
setup is shared by ``Example_3D_GPU.py`` so CPU and GPU results can be compared
without changing geometry, excitation, monitors, or post-processing.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

# Allow both ``python FDTD_3D/Example_3D.py`` and module-style execution from
# a source checkout without requiring an editable installation.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from FDTD_3D import FDTD_3D


def build_simulation(cells, steps, output_dir, backend="cpu"):
    """Build the material, scattering geometry, sources, and monitors."""
    if cells < 100:
        raise ValueError("The 3D examples require at least 100 cells per axis.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sim = FDTD_3D(
        x_range=100e-3,
        y_range=100e-3,
        z_range=100e-3,
        Nx=cells,
        Ny=cells,
        Nz=cells,
        f_min=6e9,
        f_max=10e9,
        Nt=steps,
        subpixel=2,
    )
    sim.config(backend)
    if backend == "gpu" and sim.backend != "numba_cuda":
        raise RuntimeError(
            "This example requires Numba-CUDA and an available CUDA GPU. "
            "Run Example_3D.py for the CPU version.")

    pml_width = max(12, int(round(0.12 * cells)))
    sim.add_PML(pml_width)

    dielectric = sim.add_material(
        "lossy_dielectric", epsilon_r=2.2, mu_r=1.0,
        sigma_e=2e-4, sigma_m=0.0)
    sim.add_block(
        dielectric,
        x=(44e-3, 56e-3),
        y=(43e-3, 57e-3),
        z=(46e-3, 54e-3),
    )
    sim.add_sphere(
        "PEC", center=(50e-3, 50e-3, 50e-3), radius=3e-3)

    transverse = (int(round(0.18 * cells)), int(round(0.82 * cells)))
    sim.add_source(
        "plane",
        x=transverse,
        y=transverse,
        z=int(round(0.22 * cells)),
        polarization="x",
        amplitude=1.0,
        t0=0.6e-9,
        tw=0.18e-9,
        f_min=6e9,
        f_max=10e9,
    )

    # The source is outside this closed equivalence surface; every scatterer
    # is enclosed and the complete surface remains well inside the CPML.
    box_span = (int(round(0.30 * cells)), int(round(0.70 * cells)))
    nf_box = sim.add_nf2ff_box(
        x=box_span, y=box_span, z=box_span, start_index=10)

    interior_span = (pml_width, cells - pml_width)
    saved_monitor = sim.add_plane_monitor(
        "z",
        position=int(round(0.75 * cells)),
        first=interior_span,
        second=interior_span,
        index=1,
        normal="+",
        save_path=output_dir / "transmitted_plane.h5",
    )
    return sim, nf_box, saved_monitor


def run_example(backend="cpu", argv=None):
    """Run the selected backend and save monitor, spectrum, and NF2FF plots."""
    device = "GPU/CUDA" if backend == "gpu" else "CPU/Cython"
    parser = argparse.ArgumentParser(
        description=(
            f"Million-cell 3D {device} FDTD scattering example with HDF5 "
            "monitoring, power spectrum, and 3D dB NF2FF output."))
    parser.add_argument(
        "--cells", type=int, default=100,
        help="cells per axis; values below 100 are rejected (default: 100)")
    parser.add_argument(
        "--steps", type=int, default=800,
        help="number of FDTD time steps (default: 800)")
    parser.add_argument(
        "--record-stride", type=int, default=8,
        help="record every Nth step to control monitor memory (default: 8)")
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("output") / f"3d_{backend}",
        help="directory for HDF5 and plot output")
    parser.add_argument(
        "--animate", action="store_true",
        help="also export the transmitted-plane animation as a GIF")
    parser.add_argument(
        "--no-show", action="store_true",
        help="save plots without opening Matplotlib windows")
    args = parser.parse_args(argv)
    if args.cells < 100:
        parser.error("--cells must be at least 100 for this example")
    if args.steps < 1 or args.record_stride < 1:
        parser.error("--steps and --record-stride must be positive")

    sim, nf_box, saved_monitor = build_simulation(
        args.cells, args.steps, args.output_dir, backend=backend)
    print(
        f"Running {args.cells} x {args.cells} x {args.cells} cells "
        f"with backend={sim.backend!r}")
    sim.run(
        record_stride=args.record_stride,
        progress=True,
        progress_desc=f"3D FDTD ({backend.upper()})",
    )

    frequencies = np.linspace(6e9, 10e9, 41)
    power = sim.power_spectrum(
        nf_box["z_max"], frequencies, source_index=0, window="hann")
    power_figure, _ = sim.plot_power_spectrum(power, db=True)
    power_figure.savefig(args.output_dir / "power_spectrum.png", dpi=180)

    far_field = sim.NF2FF(
        nf_box,
        freqs=[8e9],
        theta=np.linspace(0, np.pi, 61),
        phi=np.linspace(0, 2 * np.pi, 120, endpoint=False),
        source_index=0,
        window="hann",
    )
    farfield_figure, _ = sim.plot_nf2ff(
        far_field, db=True, db_floor=-40.0)
    farfield_figure.savefig(args.output_dir / "farfield_8GHz_db.png", dpi=180)

    monitor_figure, _ = sim.plot_plane_monitor(
        saved_monitor, component="Ex", time_index=-1)
    monitor_figure.savefig(
        args.output_dir / "transmitted_plane_final.png", dpi=180)

    if args.animate:
        _, monitor_animation = sim.animate_plane_monitor(
            saved_monitor,
            component="Ex",
            frame_stride=1,
            interval=50,
            save_path=args.output_dir / "transmitted_plane.gif",
        )
        # Keep a live reference until the optional writer has finished.
        monitor_animation._fdtd_example_owner = sim

    if args.no_show:
        plt.close("all")
    else:
        plt.show()
    return sim


if __name__ == "__main__":
    run_example("cpu")
