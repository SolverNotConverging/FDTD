"""Simulate a finite light packet near the Schwarzschild photon sphere."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

# Import the package (rather than the same-named module file) when this script
# is launched directly from a source checkout.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from FDTD_2D_GR import FDTD_2D_GR


# Edit these values directly, like the other FDTD examples.
BACKEND = "cpu"  # "cpu" (Cython), "gpu" (Numba-CUDA), or "python" (NumPy)
NR = 320
NPHI = 640
RHO_MIN = 0.55
RHO_MAX = 10.0
COURANT = 0.8

AZIMUTHAL_MODE = 20
RADIAL_WIDTH = 0.35
ANGULAR_WIDTH = 0.32
DIRECTION = +1  # +1 counter-clockwise, -1 clockwise
NUMBER_OF_ORBITS = 0.5
HISTORY_SAMPLES = 200

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
SAVE_ANIMATION = True  # Set True to write photon_packet.mp4 after the run.
SHOW_PLOTS = True


sim = FDTD_2D_GR(
    rho_min=RHO_MIN,
    rho_max=RHO_MAX,
    Nr=NR,
    Nphi=NPHI,
    courant=COURANT,
).config(BACKEND)

packet = sim.initialize_orbiting_packet(
    direction=DIRECTION,
    azimuthal_mode=AZIMUTHAL_MODE,
    radial_width=RADIAL_WIDTH,
    angular_width=ANGULAR_WIDTH,
)
initial_energy = sim.energy_density().copy()

duration = NUMBER_OF_ORBITS * sim.photon_orbit_period
steps = int(np.ceil(duration / sim.dt))
record_stride = max(1, int(np.ceil(steps / (HISTORY_SAMPLES - 1))))
history = sim.run(
    steps=steps,
    record_stride=record_stride,
    store_snapshots=SAVE_ANIMATION,
    snapshot_quantity="energy",
    progress=True,
)

# All plots and the optional animation are generated after time stepping.
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
initial_path = OUTPUT_DIR / "photon_packet_initial.png"
final_path = OUTPUT_DIR / "photon_packet_final.png"
diagnostics_path = OUTPUT_DIR / "photon_packet_diagnostics.png"
data_path = OUTPUT_DIR / "photon_packet_diagnostics.npz"

initial_axis, _ = sim.plot_snapshot(
    initial_energy,
    quantity="energy",
    view_radius=4.0,
    log_scale=True,
    title="Initial packet at the Schwarzschild photon sphere",
)
initial_axis.figure.savefig(initial_path, dpi=180, bbox_inches="tight")

final_axis, _ = sim.plot_snapshot(
    quantity="energy",
    view_radius=4.0,
    log_scale=True,
    title=f"Packet after {NUMBER_OF_ORBITS:g} ideal photon periods",
)
final_axis.figure.savefig(final_path, dpi=180, bbox_inches="tight")

diagnostics_figure, _ = sim.plot_diagnostics(history)
diagnostics_figure.savefig(diagnostics_path, dpi=180, bbox_inches="tight")

np.savez_compressed(
    data_path,
    **{
        key: value
        for key, value in history.items()
        if isinstance(value, np.ndarray)
    },
)

if SAVE_ANIMATION:
    sim.save_animation(
        OUTPUT_DIR / "photon_packet.mp4",
        history,
        fps=24,
        view_radius=4.0,
    )

final_energy_fraction = history["energy"][-1] / history["energy"][0]
expected_angle = packet["angular_velocity"] * history["time"][-1]
measured_angle = history["phi_unwrapped"][-1] - history["phi_unwrapped"][0]
angular_coherence = history["phi_coherence"][-1]
print(f"CFL dt/M: {sim.dt / sim.mass:.6g}")
print(f"Backend: {sim.backend}")
print(f"Ideal photon period T/M: {sim.photon_orbit_period / sim.mass:.6g}")
print(f"Expected ideal-ray angle: {expected_angle:.6g} rad")
print(f"Whole-wave centroid angle: {measured_angle:.6g} rad")
print(f"Final angular centroid coherence: {angular_coherence:.6g}")
print(f"Field energy retained in domain: {final_energy_fraction:.6g}")
print("Interpret the centroid cautiously when its angular coherence is small.")
print(f"Wrote outputs to {OUTPUT_DIR.resolve()}")

if SHOW_PLOTS:
    plt.show()
else:
    plt.close("all")
