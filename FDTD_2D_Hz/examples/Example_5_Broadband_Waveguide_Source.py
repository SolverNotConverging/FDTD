from FDTD_2D_Hz import FDTD_2D_Hz


# Two identical dielectric strip waveguides propagate in parallel toward +x.
# For this grid and 4 mm modal aperture, the Hz fundamental mode cuts on just
# below 35 GHz, so the first broadband sample deliberately lies near cutoff.
sim = FDTD_2D_Hz(
    x_range=16e-3,
    y_range=14e-3,
    Nx=160,
    Ny=140,
    f_min=50e9,
    f_max=90e9,
    Nt=5000,
)
sim.config(backend="gpu")
sim.periodic = ""
sim.add_PML(
    pml_width=13,
    order=3,
    direction="xy",
    kappa_max=7,
    alpha_max=0.025,
)

# Lower and upper parallel dielectric waveguides.
sim.add_rectangle(
    ER=5,
    MR=1,
    x_position=(0.0, 16e-3),
    y_position=(3.5e-3, 4.5e-3),
)
sim.add_rectangle(
    ER=5,
    MR=1,
    x_position=(0.0, 16e-3),
    y_position=(9.5e-3, 10.5e-3),
)

# Upper guide: a Gaussian pulse whose spatial profile comes from one
# eigenmode solve at the center frequency. The same profile is reused across
# the pulse spectrum; only the modal calculation is single-frequency.
sim.add_source(
    "waveguide-x",
    x=3.0e-3,
    y=(7.5e-3, 12.5e-3),
    f_min=50e9,
    f_max=90e9,
    amplitude=1,
    mode_index=0,
    modes_to_show=3,
    is_show=True,
)

# Lower guide: explicitly selected modal anchors. Their fields and beta
# are linearly interpolated across the dense source spectrum before the
# frequency-dependent Yee half-cell phase and inverse FFT are applied.
sim.add_source(
    "waveguide-x",
    x=3.0e-3,
    y=(1.5e-3, 6.5e-3),
    amplitude=1.0,
    broadband=True,
    frequency_mode_pairs=[
        (50e9, 0),
        (55e9, 0),
        (60e9, 0),
        (70e9, 0),
        (80e9, 0),
        (90e9, 0),
    ],
    modes_to_show=3,
    is_show=True,
)

# Each guide gets a monitor before and after the common source plane. Before
# monitors use a -x normal so backward power is reported as a positive level;
# after monitors use the default +x normal for forward power.
broadband_backward_monitor = sim.add_line_monitor(
    x=2.5e-3, y=(1.5e-3, 6.5e-3), index=10)
broadband_forward_monitor = sim.add_line_monitor(
    x=3.5e-3, y=(1.5e-3, 6.5e-3), index=20)
single_solve_backward_monitor = sim.add_line_monitor(
    x=2.5e-3, y=(7.5e-3, 12.5e-3), index=30)
single_solve_forward_monitor = sim.add_line_monitor(
    x=3.5e-3, y=(7.5e-3, 12.5e-3), index=40)

sim.run(record_stride=6)

comparison_frequencies = sim.sources[1]["broadband_interpolated_frequencies"]

# Each pair is normalized by its own source: source 1 for the broadband guide
# and source 0 for the center-frequency-mode Gaussian guide. Evaluating both
# over the same dense grid makes their spectral performance directly visible.
broadband_forward_power = sim.power_spectrum(
    broadband_forward_monitor, comparison_frequencies,
    source_index=1, normal_sign=1.0)
broadband_backward_power = sim.power_spectrum(
    broadband_backward_monitor, comparison_frequencies,
    source_index=1, normal_sign=-1.0)
single_solve_forward_power = sim.power_spectrum(
    single_solve_forward_monitor, comparison_frequencies,
    source_index=0, normal_sign=1.0)
single_solve_backward_power = sim.power_spectrum(
    single_solve_backward_monitor, comparison_frequencies,
    source_index=0, normal_sign=-1.0)

power_figure, power_axis = sim.plot_power_spectrum(
    broadband_forward_power, db=True)
sim.plot_power_spectrum(broadband_backward_power, db=True, ax=power_axis)
sim.plot_power_spectrum(single_solve_forward_power, db=True, ax=power_axis)
sim.plot_power_spectrum(single_solve_backward_power, db=True, ax=power_axis)
labels = (
    "five-mode-solve forward (+x)",
    "five-mode-solve backward (-x)",
    "single-mode-solve forward (+x)",
    "single-mode-solve backward (-x)",
)
for line, label in zip(power_axis.lines[-4:], labels):
    line.set_label(label)
power_axis.lines[-3].set_linestyle("--")
power_axis.lines[-1].set_linestyle("--")
power_axis.set_title("Source-normalized forward/backward modal power")
power_axis.legend()
power_figure.tight_layout()

sim.show_animation(fps=30, dynamic_clim=False, clim_smooth=0.25)
