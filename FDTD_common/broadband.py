"""Helpers for user-sampled broadband modal sources."""

from __future__ import annotations

import numpy as np


def validate_frequency_mode_pairs(pairs, *, f_min, f_max, dt):
    """Validate and sort ``(frequency, mode_index)`` modal samples."""
    if pairs is None:
        raise ValueError(
            "broadband=True requires frequency_mode_pairs=[(frequency, mode_index), ...].")

    try:
        parsed = []
        for frequency, mode_index in pairs:
            numeric_mode = float(mode_index)
            if not numeric_mode.is_integer():
                raise ValueError("mode index is not an integer")
            parsed.append((float(frequency), int(numeric_mode)))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "frequency_mode_pairs must be an iterable of "
            "(positive_frequency, non_negative_mode_index) pairs.") from exc

    if len(parsed) < 2:
        raise ValueError("A broadband modal source requires at least two frequency-mode pairs.")

    parsed.sort(key=lambda item: item[0])
    frequencies = np.asarray([item[0] for item in parsed], dtype=float)
    mode_indices = np.asarray([item[1] for item in parsed], dtype=int)

    if not np.all(np.isfinite(frequencies)) or np.any(frequencies <= 0.0):
        raise ValueError("Broadband modal frequencies must be finite and positive.")
    if np.any(mode_indices < 0):
        raise ValueError("Broadband modal mode indices must be non-negative.")
    if np.any(np.diff(frequencies) <= 0.0):
        raise ValueError("Broadband modal frequencies must be unique.")
    # if f_min is not None and frequencies[0] < float(f_min):
    #     raise ValueError(
    #         f"Broadband modal frequency {frequencies[0]:.6g} is below f_min={float(f_min):.6g}.")
    # if f_max is not None and frequencies[-1] > float(f_max):
    #     raise ValueError(
    #         f"Broadband modal frequency {frequencies[-1]:.6g} exceeds f_max={float(f_max):.6g}.")

    nyquist = 0.5 / float(dt)
    if frequencies[-1] >= nyquist:
        raise ValueError(
            f"Broadband modal frequencies must be below the Nyquist frequency "
            f"{nyquist:.6g} Hz.")

    return frequencies, mode_indices


def align_modal_anchor_phases(electric_profiles, magnetic_profiles):
    """Apply a continuous phase gauge without changing the selected modes."""
    electric = np.asarray(electric_profiles, dtype=complex).copy()
    magnetic = np.asarray(magnetic_profiles, dtype=complex).copy()
    if electric.ndim != 2 or magnetic.shape != electric.shape:
        raise ValueError("Modal anchor profiles must be matching 2D arrays.")

    phase_factors = np.ones(electric.shape[0], dtype=complex)
    for index in range(1, electric.shape[0]):
        previous_e = electric[index - 1]
        previous_h = magnetic[index - 1]
        current_e = electric[index]
        current_h = magnetic[index]

        electric_scale = np.linalg.norm(previous_e) * np.linalg.norm(current_e)
        magnetic_scale = np.linalg.norm(previous_h) * np.linalg.norm(current_h)
        overlap = 0.0j
        if electric_scale > 1e-30:
            overlap += np.vdot(previous_e, current_e) / electric_scale
        if magnetic_scale > 1e-30:
            overlap += np.vdot(previous_h, current_h) / magnetic_scale
        if abs(overlap) <= 1e-30:
            raise ValueError(
                f"Cannot phase-align broadband modal anchors {index - 1} and {index}; "
                "their selected fields have negligible overlap.")

        factor = np.exp(-1j * np.angle(overlap))
        electric[index] *= factor
        magnetic[index] *= factor
        phase_factors[index] = factor

    return electric, magnetic, phase_factors


def interpolate_modal_anchors(anchor_frequencies, anchor_values, frequencies):
    """Linearly interpolate complex-valued columns along frequency."""
    anchor_values = np.asarray(anchor_values, dtype=complex)
    real = np.empty((len(frequencies), anchor_values.shape[1]), dtype=float)
    imag = np.empty_like(real)
    for position in range(anchor_values.shape[1]):
        real[:, position] = np.interp(
            frequencies, anchor_frequencies, anchor_values[:, position].real)
        imag[:, position] = np.interp(
            frequencies, anchor_frequencies, anchor_values[:, position].imag)
    return real + 1j * imag


def synthesize_modal_drives(*, time, waveform, frequencies, electric_profiles,
                            magnetic_profiles, n_effs, half_cell, c0):
    """
    Synthesize real Yee-time aperture drives from modal frequency anchors.

    The selected anchor fields and propagation constants are linearly
    interpolated onto every real-FFT bin inside the anchor interval. Electric
    samples are evaluated at ``time``. Magnetic samples include the existing
    solver convention: an advance of ``dt/2`` in time and a
    frequency-dependent propagation phase through ``half_cell``.
    """
    time = np.asarray(time, dtype=float)
    waveform = np.asarray(waveform, dtype=float)
    frequencies = np.asarray(frequencies, dtype=float)
    electric_profiles = np.asarray(electric_profiles, dtype=complex)
    magnetic_profiles = np.asarray(magnetic_profiles, dtype=complex)
    n_effs = np.asarray(n_effs, dtype=float)

    if time.ndim != 1 or waveform.shape != time.shape:
        raise ValueError("time and waveform must be matching one-dimensional arrays.")
    if electric_profiles.ndim != 2 or electric_profiles.shape[0] != frequencies.size:
        raise ValueError("electric_profiles must have shape (frequency, aperture).")
    if magnetic_profiles.shape != electric_profiles.shape:
        raise ValueError("electric and magnetic modal profiles must have matching shapes.")
    if n_effs.shape != frequencies.shape:
        raise ValueError("n_effs must contain one value per modal frequency.")
    if time.size < 2:
        raise ValueError("At least two time samples are required.")

    dt = float(time[1] - time[0])
    fft_frequencies = np.fft.rfftfreq(time.size, d=dt)
    source_spectrum = np.fft.rfft(waveform)
    active = (
        (fft_frequencies >= frequencies[0])
        & (fft_frequencies <= frequencies[-1])
    )
    if not np.any(active):
        raise ValueError(
            "No rFFT frequency bin lies inside the broadband modal anchor range. "
            "Increase Nt or widen the anchor interval.")

    modal_frequencies = fft_frequencies[active]
    interpolated_electric = interpolate_modal_anchors(
        frequencies, electric_profiles, modal_frequencies)
    interpolated_magnetic = interpolate_modal_anchors(
        frequencies, magnetic_profiles, modal_frequencies)

    anchor_beta = 2.0 * np.pi * frequencies * n_effs / float(c0)
    interpolated_beta = np.interp(modal_frequencies, frequencies, anchor_beta)
    omega = 2.0 * np.pi * modal_frequencies
    magnetic_phase = np.exp(
        1j * (omega * dt / 2.0 + interpolated_beta * float(half_cell)))

    aperture = electric_profiles.shape[1]
    electric_spectrum = np.zeros(
        (fft_frequencies.size, aperture), dtype=complex)
    magnetic_spectrum = np.zeros_like(electric_spectrum)
    electric_spectrum[active] = (
        source_spectrum[active, None] * interpolated_electric)
    magnetic_spectrum[active] = (
        source_spectrum[active, None]
        * interpolated_magnetic
        * magnetic_phase[:, None]
    )

    reference_spectrum = np.zeros_like(source_spectrum)
    reference_spectrum[active] = source_spectrum[active]
    electric_drive = np.fft.irfft(electric_spectrum, n=time.size, axis=0)
    magnetic_drive = np.fft.irfft(magnetic_spectrum, n=time.size, axis=0)
    reference_drive = np.fft.irfft(reference_spectrum, n=time.size)

    # Used only to select a representative stored anchor profile.
    analysis = np.exp(
        -1j * 2.0 * np.pi * frequencies[:, None] * time[None, :])
    anchor_source_spectrum = dt * (analysis @ waveform)
    anchor_magnetic_phase = np.exp(
        1j * (
            2.0 * np.pi * frequencies * dt / 2.0
            + anchor_beta * float(half_cell)
        )
    )

    return {
        "electric": electric_drive,
        "magnetic": magnetic_drive,
        "reference": reference_drive,
        "source_spectrum": source_spectrum,
        "anchor_source_spectrum": anchor_source_spectrum,
        "fft_frequencies": fft_frequencies,
        "active_frequency_mask": active,
        "interpolated_frequencies": modal_frequencies,
        "interpolated_electric_profiles": interpolated_electric,
        "interpolated_magnetic_profiles": interpolated_magnetic,
        "interpolated_beta": interpolated_beta,
        "magnetic_phase": anchor_magnetic_phase,
        "fft_magnetic_phase": magnetic_phase,
    }


def plot_modal_grid(*, frequencies, first_modes, second_modes, n_effs, axis,
                    first_label, second_label, title, mode_indices=None):
    """Plot frequency rows by mode-index columns using compact subplots."""
    import matplotlib.pyplot as plt

    frequencies = np.asarray(frequencies, dtype=float)
    rows = len(first_modes)
    available = min(mode_set.shape[0] for mode_set in first_modes)
    if mode_indices is None:
        mode_indices = list(range(available))
    else:
        mode_indices = [int(index) for index in mode_indices]
    if not mode_indices or min(mode_indices) < 0 or max(mode_indices) >= available:
        raise ValueError("Preview mode indices must refer to available solved modes.")
    columns = len(mode_indices)
    fig, axes = plt.subplots(
        rows, columns,
        figsize=(max(3.0, 2.55 * columns), max(2.2, 1.85 * rows)),
        sharex=True, squeeze=False,
    )

    for row in range(rows):
        for column, mode_index in enumerate(mode_indices):
            ax = axes[row, column]
            twin = ax.twinx()
            ax.plot(axis, first_modes[row][mode_index], linewidth=1.05, color="C0")
            twin.plot(
                axis, second_modes[row][mode_index],
                linewidth=0.9, linestyle="--", color="C1",
            )
            ax.set_title(
                f"{frequencies[row] / 1e9:.4g} GHz · m={mode_index}\n"
                f"$n_{{eff}}$={n_effs[row][mode_index]:.5g}",
                fontsize=7,
            )
            ax.tick_params(labelsize=6, length=2)
            twin.tick_params(labelsize=6, length=2)
            ax.grid(True, alpha=0.2)
            if column == 0:
                ax.set_ylabel(first_label, fontsize=7, color="C0")
            if column == columns - 1:
                twin.set_ylabel(second_label, fontsize=7, color="C1")
            if row == rows - 1:
                ax.set_xlabel("position (m)", fontsize=7)

    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    plt.show()
    return fig, axes
