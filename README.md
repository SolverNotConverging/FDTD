# FDTD
A compact, research-oriented FDTD (finite-difference time-domain) solver for Maxwell's equations. It supports 1D and 2D time-domain simulations with CPML absorbing boundaries, anisotropic materials, multiple source types, and frequency-domain post-processing. GPU acceleration is available for the 2D solvers via PyTorch. A 3D solver is included as a development skeleton.

**What It Does**
- Solves 1D and 2D electromagnetic wave propagation in the time domain.
- 2D modes include TMz (`FDTD_2D_Ez`) and TEz (`FDTD_2D_Hz`).
- CPML boundaries for low-reflection truncation.
- Material modeling with rectangles/circles, including anisotropy and subpixel smoothing.
- Source options: point, line-soft, TF/SF, and waveguide eigenmode ports.
- Monitors and analysis: line monitors, FFT-based power, and 2D NF2FF.
- GPU execution for 2D via `FDTD_2D_Ez_GPU` and `FDTD_2D_Hz_GPU` (PyTorch required).
- Rust-accelerated CPU kernels for core field updates in:
  - `FDTD_1D/FDTD_1D.py`
  - `FDTD_2D_Ez/FDTD_2D_Ez.py`
  - `FDTD_2D_Hz/FDTD_2D_Hz.py`

**Project Layout**
- `FDTD_1D`: 1D solver and example (`FDTD_1D_example.py`).
- `FDTD_2D_Ez`: 2D TMz solver, GPU variant, and examples.
- `FDTD_2D_Hz`: 2D TEz solver, GPU variant, and examples.
- `FDTD_2D_Ez_Legacy`: older 2D Ez implementation and examples.
- `FDTD_3D`: 3D development skeleton (CPML/source scaffolding).
- Rust kernel sources/builds:
  - `FDTD_1D/rust_kernel`
  - `FDTD_2D_Ez/rust_kernel`
  - `FDTD_2D_Hz/rust_kernel`

**Quick Start**
```bash
python FDTD_1D/FDTD_1D_example.py
python FDTD_2D_Ez/Example_1_Simple_Source.py
python FDTD_2D_Hz/Example_1_Simple_Source.py
```

## Rust CPU Kernel (Default Solver) Guide

### How It Works
- The default solvers (`FDTD_1D`, `FDTD_2D_Ez`, `FDTD_2D_Hz`) try to load a compiled Rust shared library at startup.
- If found, Rust is used for core interior update loops/curl loops.
- If not found (or load fails), solver automatically falls back to pure Python loops.
- No API change is required in user scripts.

### Where the Solvers Look for Compiled Kernels
- `FDTD_1D/rust_kernel/target/release/`
- `FDTD_2D_Ez/rust_kernel/target/release/`
- `FDTD_2D_Hz/rust_kernel/target/release/`

Expected library names by platform:
- macOS: `lib*.dylib`
- Linux: `lib*.so`
- Windows: `*.dll` (both `name.dll` and `libname.dll` are accepted)

### Build Rust Kernels

Prerequisites:
- Install Rust toolchain (`cargo` + `rustc`) from `https://rustup.rs/`.
- Use a Python environment with `numpy` (and your normal solver deps).

Build commands:

macOS/Linux (bash/zsh):
```bash
cd FDTD/FDTD_1D/rust_kernel && cargo build --release
cd ../../FDTD_2D_Ez/rust_kernel && cargo build --release
cd ../../FDTD_2D_Hz/rust_kernel && cargo build --release
```

Windows (PowerShell):
```powershell
cd FDTD\FDTD_1D\rust_kernel; cargo build --release
cd ..\..\FDTD_2D_Ez\rust_kernel; cargo build --release
cd ..\..\FDTD_2D_Hz\rust_kernel; cargo build --release
```

### Verify Rust Kernel Is Active

Check from Python:
```python
from FDTD_2D_Ez import FDTD_2D_Ez
sim = FDTD_2D_Ez(x_range=1e-3, y_range=1e-3, Nx=50, Ny=50, f_max=1e11, Nt=10)
print(sim._use_rust_kernel)  # True if Rust kernel loaded
```

Same approach works for `FDTD_1D` and `FDTD_2D_Hz`.

### Examples Policy
- `GPU_exmaple.py` files use GPU solvers (`FDTD_2D_Ez_GPU` / `FDTD_2D_Hz_GPU`).
- All other 1D/2D examples use default solvers (CPU + Rust if compiled).

### Cross-Platform Notes
- Loader code is platform-aware and supports macOS/Linux/Windows shared library names.
- If Rust is unavailable on a machine, the simulation still runs via Python fallback.
- When moving project folders between machines, rebuild `rust_kernel` on that machine.

**Typical Workflow**
1. Create a solver with domain size, grid resolution, and frequency bounds.
2. Configure boundaries (CPML/periodic).
3. Add geometry (rectangles, circles, or custom material regions).
4. Add sources (point, line-soft, TF/SF, or waveguide modes).
5. Add monitors if needed (line monitors, NF2FF, FFT power).
6. Run, visualize, and post-process (animations and FFT plots).


⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠁⡇⠀⠀⠀⠀⠀⠀⠀⠸⣿⡇⠀⠚⠉⣸⣿⣅⣀⠀⠀
⠙⠻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⢸⠀⠀⠀⠀⠀⠀⠀⠀⢠⣿⣷⣾⠿⠿⠿⠿⠛⠉⠀⠀
⠀⠀⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇⡎⠀⠀⠀⠀⠀⠀⠀⠀⢸⣿⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠿⠟⠛⠛⠛⠛⠛⠛⢛⠛⠛⠻⠿⠿⠿⠿⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣦⣀⡀⠀⠀⠀⠀⢠⡿⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⣰
⣿⣿⣿⣿⣿⣿⠿⠛⠛⠉⠉⢀⣤⠞⠋⠀⠀⠀⠀⢀⡤⢀⡤⠚⠁⠀⠀⠀⠀⠀⠀⣠⠴⠂⠀⠈⠉⠉⠛⠻⠿⣿⣿⣿⣿⣿⣿⣿⣶⣤⡀⢀⡟⠁⠀⠀⠀⠀⠀⠀⠀⠀⢀⣼⣿
⣿⣿⣿⣿⠏⣠⠀⠀⠀⠀⣰⠟⠁⠀⠀⠀⠀⣠⠖⠉⠀⠀⠀⠀⠀⠀⠀⠀⢀⡴⠚⠁⠀⠀⠀⠀⠀⠀⠀⠀⢀⡟⠈⠉⠛⠿⣿⣿⣿⣿⣿⣿⣄⡀⠀⠀⠀⠀⠀⠀⠀⣠⣾⣿⣿
⣿⣿⣿⢏⡴⠃⠀⠀⢠⡞⠁⠀⣠⠔⠀⣠⢞⣡⠞⠁⠀⠀⠀⠀⠀⠀⢀⡴⠋⠀⠀⠀⠀⠀⠀⠀⣠⠄⠀⡠⢻⠁⠀⠀⠀⠀⠀⠙⠻⢿⣿⣿⣿⣿⣦⡀⠀⠀⠀⢀⣴⣿⣿⠉⠉
⣿⣿⡷⠋⠀⠀⠀⣰⠋⣀⠴⠋⠀⣠⠞⢡⠟⠁⠀⠀⠀⠀⠀⠀⠀⣠⠞⠁⠀⠀⠀⠀⠀⣀⡴⠊⢁⡴⠊⢁⡏⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠻⣿⣿⣿⣿⣦⡀⣠⣾⣿⡟⠉⠀⢀
⡿⠏⡀⠀⢀⣠⣾⠷⠋⠁⠀⣠⠞⠁⢠⠏⠀⠀⠀⠀⠀⠀⠀⢀⡼⠃⠀⠀⠀⠀⣀⡴⠚⠁⣠⠔⠋⠀⠀⣸⠁⠀⠀⠀⠀⠀⠀⠀⠀⣀⠀⣸⠁⠙⢿⣿⣿⣿⣿⣿⣿⡀⠀⠀⠃
⣁⣠⠵⠞⠛⠉⠀⠀⠀⣠⡾⣡⠄⢠⡟⠀⠀⠀⠀⠀⠀⠀⢠⡞⠀⠀⢀⣠⠴⠚⢁⣤⣶⣋⡁⠀⠀⠀⢀⡇⠀⠀⠀⠀⠀⠀⠀⠀⡼⠁⢠⡇⠀⠀⠀⠙⢿⣿⣿⣿⣿⣷⣦⡤⠂
⠁⠀⠀⠀⠀⢀⣠⢴⡾⣫⣾⠃⢠⣿⠁⠀⠀⠀⠀⠀⡀⢠⣏⡠⠴⢚⣉⠤⠖⠛⠁⠀⠀⠀⠈⠙⠳⠦⣼⠁⠀⠀⠀⠀⠀⠀⠀⡜⠁⢠⢿⡇⠀⡜⢠⠀⠀⠙⢿⣿⣿⣿⣿⠀⠀
⠀⠀⠀⠀⢠⠞⢁⣞⡽⢡⡏⠀⡞⡞⠀⠀⠀⠀⠀⠀⢷⣿⠵⠒⣯⡉⠀⢠⡄⠤⠤⠤⣀⣀⠀⠀⠀⠀⡾⠀⠀⠀⠀⠀⠀⠀⠀⠀⣰⡟⢸⠁⠀⠀⡟⢠⢀⠀⡀⠻⣿⣿⣿⣷⣾
⠀⠀⠀⠀⠀⢀⣾⡿⠁⣼⠀⢸⢡⡇⠀⠀⠀⠀⠀⠀⡞⣿⣿⡿⠿⠿⠿⠿⠿⣿⣿⣶⣤⣄⠉⠳⣄⡀⡇⠀⠀⠀⠀⠀⠀⠀⡼⣲⠋⡇⢸⠀⠀⢰⠃⡆⢸⢠⢹⠀⠙⣿⣿⣿⡏
⠀⠀⢀⣠⠴⠛⡿⣧⣴⡏⠀⡏⢸⠇⠀⠀⠀⠀⠀⢸⠁⢹⡇⠀⠀⣠⠖⢻⣟⠲⢮⡙⢿⣿⡗⢆⠘⡅⡇⠀⠀⠀⠀⠀⢀⡞⣰⠇⠀⣷⢸⠀⠀⣾⠀⢧⠈⣿⢸⢠⠀⠸⣿⣿⣿
⣶⣚⣉⠤⠤⠾⣧⠘⣇⣷⠀⠃⢸⡀⠀⠀⠀⠀⠀⣿⠀⠘⡇⠀⡾⠥⠞⠉⠁⠹⣾⣷⠀⠘⢿⡌⠀⢹⡇⠀⠀⠀⠀⠀⡼⢠⠏⠉⠛⢿⣼⠀⢰⣇⠀⠘⠦⣿⠀⢿⠀⠀⢿⣿⣿
⠀⠀⠀⠀⠀⢀⢿⣦⡈⠻⣇⠀⠸⡇⠀⠀⠀⠀⠀⣿⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⢸⣼⠀⠀⠈⠧⠀⠀⡇⠀⠀⠀⠀⢸⢣⡏⠀⠀⠀⢸⣿⠀⢸⢻⠀⠀⢸⣿⡀⠈⠐⡆⢸⣿⣿
⠀⠀⠀⠀⠀⠁⢈⣯⢳⣄⠘⡄⠀⢷⠀⠀⠀⢸⡀⢹⡀⠀⠀⠀⠹⠤⣀⡀⠀⠀⢀⡟⠀⠀⠀⠀⠀⠀⣷⠀⠀⢰⠂⡏⡾⠀⠀⣀⠀⠀⣿⡇⡾⣾⡄⠀⢸⠃⠃⠀⠀⣧⢸⣿⣿
⠀⠀⠀⠀⠀⣠⠞⢈⣿⡉⠛⣿⣆⠘⣆⠀⠀⠀⢳⡀⢷⡀⠀⠀⠀⠀⠀⠉⠓⠚⠉⠀⠀⠀⠀⠀⠀⠀⢸⡄⠀⢸⣶⣿⠇⣀⡬⢭⣉⠲⣼⣇⡇⠘⡇⢐⡿⠀⠀⠀⢰⡟⢸⣿⣿
⢦⣀⠀⣠⣾⠏⢀⣾⣿⡇⠀⣿⢯⣧⣘⢦⡀⠀⠀⢻⣦⣳⣄⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⣧⠀⢸⢻⣿⣴⣾⣿⡿⢿⣷⣌⢿⡇⠀⣧⣼⠁⠀⠀⠀⣼⡇⣸⣿⣿
⡄⠈⠛⠳⠧⣤⣾⠿⠋⠀⠀⡿⠈⢣⡘⣿⠿⠦⣤⣬⡄⠈⠙⠛⠛⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢹⡆⢿⠈⣿⡯⠿⠳⣝⣦⠙⣿⡾⠃⠀⣿⠃⠀⠀⠀⢠⡿⢡⣿⣿⣿
⠛⠷⣤⡀⠀⠀⠀⠀⠀⠀⣰⠇⡇⠀⢳⡘⢦⠀⠀⠀⣧⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢳⣸⡾⠁⠱⠄⠀⢻⣿⠀⠸⣿⣖⡾⠃⠀⠀⠀⠀⡼⢣⣾⣿⣿⣿
⠀⠀⠈⠛⢶⣤⣤⣤⠤⢾⣿⡄⢡⠀⠀⠙⣎⢳⡀⠀⢸⡄⠀⠀⠀⠀⢀⣀⣀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢻⣿⣄⠀⢀⡐⢻⡟⠀⢠⣿⡿⠷⠄⠀⢀⣠⢞⣵⠿⣿⣟⢻⣿
⣦⡀⠀⠀⠀⢻⣿⣦⣴⠋⡏⣇⠘⢧⡀⠀⣏⣷⣽⣦⣄⠹⣄⠀⠀⢀⡟⠃⠈⠉⠹⠗⠢⣄⡀⠀⠀⠀⠀⠀⠀⠀⠙⠻⠷⢤⠤⠊⢀⣴⡿⠋⢀⡼⠒⠚⠿⠚⢋⡇⠘⣿⠻⣿⣿
⢹⣿⣦⡀⠀⠀⠹⣿⣿⠀⠹⣼⡄⠈⠛⡄⢹⣼⡈⢿⠈⠙⠛⠓⠒⠘⡇⠀⠀⠀⠀⠀⠈⠲⡍⠳⢄⡀⠀⠀⠀⠀⠀⠀⠀⠠⢄⣤⣖⣋⣤⢶⡿⠁⠀⠀⢀⡾⣸⠀⠀⣏⠀⠙⣟
⠀⠈⢿⡷⣄⠀⠀⢿⣿⣧⠀⠹⣧⠀⠀⠃⠆⢿⣧⠈⠀⠀⠀⠀⠀⠀⢣⠀⠀⠀⠀⠀⠀⠀⠈⠆⠀⠹⡄⠀⠀⠀⠀⠀⠀⠀⠀⣰⢿⣟⡵⠋⠀⠀⠀⢀⣾⠁⡟⠀⠀⣿⡀⠀⢈
⠀⢀⣽⣧⠈⢷⡀⠸⡎⢻⣷⡀⠘⢧⡀⠀⠀⠸⣿⣆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡼⠁⠀⠀⠀⢠⣀⠀⢀⣾⣿⠿⠋⠀⠀⠀⠀⣠⣟⣽⣸⠁⠀⠀⣿⡇⠀⠀
⡁⠊⢿⣿⠀⠀⠙⡄⢷⠀⠙⣿⣦⡈⠛⣦⡀⠀⠹⣿⢧⡀⠀⠀⠀⠀⠀⠀⠀⠀⠠⣄⣀⣠⠤⠚⠉⠀⠀⠀⠀⠀⠀⠙⣿⣭⡉⠀⠀⠀⢀⣠⣴⣿⣿⣿⣿⡏⠀⠀⢸⣿⡇⠀⠐
⣶⣿⣿⣿⡆⠀⠀⡀⢸⡆⠀⠈⢻⣿⣦⡀⠙⢦⣀⢻⡀⠹⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣠⢾⠁⢸⢹⠙⠛⢉⠹⣿⣿⣿⣿⣿⣿⣆⠀⠀⣾⡇⢹⠀⠀
⣿⣿⣿⣿⡇⠀⠀⣷⠀⡇⠀⢀⡀⢿⣿⣿⣦⡀⠙⠻⣿⣆⠈⢷⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣀⣀⣤⠴⠚⠋⠀⢸⠀⣿⠘⡆⠀⢠⢸⣿⢿⣿⣿⣿⠿⣿⣆⣾⡟⠀⢸⠀⡇
⣿⣿⣿⣿⡇⠀⠀⡟⠀⣷⠀⡾⠀⠘⣿⣿⣿⣷⡄⠀⠈⠻⣧⡀⣹⡷⢤⣀⣤⡤⠶⠶⠞⠛⠋⠉⠉⠁⢀⠀⠀⠀⠀⢸⠀⣿⠀⢱⡀⣘⢸⢘⣿⣿⣿⡋⠄⠿⢿⣯⠀⠀⣼⡐⢳
⣿⣿⣿⣿⡇⠀⢀⡇⠀⢺⠀⡇⠀⠀⢻⣿⣿⣿⣿⡀⠀⠀⠈⠻⣿⣷⣾⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⡀⠀⠀⠀⢸⠀⡇⠀⠀⢳⡈⢹⡉⠸⡽⣜⣧⠀⠀⠀⠙⢷⡀⠻⠒⠋
⣿⣿⣿⣿⣧⠀⣸⠁⠀⢸⢸⠃⠀⠀⢸⣿⣿⣿⣿⣷⠀⠀⠀⢀⡘⣿⣿⡿⠳⣦⠀⠀⠀⠀⠀⠀⠀⠀⠀⠃⠀⠀⠀⠈⡇⣧⠀⠀⠀⠹⣌⢧⠀⢳⠙⣎⢷⡀⠀⠀⠈⠻⣄⠀⠀
⣿⣿⣿⣿⡿⢀⡏⠀⠀⣼⣾⠀⠀⠀⠈⣿⣿⣿⣿⣿⣇⠀⠀⠀⠙⠞⢞⢿⣖⣿⣷⣤⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢳⢻⡀⠀⠀⠀⠈⠻⣷⣤⣧⠘⢯⣳⣄⠀⠀⠀⠙⢧⡀
