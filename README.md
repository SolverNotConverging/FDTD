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

**Project Layout**
- `FDTD_1D`: 1D solver and example (`FDTD_1D_example.py`).
- `FDTD_2D_Ez`: 2D TMz solver, GPU variant, and examples.
- `FDTD_2D_Hz`: 2D TEz solver, GPU variant, and examples.
- `FDTD_2D_Ez_Legacy`: older 2D Ez implementation and examples.
- `FDTD_3D`: 3D development skeleton (CPML/source scaffolding).

**Quick Start**
```bash
python FDTD_1D/FDTD_1D_example.py
python FDTD_2D_Ez/Example_1_Simple_Source.py
python FDTD_2D_Hz/Example_1_Simple_Source.py
```

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
