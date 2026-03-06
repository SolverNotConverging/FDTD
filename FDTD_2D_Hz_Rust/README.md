# FDTD_2D_Hz_Rust

Rust core:
- 2D Hz updates
- point + line-soft sources
- line monitors
- source/monitor FFT power in Rust (one-sided, normalized by Nt)

Python:
- source time/frequency preview (`is_show`)
- 2x2 animation matching original style (n-map, Ex, Ey, Hz)
- FFT plot from Rust CSV outputs

Run examples:
```powershell
cd FDTD_2D_Hz_Rust
python example_run.py --config config/example_1_simple_source.json
python example_run.py --config config/example_2_tfsf_like.json
python example_run.py --config config/example_3_waveguide_like.json
python example_run.py --config config/example_4_farfield_like.json
```
