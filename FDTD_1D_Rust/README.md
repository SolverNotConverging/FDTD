# FDTD 1D Rust Core + Python Example

Rust is used for the core 1D FDTD update loop. Python is used for orchestration and plotting.

## Files
- `src/main.rs`: Rust core solver
- `config/example_1d.json`: simulation + plotting setup
- `example_1d.py`: show source profile (optional), run Rust solver, then plot
- `plot_results.py`: plotting utilities

## Config options
- `plot_fps`: default FPS for animation
- `source.is_show`: if `true`, show source time/frequency profile before simulation

## Run (Rust only)
```powershell
cd FDTD_1D_Rust
cargo run -- --config config/example_1d.json
```

## Run (Python example: source preview + solve + plot)
```powershell
cd FDTD_1D_Rust
python example_1d.py --mode animate
python example_1d.py --mode spectrum
```

To override config FPS:
```powershell
python example_1d.py --mode animate --fps 30
```

If your Python executable is `py`:
```powershell
py example_1d.py --mode animate
```

## Notes
- Animation now has 3 subplots: fields, material profiles, transmission/reflection.
- Material profiles are exported by Rust as `er_profile.csv` and `mr_profile.csv`.
- Outputs are CSV files in `output_dir` (default `output`).
