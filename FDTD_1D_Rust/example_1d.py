import argparse
import json
import subprocess
import sys
from pathlib import Path

import plot_results


def compute_dt(cfg: dict) -> float:
    if cfg.get("dt") is not None:
        return float(cfg["dt"])

    eps0 = 8.85e-12
    mu0 = 4e-7 * 3.141592653589793
    c0 = 1.0 / (eps0 * mu0) ** 0.5

    z_range = float(cfg["z_range"])
    nz = int(cfg["nz"])
    f_max = float(cfg["f_max"])

    dz = z_range / nz
    dt_cfl = dz / (2.0 * c0)
    dt_freq_sampling = 1.0 / (20.0 * f_max)
    return min(dt_cfl, dt_freq_sampling)


def show_source_if_requested(cfg: dict):
    src = cfg.get("source", {})
    if not bool(src.get("is_show", False)):
        return

    dt = compute_dt(cfg)
    nt = int(cfg["nt"])
    f_max = float(cfg["f_max"])
    amplitude = float(src.get("amplitude", 1.0))
    tw = src.get("tw")
    tw = float(tw) if tw is not None else 0.5 / f_max
    t0 = src.get("t0")
    t0 = float(t0) if t0 is not None else 4.0 * tw

    plot_results.show_source_profile(dt=dt, nt=nt, f_max=f_max, amplitude=amplitude, t0=t0, tw=tw)


def run_rust_solver(project_dir: Path, config_path: Path, release: bool):
    cmd = ["cargo", "run"]
    if release:
        cmd.append("--release")
    cmd.extend(["--", "--config", str(config_path)])

    print("Running Rust solver:", " ".join(cmd))
    subprocess.run(cmd, cwd=project_dir, check=True)


def output_dir_from_config(config_path: Path, project_dir: Path) -> Path:
    cfg = json.loads(config_path.read_text())
    out = cfg.get("output_dir", "output")
    out_path = Path(out)
    if out_path.is_absolute():
        return out_path
    return (project_dir / out_path).resolve()


def fps_from_config(cfg: dict, cli_fps):
    if cli_fps is not None:
        return int(cli_fps)
    return int(cfg.get("plot_fps", 30))


def main():
    parser = argparse.ArgumentParser(description="Run Rust FDTD core and plot results")
    parser.add_argument("--config", default="config/example_1d.json", help="Path to JSON config")
    parser.add_argument("--mode", choices=["animate", "spectrum"], default="animate")
    parser.add_argument("--fps", type=int, default=None, help="Override config plot_fps")
    parser.add_argument("--release", action="store_true", help="Run Rust solver in release mode")
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parent
    config_path = (project_dir / args.config).resolve()

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    cfg = json.loads(config_path.read_text())

    show_source_if_requested(cfg)
    run_rust_solver(project_dir, config_path, args.release)

    output_dir = output_dir_from_config(config_path, project_dir)
    fps = fps_from_config(cfg, args.fps)

    if args.mode == "animate":
        plot_results.animate(output_dir, fps)
    else:
        plot_results.spectrum(output_dir)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(f"Rust solver failed with exit code {exc.returncode}", file=sys.stderr)
        sys.exit(exc.returncode)
