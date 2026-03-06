import argparse
import json
import subprocess
import sys
from pathlib import Path

import plot_results


def compute_dt(cfg):
    if cfg.get("dt") is not None:
        return float(cfg["dt"])
    eps0 = 8.85e-12
    mu0 = 4e-7 * 3.141592653589793
    c0 = 1.0 / (eps0 * mu0) ** 0.5
    dx = float(cfg["x_range"]) / int(cfg["nx"])
    dy = float(cfg["y_range"]) / int(cfg["ny"])
    dt_cfl = (dx * dx + dy * dy) ** 0.5 / (2.0 * c0)
    dt_fs = 1.0 / (20.0 * float(cfg["f_max"]))
    return min(dt_cfl, dt_fs)


def run_rust(project_dir: Path, config_path: Path, release: bool):
    cmd = ["cargo", "run"]
    if release:
        cmd.append("--release")
    cmd.extend(["--", "--config", str(config_path)])
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=project_dir, check=True)


def out_dir(cfg, project_dir: Path):
    v = cfg.get("output_dir", "output")
    p = Path(v)
    if p.is_absolute():
        return p
    return (project_dir / p).resolve()


def main():
    parser = argparse.ArgumentParser(description="2D Ez Rust-core example")
    parser.add_argument("--config", default="config/example_2d_ez.json")
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--release", action="store_true")
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parent
    config_path = (project_dir / args.config).resolve()
    cfg = json.loads(config_path.read_text())

    src = cfg.get("source", {})
    if bool(src.get("is_show", False)):
        dt = compute_dt(cfg)
        nt = int(cfg["nt"])
        f_max = float(cfg["f_max"])
        amp = float(src.get("amplitude", 1.0))
        tw = src.get("tw")
        tw = float(tw) if tw is not None else 0.5 / f_max
        t0 = src.get("t0")
        t0 = float(t0) if t0 is not None else 4.0 * tw
        plot_results.show_source_profile(dt, nt, f_max, amp, t0, tw)

    run_rust(project_dir, config_path, args.release)

    fps = int(args.fps) if args.fps is not None else int(cfg.get("plot_fps", 60))
    plot_results.animate(out_dir(cfg, project_dir), fps)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(f"Rust solver failed with exit code {exc.returncode}", file=sys.stderr)
        sys.exit(exc.returncode)
