import argparse
import json
import subprocess
from pathlib import Path

import plot_results


def main():
    parser = argparse.ArgumentParser(description="Run Rust core then plot for 2D Hz")
    parser.add_argument("--config", default="config/example_1_simple_source.json")
    parser.add_argument("--release", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    cfg_path = (root / args.config).resolve()
    cfg = json.loads(cfg_path.read_text())
    cfg = plot_results.prepare_waveguide_profiles_from_config(cfg)
    if "postprocessing" in cfg and "nf2ff" in cfg["postprocessing"] and isinstance(cfg["postprocessing"]["nf2ff"], dict):
        cfg["postprocessing"]["nf2ff"]["enabled"] = False
    resolved_cfg_path = root / "_resolved_config_runtime.json"
    resolved_cfg_path.write_text(json.dumps(cfg, indent=2))

    if cfg.get("plot", {}).get("show_source_profiles", True):
        plot_results.show_source_profiles_from_config(cfg)

    cmd = ["cargo", "run"]
    if args.release:
        cmd.append("--release")
    cmd += ["--", "--config", str(resolved_cfg_path)]
    subprocess.run(cmd, cwd=root, check=True)

    out_dir = Path(cfg.get("output_dir", "output"))
    if not out_dir.is_absolute():
        out_dir = root / out_dir

    meta = plot_results.parse_metadata(out_dir / "metadata.txt")
    fps = int(meta.get("plot_fps", cfg.get("plot", {}).get("fps", 60)))
    dynamic = str(meta.get("dynamic_clim", cfg.get("plot", {}).get("dynamic_clim", True))).lower() == "true"
    clim_smooth = float(meta.get("clim_smooth", cfg.get("plot", {}).get("clim_smooth", 0.25)))

    if cfg.get("plot", {}).get("show_animation", True):
        plot_results.animate(out_dir, cfg, fps=fps, dynamic_clim=dynamic, clim_smooth=clim_smooth)
    if cfg.get("plot", {}).get("show_fft", True):
        plot_results.show_fft(out_dir, cfg)
    if cfg.get("plot", {}).get("show_nf2ff", False):
        plot_results.compute_nf2ff_python(out_dir, cfg)
        plot_results.show_nf2ff(out_dir)


if __name__ == "__main__":
    main()

