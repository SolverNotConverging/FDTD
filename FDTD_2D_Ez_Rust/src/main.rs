use num_complex::Complex64;
use rustfft::FftPlanner;
use serde::Deserialize;
use std::env;
use std::error::Error;
use std::f64::consts::PI;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
enum CoordSpec {
    Scalar(f64),
    Range([f64; 2]),
}

#[derive(Debug, Clone, Deserialize)]
struct ObjectConfig {
    er: f64,
    mr: f64,
    x_range: [f64; 2],
    y_range: [f64; 2],
}

#[derive(Debug, Clone, Deserialize)]
struct SourceConfig {
    kind: String,
    x: CoordSpec,
    y: CoordSpec,
    amplitude: Option<f64>,
    t0: Option<f64>,
    tw: Option<f64>,
    f_min: Option<f64>,
    f_max: Option<f64>,
    f0: Option<f64>,
    angle: Option<f64>,
    mode_index: Option<usize>,
    n_eff: Option<f64>,
    profile: Option<Vec<f64>>,
    profile_e: Option<Vec<f64>>,
    profile_h: Option<Vec<f64>>,
    is_show: Option<bool>,
    name: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct MonitorConfig {
    x: CoordSpec,
    y: CoordSpec,
    name: Option<String>,
    normal_sign: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct PlotConfig {
    fps: Option<u32>,
    dynamic_clim: Option<bool>,
    clim_smooth: Option<f64>,
    show_source_profiles: Option<bool>,
    show_animation: Option<bool>,
    show_fft: Option<bool>,
    show_nf2ff: Option<bool>,
}

#[derive(Debug, Clone, Deserialize)]
struct PmlConfig {
    enabled: Option<bool>,
    width_cells: Option<usize>,
    strength: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct Nf2ffConfig {
    enabled: Option<bool>,
    top: Option<usize>,
    bottom: Option<usize>,
    left: Option<usize>,
    right: Option<usize>,
    nphi: Option<usize>,
    freq_count: Option<usize>,
    source_index: Option<usize>,
}

#[derive(Debug, Clone, Deserialize)]
struct PostProcessingConfig {
    compute_source_fft: Option<bool>,
    compute_monitor_fft: Option<bool>,
    nf2ff: Option<Nf2ffConfig>,
}

#[derive(Debug, Clone, Deserialize)]
struct Config {
    x_range: f64,
    y_range: f64,
    nx: usize,
    ny: usize,
    f_min: Option<f64>,
    f_max: f64,
    nt: usize,
    dt: Option<f64>,
    record_stride: Option<usize>,
    objects: Vec<ObjectConfig>,
    sources: Vec<SourceConfig>,
    monitors: Option<Vec<MonitorConfig>>,
    pml: Option<PmlConfig>,
    postprocessing: Option<PostProcessingConfig>,
    output_dir: Option<String>,
    plot: Option<PlotConfig>,
}

#[derive(Clone)]
enum SourceKind {
    Point,
    LineSoft,
    Sftf,
    WaveguideX,
    WaveguideY,
}

#[derive(Clone)]
struct SourcePrepared {
    kind: SourceKind,
    points: Vec<usize>,
    along_x: bool,
    e_weights: Vec<f64>,
    h_weights: Vec<f64>,
    delays: Vec<f64>,
    amp: f64,
    t0: f64,
    tw: f64,
    ix0: usize,
    ix1: usize,
    iy0: usize,
    iy1: usize,
    angle: f64,
    n_eff: f64,
    f_min: Option<f64>,
    f_max: Option<f64>,
    name: String,
}

#[derive(Clone)]
struct MonitorPrepared {
    points: Vec<usize>,
    x_m: Vec<f64>,
    y_m: Vec<f64>,
    orientation: String,
    normal_sign: f64,
    name: String,
}

struct MonitorBuffers {
    ez: Vec<f64>,
    hx: Vec<f64>,
    hy: Vec<f64>,
    nline: usize,
}

struct Sim2DEz {
    eps0: f64,
    mu0: f64,
    eta0: f64,
    c0: f64,
    x_range: f64,
    y_range: f64,
    nx: usize,
    ny: usize,
    dx: f64,
    dy: f64,
    nt: usize,
    dt: f64,
    f_min: Option<f64>,
    f_max: f64,
    record_stride: usize,
    nt_rec: usize,

    erzz: Vec<f64>,
    mrxx: Vec<f64>,
    mryy: Vec<f64>,

    bx: Vec<f64>,
    by: Vec<f64>,
    hx: Vec<f64>,
    hy: Vec<f64>,
    dzf: Vec<f64>,
    ez: Vec<f64>,

    sources: Vec<SourcePrepared>,
    monitors: Vec<MonitorPrepared>,
    monitor_buffers: Vec<MonitorBuffers>,

    hx_hist: Vec<f32>,
    hy_hist: Vec<f32>,
    ez_hist: Vec<f32>,
    pml_mask: Vec<f64>,
}

impl Sim2DEz {
    fn avg_with_neighbor(arr: &[f64], nx: usize, ny: usize, axis: usize, direction: isize) -> Vec<f64> {
        let mut out = vec![0.0_f64; nx * ny];
        for i in 0..nx {
            for j in 0..ny {
                let (ii, jj) = if axis == 0 {
                    let ni = i as isize + direction;
                    if ni < 0 || ni >= nx as isize {
                        (-1, -1)
                    } else {
                        (ni as isize, j as isize)
                    }
                } else {
                    let nj = j as isize + direction;
                    if nj < 0 || nj >= ny as isize {
                        (-1, -1)
                    } else {
                        (i as isize, nj as isize)
                    }
                };
                let k = i * ny + j;
                let nei = if ii < 0 { 0.0 } else { arr[ii as usize * ny + jj as usize] };
                out[k] = 0.5 * (arr[k] + nei);
            }
        }
        out
    }

    fn new(cfg: &Config) -> Self {
        let eps0 = 8.85e-12;
        let mu0 = 4e-7 * PI;
        let c0 = 1.0 / (eps0 * mu0).sqrt();
        let eta0 = (mu0 / eps0).sqrt();

        let dx = cfg.x_range / cfg.nx as f64;
        let dy = cfg.y_range / cfg.ny as f64;

        let dt_cfl = (dx * dx + dy * dy).sqrt() / (2.0 * c0);
        let dt_fs = 1.0 / (20.0 * cfg.f_max);
        let dt = cfg.dt.unwrap_or(dt_cfl.min(dt_fs));

        let record_stride = cfg.record_stride.unwrap_or(1).max(1);
        let nt_rec = (cfg.nt + record_stride - 1) / record_stride;

        let n = cfg.nx * cfg.ny;

        Self {
            eps0,
            mu0,
            eta0,
            c0,
            x_range: cfg.x_range,
            y_range: cfg.y_range,
            nx: cfg.nx,
            ny: cfg.ny,
            dx,
            dy,
            nt: cfg.nt,
            dt,
            f_min: cfg.f_min,
            f_max: cfg.f_max,
            record_stride,
            nt_rec,

            erzz: vec![1.0; n],
            mrxx: vec![1.0; n],
            mryy: vec![1.0; n],

            bx: vec![0.0; n],
            by: vec![0.0; n],
            hx: vec![0.0; n],
            hy: vec![0.0; n],
            dzf: vec![0.0; n],
            ez: vec![0.0; n],

            sources: Vec::new(),
            monitors: Vec::new(),
            monitor_buffers: Vec::new(),

            hx_hist: Vec::with_capacity(nt_rec * n),
            hy_hist: Vec::with_capacity(nt_rec * n),
            ez_hist: Vec::with_capacity(nt_rec * n),
            pml_mask: vec![1.0; n],
        }
    }

    fn idx(&self, i: usize, j: usize) -> usize {
        i * self.ny + j
    }

    fn to_index_x(&self, x: f64) -> usize {
        ((x / self.dx).round() as isize).clamp(0, (self.nx - 1) as isize) as usize
    }

    fn to_index_y(&self, y: f64) -> usize {
        ((y / self.dy).round() as isize).clamp(0, (self.ny - 1) as isize) as usize
    }

    fn add_rectangle(&mut self, obj: &ObjectConfig) {
        let mut x0 = self.to_index_x(obj.x_range[0]);
        let mut x1 = ((obj.x_range[1] / self.dx).round() as isize).clamp(0, self.nx as isize) as usize;
        let mut y0 = self.to_index_y(obj.y_range[0]);
        let mut y1 = ((obj.y_range[1] / self.dy).round() as isize).clamp(0, self.ny as isize) as usize;
        if x1 < x0 {
            std::mem::swap(&mut x0, &mut x1);
        }
        if y1 < y0 {
            std::mem::swap(&mut y0, &mut y1);
        }

        for i in x0..x1 {
            for j in y0..y1 {
                let k = self.idx(i, j);
                self.erzz[k] = obj.er;
                self.mrxx[k] = obj.mr;
                self.mryy[k] = obj.mr;
            }
        }
    }

    fn init_pml(&mut self, cfg: &Config) {
        let Some(pml) = &cfg.pml else {
            return;
        };
        if !pml.enabled.unwrap_or(false) {
            return;
        }
        let width = pml.width_cells.unwrap_or(20).max(1);
        let strength = pml.strength.unwrap_or(0.15);
        for i in 0..self.nx {
            for j in 0..self.ny {
                let di = i.min(self.nx - 1 - i);
                let dj = j.min(self.ny - 1 - j);
                let mut sigma = 0.0_f64;
                if di < width {
                    let t = (width - di) as f64 / width as f64;
                    sigma += t * t;
                }
                if dj < width {
                    let t = (width - dj) as f64 / width as f64;
                    sigma += t * t;
                }
                let k = self.idx(i, j);
                self.pml_mask[k] = (-strength * sigma).exp();
            }
        }
    }

    fn spec_to_range_x(&self, spec: &CoordSpec) -> (usize, usize) {
        match spec {
            CoordSpec::Scalar(v) => {
                let i = self.to_index_x(*v);
                (i, i)
            }
            CoordSpec::Range(r) => {
                let mut i0 = self.to_index_x(r[0]);
                let mut i1 = self.to_index_x(r[1]);
                if i1 < i0 {
                    std::mem::swap(&mut i0, &mut i1);
                }
                (i0, i1)
            }
        }
    }

    fn spec_to_range_y(&self, spec: &CoordSpec) -> (usize, usize) {
        match spec {
            CoordSpec::Scalar(v) => {
                let j = self.to_index_y(*v);
                (j, j)
            }
            CoordSpec::Range(r) => {
                let mut j0 = self.to_index_y(r[0]);
                let mut j1 = self.to_index_y(r[1]);
                if j1 < j0 {
                    std::mem::swap(&mut j0, &mut j1);
                }
                (j0, j1)
            }
        }
    }

    fn prepare_source(&self, s: &SourceConfig, idx: usize) -> SourcePrepared {
        let kind = match s.kind.to_ascii_lowercase().as_str() {
            "point" => SourceKind::Point,
            "line-soft" => SourceKind::LineSoft,
            "sftf" => SourceKind::Sftf,
            "waveguide-x" => SourceKind::WaveguideX,
            "waveguide-y" => SourceKind::WaveguideY,
            _ => panic!("Unsupported source kind for Rust core: {}", s.kind),
        };

        let (x0, x1) = self.spec_to_range_x(&s.x);
        let (y0, y1) = self.spec_to_range_y(&s.y);

        let mut points = Vec::new();
        let mut along_x = false;
        let mut e_weights = Vec::new();
        let mut h_weights = Vec::new();
        let mut delays = Vec::new();

        match kind {
            SourceKind::Point => {
                points.push(self.idx(x0, y0));
                e_weights.push(1.0);
                h_weights.push(1.0);
                delays.push(0.0);
            }
            SourceKind::LineSoft | SourceKind::WaveguideX | SourceKind::WaveguideY => {
                if x0 != x1 && y0 == y1 {
                    along_x = true;
                    for i in x0..x1 {
                        points.push(self.idx(i, y0));
                        delays.push(0.0);
                    }
                } else if x0 == x1 && y0 != y1 {
                    along_x = false;
                    for j in y0..y1 {
                        points.push(self.idx(x0, j));
                        delays.push(0.0);
                    }
                } else {
                    panic!("line-soft source must have either x-range or y-range.");
                }
                if matches!(kind, SourceKind::WaveguideX | SourceKind::WaveguideY) {
                    let p_e = s.profile_e.as_ref().or(s.profile.as_ref());
                    let p_h = s.profile_h.as_ref().or(s.profile.as_ref());
                    if let (Some(pe), Some(ph)) = (p_e, p_h) {
                        if pe.len() == points.len() && ph.len() == points.len() {
                            e_weights = pe.clone();
                            h_weights = ph.clone();
                        }
                    }
                    if e_weights.is_empty() || h_weights.is_empty() {
                        let m = s.mode_index.unwrap_or(0) + 1;
                        let n = points.len().max(2);
                        for i in 0..points.len() {
                            let xi = i as f64 / (n as f64 - 1.0);
                            let w = (m as f64 * PI * xi).sin();
                            e_weights.push(w);
                            h_weights.push(w);
                        }
                    }
                } else {
                    e_weights.resize(points.len(), 1.0);
                    h_weights.resize(points.len(), 1.0);
                }
            }
            SourceKind::Sftf => {
                let angle = s.angle.unwrap_or(0.0);
                let ca = angle.cos();
                let sa = angle.sin();
                for i in x0..=x1 {
                    for &j in &[y0, y1] {
                        points.push(self.idx(i, j));
                        e_weights.push(1.0);
                        h_weights.push(1.0);
                        let x = i as f64 * self.dx;
                        let y = j as f64 * self.dy;
                        delays.push((x * ca + y * sa) / self.c0);
                    }
                }
                for j in y0..=y1 {
                    for &i in &[x0, x1] {
                        points.push(self.idx(i, j));
                        e_weights.push(1.0);
                        h_weights.push(1.0);
                        let x = i as f64 * self.dx;
                        let y = j as f64 * self.dy;
                        delays.push((x * ca + y * sa) / self.c0);
                    }
                }
            }
        }

        let amp = s.amplitude.unwrap_or(1.0);
        let _preview = s.is_show.unwrap_or(false);
        let fmax = s.f_max.or(s.f0).unwrap_or(self.f_max);
        let tw = s.tw.unwrap_or(0.5 / fmax.max(1e-30));
        let t0 = s.t0.unwrap_or(4.0 * tw);

        SourcePrepared {
            kind,
            points,
            along_x,
            e_weights,
            h_weights,
            delays,
            amp,
            t0,
            tw,
            ix0: x0,
            ix1: x1,
            iy0: y0,
            iy1: y1,
            // NOTE: config angle is in degrees for TF/SF sources.
            angle: s.angle.unwrap_or(0.0).to_radians(),
            n_eff: s.n_eff.unwrap_or(1.0),
            f_min: s.f_min,
            f_max: s.f_max.or(s.f0),
            name: s.name.clone().unwrap_or_else(|| format!("source_{idx}")),
        }
    }

    fn prepare_monitor(&self, m: &MonitorConfig, idx: usize) -> MonitorPrepared {
        let (x0, x1) = self.spec_to_range_x(&m.x);
        let (y0, y1) = self.spec_to_range_y(&m.y);

        let mut points = Vec::new();
        let mut x_m = Vec::new();
        let mut y_m = Vec::new();
        let orientation = if x0 != x1 && y0 == y1 {
            for i in x0..x1 {
                points.push(self.idx(i, y0));
                x_m.push(i as f64 * self.dx);
                y_m.push(y0 as f64 * self.dy);
            }
            "horizontal".to_string()
        } else if x0 == x1 && y0 != y1 {
            for j in y0..y1 {
                points.push(self.idx(x0, j));
                x_m.push(x0 as f64 * self.dx);
                y_m.push(j as f64 * self.dy);
            }
            "vertical".to_string()
        } else {
            panic!("Monitor must be horizontal or vertical line.");
        };

        MonitorPrepared {
            points,
            x_m,
            y_m,
            orientation,
            normal_sign: m.normal_sign.unwrap_or(1.0),
            name: m.name.clone().unwrap_or_else(|| format!("monitor_{idx}")),
        }
    }

    fn g(src: &SourcePrepared, t: f64, dt: f64) -> f64 {
        if src.f_min.is_none() {
            return src.amp * (-((t - src.t0) / src.tw).powi(2)).exp();
        }
        let fmin = src.f_min.unwrap();
        let fmax = src.f_max.unwrap_or(fmin);
        if (fmin - fmax).abs() <= 1e-12 * fmax.abs().max(1.0) {
            let tr = (1.0 / fmax.max(1e-30)).max(dt);
            let tau = (t - src.t0).max(0.0);
            let ramp = 1.0 - (-(tau / tr).powi(3)).exp();
            return src.amp * ramp * (2.0 * PI * fmax * (t - src.t0)).sin();
        }
        let f0 = 0.5 * (fmin + fmax);
        src.amp
            * (2.0 * PI * f0 * (t - src.t0)).sin()
            * (-((t - src.t0) / src.tw).powi(2)).exp()
    }

    fn init_scene(&mut self, cfg: &Config) {
        for obj in &cfg.objects {
            self.add_rectangle(obj);
        }

        self.sources = cfg
            .sources
            .iter()
            .enumerate()
            .map(|(i, s)| self.prepare_source(s, i))
            .collect();

        self.monitors = cfg
            .monitors
            .clone()
            .unwrap_or_default()
            .iter()
            .enumerate()
            .map(|(i, m)| self.prepare_monitor(m, i))
            .collect();

        self.monitor_buffers = self
            .monitors
            .iter()
            .map(|m| MonitorBuffers {
                ez: Vec::with_capacity(self.nt * m.points.len()),
                hx: Vec::with_capacity(self.nt * m.points.len()),
                hy: Vec::with_capacity(self.nt * m.points.len()),
                nline: m.points.len(),
            })
            .collect();

        self.init_pml(cfg);
    }

    fn run(&mut self) {
        let mcoef = self.c0 * self.dt;
        let mut hx_prev = vec![0.0_f64; self.nx * self.ny];
        let mut hy_prev = vec![0.0_f64; self.nx * self.ny];

        for t_idx in 0..self.nt {
            for i in 0..self.nx {
                for j in 0..self.ny {
                    let k = self.idx(i, j);
                    let ez_y_hi = if j + 1 < self.ny { self.ez[self.idx(i, j + 1)] } else { 0.0 };
                    let ez_x_hi = if i + 1 < self.nx { self.ez[self.idx(i + 1, j)] } else { 0.0 };

                    let d_ez_y = (ez_y_hi - self.ez[k]) / self.dy;
                    let d_ez_x = (ez_x_hi - self.ez[k]) / self.dx;

                    self.bx[k] -= mcoef * d_ez_y;
                    self.by[k] += mcoef * d_ez_x;
                }
            }

            let t = t_idx as f64 * self.dt;
            // E-side injection for advanced sources (matches Python curl-E placement).
            for s in &self.sources {
                match s.kind {
                    SourceKind::WaveguideY => {
                        let e_src = Self::g(s, t, self.dt);
                        for p in 0..s.points.len() {
                            let k = s.points[p];
                            let i = k / self.ny;
                            let j = k % self.ny;
                            if j > 0 {
                                let kk = self.idx(i, j - 1);
                                self.bx[kk] += mcoef * e_src * s.e_weights[p] / self.dy;
                            }
                        }
                    }
                    SourceKind::WaveguideX => {
                        let e_src = Self::g(s, t, self.dt);
                        for p in 0..s.points.len() {
                            let k = s.points[p];
                            let i = k / self.ny;
                            let j = k % self.ny;
                            if i > 0 {
                                let kk = self.idx(i - 1, j);
                                self.by[kk] -= mcoef * e_src * s.e_weights[p] / self.dx;
                            }
                        }
                    }
                    SourceKind::Sftf => {
                        let ix_lo = s.ix0.min(s.ix1);
                        let ix_hi = s.ix0.max(s.ix1);
                        let iy_lo = s.iy0.min(s.iy1);
                        let iy_hi = s.iy0.max(s.iy1);
                        let kx = s.angle.cos();
                        let ky = s.angle.sin();

                        if ix_lo > 0 {
                            for j in iy_lo..=iy_hi {
                                let delay = (kx * ix_lo as f64 * self.dx + ky * j as f64 * self.dy) / self.c0;
                                let ezs = Self::g(s, t - delay, self.dt);
                                let kk = self.idx(ix_lo - 1, j);
                                self.by[kk] -= mcoef * ezs / self.dx;
                            }
                        }
                        if ix_hi < self.nx {
                            for j in iy_lo..=iy_hi {
                                let delay = (kx * (ix_hi + 1) as f64 * self.dx + ky * j as f64 * self.dy) / self.c0;
                                let ezs = Self::g(s, t - delay, self.dt);
                                let kk = self.idx(ix_hi, j);
                                self.by[kk] += mcoef * ezs / self.dx;
                            }
                        }
                        if iy_lo > 0 {
                            for i in ix_lo..=ix_hi {
                                let delay = (kx * i as f64 * self.dx + ky * iy_lo as f64 * self.dy) / self.c0;
                                let ezs = Self::g(s, t - delay, self.dt);
                                let kk = self.idx(i, iy_lo - 1);
                                self.bx[kk] += mcoef * ezs / self.dy;
                            }
                        }
                        if iy_hi < self.ny {
                            for i in ix_lo..=ix_hi {
                                let delay = (kx * i as f64 * self.dx + ky * (iy_hi + 1) as f64 * self.dy) / self.c0;
                                let ezs = Self::g(s, t - delay, self.dt);
                                let kk = self.idx(i, iy_hi);
                                self.bx[kk] -= mcoef * ezs / self.dy;
                            }
                        }
                    }
                    _ => {}
                }
            }

            for i in 0..self.nx {
                for j in 0..self.ny {
                    let k = self.idx(i, j);
                    self.hx[k] = self.bx[k] / self.mrxx[k];
                    self.hy[k] = self.by[k] / self.mryy[k];
                }
            }

            for i in 0..self.nx {
                for j in 0..self.ny {
                    let k = self.idx(i, j);
                    let hx_lo = if j > 0 { self.hx[self.idx(i, j - 1)] } else { 0.0 };
                    let hy_lo = if i > 0 { self.hy[self.idx(i - 1, j)] } else { 0.0 };
                    let d_hx_y = (self.hx[k] - hx_lo) / self.dy;
                    let d_hy_x = (self.hy[k] - hy_lo) / self.dx;
                    self.dzf[k] += mcoef * (d_hy_x - d_hx_y);
                }
            }

            let t_half = t + 0.5 * self.dt;
            // H-side injection for advanced sources (matches Python curl-H placement).
            for s in &self.sources {
                match s.kind {
                    SourceKind::WaveguideY => {
                        let h_src = -Self::g(s, t + self.dy * s.n_eff / (2.0 * self.c0) + 0.5 * self.dt, self.dt);
                        for p in 0..s.points.len() {
                            let k = s.points[p];
                            let i = k / self.ny;
                            let j = k % self.ny;
                            let kk = self.idx(i, j);
                            self.dzf[kk] += mcoef * h_src * s.h_weights[p] / self.dy;
                        }
                    }
                    SourceKind::WaveguideX => {
                        let h_src = -Self::g(s, t + self.dx * s.n_eff / (2.0 * self.c0) + 0.5 * self.dt, self.dt);
                        for p in 0..s.points.len() {
                            let k = s.points[p];
                            let i = k / self.ny;
                            let j = k % self.ny;
                            let kk = self.idx(i, j);
                            self.dzf[kk] += mcoef * h_src * s.h_weights[p] / self.dx;
                        }
                    }
                    SourceKind::Sftf => {
                        let ix_lo = s.ix0.min(s.ix1);
                        let ix_hi = s.ix0.max(s.ix1);
                        let iy_lo = s.iy0.min(s.iy1);
                        let iy_hi = s.iy0.max(s.iy1);
                        let kx = s.angle.cos();
                        let ky = s.angle.sin();
                        if ix_lo < self.nx {
                            for j in iy_lo..=iy_hi {
                                let delay = (kx * (ix_lo as f64 - 0.5) * self.dx + ky * j as f64 * self.dy) / self.c0;
                                let hys = -kx * Self::g(s, t_half - delay, self.dt);
                                let kk = self.idx(ix_lo, j);
                                self.dzf[kk] += mcoef * (-hys / self.dx);
                            }
                        }
                        if ix_hi + 1 < self.nx {
                            for j in iy_lo..=iy_hi {
                                let delay = (kx * (ix_hi as f64 + 0.5) * self.dx + ky * j as f64 * self.dy) / self.c0;
                                let hys = -kx * Self::g(s, t_half - delay, self.dt);
                                let kk = self.idx(ix_hi + 1, j);
                                self.dzf[kk] += mcoef * (hys / self.dx);
                            }
                        }
                        if iy_lo < self.ny {
                            for i in ix_lo..=ix_hi {
                                let delay = (kx * i as f64 * self.dx + ky * (iy_lo as f64 - 0.5) * self.dy) / self.c0;
                                let hxs = ky * Self::g(s, t_half - delay, self.dt);
                                let kk = self.idx(i, iy_lo);
                                self.dzf[kk] += mcoef * (hxs / self.dy);
                            }
                        }
                        if iy_hi + 1 < self.ny {
                            for i in ix_lo..=ix_hi {
                                let delay = (kx * i as f64 * self.dx + ky * (iy_hi as f64 + 0.5) * self.dy) / self.c0;
                                let hxs = ky * Self::g(s, t_half - delay, self.dt);
                                let kk = self.idx(i, iy_hi + 1);
                                self.dzf[kk] += mcoef * (-hxs / self.dy);
                            }
                        }
                    }
                    _ => {}
                }
            }

            // Soft source injection into D (point/line-soft).
            for s in &self.sources {
                match s.kind {
                    SourceKind::Point | SourceKind::LineSoft => {
                        for p in 0..s.points.len() {
                            let val = Self::g(s, t - s.delays[p], self.dt) * s.e_weights[p];
                            self.dzf[s.points[p]] += val;
                        }
                    }
                    _ => {}
                }
            }

            for i in 0..self.nx {
                for j in 0..self.ny {
                    let k = self.idx(i, j);
                    self.ez[k] = self.dzf[k] / self.erzz[k];
                }
            }

            for k in 0..self.pml_mask.len() {
                let d = self.pml_mask[k];
                self.bx[k] *= d;
                self.by[k] *= d;
                self.hx[k] *= d;
                self.hy[k] *= d;
                self.dzf[k] *= d;
                self.ez[k] *= d;
            }

            let mut hx_half = vec![0.0_f64; self.nx * self.ny];
            let mut hy_half = vec![0.0_f64; self.nx * self.ny];
            for k in 0..hx_half.len() {
                hx_half[k] = 0.5 * (self.hx[k] + hx_prev[k]);
                hy_half[k] = 0.5 * (self.hy[k] + hy_prev[k]);
            }
            let hx_center = Self::avg_with_neighbor(&hx_half, self.nx, self.ny, 1, -1);
            let hy_center = Self::avg_with_neighbor(&hy_half, self.nx, self.ny, 0, -1);

            for (mi, m) in self.monitors.iter().enumerate() {
                let buf = &mut self.monitor_buffers[mi];
                for &k in &m.points {
                    buf.ez.push(self.ez[k]);
                    buf.hx.push(hx_center[k]);
                    buf.hy.push(hy_center[k]);
                }
            }

            hx_prev.clone_from_slice(&self.hx);
            hy_prev.clone_from_slice(&self.hy);

            if t_idx % self.record_stride == 0 {
                for i in 0..self.nx {
                    for j in 0..self.ny {
                        let k = self.idx(i, j);
                        self.hx_hist.push(self.hx[k] as f32);
                        self.hy_hist.push(self.hy[k] as f32);
                        self.ez_hist.push(self.ez[k] as f32);
                    }
                }
            }

            if t_idx % 200 == 0 || t_idx + 1 == self.nt {
                println!("step {}/{}", t_idx + 1, self.nt);
            }
        }
    }

    fn rfft_real(&self, x: &[f64]) -> Vec<Complex64> {
        let n = x.len();
        let mut planner = FftPlanner::<f64>::new();
        let fft = planner.plan_fft_forward(n);
        let mut data = x.iter().map(|&v| Complex64::new(v, 0.0)).collect::<Vec<_>>();
        fft.process(&mut data);
        let scale = 1.0 / n as f64;
        let nfreq = n / 2 + 1;
        data[..nfreq].iter().map(|v| *v * scale).collect()
    }

    fn hanning(n: usize) -> Vec<f64> {
        if n <= 1 {
            return vec![1.0; n];
        }
        (0..n)
            .map(|i| 0.5 - 0.5 * (2.0 * PI * i as f64 / (n as f64 - 1.0)).cos())
            .collect()
    }

    fn compute_source_fft(&self, out: &Path) -> Result<(), Box<dyn Error>> {
        let n = self.nt;
        let freq_step = 1.0 / (n as f64 * self.dt);
        let nfreq = n / 2 + 1;

        for (si, s) in self.sources.iter().enumerate() {
            let mut g = vec![0.0_f64; n];
            for t_idx in 0..n {
                g[t_idx] = Self::g(s, t_idx as f64 * self.dt, self.dt);
            }
            let mean = g.iter().sum::<f64>() / n as f64;
            for v in &mut g {
                *v -= mean;
            }
            let spec = self.rfft_real(&g);

            let pscale = 0.5 / self.eta0;
            let geom = match s.kind {
                SourceKind::Point => pscale * self.dx * self.dy,
                SourceKind::LineSoft | SourceKind::WaveguideX | SourceKind::WaveguideY => {
                    if s.along_x {
                        pscale * s.points.len() as f64 * self.dx
                    } else {
                        pscale * s.points.len() as f64 * self.dy
                    }
                }
                SourceKind::Sftf => pscale * s.points.len() as f64 * 0.5 * (self.dx + self.dy),
            };

            let mut f = BufWriter::new(File::create(out.join(format!("source_fft_{si}.csv")))?);
            writeln!(f, "freq_hz,power,waveform_power,real,imag")?;
            for k in 0..nfreq {
                let wf = spec[k].norm_sqr();
                let p = wf * geom;
                writeln!(
                    f,
                    "{:.12e},{:.12e},{:.12e},{:.12e},{:.12e}",
                    k as f64 * freq_step,
                    p,
                    wf,
                    spec[k].re,
                    spec[k].im
                )?;
            }
        }
        Ok(())
    }

    fn compute_monitor_fft(&self, out: &Path) -> Result<(), Box<dyn Error>> {
        let n = self.nt;
        let freq_step = 1.0 / (n as f64 * self.dt);
        let nfreq = n / 2 + 1;

        let w = vec![1.0_f64; n];
        let _ = Self::hanning(8);

        for (mi, m) in self.monitors.iter().enumerate() {
            let buf = &self.monitor_buffers[mi];
            let l = buf.nline;

            let mut ez_f = vec![vec![Complex64::new(0.0, 0.0); l]; nfreq];
            let mut hx_f = vec![vec![Complex64::new(0.0, 0.0); l]; nfreq];
            let mut hy_f = vec![vec![Complex64::new(0.0, 0.0); l]; nfreq];

            for p in 0..l {
                let mut ez_col = vec![0.0_f64; n];
                let mut hx_col = vec![0.0_f64; n];
                let mut hy_col = vec![0.0_f64; n];
                for t in 0..n {
                    ez_col[t] = buf.ez[t * l + p];
                    hx_col[t] = buf.hx[t * l + p];
                    hy_col[t] = buf.hy[t * l + p];
                }

                let me = ez_col.iter().sum::<f64>() / n as f64;
                let mhx = hx_col.iter().sum::<f64>() / n as f64;
                let mhy = hy_col.iter().sum::<f64>() / n as f64;

                for t in 0..n {
                    ez_col[t] = (ez_col[t] - me) * w[t];
                    hx_col[t] = (hx_col[t] - mhx) * w[t];
                    hy_col[t] = (hy_col[t] - mhy) * w[t];
                }

                let ezs = self.rfft_real(&ez_col);
                let hxs = self.rfft_real(&hx_col);
                let hys = self.rfft_real(&hy_col);

                for k in 0..nfreq {
                    ez_f[k][p] = ezs[k];
                    hx_f[k][p] = hxs[k];
                    hy_f[k][p] = hys[k];
                }
            }

            let d_l = if m.orientation == "horizontal" { self.dx } else { self.dy };
            let mut fout = BufWriter::new(File::create(out.join(format!("monitor_fft_{mi}.csv")))?);
            writeln!(fout, "freq_hz,power,complex_real,complex_imag")?;

            for k in 0..nfreq {
                let mut comp = Complex64::new(0.0, 0.0);
                for p in 0..l {
                    let pd = if m.orientation == "horizontal" {
                        (m.normal_sign * (0.5 / self.eta0)) * ez_f[k][p] * hx_f[k][p].conj()
                    } else {
                        (m.normal_sign * (-0.5 / self.eta0)) * ez_f[k][p] * hy_f[k][p].conj()
                    };
                    comp += pd;
                }
                comp *= d_l;
                writeln!(
                    fout,
                    "{:.12e},{:.12e},{:.12e},{:.12e}",
                    k as f64 * freq_step,
                    comp.re,
                    comp.re,
                    comp.im
                )?;
            }
        }
        Ok(())
    }

    fn phasor_from_series(signal: &[f64], ntime: usize, nline: usize, freqs: &[f64], dt: f64) -> Vec<Vec<Complex64>> {
        let mut out = vec![vec![Complex64::new(0.0, 0.0); nline]; freqs.len()];
        for (fi, &f) in freqs.iter().enumerate() {
            for t in 0..ntime {
                let ph = Complex64::from_polar(1.0, -2.0 * PI * f * (t as f64 * dt)) * dt;
                let base = t * nline;
                for l in 0..nline {
                    out[fi][l] += ph * signal[base + l];
                }
            }
        }
        out
    }

    fn compute_nf2ff(&self, out: &Path, nf: &Nf2ffConfig) -> Result<(), Box<dyn Error>> {
        let sides = [nf.top, nf.bottom, nf.left, nf.right];
        if sides.iter().all(|s| s.is_none()) || self.monitors.is_empty() {
            return Ok(());
        }

        let nphi = nf.nphi.unwrap_or(361).max(16);
        let nfreq = nf.freq_count.unwrap_or(10).max(1);
        let f0 = self.f_min.unwrap_or(0.0);
        let f1 = self.f_max.max(f0 + 1.0);
        let freqs: Vec<f64> = (0..nfreq)
            .map(|i| f0 + (f1 - f0) * i as f64 / (nfreq.saturating_sub(1).max(1) as f64))
            .collect();
        let phi: Vec<f64> = (0..nphi).map(|i| 2.0 * PI * i as f64 / nphi as f64).collect();

        let mut side_ez: [Option<Vec<Vec<Complex64>>>; 4] = [None, None, None, None];
        let mut side_hx: [Option<Vec<Vec<Complex64>>>; 4] = [None, None, None, None];
        let mut side_hy: [Option<Vec<Vec<Complex64>>>; 4] = [None, None, None, None];

        for (si, m_idx_opt) in sides.iter().enumerate() {
            if let Some(m_idx) = *m_idx_opt {
                if m_idx < self.monitor_buffers.len() {
                    let mb = &self.monitor_buffers[m_idx];
                    let mut ezp = Self::phasor_from_series(&mb.ez, self.nt, mb.nline, &freqs, self.dt);
                    for row in &mut ezp {
                        for v in row {
                            *v *= self.eta0;
                        }
                    }
                    side_ez[si] = Some(ezp);
                    side_hx[si] = Some(Self::phasor_from_series(&mb.hx, self.nt, mb.nline, &freqs, self.dt));
                    side_hy[si] = Some(Self::phasor_from_series(&mb.hy, self.nt, mb.nline, &freqs, self.dt));
                }
            }
        }

        let mut gsrc = vec![Complex64::new(1.0, 0.0); nfreq];
        if let Some(src_i) = nf.source_index {
            if src_i < self.sources.len() {
                let s = &self.sources[src_i];
                for (fi, &f) in freqs.iter().enumerate() {
                    let mut acc = Complex64::new(0.0, 0.0);
                    for t in 0..self.nt {
                        let gt = Self::g(s, t as f64 * self.dt, self.dt);
                        acc += Complex64::from_polar(1.0, -2.0 * PI * f * (t as f64 * self.dt)) * (gt * self.dt);
                    }
                    gsrc[fi] = acc;
                }
            }
        }

        let mut f = BufWriter::new(File::create(out.join("nf2ff.csv"))?);
        writeln!(f, "freq_hz,phi_rad,e_theta_re,e_theta_im,h_phi_re,h_phi_im,p_theta_re")?;

        for (fi, &fr) in freqs.iter().enumerate() {
            let k0 = 2.0 * PI * fr / self.c0;
            let gmax = gsrc
                .iter()
                .map(|v| v.norm())
                .fold(0.0_f64, |a, b| if a > b { a } else { b });
            for &ph in &phi {
                let cph = ph.cos();
                let sph = ph.sin();
                let mut n_theta = Complex64::new(0.0, 0.0);
                let mut l_phi = Complex64::new(0.0, 0.0);

                if let Some(mi) = nf.bottom {
                    if mi < self.monitors.len() && side_hx[1].is_some() {
                        let mon = &self.monitors[mi];
                        let hx = side_hx[1].as_ref().unwrap();
                        let ez = side_ez[1].as_ref().unwrap();
                        for l in 0..mon.points.len() {
                            let phase = Complex64::from_polar(1.0, k0 * (mon.x_m[l] * cph + mon.y_m[l] * sph));
                            n_theta -= hx[fi][l] * phase * self.dx;
                            l_phi -= Complex64::new(sph, 0.0) * ez[fi][l] * phase * self.dx;
                        }
                    }
                }
                if let Some(mi) = nf.top {
                    if mi < self.monitors.len() && side_hx[0].is_some() {
                        let mon = &self.monitors[mi];
                        let hx = side_hx[0].as_ref().unwrap();
                        let ez = side_ez[0].as_ref().unwrap();
                        for l in 0..mon.points.len() {
                            let phase = Complex64::from_polar(1.0, k0 * (mon.x_m[l] * cph + mon.y_m[l] * sph));
                            n_theta += hx[fi][l] * phase * self.dx;
                            l_phi += Complex64::new(sph, 0.0) * ez[fi][l] * phase * self.dx;
                        }
                    }
                }
                if let Some(mi) = nf.right {
                    if mi < self.monitors.len() && side_hy[3].is_some() {
                        let mon = &self.monitors[mi];
                        let hy = side_hy[3].as_ref().unwrap();
                        let ez = side_ez[3].as_ref().unwrap();
                        for l in 0..mon.points.len() {
                            let phase = Complex64::from_polar(1.0, k0 * (mon.x_m[l] * cph + mon.y_m[l] * sph));
                            n_theta -= hy[fi][l] * phase * self.dy;
                            l_phi += Complex64::new(cph, 0.0) * ez[fi][l] * phase * self.dy;
                        }
                    }
                }
                if let Some(mi) = nf.left {
                    if mi < self.monitors.len() && side_hy[2].is_some() {
                        let mon = &self.monitors[mi];
                        let hy = side_hy[2].as_ref().unwrap();
                        let ez = side_ez[2].as_ref().unwrap();
                        for l in 0..mon.points.len() {
                            let phase = Complex64::from_polar(1.0, k0 * (mon.x_m[l] * cph + mon.y_m[l] * sph));
                            n_theta += hy[fi][l] * phase * self.dy;
                            l_phi -= Complex64::new(cph, 0.0) * ez[fi][l] * phase * self.dy;
                        }
                    }
                }

                let mut et = Complex64::new(k0, 0.0) * (self.eta0 * n_theta + l_phi);
                let mut hp = Complex64::new(k0, 0.0) * (l_phi / self.eta0 + n_theta);
                let gmag = gsrc[fi].norm();
                let good = gmag >= 1e-6 * gmax && gmax > 0.0;
                if good {
                    et /= gsrc[fi];
                    hp /= gsrc[fi];
                } else {
                    et = Complex64::new(0.0, 0.0);
                    hp = Complex64::new(0.0, 0.0);
                }
                let pt = 0.5 * (et * hp).re;
                writeln!(f, "{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e}", fr, ph, et.re, et.im, hp.re, hp.im, pt)?;
            }
        }
        Ok(())
    }

    fn write_outputs<P: AsRef<Path>>(&self, out_dir: P, cfg: &Config) -> Result<(), Box<dyn Error>> {
        let out = out_dir.as_ref();
        fs::create_dir_all(out)?;

        write_f32_bin(out.join("hx_history.bin"), &self.hx_hist)?;
        write_f32_bin(out.join("hy_history.bin"), &self.hy_hist)?;
        write_f32_bin(out.join("ez_history.bin"), &self.ez_hist)?;
        write_f64_bin(out.join("erzz.bin"), &self.erzz)?;

        let pp = cfg.postprocessing.clone().unwrap_or(PostProcessingConfig {
            compute_source_fft: Some(true),
            compute_monitor_fft: Some(true),
            nf2ff: None,
        });
        if pp.compute_source_fft.unwrap_or(true) {
            self.compute_source_fft(out)?;
        }
        if pp.compute_monitor_fft.unwrap_or(true) {
            self.compute_monitor_fft(out)?;
        }
        if let Some(nf) = pp.nf2ff {
            if nf.enabled.unwrap_or(false) {
                self.compute_nf2ff(out, &nf)?;
            }
        }

        for (mi, mb) in self.monitor_buffers.iter().enumerate() {
            write_f64_bin(out.join(format!("monitor_time_ez_{mi}.bin")), &mb.ez)?;
            write_f64_bin(out.join(format!("monitor_time_hx_{mi}.bin")), &mb.hx)?;
            write_f64_bin(out.join(format!("monitor_time_hy_{mi}.bin")), &mb.hy)?;
        }

        let mut meta = BufWriter::new(File::create(out.join("metadata.txt"))?);
        writeln!(meta, "nx={}", self.nx)?;
        writeln!(meta, "ny={}", self.ny)?;
        writeln!(meta, "nt={}", self.nt)?;
        writeln!(meta, "nt_rec={}", self.nt_rec)?;
        writeln!(meta, "record_stride={}", self.record_stride)?;
        writeln!(meta, "x_range_m={}", self.x_range)?;
        writeln!(meta, "y_range_m={}", self.y_range)?;
        writeln!(meta, "dt_s={}", self.dt)?;
        writeln!(meta, "f_min_hz={}", self.f_min.unwrap_or(0.0))?;
        writeln!(meta, "f_max_hz={}", self.f_max)?;
        writeln!(meta, "eps0={}", self.eps0)?;
        writeln!(meta, "mu0={}", self.mu0)?;
        writeln!(meta, "num_sources={}", self.sources.len())?;
        writeln!(meta, "num_monitors={}", self.monitors.len())?;
        for (i, m) in self.monitors.iter().enumerate() {
            writeln!(meta, "monitor_{}_nline={}", i, m.points.len())?;
        }
        if let Some(plot) = &cfg.plot {
            writeln!(meta, "plot_fps={}", plot.fps.unwrap_or(60))?;
            writeln!(meta, "dynamic_clim={}", plot.dynamic_clim.unwrap_or(true))?;
            writeln!(meta, "clim_smooth={}", plot.clim_smooth.unwrap_or(0.25))?;
            writeln!(meta, "plot_show_source_profiles={}", plot.show_source_profiles.unwrap_or(true))?;
            writeln!(meta, "plot_show_animation={}", plot.show_animation.unwrap_or(true))?;
            writeln!(meta, "plot_show_fft={}", plot.show_fft.unwrap_or(true))?;
            writeln!(meta, "plot_show_nf2ff={}", plot.show_nf2ff.unwrap_or(false))?;
        }

        let mut idx = BufWriter::new(File::create(out.join("fft_index.txt"))?);
        for (i, s) in self.sources.iter().enumerate() {
            writeln!(idx, "source,{i},{}", s.name)?;
        }
        for (i, m) in self.monitors.iter().enumerate() {
            writeln!(idx, "monitor,{i},{}", m.name)?;
        }

        Ok(())
    }
}

fn write_f32_bin<P: AsRef<Path>>(path: P, data: &[f32]) -> Result<(), Box<dyn Error>> {
    let mut f = BufWriter::new(File::create(path)?);
    for &v in data {
        f.write_all(&v.to_le_bytes())?;
    }
    Ok(())
}

fn write_f64_bin<P: AsRef<Path>>(path: P, data: &[f64]) -> Result<(), Box<dyn Error>> {
    let mut f = BufWriter::new(File::create(path)?);
    for &v in data {
        f.write_all(&v.to_le_bytes())?;
    }
    Ok(())
}

fn parse_args() -> PathBuf {
    let mut args = env::args().skip(1);
    let mut config_path = PathBuf::from("config/example_1_simple_source.json");
    while let Some(arg) = args.next() {
        if arg == "--config" {
            if let Some(v) = args.next() {
                config_path = PathBuf::from(v);
            }
        }
    }
    config_path
}

fn main() -> Result<(), Box<dyn Error>> {
    let config_path = parse_args();
    let cfg_txt = fs::read_to_string(config_path)?;
    let cfg: Config = serde_json::from_str(&cfg_txt)?;

    let mut sim = Sim2DEz::new(&cfg);
    sim.init_scene(&cfg);
    sim.run();

    let out = cfg.output_dir.clone().unwrap_or_else(|| "output".to_string());
    sim.write_outputs(&out, &cfg)?;

    println!("Done. Outputs written to ./{out}");
    Ok(())
}







