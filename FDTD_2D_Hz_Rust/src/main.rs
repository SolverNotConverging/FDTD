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
    hz: Vec<f64>,
    ex: Vec<f64>,
    ey: Vec<f64>,
    nline: usize,
}

struct Sim2DHz {
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

    erxx: Vec<f64>,
    eryy: Vec<f64>,
    mrzz: Vec<f64>,

    dxf: Vec<f64>,
    dyf: Vec<f64>,
    ex: Vec<f64>,
    ey: Vec<f64>,
    bz: Vec<f64>,
    hz: Vec<f64>,

    sources: Vec<SourcePrepared>,
    monitors: Vec<MonitorPrepared>,
    monitor_buffers: Vec<MonitorBuffers>,

    ex_hist: Vec<f32>,
    ey_hist: Vec<f32>,
    hz_hist: Vec<f32>,
    pml_mask: Vec<f64>,
}

impl Sim2DHz {
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

            erxx: vec![1.0; n],
            eryy: vec![1.0; n],
            mrzz: vec![1.0; n],

            dxf: vec![0.0; n],
            dyf: vec![0.0; n],
            ex: vec![0.0; n],
            ey: vec![0.0; n],
            bz: vec![0.0; n],
            hz: vec![0.0; n],

            sources: Vec::new(),
            monitors: Vec::new(),
            monitor_buffers: Vec::new(),

            ex_hist: Vec::with_capacity(nt_rec * n),
            ey_hist: Vec::with_capacity(nt_rec * n),
            hz_hist: Vec::with_capacity(nt_rec * n),
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
                self.erxx[k] = obj.er;
                self.eryy[k] = obj.er;
                self.mrzz[k] = obj.mr;
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
                hz: Vec::with_capacity(self.nt * m.points.len()),
                ex: Vec::with_capacity(self.nt * m.points.len()),
                ey: Vec::with_capacity(self.nt * m.points.len()),
                nline: m.points.len(),
            })
            .collect();

        self.init_pml(cfg);
    }

    fn run(&mut self) {
        let mcoef = self.c0 * self.dt;
        let mut hz_prev = vec![0.0_f64; self.nx * self.ny];
        for t_idx in 0..self.nt {
            for i in 0..self.nx {
                for j in 0..self.ny {
                    let k = self.idx(i, j);
                    let ex_y_hi = if j + 1 < self.ny { self.ex[self.idx(i, j + 1)] } else { 0.0 };
                    let ey_x_hi = if i + 1 < self.nx { self.ey[self.idx(i + 1, j)] } else { 0.0 };
                    let d_ex_y = (ex_y_hi - self.ex[k]) / self.dy;
                    let d_ey_x = (ey_x_hi - self.ey[k]) / self.dx;
                    self.bz[k] -= mcoef * (d_ey_x - d_ex_y);
                }
            }

            let t = t_idx as f64 * self.dt;
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
                                self.bz[kk] -= mcoef * e_src * s.e_weights[p] / self.dy;
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
                                self.bz[kk] += mcoef * e_src * s.e_weights[p] / self.dx;
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
                                let delay = (kx * (ix_lo as f64 + 0.5) * self.dx + ky * j as f64 * self.dy) / self.c0;
                                let ey_inc = kx * Self::g(s, t - delay, self.dt);
                                let kk = self.idx(ix_lo - 1, j);
                                self.bz[kk] += mcoef * ey_inc / self.dx;
                            }
                        }
                        if ix_hi < self.nx {
                            for j in iy_lo..=iy_hi {
                                let delay = (kx * (ix_hi as f64 + 1.5) * self.dx + ky * j as f64 * self.dy) / self.c0;
                                let ey_inc = kx * Self::g(s, t - delay, self.dt);
                                let kk = self.idx(ix_hi, j);
                                self.bz[kk] -= mcoef * ey_inc / self.dx;
                            }
                        }
                        if iy_lo > 0 {
                            for i in ix_lo..=ix_hi {
                                let delay = (kx * i as f64 * self.dx + ky * (iy_lo as f64 + 0.5) * self.dy) / self.c0;
                                let ex_inc = -ky * Self::g(s, t - delay, self.dt);
                                let kk = self.idx(i, iy_lo - 1);
                                self.bz[kk] += mcoef * ex_inc / self.dy;
                            }
                        }
                        if iy_hi < self.ny {
                            for i in ix_lo..=ix_hi {
                                let delay = (kx * i as f64 * self.dx + ky * (iy_hi as f64 + 1.5) * self.dy) / self.c0;
                                let ex_inc = -ky * Self::g(s, t - delay, self.dt);
                                let kk = self.idx(i, iy_hi);
                                self.bz[kk] -= mcoef * ex_inc / self.dy;
                            }
                        }
                    }
                    SourceKind::Point | SourceKind::LineSoft => {
                        for p in 0..s.points.len() {
                            let val = Self::g(s, t - s.delays[p], self.dt) * s.e_weights[p];
                            self.bz[s.points[p]] += val;
                        }
                    }
                }
            }

            for i in 0..self.nx {
                for j in 0..self.ny {
                    let k = self.idx(i, j);
                    self.hz[k] = self.bz[k] / self.mrzz[k];
                }
            }

            for i in 0..self.nx {
                for j in 0..self.ny {
                    let k = self.idx(i, j);
                    let hz_lo_y = if j > 0 { self.hz[self.idx(i, j - 1)] } else { 0.0 };
                    let hz_lo_x = if i > 0 { self.hz[self.idx(i - 1, j)] } else { 0.0 };

                    let d_hz_y = (self.hz[k] - hz_lo_y) / self.dy;
                    let d_hz_x = (self.hz[k] - hz_lo_x) / self.dx;
                    self.dxf[k] += mcoef * d_hz_y;
                    self.dyf[k] -= mcoef * d_hz_x;
                }
            }

            let t_half = t + 0.5 * self.dt;
            for s in &self.sources {
                match s.kind {
                    SourceKind::WaveguideY => {
                        let h_src = -Self::g(s, t + self.dy * s.n_eff / (2.0 * self.c0) + 0.5 * self.dt, self.dt);
                        for p in 0..s.points.len() {
                            let k = s.points[p];
                            let i = k / self.ny;
                            let j = k % self.ny;
                            let kk = self.idx(i, j);
                            self.dxf[kk] += mcoef * h_src * s.h_weights[p] / self.dy;
                        }
                    }
                    SourceKind::WaveguideX => {
                        let h_src = -Self::g(s, t + self.dx * s.n_eff / (2.0 * self.c0) + 0.5 * self.dt, self.dt);
                        for p in 0..s.points.len() {
                            let k = s.points[p];
                            let i = k / self.ny;
                            let j = k % self.ny;
                            let kk = self.idx(i, j);
                            self.dyf[kk] += mcoef * h_src * s.h_weights[p] / self.dx;
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
                                let delay = (kx * ix_lo as f64 * self.dx + ky * j as f64 * self.dy) / self.c0;
                                let hz_inc = Self::g(s, t_half - delay, self.dt);
                                let kk = self.idx(ix_lo, j);
                                self.dyf[kk] += mcoef * hz_inc / self.dx;
                            }
                        }
                        if ix_hi + 1 < self.nx {
                            for j in iy_lo..=iy_hi {
                                let delay = (kx * (ix_hi as f64 + 1.0) * self.dx + ky * j as f64 * self.dy) / self.c0;
                                let hz_inc = Self::g(s, t_half - delay, self.dt);
                                let kk = self.idx(ix_hi + 1, j);
                                self.dyf[kk] -= mcoef * hz_inc / self.dx;
                            }
                        }
                        if iy_lo < self.ny {
                            for i in ix_lo..=ix_hi {
                                let delay = (kx * i as f64 * self.dx + ky * iy_lo as f64 * self.dy) / self.c0;
                                let hz_inc = Self::g(s, t_half - delay, self.dt);
                                let kk = self.idx(i, iy_lo);
                                self.dxf[kk] += mcoef * hz_inc / self.dy;
                            }
                        }
                        if iy_hi + 1 < self.ny {
                            for i in ix_lo..=ix_hi {
                                let delay = (kx * i as f64 * self.dx + ky * (iy_hi as f64 + 1.0) * self.dy) / self.c0;
                                let hz_inc = Self::g(s, t_half - delay, self.dt);
                                let kk = self.idx(i, iy_hi + 1);
                                self.dxf[kk] -= mcoef * hz_inc / self.dy;
                            }
                        }
                    }
                    _ => {}
                }
            }

            for i in 0..self.nx {
                for j in 0..self.ny {
                    let k = self.idx(i, j);
                    self.ex[k] = self.dxf[k] / self.erxx[k];
                    self.ey[k] = self.dyf[k] / self.eryy[k];
                }
            }

            for k in 0..self.pml_mask.len() {
                let d = self.pml_mask[k];
                self.bz[k] *= d;
                self.hz[k] *= d;
                self.dxf[k] *= d;
                self.dyf[k] *= d;
                self.ex[k] *= d;
                self.ey[k] *= d;
            }

            let mut hz_center = vec![0.0_f64; self.nx * self.ny];
            for k in 0..hz_center.len() {
                hz_center[k] = 0.5 * (self.hz[k] + hz_prev[k]);
            }
            let ex_center = Self::avg_with_neighbor(&self.ex, self.nx, self.ny, 1, 1);
            let ey_center = Self::avg_with_neighbor(&self.ey, self.nx, self.ny, 0, 1);

            for (mi, m) in self.monitors.iter().enumerate() {
                let buf = &mut self.monitor_buffers[mi];
                for &k in &m.points {
                    buf.hz.push(hz_center[k]);
                    buf.ex.push(ex_center[k]);
                    buf.ey.push(ey_center[k]);
                }
            }

            hz_prev.clone_from_slice(&self.hz);

            if t_idx % self.record_stride == 0 {
                for i in 0..self.nx {
                    for j in 0..self.ny {
                        let k = self.idx(i, j);
                        self.ex_hist.push(self.ex[k] as f32);
                        self.ey_hist.push(self.ey[k] as f32);
                        self.hz_hist.push(self.hz[k] as f32);
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

        for (mi, m) in self.monitors.iter().enumerate() {
            let buf = &self.monitor_buffers[mi];
            let l = buf.nline;

            let mut hz_f = vec![vec![Complex64::new(0.0, 0.0); l]; nfreq];
            let mut ex_f = vec![vec![Complex64::new(0.0, 0.0); l]; nfreq];
            let mut ey_f = vec![vec![Complex64::new(0.0, 0.0); l]; nfreq];

            for p in 0..l {
                let mut hz_col = vec![0.0_f64; n];
                let mut ex_col = vec![0.0_f64; n];
                let mut ey_col = vec![0.0_f64; n];
                for t in 0..n {
                    hz_col[t] = buf.hz[t * l + p];
                    ex_col[t] = buf.ex[t * l + p];
                    ey_col[t] = buf.ey[t * l + p];
                }

                let mhz = hz_col.iter().sum::<f64>() / n as f64;
                let mex = ex_col.iter().sum::<f64>() / n as f64;
                let mey = ey_col.iter().sum::<f64>() / n as f64;

                for t in 0..n {
                    hz_col[t] -= mhz;
                    ex_col[t] -= mex;
                    ey_col[t] -= mey;
                }

                let hzs = self.rfft_real(&hz_col);
                let exs = self.rfft_real(&ex_col);
                let eys = self.rfft_real(&ey_col);

                for k in 0..nfreq {
                    hz_f[k][p] = hzs[k];
                    ex_f[k][p] = exs[k];
                    ey_f[k][p] = eys[k];
                }
            }

            let d_l = if m.orientation == "horizontal" { self.dx } else { self.dy };
            let mut fout = BufWriter::new(File::create(out.join(format!("monitor_fft_{mi}.csv")))?);
            writeln!(fout, "freq_hz,power,complex_real,complex_imag")?;

            for k in 0..nfreq {
                let mut comp = Complex64::new(0.0, 0.0);
                for p in 0..l {
                    let pd = if m.orientation == "horizontal" {
                        (m.normal_sign * (-0.5 / self.eta0)) * ex_f[k][p] * hz_f[k][p].conj()
                    } else {
                        (m.normal_sign * (0.5 / self.eta0)) * ey_f[k][p] * hz_f[k][p].conj()
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

        let mut side_hz: [Option<Vec<Vec<Complex64>>>; 4] = [None, None, None, None];
        let mut side_ex: [Option<Vec<Vec<Complex64>>>; 4] = [None, None, None, None];
        let mut side_ey: [Option<Vec<Vec<Complex64>>>; 4] = [None, None, None, None];

        for (si, m_idx_opt) in sides.iter().enumerate() {
            if let Some(m_idx) = *m_idx_opt {
                if m_idx < self.monitor_buffers.len() {
                    let mb = &self.monitor_buffers[m_idx];
                    side_hz[si] = Some(Self::phasor_from_series(&mb.hz, self.nt, mb.nline, &freqs, self.dt));
                    side_ex[si] = Some(Self::phasor_from_series(&mb.ex, self.nt, mb.nline, &freqs, self.dt));
                    side_ey[si] = Some(Self::phasor_from_series(&mb.ey, self.nt, mb.nline, &freqs, self.dt));
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
        writeln!(f, "freq_hz,phi_rad,e_phi_re,e_phi_im,h_theta_re,h_theta_im,p_phi_re")?;

        for (fi, &fr) in freqs.iter().enumerate() {
            let k0 = 2.0 * PI * fr / self.c0;
            let gmax = gsrc
                .iter()
                .map(|v| v.norm())
                .fold(0.0_f64, |a, b| if a > b { a } else { b });
            for &ph in &phi {
                let cph = ph.cos();
                let sph = ph.sin();
                let mut n_phi = Complex64::new(0.0, 0.0);
                let mut l_theta = Complex64::new(0.0, 0.0);

                if let Some(mi) = nf.bottom {
                    if mi < self.monitors.len() && side_ex[1].is_some() {
                        let mon = &self.monitors[mi];
                        let ex = side_ex[1].as_ref().unwrap();
                        let hz = side_hz[1].as_ref().unwrap();
                        for l in 0..mon.points.len() {
                            let x = mon.x_m[l] + 0.5 * self.dx;
                            let y = mon.y_m[l] + 0.5 * self.dy;
                            let phase = Complex64::from_polar(1.0, k0 * (x * cph + y * sph));
                            n_phi += Complex64::new(sph, 0.0) * hz[fi][l] * phase * self.dx;
                            l_theta += ex[fi][l] * self.eta0 * phase * self.dx;
                        }
                    }
                }
                if let Some(mi) = nf.top {
                    if mi < self.monitors.len() && side_ex[0].is_some() {
                        let mon = &self.monitors[mi];
                        let ex = side_ex[0].as_ref().unwrap();
                        let hz = side_hz[0].as_ref().unwrap();
                        for l in 0..mon.points.len() {
                            let x = mon.x_m[l] + 0.5 * self.dx;
                            let y = mon.y_m[l] + 0.5 * self.dy;
                            let phase = Complex64::from_polar(1.0, k0 * (x * cph + y * sph));
                            n_phi -= Complex64::new(sph, 0.0) * hz[fi][l] * phase * self.dx;
                            l_theta -= ex[fi][l] * self.eta0 * phase * self.dx;
                        }
                    }
                }
                if let Some(mi) = nf.right {
                    if mi < self.monitors.len() && side_ey[3].is_some() {
                        let mon = &self.monitors[mi];
                        let ey = side_ey[3].as_ref().unwrap();
                        let hz = side_hz[3].as_ref().unwrap();
                        for l in 0..mon.points.len() {
                            let x = mon.x_m[l] + 0.5 * self.dx;
                            let y = mon.y_m[l] + 0.5 * self.dy;
                            let phase = Complex64::from_polar(1.0, k0 * (x * cph + y * sph));
                            n_phi -= Complex64::new(cph, 0.0) * hz[fi][l] * phase * self.dy;
                            l_theta += ey[fi][l] * self.eta0 * phase * self.dy;
                        }
                    }
                }
                if let Some(mi) = nf.left {
                    if mi < self.monitors.len() && side_ey[2].is_some() {
                        let mon = &self.monitors[mi];
                        let ey = side_ey[2].as_ref().unwrap();
                        let hz = side_hz[2].as_ref().unwrap();
                        for l in 0..mon.points.len() {
                            let x = mon.x_m[l] + 0.5 * self.dx;
                            let y = mon.y_m[l] + 0.5 * self.dy;
                            let phase = Complex64::from_polar(1.0, k0 * (x * cph + y * sph));
                            n_phi += Complex64::new(cph, 0.0) * hz[fi][l] * phase * self.dy;
                            l_theta -= ey[fi][l] * self.eta0 * phase * self.dy;
                        }
                    }
                }

                let mut ephi = Complex64::new(k0, 0.0) * (self.eta0 * n_phi - l_theta);
                let mut htheta = Complex64::new(k0, 0.0) * (l_theta / self.eta0 - n_phi);
                let gmag = gsrc[fi].norm();
                let good = gmag >= 1e-6 * gmax && gmax > 0.0;
                if good {
                    ephi /= gsrc[fi];
                    htheta /= gsrc[fi];
                } else {
                    ephi = Complex64::new(0.0, 0.0);
                    htheta = Complex64::new(0.0, 0.0);
                }
                let pt = 0.5 * (ephi * htheta).re;
                writeln!(
                    f,
                    "{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e}",
                    fr, ph, ephi.re, ephi.im, htheta.re, htheta.im, pt
                )?;
            }
        }
        Ok(())
    }

    fn write_outputs<P: AsRef<Path>>(&self, out_dir: P, cfg: &Config) -> Result<(), Box<dyn Error>> {
        let out = out_dir.as_ref();
        fs::create_dir_all(out)?;

        write_f32_bin(out.join("ex_history.bin"), &self.ex_hist)?;
        write_f32_bin(out.join("ey_history.bin"), &self.ey_hist)?;
        write_f32_bin(out.join("hz_history.bin"), &self.hz_hist)?;
        write_f64_bin(out.join("eravg.bin"), &self.erxx)?;

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
            write_f64_bin(out.join(format!("monitor_time_hz_{mi}.bin")), &mb.hz)?;
            write_f64_bin(out.join(format!("monitor_time_ex_{mi}.bin")), &mb.ex)?;
            write_f64_bin(out.join(format!("monitor_time_ey_{mi}.bin")), &mb.ey)?;
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

    let mut sim = Sim2DHz::new(&cfg);
    sim.init_scene(&cfg);
    sim.run();

    let out = cfg.output_dir.clone().unwrap_or_else(|| "output".to_string());
    sim.write_outputs(&out, &cfg)?;

    println!("Done. Outputs written to ./{out}");
    Ok(())
}







