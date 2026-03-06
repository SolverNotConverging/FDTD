use num_complex::Complex64;
use serde::Deserialize;
use std::env;
use std::error::Error;
use std::f64::consts::PI;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

#[derive(Clone, Copy)]
enum Boundary {
    Absorbing,
    Electric,
    Magnetic,
}

impl Boundary {
    fn from_str(value: &str) -> Result<Self, Box<dyn Error>> {
        match value.to_ascii_lowercase().as_str() {
            "absorbing" | "a" => Ok(Self::Absorbing),
            "electric" | "e" => Ok(Self::Electric),
            "magnetic" | "m" => Ok(Self::Magnetic),
            _ => Err(format!("Unknown boundary: {value}").into()),
        }
    }
}

#[derive(Debug, Deserialize)]
struct SimulationConfig {
    z_range: f64,
    nz: usize,
    f_max: f64,
    nt: usize,
    dt: Option<f64>,
    objects: Vec<ObjectConfig>,
    boundary: BoundaryConfig,
    source: SourceConfig,
    output_dir: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ObjectConfig {
    er: f64,
    mr: f64,
    region: [f64; 2],
}

#[derive(Debug, Deserialize)]
struct BoundaryConfig {
    left: String,
    right: String,
}

#[derive(Debug, Deserialize)]
struct SourceConfig {
    position: f64,
    amplitude: Option<f64>,
    t0: Option<f64>,
    tw: Option<f64>,
}

struct FDTD1D {
    eps0: f64,
    mu0: f64,
    c0: f64,
    z_range: f64,
    nz: usize,
    dz: f64,
    er: Vec<f64>,
    mr: Vec<f64>,
    mey: Vec<f64>,
    mhx: Vec<f64>,
    ey: Vec<f64>,
    hx: Vec<f64>,
    nt: usize,
    f_max: f64,
    dt: f64,
    src_index: Option<usize>,
    src_amplitude: f64,
    src_t0: f64,
    src_tw: f64,
    left_absorbing_boundary: bool,
    right_absorbing_boundary: bool,
    ey_past: f64,
    hx_past: f64,
    nf: usize,
    ref_acc: Vec<Complex64>,
    trn_acc: Vec<Complex64>,
    src_acc: Vec<Complex64>,
    ey_history: Vec<Vec<f64>>,
    hx_history: Vec<Vec<f64>>,
    ref_power_history: Vec<Vec<f64>>,
    trn_power_history: Vec<Vec<f64>>,
    frequencies: Vec<f64>,
}

impl FDTD1D {
    fn new(z_range: f64, nz: usize, f_max: f64, nt: usize, dt: Option<f64>) -> Self {
        let eps0 = 8.85e-12;
        let mu0 = 4e-7 * PI;
        let c0 = 1.0 / (eps0 * mu0).sqrt();
        let dz = z_range / nz as f64;

        let dt_cfl = dz / (2.0 * c0);
        let dt_freq_sampling = 1.0 / (20.0 * f_max);
        let dt = dt.unwrap_or(dt_cfl.min(dt_freq_sampling));

        let nf = nt.min(100);
        let frequencies = (0..nf)
            .map(|i| f_max * i as f64 / (nf.saturating_sub(1).max(1) as f64))
            .collect::<Vec<_>>();

        Self {
            eps0,
            mu0,
            c0,
            z_range,
            nz,
            dz,
            er: vec![1.0; nz],
            mr: vec![1.0; nz],
            mey: vec![1.0; nz],
            mhx: vec![1.0; nz],
            ey: vec![0.0; nz],
            hx: vec![0.0; nz],
            nt,
            f_max,
            dt,
            src_index: None,
            src_amplitude: 1.0,
            src_t0: 0.0,
            src_tw: 0.0,
            left_absorbing_boundary: false,
            right_absorbing_boundary: false,
            ey_past: 0.0,
            hx_past: 0.0,
            nf,
            ref_acc: vec![Complex64::new(0.0, 0.0); nf],
            trn_acc: vec![Complex64::new(0.0, 0.0); nf],
            src_acc: vec![Complex64::new(0.0, 0.0); nf],
            ey_history: Vec::with_capacity(nt),
            hx_history: Vec::with_capacity(nt),
            ref_power_history: Vec::with_capacity(nt),
            trn_power_history: Vec::with_capacity(nt),
            frequencies,
        }
    }

    fn init_mey_mhx(&mut self) {
        for i in 0..self.nz {
            self.mey[i] = self.c0 * self.dt / self.er[i];
            self.mhx[i] = self.c0 * self.dt / self.mr[i];
        }
    }

    fn indices_from_z(&self, z_start: f64, z_end: f64) -> (usize, usize) {
        let mut i0 = (z_start / self.dz).round() as isize;
        let mut i1 = (z_end / self.dz).round() as isize;

        i0 = i0.clamp(0, (self.nz - 1) as isize);
        i1 = i1.clamp(0, self.nz as isize);

        let (mut a, mut b) = (i0 as usize, i1 as usize);
        if b < a {
            std::mem::swap(&mut a, &mut b);
        }
        (a, b)
    }

    fn add_object(&mut self, er: f64, mr: f64, region: (f64, f64)) {
        let (i0, i1) = self.indices_from_z(region.0, region.1);
        for i in i0..i1 {
            self.er[i] = er;
            self.mr[i] = mr;
        }
    }

    fn set_boundary(&mut self, left: Boundary, right: Boundary) {
        match left {
            Boundary::Absorbing => self.left_absorbing_boundary = true,
            Boundary::Electric => {
                self.er[0] = 1e8;
                self.left_absorbing_boundary = false;
            }
            Boundary::Magnetic => {
                self.mr[0] = 1e8;
                self.left_absorbing_boundary = false;
            }
        }

        match right {
            Boundary::Absorbing => self.right_absorbing_boundary = true,
            Boundary::Electric => {
                self.er[self.nz - 1] = 1e8;
                self.right_absorbing_boundary = false;
            }
            Boundary::Magnetic => {
                self.mr[self.nz - 1] = 1e8;
                self.right_absorbing_boundary = false;
            }
        }
    }

    fn add_source(&mut self, src_position_m: f64, amplitude: f64, t0: Option<f64>, tw: Option<f64>) {
        let idx = (src_position_m / self.dz).round() as isize;
        if idx < 0 || idx as usize >= self.nz {
            panic!("Source position is out of range.");
        }

        self.src_index = Some(idx as usize);
        self.src_amplitude = amplitude;
        self.src_tw = tw.unwrap_or(0.5 / self.f_max);
        self.src_t0 = t0.unwrap_or(4.0 * self.src_tw);

        let pulse0 = self.pulse(0.0);
        if pulse0.abs() > 1e-4 {
            panic!("Source is non-zero at t=0. Adjust t0/tw.");
        }
    }

    fn pulse(&self, t: f64) -> f64 {
        self.src_amplitude * (-((t - self.src_t0) / self.src_tw).powi(2)).exp()
    }

    fn h_update(&mut self) {
        for nz in 0..(self.nz - 1) {
            self.hx[nz] += self.mhx[nz] * (self.ey[nz + 1] - self.ey[nz]) / self.dz;
        }

        if !self.right_absorbing_boundary {
            self.hx[self.nz - 1] += self.mhx[self.nz - 1] * (-self.ey[self.nz - 1]) / self.dz;
        } else {
            let s = self.c0 * self.dt
                / (self.dz * (self.er[self.nz - 1] * self.mr[self.nz - 1]).sqrt());
            self.hx[self.nz - 1] =
                self.hx_past + ((s - 1.0) / (s + 1.0)) * (self.hx[self.nz - 2] - self.hx[self.nz - 1]);
            self.hx_past = self.hx[self.nz - 2];
        }
    }

    fn e_update(&mut self) {
        for nz in 1..self.nz {
            self.ey[nz] += self.mey[nz] * (self.hx[nz] - self.hx[nz - 1]) / self.dz;
        }

        if !self.left_absorbing_boundary {
            self.ey[0] += self.mey[0] * self.hx[0] / self.dz;
        } else {
            let s = self.c0 * self.dt / (self.dz * (self.er[0] * self.mr[0]).sqrt());
            self.ey[0] = self.ey_past + ((s - 1.0) / (s + 1.0)) * (self.ey[1] - self.ey[0]);
            self.ey_past = self.ey[1];
        }
    }

    fn run(&mut self) {
        self.init_mey_mhx();

        let src_index = self
            .src_index
            .expect("Source must be added before calling run().");
        if src_index == 0 {
            panic!("src_index must be at least 1 because H injection uses src_index - 1.");
        }

        let kn = self
            .frequencies
            .iter()
            .map(|f| Complex64::from_polar(1.0, -2.0 * PI * f * self.dt))
            .collect::<Vec<_>>();
        let mut kn_pow = vec![Complex64::new(1.0, 0.0); self.nf];

        let c_ref = (self.mr[1] / self.er[1]).sqrt().sqrt();
        let c_trn = (self.mr[self.nz - 2] / self.er[self.nz - 2]).sqrt().sqrt();
        let c_src = (self.mr[src_index] / self.er[src_index]).sqrt().sqrt();

        for t_index in 0..self.nt {
            self.h_update();

            let t = t_index as f64 * self.dt;
            let ey_src = self.pulse(t);
            self.hx[src_index - 1] -= self.mhx[src_index - 1] / self.dz * ey_src;

            self.e_update();

            let eta_r_inv = (self.er[src_index] / self.mr[src_index]).sqrt();
            let n = (self.er[src_index] * self.mr[src_index]).sqrt();
            let hx_src = -eta_r_inv * self.pulse(t + n * self.dz / (2.0 * self.c0) + self.dt / 2.0);
            self.ey[src_index] -= self.mey[src_index] / self.dz * hx_src;

            let mut ref_row = vec![0.0; self.nf];
            let mut trn_row = vec![0.0; self.nf];

            for nf in 0..self.nf {
                self.ref_acc[nf] += kn_pow[nf] * (self.dt * self.ey[1] / c_ref);
                self.trn_acc[nf] += kn_pow[nf] * (self.dt * self.ey[self.nz - 2] / c_trn);
                self.src_acc[nf] += kn_pow[nf] * (self.dt * ey_src / c_src);

                let src_norm = self.src_acc[nf].norm_sqr().max(1e-30);
                ref_row[nf] = self.ref_acc[nf].norm_sqr() / src_norm;
                trn_row[nf] = self.trn_acc[nf].norm_sqr() / src_norm;

                kn_pow[nf] *= kn[nf];
            }

            self.hx_history.push(self.hx.clone());
            self.ey_history.push(self.ey.clone());
            self.ref_power_history.push(ref_row);
            self.trn_power_history.push(trn_row);

            if t_index % 250 == 0 || t_index + 1 == self.nt {
                println!("step {}/{}", t_index + 1, self.nt);
            }
        }
    }

    fn write_outputs<P: AsRef<Path>>(&self, output_dir: P) -> Result<(), Box<dyn Error>> {
        let output_dir = output_dir.as_ref();
        fs::create_dir_all(output_dir)?;

        write_matrix_csv(output_dir.join("ey_history.csv"), &self.ey_history)?;
        write_matrix_csv(output_dir.join("hx_history.csv"), &self.hx_history)?;
        write_matrix_csv(output_dir.join("ref_power_history.csv"), &self.ref_power_history)?;
        write_matrix_csv(output_dir.join("trn_power_history.csv"), &self.trn_power_history)?;

        write_vector_csv(output_dir.join("frequency_hz.csv"), &self.frequencies)?;
        write_vector_csv(output_dir.join("er_profile.csv"), &self.er)?;
        write_vector_csv(output_dir.join("mr_profile.csv"), &self.mr)?;

        let z = (0..self.nz).map(|i| i as f64 * self.dz).collect::<Vec<_>>();
        write_vector_csv(output_dir.join("z_m.csv"), &z)?;

        let mut meta = BufWriter::new(File::create(output_dir.join("metadata.txt"))?);
        writeln!(meta, "z_range_m={}", self.z_range)?;
        writeln!(meta, "nz={}", self.nz)?;
        writeln!(meta, "dz_m={}", self.dz)?;
        writeln!(meta, "nt={}", self.nt)?;
        writeln!(meta, "f_max_hz={}", self.f_max)?;
        writeln!(meta, "dt_s={}", self.dt)?;
        writeln!(meta, "eps0={}", self.eps0)?;
        writeln!(meta, "mu0={}", self.mu0)?;
        writeln!(meta, "source_index={}", self.src_index.unwrap_or(0))?;
        writeln!(meta, "source_t0_s={}", self.src_t0)?;
        writeln!(meta, "source_tw_s={}", self.src_tw)?;
        writeln!(meta, "source_amplitude={}", self.src_amplitude)?;
        writeln!(meta, "output_format=rows_are_time_steps")?;

        Ok(())
    }
}

fn write_matrix_csv<P: AsRef<Path>>(path: P, matrix: &[Vec<f64>]) -> Result<(), Box<dyn Error>> {
    let mut file = BufWriter::new(File::create(path)?);
    for row in matrix {
        for (i, v) in row.iter().enumerate() {
            if i > 0 {
                write!(file, ",")?;
            }
            write!(file, "{:.12e}", v)?;
        }
        writeln!(file)?;
    }
    Ok(())
}

fn write_vector_csv<P: AsRef<Path>>(path: P, values: &[f64]) -> Result<(), Box<dyn Error>> {
    let mut file = BufWriter::new(File::create(path)?);
    for v in values {
        writeln!(file, "{:.12e}", v)?;
    }
    Ok(())
}

fn parse_args() -> PathBuf {
    let mut args = env::args().skip(1);
    let mut config_path = PathBuf::from("config/example_1d.json");

    while let Some(arg) = args.next() {
        if arg == "--config" {
            if let Some(path) = args.next() {
                config_path = PathBuf::from(path);
            }
        }
    }

    config_path
}

fn main() -> Result<(), Box<dyn Error>> {
    let config_path = parse_args();
    let raw = fs::read_to_string(&config_path)?;
    let cfg: SimulationConfig = serde_json::from_str(&raw)?;

    let mut sim = FDTD1D::new(cfg.z_range, cfg.nz, cfg.f_max, cfg.nt, cfg.dt);

    for obj in cfg.objects {
        sim.add_object(obj.er, obj.mr, (obj.region[0], obj.region[1]));
    }

    let left = Boundary::from_str(&cfg.boundary.left)?;
    let right = Boundary::from_str(&cfg.boundary.right)?;
    sim.set_boundary(left, right);

    sim.add_source(
        cfg.source.position,
        cfg.source.amplitude.unwrap_or(1.0),
        cfg.source.t0,
        cfg.source.tw,
    );

    sim.run();

    let out = cfg.output_dir.unwrap_or_else(|| "output".to_string());
    sim.write_outputs(&out)?;

    println!("Done. Outputs written to ./{out}");
    Ok(())
}

