//! torus_stream — streaming reference: the cascade as a Unix pipe.
//!
//! Cascade position: stdin frames → bridge (DFT) → discover (analyze) →
//! stdout JSON. The main() function IS a cascade: ingest (read frame) →
//! discover (analyze) → output (write JSON line).
//!
//! Reads raw grayscale frames from stdin (width*height bytes each),
//! outputs one JSON line per frame to stdout with torus analysis.
//!
//! Protocol: first line of stdin = "WIDTHxHEIGHT\n", then raw bytes.
//!
//! Build: rustc -O lib/torus_stream.rs -o torus_stream.exe
//! Usage: (echo 80x45 && cat frames.raw) | ./torus_stream

use std::f64::consts::PI;
use std::io::{self, Read, Write, BufWriter, BufRead};

#[derive(Clone, Copy)]
struct C { re: f64, im: f64 }

impl C {
    fn new(re: f64, im: f64) -> Self { Self { re, im } }
    fn zero() -> Self { Self { re: 0.0, im: 0.0 } }
    fn mag_sq(&self) -> f64 { self.re * self.re + self.im * self.im }
}

fn dft_1d(input: &[C]) -> Vec<C> {
    let n = input.len();
    let mut out = vec![C::zero(); n];
    for k in 0..n {
        let mut re = 0.0f64;
        let mut im = 0.0f64;
        for j in 0..n {
            let angle = -2.0 * PI * (k as f64) * (j as f64) / (n as f64);
            let (sin_a, cos_a) = angle.sin_cos();
            re += input[j].re * cos_a - input[j].im * sin_a;
            im += input[j].re * sin_a + input[j].im * cos_a;
        }
        out[k] = C::new(re, im);
    }
    out
}

fn torus_fft_2d(pixels: &[u8], w: usize, h: usize) -> Vec<C> {
    let mut grid: Vec<C> = pixels.iter().map(|&p| C::new(p as f64, 0.0)).collect();

    for y in 0..h {
        let s = y * w;
        let row: Vec<C> = grid[s..s + w].to_vec();
        let t = dft_1d(&row);
        for x in 0..w { grid[s + x] = t[x]; }
    }

    for x in 0..w {
        let col: Vec<C> = (0..h).map(|y| grid[y * w + x]).collect();
        let t = dft_1d(&col);
        for y in 0..h { grid[y * w + x] = t[y]; }
    }

    grid
}

fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 { let t = b; b = a % b; a = t; }
    a
}

struct Analysis {
    dc_fraction: f64,
    coprime_fraction: f64,
    effective_modes: f64,
    mode_diversity: f64,
    total_energy: f64,
}

fn analyze(spectrum: &[C], w: usize, h: usize) -> Analysis {
    let mag_sq: Vec<f64> = spectrum.iter().map(|c| c.mag_sq()).collect();
    let total: f64 = mag_sq.iter().sum();
    let dc = mag_sq[0];

    let mut coprime = 0.0f64;
    let half_w = w / 2 + 1;
    let half_h = h / 2 + 1;
    for ky in 0..half_h {
        for kx in 0..half_w {
            if kx > 0 && ky > 0 && gcd(kx, ky) == 1 {
                coprime += mag_sq[ky * w + kx];
            }
        }
    }

    let mut entropy = 0.0f64;
    if total > 1e-12 {
        for &e in &mag_sq {
            let p = e / total;
            if p > 1e-12 { entropy -= p * p.ln(); }
        }
    }
    let eff = entropy.exp();
    let max_m = (w * h) as f64;

    Analysis {
        dc_fraction: if total > 1e-12 { dc / total } else { 0.0 },
        coprime_fraction: if total > 1e-12 { coprime / total } else { 0.0 },
        effective_modes: eff,
        mode_diversity: eff / max_m,
        total_energy: total,
    }
}

fn main() {
    let stdin = io::stdin();
    let mut reader = stdin.lock();

    let mut header = String::new();
    reader.read_line(&mut header).expect("need WIDTHxHEIGHT header");
    let header = header.trim();
    let parts: Vec<&str> = header.split('x').collect();
    if parts.len() != 2 {
        eprintln!("header must be WIDTHxHEIGHT, got: {}", header);
        std::process::exit(1);
    }
    let w: usize = parts[0].parse().expect("bad width");
    let h: usize = parts[1].parse().expect("bad height");
    let frame_size = w * h;

    let stdout = io::stdout();
    let mut out = BufWriter::new(stdout.lock());

    let mut buf = vec![0u8; frame_size];
    let mut prior: Option<Vec<C>> = None;
    let mut frame_n: usize = 0;

    loop {
        match reader.read_exact(&mut buf) {
            Ok(()) => {},
            Err(_) => break,
        }

        frame_n += 1;
        let spectrum = torus_fft_2d(&buf, w, h);
        let a = analyze(&spectrum, w, h);

        let has_delta = prior.is_some();
        let (d_sum, d_max, d_mean) = if let Some(ref p) = prior {
            let mut s = 0.0f64;
            let mut mx = 0.0f64;
            for (now, was) in spectrum.iter().zip(p.iter()) {
                let dr = now.re - was.re;
                let di = now.im - was.im;
                let d = (dr * dr + di * di).sqrt();
                s += d;
                if d > mx { mx = d; }
            }
            let mean = if spectrum.len() > 0 { s / spectrum.len() as f64 } else { 0.0 };
            (s, mx, mean)
        } else {
            (0.0, 0.0, 0.0)
        };

        prior = Some(spectrum);

        if has_delta {
            write!(out, "{{\"frame\":{},\"dc\":{:.4},\"coprime\":{:.4},\"modes\":{:.2},\"diversity\":{:.6},\"energy\":{:.0},\"delta_sum\":{:.0},\"delta_max\":{:.0},\"delta_mean\":{:.2}}}\n",
                frame_n, a.dc_fraction, a.coprime_fraction,
                a.effective_modes, a.mode_diversity, a.total_energy,
                d_sum, d_max, d_mean).ok();
        } else {
            write!(out, "{{\"frame\":{},\"dc\":{:.4},\"coprime\":{:.4},\"modes\":{:.2},\"diversity\":{:.6},\"energy\":{:.0}}}\n",
                frame_n, a.dc_fraction, a.coprime_fraction,
                a.effective_modes, a.mode_diversity, a.total_energy).ok();
        }
        out.flush().ok();
    }
}
