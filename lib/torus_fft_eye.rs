//! torus_fft_eye — reference implementation of the cascade's bridge and
//! discover stages in Rust.
//!
//! Cascade position: sizes the DFT (bridge-level) → sigma/omega extraction
//! (bridge→discover boundary) → winding analysis (discover-level) → TorusEye
//! (stateful accrual across frames). Function order matches cascade timing.
//!
//! Any 2D grid is a flat torus T² = S¹ × S¹:
//!   x-axis: pixel columns wrap (no left/right edge in code)
//!   y-axis: pixel rows wrap (no top/bottom edge in code)
//!   K = 0 intrinsically — all visible curvature belongs to the display
//!
//! The 2D DFT on this torus decomposes the visual field into spatial
//! frequency modes. Each coefficient (k_x, k_y) is a winding number pair:
//!   σ = magnitude (what spatial patterns are present, how strong)
//!   ω = phase (the turn memory of each pattern — where it sits)
//!
//! Together: 2D position + 2D frequency = 4D view of 3D space
//! projected onto the 2D display. No edge bound because the torus wraps.
//!
//! Pure Rust, no dependencies. Compiles with `rustc lib/torus_fft_eye.rs`.

use std::f64::consts::PI;

/// A complex number — the σ+iω carrier.
#[derive(Clone, Copy, Debug)]
pub struct Complex {
    pub re: f64,
    pub im: f64,
}

impl Complex {
    pub fn new(re: f64, im: f64) -> Self { Self { re, im } }
    pub fn zero() -> Self { Self { re: 0.0, im: 0.0 } }
    pub fn magnitude(&self) -> f64 { (self.re * self.re + self.im * self.im).sqrt() }
    pub fn phase(&self) -> f64 { self.im.atan2(self.re) }

    pub fn add(self, other: Self) -> Self {
        Self { re: self.re + other.re, im: self.im + other.im }
    }
    pub fn mul(self, other: Self) -> Self {
        Self {
            re: self.re * other.re - self.im * other.im,
            im: self.re * other.im + self.im * other.re,
        }
    }
}

/// 1D DFT — the base transform. Pure, no allocation beyond the output.
/// N can be any size (not restricted to powers of 2).
pub fn dft_1d(input: &[Complex]) -> Vec<Complex> {
    let n = input.len();
    let mut output = vec![Complex::zero(); n];
    for k in 0..n {
        let mut sum = Complex::zero();
        for j in 0..n {
            let angle = -2.0 * PI * (k as f64) * (j as f64) / (n as f64);
            let twiddle = Complex::new(angle.cos(), angle.sin());
            sum = sum.add(input[j].mul(twiddle));
        }
        output[k] = sum;
    }
    output
}

/// 1D inverse DFT.
pub fn idft_1d(input: &[Complex]) -> Vec<Complex> {
    let n = input.len();
    let mut output = vec![Complex::zero(); n];
    for k in 0..n {
        let mut sum = Complex::zero();
        for j in 0..n {
            let angle = 2.0 * PI * (k as f64) * (j as f64) / (n as f64);
            let twiddle = Complex::new(angle.cos(), angle.sin());
            sum = sum.add(input[j].mul(twiddle));
        }
        output[k] = Complex::new(sum.re / n as f64, sum.im / n as f64);
    }
    output
}

/// 2D DFT on the torus — the eye's frequency transform.
/// Input: row-major grayscale pixels (height rows × width cols), values 0–255.
/// Output: complex spectrum (height × width), row-major.
///
/// This IS the torus: the 2D DFT treats the grid as periodic in both axes,
/// which is exactly T² = S¹ × S¹. No edge bound — the transform wraps.
pub fn torus_fft_2d(pixels: &[u8], width: usize, height: usize) -> Vec<Complex> {
    let mut grid: Vec<Complex> = pixels.iter()
        .map(|&p| Complex::new(p as f64, 0.0))
        .collect();

    // Transform rows (x-axis windings)
    for y in 0..height {
        let row_start = y * width;
        let row: Vec<Complex> = grid[row_start..row_start + width].to_vec();
        let transformed = dft_1d(&row);
        for x in 0..width {
            grid[row_start + x] = transformed[x];
        }
    }

    // Transform columns (y-axis windings)
    for x in 0..width {
        let col: Vec<Complex> = (0..height).map(|y| grid[y * width + x]).collect();
        let transformed = dft_1d(&col);
        for y in 0..height {
            grid[y * width + x] = transformed[y];
        }
    }

    grid
}

/// Inverse 2D DFT — reconstruct pixels from the torus spectrum.
pub fn torus_ifft_2d(spectrum: &[Complex], width: usize, height: usize) -> Vec<u8> {
    let mut grid = spectrum.to_vec();

    for x in 0..width {
        let col: Vec<Complex> = (0..height).map(|y| grid[y * width + x]).collect();
        let transformed = idft_1d(&col);
        for y in 0..height {
            grid[y * width + x] = transformed[y];
        }
    }

    for y in 0..height {
        let row_start = y * width;
        let row: Vec<Complex> = grid[row_start..row_start + width].to_vec();
        let transformed = idft_1d(&row);
        for x in 0..width {
            grid[row_start + x] = transformed[x];
        }
    }

    grid.iter().map(|c| c.re.round().max(0.0).min(255.0) as u8).collect()
}

/// The σ+iω separation — the theorem's imaginary calculus made spatial.
///
/// Returns (sigma, omega) where:
///   sigma[k] = magnitude of mode k (what spatial frequency is present)
///   omega[k] = phase of mode k (where in the cycle this pattern sits)
///
/// Together they ARE the 4D view: each (k_x, k_y) mode gives a winding
/// number pair on the torus, and σ+iω at that mode gives the amplitude
/// and phase of that winding.
pub fn sigma_omega(spectrum: &[Complex]) -> (Vec<f64>, Vec<f64>) {
    let sigma: Vec<f64> = spectrum.iter().map(|c| c.magnitude()).collect();
    let omega: Vec<f64> = spectrum.iter().map(|c| c.phase()).collect();
    (sigma, omega)
}

/// Winding analysis — which spatial frequencies dominate the visual field.
///
/// coprime_fraction: energy in modes where gcd(k_x, k_y) = 1
///   These are irreducible rational windings — the simplest closed
///   orbits on T². High = clean periodic structure (rational winding).
///   Low = energy in harmonic multiples of simpler patterns.
///
/// mode_diversity: effective_modes / max_modes. The actual complexity
///   measure — energy spread across many modes approaches irrational
///   winding (dense orbit, never closing). High diversity = complex.
///
/// dc_fraction: energy in the (0,0) mode = the constant brightness.
///
/// effective_modes: entropy-based count of how many spatial frequencies
///   carry meaningful energy.
pub struct WindingAnalysis {
    pub dc_fraction: f64,
    pub coprime_fraction: f64,
    pub effective_modes: f64,
    pub max_modes: usize,
    pub mode_diversity: f64,
    pub total_energy: f64,
}

fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 { let t = b; b = a % b; a = t; }
    a
}

pub fn winding_analysis(spectrum: &[Complex], width: usize, height: usize) -> WindingAnalysis {
    let total_energy: f64 = spectrum.iter().map(|c| c.magnitude().powi(2)).sum();
    let dc_energy = spectrum[0].magnitude().powi(2);

    let mut coprime_energy = 0.0f64;
    let half_w = width / 2 + 1;
    let half_h = height / 2 + 1;
    for ky in 0..half_h {
        for kx in 0..half_w {
            if kx > 0 && ky > 0 && gcd(kx, ky) == 1 {
                let e = spectrum[ky * width + kx].magnitude().powi(2);
                coprime_energy += e;
            }
        }
    }

    // Effective modes via entropy
    let mag_sq: Vec<f64> = spectrum.iter().map(|c| c.magnitude().powi(2)).collect();
    let sum_sq: f64 = mag_sq.iter().sum();
    let entropy: f64 = if sum_sq > 1e-12 {
        mag_sq.iter()
            .map(|&e| { let p = e / sum_sq; if p > 1e-12 { -p * p.ln() } else { 0.0 } })
            .sum()
    } else {
        0.0
    };
    let effective_modes = entropy.exp();
    let max_modes = width * height;

    WindingAnalysis {
        dc_fraction: if total_energy > 1e-12 { dc_energy / total_energy } else { 0.0 },
        coprime_fraction: if total_energy > 1e-12 { coprime_energy / total_energy } else { 0.0 },
        effective_modes,
        max_modes,
        mode_diversity: effective_modes / max_modes as f64,
        total_energy,
    }
}

/// The torus eye — holds accrued state across frames.
///
/// The prior frame's spectrum is held so that `changed` regions can be
/// identified in frequency domain (not just position domain). A spatial
/// change that affects only high-frequency modes (edge movement) looks
/// different from one that affects low-frequency modes (brightness shift)
/// — the torus sees WHAT KIND of change, not just WHERE.
pub struct TorusEye {
    pub width: usize,
    pub height: usize,
    prior_spectrum: Option<Vec<Complex>>,
    history_count: usize,
}

impl TorusEye {
    pub fn new(width: usize, height: usize) -> Self {
        Self { width, height, prior_spectrum: None, history_count: 0 }
    }

    /// Ingest a frame. Returns the σ+iω decomposition and the spectral
    /// delta (which frequency modes changed since last frame).
    pub fn ingest(&mut self, pixels: &[u8]) -> TorusFrame {
        let spectrum = torus_fft_2d(pixels, self.width, self.height);
        let (sigma, omega) = sigma_omega(&spectrum);

        let delta = self.prior_spectrum.as_ref().map(|prior| {
            spectrum.iter().zip(prior.iter())
                .map(|(now, was)| {
                    let d = Complex::new(now.re - was.re, now.im - was.im);
                    d.magnitude()
                })
                .collect::<Vec<f64>>()
        });

        let analysis = winding_analysis(&spectrum, self.width, self.height);
        self.prior_spectrum = Some(spectrum);
        self.history_count += 1;

        TorusFrame { sigma, omega, delta, analysis, frame_number: self.history_count }
    }
}

pub struct TorusFrame {
    pub sigma: Vec<f64>,
    pub omega: Vec<f64>,
    pub delta: Option<Vec<f64>>,
    pub analysis: WindingAnalysis,
    pub frame_number: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_identity() {
        let w = 8;
        let h = 8;
        let pixels: Vec<u8> = (0..w*h).map(|i| (i * 4) as u8).collect();
        let spectrum = torus_fft_2d(&pixels, w, h);
        let recovered = torus_ifft_2d(&spectrum, w, h);
        for (a, b) in pixels.iter().zip(recovered.iter()) {
            assert!((*a as i16 - *b as i16).abs() <= 1,
                "roundtrip diverged: {} vs {}", a, b);
        }
    }

    #[test]
    fn constant_image_is_dc_only() {
        let w = 4;
        let h = 4;
        let pixels = vec![128u8; w * h];
        let spectrum = torus_fft_2d(&pixels, w, h);
        let analysis = winding_analysis(&spectrum, w, h);
        assert!(analysis.dc_fraction > 0.99,
            "constant image should be nearly all DC, got {}", analysis.dc_fraction);
    }

    #[test]
    fn sigma_omega_separation() {
        let w = 4;
        let h = 4;
        let pixels: Vec<u8> = (0..w*h).map(|i| ((i % 4) * 64) as u8).collect();
        let spectrum = torus_fft_2d(&pixels, w, h);
        let (sigma, omega) = sigma_omega(&spectrum);
        assert_eq!(sigma.len(), w * h);
        assert_eq!(omega.len(), w * h);
        assert!(sigma.iter().any(|&s| s > 0.0));
    }

    #[test]
    fn torus_eye_accrual() {
        let w = 8;
        let h = 8;
        let mut eye = TorusEye::new(w, h);

        let frame1_pixels = vec![100u8; w * h];
        let f1 = eye.ingest(&frame1_pixels);
        assert!(f1.delta.is_none());
        assert_eq!(f1.frame_number, 1);

        let mut frame2_pixels = vec![100u8; w * h];
        frame2_pixels[0] = 200;
        let f2 = eye.ingest(&frame2_pixels);
        assert!(f2.delta.is_some());
        let delta = f2.delta.unwrap();
        assert!(delta.iter().any(|&d| d > 0.0));
        assert_eq!(f2.frame_number, 2);
    }

    #[test]
    fn no_edge_bound() {
        // The torus wraps — adjacent pixels at opposite edges should
        // produce nearby frequency content, not edge artifacts.
        let w = 8;
        let h = 8;
        let mut pixels = vec![0u8; w * h];
        // Gradient that wraps: right edge meets left edge
        for y in 0..h {
            for x in 0..w {
                pixels[y * w + x] = ((x * 32) % 256) as u8;
            }
        }
        let spectrum = torus_fft_2d(&pixels, w, h);
        let analysis = winding_analysis(&spectrum, w, h);
        // A wrapping gradient should concentrate energy in a few modes
        assert!(analysis.effective_modes < (w * h) as f64 / 2.0,
            "wrapping gradient should not spread energy across all modes");
    }
}

fn main() {
    let w = 16;
    let h = 16;
    let pixels: Vec<u8> = (0..w*h).map(|i| ((i * 3) % 256) as u8).collect();

    let mut eye = TorusEye::new(w, h);
    let frame = eye.ingest(&pixels);

    println!("Torus Eye — frame {}", frame.frame_number);
    println!("  total_energy:    {:.2}", frame.analysis.total_energy);
    println!("  dc_fraction:     {:.4}", frame.analysis.dc_fraction);
    println!("  coprime_fraction:{:.4}", frame.analysis.coprime_fraction);
    println!("  effective_modes: {:.2}", frame.analysis.effective_modes);
    println!("  mode_diversity:  {:.4}", frame.analysis.mode_diversity);
    println!("  sigma[0] (DC):   {:.2}", frame.sigma[0]);
    println!("  omega[0] (DC):   {:.4}", frame.omega[0]);
}
