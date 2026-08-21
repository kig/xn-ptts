//! Post-processing for synthesized speech: spectral denoising, loudness
//! normalization and soft limiting.
//!
//! Designed for the output stage of the TTS pipeline (streaming servers can
//! buffer one utterance and process it as a whole). All stages are
//! deterministic, in-place and cheap: measured CPU cost is a few percent of
//! one core per 30s utterance on the EPYC 7302 (see experiments/notebook.md).

use xn::Result;

/// Denoise via spectral gating (per-bin noise-floor tracking with
/// over-subtraction), implemented with a Hann-windowed overlap-add FFT.
///
/// Parameters tuned for Mimi-decoder hiss/background: frame 2048 samples
/// (85ms @ 24kHz), 50% hop, over-subtraction factor 2, gain floor -20dB,
/// noise-floor upward drift ~1 dB/s so the estimate follows slow noise
/// changes without chasing speech.
pub struct SpectralGate {
    frame: usize,
    hop: usize,
    fft_fwd: std::sync::Arc<dyn realfft::RealToComplex<f32>>,
    fft_inv: std::sync::Arc<dyn realfft::ComplexToReal<f32>>,
    window: Vec<f32>,
    noise_floor: Vec<f32>,
    initialized: bool,
}

impl SpectralGate {
    /// `frame` should be a power of two (2048 for 85ms @ 24kHz).
    pub fn new(frame: usize) -> Self {
        let mut planner = realfft::RealFftPlanner::<f32>::new();
        let fft_fwd = planner.plan_fft_forward(frame);
        let fft_inv = planner.plan_fft_inverse(frame);
        // Hann window.
        let window: Vec<f32> =
            (0..frame).map(|i| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / frame as f32).cos()).collect();
        let noise_floor = vec![0.0; frame / 2 + 1];
        Self { frame, hop: frame / 2, fft_fwd, fft_inv, window, noise_floor, initialized: false }
    }

    /// Denoise `pcm` in place.
    pub fn process(&mut self, pcm: &mut [f32]) {
        if pcm.len() < self.frame {
            return;
        }
        let frame = self.frame;
        let hop = self.hop;
        let mut spec = self.fft_fwd.make_output_vec();
        let mut input = self.fft_fwd.make_input_vec();
        let mut time_out = self.fft_inv.make_output_vec();
        let mut acc = vec![0.0f32; pcm.len()];
        let mut acc_w = vec![0.0f32; pcm.len()];
        // ~1 dB/s upward drift of the noise floor.
        let drift = 10f32.powf((1.0 / 20.0) * (hop as f32 / 24000.0));
        let n_bins = frame / 2 + 1;
        let mut frame_buf = vec![0.0f32; frame];
        let mut start = 0usize;
        while start + frame <= pcm.len() {
            for i in 0..frame {
                frame_buf[i] = pcm[start + i] * self.window[i];
            }
            input.copy_from_slice(&frame_buf);
            self.fft_fwd.process(&mut input, &mut spec).expect("fft forward");
            // spec is complex [Complex<f32>; n_bins]
            let mut denoised = vec![realfft::num_complex::Complex::new(0.0f32, 0.0f32); n_bins];
            for b in 0..n_bins {
                let re = spec[b].re;
                let im = spec[b].im;
                let mag = (re * re + im * im).sqrt();
                if !self.initialized {
                    // Start from the first frame (utterance onset is silence).
                    self.noise_floor[b] = mag;
                } else {
                    let nf = self.noise_floor[b];
                    self.noise_floor[b] = if mag < nf { mag } else { nf * drift };
                }
                let nf = self.noise_floor[b];
                let gain = if mag <= 1e-9 {
                    1.0
                } else {
                    let g = 1.0 - 2.0 * (nf / mag);
                    g.clamp(0.1, 1.0)
                };
                denoised[b] = realfft::num_complex::Complex::new(re * gain, im * gain);
            }
            self.fft_inv
                .process(&mut denoised, &mut time_out)
                .expect("fft inverse");
            // realfft's inverse is unnormalized: scale by 1/frame.
            let inv_n = 1.0 / frame as f32;
            for i in 0..frame {
                acc[start + i] += time_out[i] * inv_n * self.window[i];
                acc_w[start + i] += self.window[i] * self.window[i];
            }
            self.initialized = true;
            start += hop;
        }
        for i in 0..pcm.len() {
            if acc_w[i] > 1e-6 {
                pcm[i] = acc[i] / acc_w[i];
            }
        }
    }
}

/// Normalize `pcm` to `target_lufs` (integrated, ITU-R BS.1770-4) with DC
/// removal, an energy floor (no boosting of near-silence) and a `max_gain_db`
/// cap. Mirrors the loudness logic in `utils::normalize_loudness` but with an
/// explicit target and gain cap.
pub fn normalize_to_lufs(pcm: &mut [f32], sample_rate: u32, target_lufs: f64, max_gain_db: f64) -> Result<()> {
    if pcm.is_empty() {
        return Ok(());
    }
    // Remove DC offset.
    let mean = (pcm.iter().map(|&s| s as f64).sum::<f64>() / pcm.len() as f64) as f32;
    for s in pcm.iter_mut() {
        *s -= mean;
    }
    let rms = (pcm.iter().map(|&s| (s as f64) * (s as f64)).sum::<f64>() / pcm.len() as f64).sqrt()
        as f32;
    if rms < 2e-3 {
        return Ok(());
    }
    let mut meter =
        ebur128::EbuR128::new(1, sample_rate, ebur128::Mode::I).map_err(xn::Error::wrap)?;
    meter.add_frames_f32(pcm).map_err(xn::Error::wrap)?;
    let input_loudness_db = match meter.loudness_global() {
        Ok(l) if l.is_finite() => l,
        _ => return Ok(()),
    };
    let delta = target_lufs - input_loudness_db;
    let gain_db = delta.clamp(-max_gain_db, max_gain_db);
    let gain = 10f64.powf(gain_db / 20.0) as f32;
    for s in pcm.iter_mut() {
        *s *= gain;
    }
    Ok(())
}

/// Soft limiter: below `threshold` passthrough; above, a smooth tanh knee that
/// asymptotes at `ceiling`. Prevents clipping and inter-sample overs without
/// hard-clip distortion.
pub fn soft_limit(pcm: &mut [f32], threshold: f32, ceiling: f32) {
    let knee = ceiling - threshold;
    for s in pcm.iter_mut() {
        let a = s.abs();
        if a > threshold {
            let y = threshold + knee * ((a - threshold) / knee).tanh();
            *s = s.signum() * y.min(ceiling);
        }
    }
}

/// Full post-processing pipeline: denoise -> normalize -> limit.
pub struct PostProcess {
    pub gate: SpectralGate,
    pub target_lufs: f64,
    pub max_gain_db: f64,
    pub limit_threshold: f32,
    pub limit_ceiling: f32,
}

/// Server-facing configuration.
#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub struct PostProcessConfig {
    pub enabled: bool,
    pub target_lufs: f64,
    pub max_gain_db: f64,
}

impl PostProcessConfig {
    pub fn disabled() -> Self {
        Self { enabled: false, target_lufs: -18.0, max_gain_db: 24.0 }
    }
}

impl PostProcess {
    pub fn new(target_lufs: f64, max_gain_db: f64) -> Self {
        Self {
            gate: SpectralGate::new(2048),
            target_lufs,
            max_gain_db,
            limit_threshold: 0.95,
            limit_ceiling: 0.999,
        }
    }

    pub fn from_config(cfg: &PostProcessConfig) -> Self {
        Self::new(cfg.target_lufs, cfg.max_gain_db)
    }

    pub fn process(&mut self, pcm: &mut [f32], sample_rate: u32) -> Result<()> {
        self.gate.process(pcm);
        normalize_to_lufs(pcm, sample_rate, self.target_lufs, self.max_gain_db)?;
        soft_limit(pcm, self.limit_threshold, self.limit_ceiling);
        Ok(())
    }
}
