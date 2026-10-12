//! Suppression of the 6-11 kHz noise bursts that the decoder emits for
//! fricative/onset transitions.
//!
//! Measured on the served output (24 kHz, 2026-04 codec, f32 and q8_0
//! weights alike): the decoder renders fricatives and the transition into
//! speech as isolated bursts of 6-11 kHz noise, 20-150 ms long, at
//! -19..-32 dBFS, i.e. 8-30 dB *above* the concurrent 300-4000 Hz speech band.
//! Paperradio's listener audit on episode 37155005 counts 86-88 of them per
//! episode (worst at 6.400 s, the onset of the second line). They are not a
//! quantization artifact (identical with f32 weights) and not chunking-related
//! (they occur in 15-token utterances).
//!
//! The filter is a soft, bounded duck of the 6-11 kHz band, applied per STFT
//! frame and only to frames whose 6-11 kHz band is itself above the burst
//! threshold *and* ≥8 dB over that frame's 300-4000 Hz band:
//!
//! * everything below ~5.2 kHz passes at unity gain (the 300-4000 Hz speech
//!   band and the 4-6 kHz lower half of the fricative spectrum are untouched);
//! * the duck is bounded (-14 dB) and the affected band is the burst band only,
//!   so a fricative keeps its body, level and duration: the burst loses its
//!   level, not the phoneme;
//! * frames that are not burst frames are reconstructed sample-exactly (Hann
//!   window, 75% overlap, weighted-overlap-add normalisation).
//!
//! Cost: one 512-point real FFT and one inverse per 128 samples -> ~10 ms of
//! CPU per 10 s utterance on one core (measured; `cargo run --release
//! --example pocket_tts` scale).

use realfft::RealFftPlanner;
use realfft::num_complex::Complex;
use std::sync::Arc;

/// Tunables for [`HfBurstSuppressor`].
#[derive(Clone, Debug)]
pub struct HfBurstConfig {
    /// STFT frame length in samples (513-point spectrum).
    pub frame: usize,
    /// STFT hop in samples (75% overlap).
    pub hop: usize,
    /// Burst detection band, lower edge (Hz).
    pub detect_lo_hz: f32,
    /// Burst detection band, upper edge (Hz).
    pub detect_hi_hz: f32,
    /// Speech band used as the detector reference, lower edge (Hz).
    pub speech_lo_hz: f32,
    /// Speech band used as the detector reference, upper edge (Hz).
    pub speech_hi_hz: f32,
    /// Minimum burst-band level for a frame to count as a burst (dBFS).
    pub min_burst_dbfs: f32,
    /// Minimum burst-band excess over the speech band (dB).
    pub min_excess_db: f32,
    /// Duck depth applied inside the affected band (dB, positive).
    pub duck_db: f32,
    /// Lower edge of the fully ducked band (Hz).
    pub gate_lo_hz: f32,
    /// Upper edge of the fully ducked band (Hz).
    pub gate_hi_hz: f32,
    /// Width of the raised-cosine transitions at both band edges (Hz).
    pub gate_ramp_hz: f32,
    /// Frames held ducked after the last detected burst frame.
    pub release_frames: usize,
}

impl Default for HfBurstConfig {
    fn default() -> Self {
        Self {
            frame: 512,
            hop: 128,
            detect_lo_hz: 6000.0,
            detect_hi_hz: 11000.0,
            speech_lo_hz: 300.0,
            speech_hi_hz: 4000.0,
            min_burst_dbfs: -35.0,
            min_excess_db: 8.0,
            duck_db: 14.0,
            gate_lo_hz: 5800.0,
            gate_hi_hz: 11500.0,
            gate_ramp_hz: 700.0,
            release_frames: 4,
        }
    }
}

/// Counters, useful for logs and tests.
#[derive(Clone, Copy, Debug, Default)]
pub struct HfBurstStats {
    /// Frames analysed.
    pub frames: u64,
    /// Frames classified as bursts.
    pub burst_frames: u64,
    /// Frames attenuated (burst frames plus releases).
    pub ducked_frames: u64,
}

/// Streaming 6-11 kHz burst suppressor.
///
/// Feed decoded PCM with [`process`](Self::process); the returned samples lag
/// the input by one frame (512 samples, 21 ms at 24 kHz) because a sample is
/// only final once every overlapping STFT frame has been seen. Call
/// [`flush`](Self::flush) at the end of an utterance to emit the tail; the
/// total number of output samples equals the number of input samples.
pub struct HfBurstSuppressor {
    cfg: HfBurstConfig,
    fwd: Arc<dyn realfft::RealToComplex<f32>>,
    inv: Arc<dyn realfft::ComplexToReal<f32>>,
    sample_rate: f32,
    window: Vec<f32>,
    window_sum: f32,
    frame_in: Vec<f32>,
    spec: Vec<Complex<f32>>,
    time_out: Vec<f32>,
    /// Weighted overlap-add accumulators; `acc[i]` is the internal sample
    /// `acc_base + i`.
    acc: Vec<f32>,
    wsum: Vec<f32>,
    acc_base: usize,
    /// Pending input; `buf[0]` is internal sample `buf_start`.
    buf: Vec<f32>,
    buf_start: usize,
    /// Internal index of the next STFT frame to process.
    next_frame: usize,
    /// Real (caller-visible) samples fed so far. Internal index `i` is real
    /// sample `i - frame` (the first `frame` internal samples are the implicit
    /// silence history).
    fed: usize,
    hold: usize,
    stats: HfBurstStats,
}

impl HfBurstSuppressor {
    /// Create a suppressor with the default tunables for `sample_rate`.
    pub fn new(sample_rate: u32) -> Self {
        Self::with_config(sample_rate, HfBurstConfig::default())
    }

    /// Create a suppressor with explicit tunables.
    pub fn with_config(sample_rate: u32, cfg: HfBurstConfig) -> Self {
        let frame = cfg.frame;
        let mut planner = RealFftPlanner::<f32>::new();
        let fwd = planner.plan_fft_forward(frame);
        let inv = planner.plan_fft_inverse(frame);
        // Periodic Hann window; with hop = frame/4 the squared-window
        // overlap-add sums to a constant, so unattenuated frames are
        // reconstructed exactly.
        let window: Vec<f32> = (0..frame)
            .map(|i| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / frame as f32).cos())
            .collect();
        let window_sum: f32 = window.iter().sum();
        Self {
            frame_in: vec![0.0; frame],
            spec: vec![Complex::new(0.0, 0.0); frame / 2 + 1],
            time_out: vec![0.0; frame],
            acc: vec![0.0; frame],
            wsum: vec![0.0; frame],
            // Implicit silence history so the first real sample is covered by
            // a full set of overlapping frames.
            buf: vec![0.0; frame],
            buf_start: 0,
            next_frame: 0,
            acc_base: 0,
            fed: 0,
            hold: 0,
            stats: HfBurstStats::default(),
            cfg,
            fwd,
            inv,
            sample_rate: sample_rate as f32,
            window,
            window_sum,
        }
    }

    /// Frame counters since construction (or the last [`reset`](Self::reset)).
    pub fn stats(&self) -> HfBurstStats {
        self.stats
    }

    /// Number of output samples currently buffered inside the filter (the
    /// algorithmic latency).
    pub fn latency(&self) -> usize {
        self.cfg.frame - self.cfg.hop
    }

    /// Drop the streaming state, keeping the configuration.
    pub fn reset(&mut self) {
        self.buf.clear();
        self.buf.resize(self.cfg.frame, 0.0);
        self.buf_start = 0;
        self.next_frame = 0;
        self.acc_base = 0;
        self.acc.iter_mut().for_each(|v| *v = 0.0);
        self.wsum.iter_mut().for_each(|v| *v = 0.0);
        self.fed = 0;
        self.hold = 0;
        self.stats = HfBurstStats::default();
    }

    /// Filter `input`, appending the final samples to `output`.
    pub fn process(&mut self, input: &[f32], output: &mut Vec<f32>) {
        self.buf.extend_from_slice(input);
        self.fed += input.len();
        let end = self.real_end();
        while self.next_frame + self.cfg.frame <= end {
            self.process_frame(output);
        }
    }

    /// Emit the last samples; the output length matches the input length.
    pub fn flush(&mut self, output: &mut Vec<f32>) {
        let end = self.real_end();
        // Pad with silence so every frame that overlaps a real sample can run.
        while self.buf_start + self.buf.len() < end + self.cfg.frame {
            self.buf.push(0.0);
        }
        while self.next_frame < end {
            self.process_frame(output);
        }
        self.emit_until(output, end);
    }

    /// Internal sample index one past the last real (caller-supplied) sample.
    fn real_end(&self) -> usize {
        self.fed + self.cfg.frame
    }

    /// Push every sample that is final (`< stop`, and all overlapping frames
    /// seen) to `output`, then drop it from the accumulators.
    fn emit_until(&mut self, output: &mut Vec<f32>, stop: usize) {
        let frame = self.cfg.frame;
        let stop = stop.min(self.acc_base + frame);
        let from = self.acc_base;
        if stop <= from {
            return;
        }
        for i in 0..(stop - from) {
            let w = self.wsum[i];
            let v = if w > 1e-9 { self.acc[i] / w } else { 0.0 };
            let internal = from + i;
            if internal >= frame {
                output.push(v);
            }
        }
        let shift = stop - from;
        self.acc.copy_within(shift..frame, 0);
        self.acc[frame - shift..].fill(0.0);
        self.wsum.copy_within(shift..frame, 0);
        self.wsum[frame - shift..].fill(0.0);
        self.acc_base = stop;
    }

    fn process_frame(&mut self, output: &mut Vec<f32>) {
        let frame = self.cfg.frame;
        let hop = self.cfg.hop;
        let f = self.next_frame;
        let off = f - self.buf_start;
        for i in 0..frame {
            self.frame_in[i] = self.buf[off + i] * self.window[i];
        }
        self.fwd.process(&mut self.frame_in, &mut self.spec).expect("realfft forward");

        let bin_hz = self.sample_rate / frame as f32;
        let burst_level = band_level_db(
            &self.spec,
            bin_hz,
            self.cfg.detect_lo_hz,
            self.cfg.detect_hi_hz,
            self.window_sum,
        );
        let speech_level = band_level_db(
            &self.spec,
            bin_hz,
            self.cfg.speech_lo_hz,
            self.cfg.speech_hi_hz,
            self.window_sum,
        );
        let is_burst = burst_level > self.cfg.min_burst_dbfs
            && burst_level - speech_level > self.cfg.min_excess_db;
        if is_burst {
            self.stats.burst_frames += 1;
            self.hold = self.cfg.release_frames;
        } else if self.hold > 0 {
            self.hold -= 1;
        }
        let ducking = is_burst || self.hold > 0;
        self.stats.frames += 1;
        if ducking {
            self.stats.ducked_frames += 1;
            let gain = 10f32.powf(-self.cfg.duck_db / 20.0);
            let cfg = &self.cfg;
            let spec = &mut self.spec;
            for (k, c) in spec.iter_mut().enumerate() {
                *c *= gate_gain(cfg, k as f32 * bin_hz, gain);
            }
        }

        self.inv.process(&mut self.spec, &mut self.time_out).expect("realfft inverse");
        // realfft's inverse is unnormalised: scale by 1/frame.
        let norm = 1.0 / frame as f32;
        for i in 0..frame {
            let w = self.window[i];
            self.acc[i] += self.time_out[i] * norm * w;
            self.wsum[i] += w * w;
        }

        self.next_frame += hop;
        let drop = self.next_frame - self.buf_start;
        self.buf.drain(..drop);
        self.buf_start = self.next_frame;
        // Samples below the start of the next frame are final.
        let stop = self.next_frame.min(self.real_end());
        self.emit_until(output, stop);
    }
}

/// Frequency response of the duck: unity outside the burst band, `gain`
/// between the band edges, raised-cosine transitions at both edges.
fn gate_gain(cfg: &HfBurstConfig, hz: f32, gain: f32) -> f32 {
    let ramp = cfg.gate_ramp_hz.max(1.0);
    if hz <= cfg.gate_lo_hz - ramp || hz >= cfg.gate_hi_hz + ramp {
        return 1.0;
    }
    let t = if hz < cfg.gate_lo_hz {
        (hz - (cfg.gate_lo_hz - ramp)) / ramp
    } else if hz > cfg.gate_hi_hz {
        ((cfg.gate_hi_hz + ramp) - hz) / ramp
    } else {
        1.0
    }
    .clamp(0.0, 1.0);
    let s = 0.5 - 0.5 * (std::f32::consts::PI * t).cos();
    1.0 - (1.0 - gain) * s
}

/// RMS level (dB, full scale = 1.0) of the band `[lo, hi]` of a real FFT
/// magnitude spectrum produced from a `window_sum`-summing window. Matching
/// the analysis convention used for the measurements: a full-scale sine reads
/// 0 dB, so this is the band's time-domain level.
fn band_level_db(spec: &[Complex<f32>], bin_hz: f32, lo: f32, hi: f32, window_sum: f32) -> f32 {
    let mut energy = 0.0f32;
    for (k, c) in spec.iter().enumerate() {
        let hz = k as f32 * bin_hz;
        if hz >= lo && hz <= hi {
            energy += c.re * c.re + c.im * c.im;
        }
    }
    20.0 * (energy.sqrt() * 2.0 / window_sum + 1e-12).log10()
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: u32 = 24000;

    /// Analysis helper for tests: level of a band in a plain signal (no
    /// window), in dBFS (full-scale sine = 0 dB).
    fn band_rms_db(x: &[f32], lo: f32, hi: f32, fft: usize) -> f32 {
        assert!(x.len() >= fft);
        let win: Vec<f32> = (0..fft)
            .map(|i| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / fft as f32).cos())
            .collect();
        let mut planner = RealFftPlanner::<f32>::new();
        let fwd = planner.plan_fft_forward(fft);
        let mut sum = 0.0f32;
        let mut n = 0;
        let mut input = vec![0.0f32; fft];
        let mut spec = fwd.make_output_vec();
        let bin_hz = SR as f32 / fft as f32;
        let mut start = 0;
        while start + fft <= x.len() {
            for i in 0..fft {
                input[i] = x[start + i] * win[i];
            }
            fwd.process(&mut input, &mut spec).unwrap();
            for (k, c) in spec.iter().enumerate() {
                let hz = k as f32 * bin_hz;
                if hz >= lo && hz <= hi {
                    sum += c.re * c.re + c.im * c.im;
                }
            }
            n += 1;
            start += fft / 2;
        }
        let window_sum: f32 = win.iter().sum();
        20.0 * ((sum / n as f32).sqrt() * 2.0 / window_sum + 1e-12).log10()
    }

    /// Speech-like test signal: harmonics of `f0` up to 3.5 kHz, amplitude
    /// `amp`, over `[t0, t1)`.
    fn add_speech(x: &mut [f32], t0: f32, t1: f32, f0: f32, amp: f32) {
        let (i0, i1) = ((t0 * SR as f32) as usize, (t1 * SR as f32) as usize);
        for i in i0..i1.min(x.len()) {
            let t = i as f32 / SR as f32;
            let mut v = 0.0;
            let mut h = 1.0;
            while h * f0 < 3500.0 {
                v += (2.0 * std::f32::consts::PI * h * f0 * t).sin() / h;
                h += 1.0;
            }
            // Slow syllabic amplitude modulation.
            let env = 0.6 + 0.4 * (2.0 * std::f32::consts::PI * 3.0 * (t - t0)).sin();
            x[i] += amp * env * v;
        }
    }

    /// Injected burst: `dur` seconds of 6.5 and 9.5 kHz tones at `amp`,
    /// starting at `t0` (the shape of the decoder's HF bursts: a band of
    /// narrow lines in the 6-11 kHz band, no low-frequency energy).
    fn add_burst(x: &mut [f32], t0: f32, dur: f32, amp: f32) {
        let (i0, i1) = ((t0 * SR as f32) as usize, ((t0 + dur) * SR as f32) as usize);
        for i in i0..i1.min(x.len()) {
            let t = i as f32 / SR as f32;
            let v = (2.0 * std::f32::consts::PI * 6500.0 * t).sin()
                + (2.0 * std::f32::consts::PI * 9500.0 * t).sin();
            x[i] += amp * v;
        }
    }

    fn filter(x: &[f32]) -> Vec<f32> {
        let mut sup = HfBurstSuppressor::new(SR);
        let mut out = Vec::with_capacity(x.len());
        sup.process(x, &mut out);
        sup.flush(&mut out);
        out
    }

    #[test]
    fn injected_burst_is_suppressed_and_speech_is_untouched() {
        let n = SR as usize; // 1 s
        let mut x = vec![0.0f32; n];
        add_speech(&mut x, 0.10, 0.35, 180.0, 0.25);
        add_speech(&mut x, 0.65, 0.90, 200.0, 0.25);
        add_burst(&mut x, 0.45, 0.06, 0.10); // -20 dBFS burst in the gap

        let y = filter(&x);
        assert_eq!(y.len(), x.len(), "filter must preserve the sample count");

        let burst_before = band_rms_db(
            &x[((0.45 * SR as f32) as usize)..((0.51 * SR as f32) as usize)],
            6000.0,
            11000.0,
            512,
        );
        let burst_after = band_rms_db(
            &y[((0.45 * SR as f32) as usize)..((0.51 * SR as f32) as usize)],
            6000.0,
            11000.0,
            512,
        );
        assert!(
            burst_before > -30.0,
            "test fixture: injected burst should be loud, got {burst_before:.1} dBFS"
        );
        assert!(
            burst_after < burst_before - 10.0,
            "burst band must drop by >=10 dB, got {burst_before:.1} -> {burst_after:.1} dBFS"
        );

        // Speech band (300-4000 Hz) is below the ducked band: unchanged.
        for (t0, t1) in [(0.12, 0.33), (0.67, 0.88)] {
            let (i0, i1) = ((t0 * SR as f32) as usize, (t1 * SR as f32) as usize);
            let before = band_rms_db(&x[i0..i1], 300.0, 4000.0, 512);
            let after = band_rms_db(&y[i0..i1], 300.0, 4000.0, 512);
            assert!(
                (after - before).abs() < 0.05,
                "speech band must be unchanged, got {before:.2} -> {after:.2} dB"
            );
        }
    }

    #[test]
    fn speech_only_signal_passes_through() {
        let n = SR as usize;
        let mut x = vec![0.0f32; n];
        add_speech(&mut x, 0.05, 0.45, 180.0, 0.3);
        add_speech(&mut x, 0.55, 0.95, 220.0, 0.3);
        let y = filter(&x);
        assert_eq!(y.len(), x.len());
        let max_diff = x
            .iter()
            .zip(&y)
            .take(n - HfBurstConfig::default().frame)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff < 1e-4, "speech must pass through, max |diff| = {max_diff}");
    }

    #[test]
    fn broadband_hf_with_loud_speech_is_kept() {
        // A fricative-like broadband HF component that does *not* dominate the
        // speech band by 8 dB must not be ducked.
        let n = SR as usize;
        let mut x = vec![0.0f32; n];
        add_speech(&mut x, 0.05, 0.95, 180.0, 0.5);
        let mut seed = 0x1234_5678u32;
        for (i, s) in x.iter_mut().enumerate() {
            // xorshift noise, band limited by construction to the HF region via
            // a 6-11 kHz tone pair at a level 3 dB below the speech band.
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            let noise = (seed as f32 / u32::MAX as f32) * 2.0 - 1.0;
            let t = i as f32 / SR as f32;
            let carrier = 0.5
                * ((2.0 * std::f32::consts::PI * 7000.0 * t).sin()
                    + (2.0 * std::f32::consts::PI * 9000.0 * t).sin());
            *s += 0.12 * (0.7 * carrier + 0.3 * noise);
        }
        let y = filter(&x);
        let (i0, i1) = ((0.2 * SR as f32) as usize, (0.8 * SR as f32) as usize);
        let before = band_rms_db(&x[i0..i1], 6000.0, 11000.0, 512);
        let after = band_rms_db(&y[i0..i1], 6000.0, 11000.0, 512);
        let speech = band_rms_db(&x[i0..i1], 300.0, 4000.0, 512);
        assert!(
            before - speech < HfBurstConfig::default().min_excess_db,
            "test fixture: HF must not be a burst, excess {:.1} dB",
            before - speech
        );
        assert!(
            (after - before).abs() < 0.5,
            "non-burst HF content must be kept, got {before:.2} -> {after:.2} dB"
        );
    }

    #[test]
    fn cost_per_utterance_is_negligible() {
        // 10 s of 24 kHz audio through the filter must cost a small fraction
        // of realtime: one 512-point real FFT and one inverse per 128 samples.
        // The bound is loose enough for debug builds; the printed value is the
        // release-mode cost (run with `--release`).
        let n = 10 * SR as usize;
        let mut x = vec![0.0f32; n];
        add_speech(&mut x, 0.0, 10.0, 180.0, 0.2);
        add_burst(&mut x, 4.0, 0.1, 0.1);
        let mut sup = HfBurstSuppressor::new(SR);
        let mut out = Vec::with_capacity(n);
        let t0 = std::time::Instant::now();
        sup.process(&x, &mut out);
        sup.flush(&mut out);
        let dt = t0.elapsed();
        println!(
            "hf_burst: 10 s of audio in {:.1} ms ({} frames analysed, {} ducked)",
            dt.as_secs_f64() * 1e3,
            sup.stats().frames,
            sup.stats().ducked_frames
        );
        assert_eq!(out.len(), n);
        assert!(
            dt.as_secs_f64() < 0.5,
            "filter must stay well under realtime, took {:.1} ms for 10 s of audio",
            dt.as_secs_f64() * 1e3
        );
    }
}
