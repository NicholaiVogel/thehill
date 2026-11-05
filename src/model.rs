//! The difficulty model: a small MLP that predicts, for every keystroke in context,
//! how long it will take and how likely it is to be wrong.
//!
//! Hand-rolled (forward, backprop, Adam) because the network is tiny and a framework
//! would cost more in build time and dependencies than the ~200 lines it replaces.

use crate::typing::{Event, Key};
use rand::{Rng, SeedableRng, rngs::StdRng, seq::SliceRandom};
use std::collections::HashSet;
use std::path::Path;

/// Token 0 is "nothing before this" (run start); 1..=95 are ASCII 32..=126 with letters lowercased.
pub const VOCAB: usize = 96;
const EMB: usize = 16;
const DENSE: usize = 6;
const IN: usize = 3 * EMB + DENSE;
const HIDDEN: usize = 64;
const OUT: usize = 2;

const O_EMB: usize = 0;
const O_W1: usize = O_EMB + VOCAB * EMB;
const O_B1: usize = O_W1 + IN * HIDDEN;
const O_W2: usize = O_B1 + HIDDEN;
const O_B2: usize = O_W2 + HIDDEN * HIDDEN;
const O_W3: usize = O_B2 + HIDDEN;
const O_B3: usize = O_W3 + HIDDEN * OUT;
const N_PARAMS: usize = O_B3 + OUT;

const MAX_INTERVAL_MS: f64 = 1500.0;
const RECENCY_HALF_LIFE: f64 = 40.0;
const EXPLORE_MS: f32 = 80.0;
const DEFAULT_ERROR_MS: f32 = 600.0;
/// v3 has v2's layout; v2 files were calibrated on all runs and get recalibrated.
const MAGIC: &[u8] = b"SSPH\x03";
const MIN_CALIBRATION_RUNS: usize = 8;
/// Shorter than training's: the exchange rate should follow how you type now.
const CALIBRATION_HALF_LIFE: f64 = 10.0;
const SHIFTED: &str = "~!@#$%^&*()_+{}|:\"<>?";

pub fn token(c: char) -> usize {
    match c {
        ' '..='~' => c.to_ascii_lowercase() as usize - 31,
        _ => 0,
    }
}

/// The model input for keystroke `idx` of `text`: the two chars before it, the char itself,
/// whether it needs shift, and where it sits in its word.
#[derive(Clone, Copy, Debug)]
pub struct Feat {
    tokens: [usize; 3],
    dense: [f32; DENSE],
}

impl Feat {
    pub fn at(text: &[char], idx: usize) -> Feat {
        let c = text[idx];
        let before = |n: usize| if idx >= n { token(text[idx - n]) } else { 0 };
        let mut dense = [0.0; DENSE];
        dense[0] = f32::from(c.is_ascii_uppercase() || SHIFTED.contains(c));
        let slot = if c == ' ' {
            1
        } else {
            let in_word = text[..idx].iter().rev().take_while(|&&p| p != ' ').count();
            2 + in_word.min(3)
        };
        dense[slot] = 1.0;
        Feat { tokens: [before(2), before(1), token(c)], dense }
    }
}

pub struct Sample {
    feat: Feat,
    /// ln(interval ms); `None` when the interval is unusable (run start, pause).
    log_ms: Option<f32>,
    error: bool,
    weight: f32,
}

/// One completed run as stored: the full text, every keypress, and the scored wpm.
pub struct RunLog {
    pub text: Vec<char>,
    pub events: Vec<Event>,
    pub wpm: f64,
    /// Played against the hill's targeted text rather than plain common words.
    pub hill: bool,
}

impl RunLog {
    /// The part of the text the user actually got to.
    pub fn reached(&self) -> &[char] {
        let end = self.events.iter().map(|e| e.idx + 1).max().unwrap_or(0);
        &self.text[..end.min(self.text.len())]
    }
}

#[derive(Clone)]
pub struct Model {
    params: Vec<f32>,
    /// What an expected error costs in scored time, fitted so predicted wpm matches the
    /// wpm you actually got (backspacing, recovering and uncorrected words all included).
    pub error_ms: f32,
    /// Mean over median interval: the latency head predicts typical (log-space) times, but
    /// wpm is driven by the mean, which hesitations pull up.
    pub ms_scale: f32,
    /// Training samples seen per (previous char, char) pair, for the exploration bonus.
    seen: Vec<u32>,
}

struct Acts {
    x: [f32; IN],
    z1: [f32; HIDDEN],
    a1: [f32; HIDDEN],
    z2: [f32; HIDDEN],
    a2: [f32; HIDDEN],
    out: [f32; OUT],
}

impl Acts {
    fn new() -> Acts {
        Acts {
            x: [0.0; IN],
            z1: [0.0; HIDDEN],
            a1: [0.0; HIDDEN],
            z2: [0.0; HIDDEN],
            a2: [0.0; HIDDEN],
            out: [0.0; OUT],
        }
    }
}

fn silu(z: f32) -> f32 {
    z / (1.0 + (-z).exp())
}

fn silu_grad(z: f32) -> f32 {
    let s = 1.0 / (1.0 + (-z).exp());
    s * (1.0 + z * (1.0 - s))
}

fn sigmoid(z: f32) -> f32 {
    1.0 / (1.0 + (-z).exp())
}

fn dense_forward(w: &[f32], b: &[f32], x: &[f32], z: &mut [f32]) {
    z.copy_from_slice(b);
    for (i, &xi) in x.iter().enumerate() {
        if xi != 0.0 {
            let row = &w[i * z.len()..(i + 1) * z.len()];
            z.iter_mut().zip(row).for_each(|(zj, wij)| *zj += xi * wij);
        }
    }
}

/// Accumulates weight/bias grads for z = xW + b and returns dL/dx.
fn dense_backward(w: &[f32], gw: &mut [f32], gb: &mut [f32], x: &[f32], dz: &[f32], dx: &mut [f32]) {
    let n = dz.len();
    gb.iter_mut().zip(dz).for_each(|(g, d)| *g += d);
    for (i, &xi) in x.iter().enumerate() {
        let row = &w[i * n..(i + 1) * n];
        let grow = &mut gw[i * n..(i + 1) * n];
        let mut acc = 0.0;
        for j in 0..n {
            grow[j] += xi * dz[j];
            acc += row[j] * dz[j];
        }
        dx[i] = acc;
    }
}

impl Model {
    pub fn new(seed: u64) -> Model {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut params = vec![0.0; N_PARAMS];
        let mut init = |range: std::ops::Range<usize>, limit: f32| {
            params[range].iter_mut().for_each(|p| *p = rng.random_range(-limit..limit));
        };
        let xavier = |fan_in: usize, fan_out: usize| (6.0 / (fan_in + fan_out) as f32).sqrt();
        init(O_EMB..O_W1, 0.1);
        init(O_W1..O_B1, xavier(IN, HIDDEN));
        init(O_W2..O_B2, xavier(HIDDEN, HIDDEN));
        init(O_W3..O_B3, xavier(HIDDEN, OUT));
        params[O_B3] = 150f32.ln();
        params[O_B3 + 1] = (0.05f32 / 0.95).ln();
        Model { params, error_ms: DEFAULT_ERROR_MS, ms_scale: 1.0, seen: vec![0; VOCAB * VOCAB] }
    }

    fn forward(&self, f: &Feat, a: &mut Acts) {
        let p = &self.params;
        for (k, &t) in f.tokens.iter().enumerate() {
            a.x[k * EMB..(k + 1) * EMB].copy_from_slice(&p[O_EMB + t * EMB..O_EMB + (t + 1) * EMB]);
        }
        a.x[3 * EMB..].copy_from_slice(&f.dense);
        dense_forward(&p[O_W1..O_B1], &p[O_B1..O_W2], &a.x, &mut a.z1);
        a.z1.iter().zip(a.a1.iter_mut()).for_each(|(z, a)| *a = silu(*z));
        dense_forward(&p[O_W2..O_B2], &p[O_B2..O_W3], &a.a1, &mut a.z2);
        a.z2.iter().zip(a.a2.iter_mut()).for_each(|(z, a)| *a = silu(*z));
        dense_forward(&p[O_W3..O_B3], &p[O_B3..], &a.a2, &mut a.out);
    }

    /// Predicted (interval ms, error probability) for one keystroke.
    pub fn predict(&self, f: &Feat) -> (f32, f32) {
        let mut a = Acts::new();
        self.forward(f, &mut a);
        (a.out[0].exp(), sigmoid(a.out[1]))
    }

    /// Expected time cost of a keystroke in ms, errors included.
    pub fn cost(&self, f: &Feat) -> f32 {
        let (ms, p_err) = self.predict(f);
        ms * self.ms_scale + p_err * self.error_ms
    }

    /// Optimism bonus for transitions the model has little data on.
    pub fn bonus(&self, f: &Feat) -> f32 {
        let n = self.seen[f.tokens[1] * VOCAB + f.tokens[2]];
        EXPLORE_MS / (1.0 + n as f32).sqrt()
    }

    /// Training samples seen for keystrokes of `c`, in any context.
    pub fn samples_for(&self, c: char) -> u32 {
        let t = token(c);
        (0..VOCAB).map(|prev| self.seen[prev * VOCAB + t]).sum()
    }

    /// Cost of every keystroke in `text`.
    pub fn costs(&self, text: &[char]) -> Vec<f32> {
        (0..text.len()).map(|i| self.cost(&Feat::at(text, i))).collect()
    }

    /// Words per minute the model expects on `text`, ignoring the untimed first keystroke.
    pub fn predicted_wpm(&self, text: &[char]) -> f64 {
        wpm_from_costs(&self.costs(text))
    }

    fn backward(&self, f: &Feat, a: &Acts, d_out: [f32; OUT], g: &mut [f32]) {
        let p = &self.params;
        let (mut d_a2, mut d_a1, mut d_x) = ([0.0; HIDDEN], [0.0; HIDDEN], [0.0; IN]);
        {
            let (gw, gb) = g[O_W3..].split_at_mut(O_B3 - O_W3);
            dense_backward(&p[O_W3..O_B3], gw, gb, &a.a2, &d_out, &mut d_a2);
        }
        let d_z2: [f32; HIDDEN] = std::array::from_fn(|j| d_a2[j] * silu_grad(a.z2[j]));
        {
            let (gw, gb) = g[O_W2..O_W3].split_at_mut(O_B2 - O_W2);
            dense_backward(&p[O_W2..O_B2], gw, gb, &a.a1, &d_z2, &mut d_a1);
        }
        let d_z1: [f32; HIDDEN] = std::array::from_fn(|j| d_a1[j] * silu_grad(a.z1[j]));
        {
            let (gw, gb) = g[O_W1..O_W2].split_at_mut(O_B1 - O_W1);
            dense_backward(&p[O_W1..O_B1], gw, gb, &a.x, &d_z1, &mut d_x);
        }
        for (k, &t) in f.tokens.iter().enumerate() {
            let row = &mut g[O_EMB + t * EMB..O_EMB + (t + 1) * EMB];
            row.iter_mut().zip(&d_x[k * EMB..(k + 1) * EMB]).for_each(|(g, d)| *g += d);
        }
    }

    /// Weighted loss and its gradient for one sample.
    fn sample_grad(&self, s: &Sample, a: &mut Acts, g: &mut [f32]) -> f32 {
        self.forward(&s.feat, a);
        let mut loss = 0.0;
        let mut d = [0.0; OUT];
        if let Some(y) = s.log_ms {
            let diff = a.out[0] - y;
            loss += 0.5 * diff * diff;
            d[0] = diff;
        }
        let p = sigmoid(a.out[1]);
        let e = f32::from(s.error);
        loss -= e * p.max(1e-7).ln() + (1.0 - e) * (1.0 - p).max(1e-7).ln();
        d[1] = p - e;
        self.backward(&s.feat, a, d.map(|v| v * s.weight), g);
        loss * s.weight
    }

    /// Trains on the logged runs (oldest first), warm-starting from the current weights.
    /// Returns the final epoch's mean loss.
    pub fn fit(&mut self, runs: &[RunLog], epochs: usize, seed: u64) -> f32 {
        let mut samples = Vec::new();
        for (age, run) in runs.iter().rev().enumerate() {
            extract(&run.text, &run.events, recency(age), &mut samples);
        }
        if samples.is_empty() {
            return 0.0;
        }
        self.seen.fill(0);
        for s in &samples {
            self.seen[s.feat.tokens[1] * VOCAB + s.feat.tokens[2]] += 1;
        }

        let mut rng = StdRng::seed_from_u64(seed);
        let mut adam = Adam::new(N_PARAMS);
        let mut grad = vec![0.0; N_PARAMS];
        let mut acts = Acts::new();
        let mut last_loss = 0.0;
        for _ in 0..epochs {
            samples.shuffle(&mut rng);
            let (mut total, mut weight) = (0.0, 0.0);
            for batch in samples.chunks(256) {
                grad.fill(0.0);
                let w: f32 = batch.iter().map(|s| s.weight).sum();
                for s in batch {
                    total += self.sample_grad(s, &mut acts, &mut grad);
                }
                weight += w;
                grad.iter_mut().for_each(|g| *g /= w);
                adam.step(&mut self.params, &grad);
            }
            last_loss = total / weight;
        }
        self.calibrate_cost(runs);
        last_loss
    }

    /// Fits `ms_scale` and `error_ms` so that predicted wpm matches scored wpm on the log.
    /// The heads learn *which* keystrokes are slow or error-prone; this sets the exchange
    /// rate into wpm. Without it, errors were charged their full recovery time on top of
    /// the keystrokes typed during that recovery, and predictions ran ~30% low.
    ///
    /// Calibrates on hill runs when there are enough. The hill picks words the model thinks
    /// you'll fumble, so on its text the error head runs high (on your log: 7-9% predicted,
    /// 2-4% actual). Fitting on plain-word runs too averaged that bias away instead of
    /// correcting it where the prediction is shown.
    pub fn calibrate_cost(&mut self, runs: &[RunLog]) {
        let hill: Vec<&RunLog> = runs.iter().filter(|r| r.hill).collect();
        let runs: Vec<&RunLog> = if hill.len() >= MIN_CALIBRATION_RUNS { hill } else { runs.iter().collect() };
        let (mut actual, mut predicted) = (0.0, 0.0);
        let mut per_run = Vec::new();
        for (age, run) in runs.iter().rev().enumerate() {
            let w = 0.5f64.powf(age as f64 / CALIBRATION_HALF_LIFE);
            let mut samples = Vec::new();
            extract(&run.text, &run.events, 1.0, &mut samples);
            for s in &samples {
                if let Some(y) = s.log_ms {
                    actual += w * (y as f64).exp();
                    predicted += w * self.predict(&s.feat).0 as f64;
                }
            }
            let text = run.reached();
            if run.wpm > 0.0 && text.len() >= 10 {
                let preds: Vec<(f32, f32)> = (1..text.len()).map(|i| self.predict(&Feat::at(text, i))).collect();
                let n = preds.len() as f64;
                let ms = preds.iter().map(|p| p.0 as f64).sum::<f64>() / n;
                let p_err = preds.iter().map(|p| p.1 as f64).sum::<f64>() / n;
                per_run.push((w, 12000.0 / run.wpm, ms, p_err));
            }
        }
        if predicted > 0.0 {
            self.ms_scale = (actual / predicted).clamp(0.5, 2.0) as f32;
        }
        // Least squares for the one remaining unknown: target = scale·ms + error_ms·p_err.
        let scale = self.ms_scale as f64;
        let num: f64 = per_run.iter().map(|(w, t, ms, p)| w * (t - scale * ms) * p).sum();
        let den: f64 = per_run.iter().map(|(w, _, _, p)| w * p * p).sum();
        if den > 1e-9 {
            self.error_ms = (num / den).clamp(0.0, 3000.0) as f32;
        }
        // Errors cost nothing and it's still slow: the latency scale has to take up the rest.
        if self.error_ms == 0.0 {
            let num: f64 = per_run.iter().map(|(w, t, ms, _)| w * t * ms).sum();
            let den: f64 = per_run.iter().map(|(w, _, ms, _)| w * ms * ms).sum();
            if den > 1e-9 {
                self.ms_scale = (num / den).clamp(0.5, 2.0) as f32;
            }
        }
    }

    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        let mut bytes = MAGIC.to_vec();
        bytes.extend(self.error_ms.to_le_bytes());
        bytes.extend(self.ms_scale.to_le_bytes());
        self.params.iter().for_each(|p| bytes.extend(p.to_le_bytes()));
        self.seen.iter().for_each(|n| bytes.extend(n.to_le_bytes()));
        let tmp = path.with_extension("tmp");
        std::fs::write(&tmp, bytes)?;
        std::fs::rename(tmp, path)
    }

    /// Loads a model. The second value is false for the old format, whose cost calibration
    /// is wrong; callers recalibrate those with `calibrate_cost` and save them again.
    pub fn load_versioned(path: &Path) -> std::io::Result<(Model, bool)> {
        let bytes = std::fs::read(path)?;
        let body = 4 * N_PARAMS + 4 * VOCAB * VOCAB;
        let v2 = bytes.starts_with(b"SSPH\x02") && bytes.len() == 5 + 8 + body;
        let current = bytes.starts_with(MAGIC) && bytes.len() == 5 + 8 + body;
        let old = bytes.starts_with(b"SSPH\x01") && bytes.len() == 5 + 4 + body;
        if !current && !old && !v2 {
            return Err(std::io::Error::other(format!("{} is not a sisyphus model", path.display())));
        }
        let f32_at = |i: usize| f32::from_le_bytes(bytes[i..i + 4].try_into().unwrap());
        let words = |b: &[u8]| -> Vec<[u8; 4]> { b.chunks_exact(4).map(|c| c.try_into().unwrap()).collect() };
        let start = if old { 9 } else { 13 };
        let (params, seen) = bytes[start..].split_at(4 * N_PARAMS);
        let model = Model {
            params: words(params).into_iter().map(f32::from_le_bytes).collect(),
            error_ms: f32_at(5),
            ms_scale: if old { 1.0 } else { f32_at(9) },
            seen: words(seen).into_iter().map(u32::from_le_bytes).collect(),
        };
        let finite = model.error_ms.is_finite() && model.ms_scale.is_finite();
        if !finite || !model.params.iter().all(|p| p.is_finite()) {
            return Err(std::io::Error::other(format!("{} has non-finite weights", path.display())));
        }
        Ok((model, current))
    }

    pub fn load(path: &Path) -> std::io::Result<Model> {
        Self::load_versioned(path).map(|(m, _)| m)
    }
}

/// How a model's predictions line up with one logged run, for `--calibrate`.
pub struct Calibration {
    /// Mean predicted and actual clean interval (ms), over the same keystrokes.
    pub predicted_ms: f64,
    pub actual_ms: f64,
    /// Mean of ln(actual) - ln(predicted): the median-level bias.
    pub log_bias: f64,
    pub predicted_err: f64,
    pub actual_err: f64,
}

impl Model {
    pub fn calibrate(&self, run: &RunLog) -> Calibration {
        let mut samples = Vec::new();
        extract(&run.text, &run.events, 1.0, &mut samples);
        let timed: Vec<(f64, f64)> = samples
            .iter()
            .filter_map(|s| s.log_ms.map(|y| (self.predict(&s.feat).0 as f64, (y as f64).exp())))
            .collect();
        let n = timed.len().max(1) as f64;
        let m = samples.len().max(1) as f64;
        Calibration {
            predicted_ms: timed.iter().map(|t| t.0).sum::<f64>() / n * self.ms_scale as f64,
            actual_ms: timed.iter().map(|t| t.1).sum::<f64>() / n,
            log_bias: timed.iter().map(|t| t.1.ln() - t.0.ln()).sum::<f64>() / n,
            predicted_err: samples.iter().map(|s| self.predict(&s.feat).1 as f64).sum::<f64>() / m,
            actual_err: samples.iter().filter(|s| s.error).count() as f64 / m,
        }
    }
}

fn recency(age: usize) -> f32 {
    0.5f64.powf(age as f64 / RECENCY_HALF_LIFE) as f32
}

pub fn wpm_from_costs(costs: &[f32]) -> f64 {
    if costs.len() < 2 {
        return 0.0;
    }
    let mean = costs[1..].iter().map(|&c| c as f64).sum::<f64>() / (costs.len() - 1) as f64;
    12000.0 / mean
}

/// Turns one run's keypresses into training samples: the first attempt at each text position.
/// Intervals after an error or correction are dropped, since they measure recovery, not typing.
fn extract(text: &[char], events: &[Event], weight: f32, out: &mut Vec<Sample>) {
    let mut attempted = HashSet::new();
    let mut prev: Option<(f64, bool)> = None;
    for e in events {
        let clean = matches!(e.key, Key::Char(_)) && e.correct;
        if let (Key::Char(_), Some(_)) = (e.key, e.expected)
            && attempted.insert(e.idx)
        {
            let log_ms = match prev {
                Some((t, true)) if e.t - t > 0.0 && e.t - t <= MAX_INTERVAL_MS => Some((e.t - t).ln() as f32),
                _ => None,
            };
            out.push(Sample { feat: Feat::at(text, e.idx), log_ms, error: !e.correct, weight });
        }
        prev = Some((e.t, clean));
    }
}

struct Adam {
    m: Vec<f32>,
    v: Vec<f32>,
    t: i32,
}

impl Adam {
    const LR: f32 = 3e-3;
    const B1: f32 = 0.9;
    const B2: f32 = 0.999;

    fn new(n: usize) -> Adam {
        Adam { m: vec![0.0; n], v: vec![0.0; n], t: 0 }
    }

    fn step(&mut self, params: &mut [f32], grad: &[f32]) {
        self.t += 1;
        let c1 = 1.0 - Self::B1.powi(self.t);
        let c2 = 1.0 - Self::B2.powi(self.t);
        for i in 0..params.len() {
            self.m[i] = Self::B1 * self.m[i] + (1.0 - Self::B1) * grad[i];
            self.v[i] = Self::B2 * self.v[i] + (1.0 - Self::B2) * grad[i] * grad[i];
            params[i] -= Self::LR * (self.m[i] / c1) / ((self.v[i] / c2).sqrt() + 1e-8);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chars(s: &str) -> Vec<char> {
        s.chars().collect()
    }

    #[test]
    fn features_encode_context_shift_and_position() {
        let text = chars("ab Cd");
        let f = Feat::at(&text, 3);
        assert_eq!(f.tokens, [token('b'), token(' '), token('c')]);
        assert_eq!(f.dense[0], 1.0, "capital needs shift");
        assert_eq!(f.dense[2], 1.0, "first char of a word");
        assert_eq!(Feat::at(&text, 2).dense[1], 1.0, "space slot");
        assert_eq!(Feat::at(&text, 0).tokens[..2], [0, 0]);
    }

    #[test]
    fn gradients_match_finite_differences() {
        let text = chars("the cat");
        let mut model = Model::new(3);
        let sample = Sample { feat: Feat::at(&text, 5), log_ms: Some(4.8), error: true, weight: 0.7 };
        let mut grad = vec![0.0; N_PARAMS];
        let mut acts = Acts::new();
        model.sample_grad(&sample, &mut acts, &mut grad);
        let probes = [O_EMB + token('a') * EMB + 2, O_W1 + 17, O_B1 + 5, O_W2 + 300, O_W3 + 7, O_B3, O_B3 + 1];
        for i in probes {
            let h = 1e-2;
            let orig = model.params[i];
            let mut loss_at = |v: f32| {
                model.params[i] = v;
                model.sample_grad(&sample, &mut Acts::new(), &mut vec![0.0; N_PARAMS])
            };
            let numeric = (loss_at(orig + h) - loss_at(orig - h)) / (2.0 * h);
            model.params[i] = orig;
            assert!((numeric - grad[i]).abs() < 2e-3 + 2e-2 * numeric.abs(), "param {i}: {numeric} vs {}", grad[i]);
        }
    }

    #[test]
    fn save_and_load_give_identical_predictions() {
        let dir = std::env::temp_dir().join(format!("sisyphus-model-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut model = Model::new(9);
        model.error_ms = 432.0;
        model.seen[5] = 7;
        let path = dir.join("m.bin");
        model.save(&path).unwrap();
        let loaded = Model::load(&path).unwrap();
        let text = chars("Hello, world 42");
        assert_eq!(model.costs(&text), loaded.costs(&text));
        assert_eq!(loaded.seen[5], 7);

        model.error_ms = f32::NAN;
        model.save(&path).unwrap();
        assert!(Model::load(&path).is_err(), "non-finite models are rejected");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn extract_takes_first_attempts() {
        let text = chars("ab");
        let ev = |t: f64, idx: usize, key: char, correct: bool| Event {
            t,
            idx,
            expected: text.get(idx).copied(),
            key: Key::Char(key),
            correct,
        };
        let events = vec![
            ev(0.0, 0, 'a', true),
            ev(100.0, 1, 'x', false),
            Event { t: 300.0, idx: 1, expected: Some('b'), key: Key::Backspace, correct: false },
            ev(400.0, 1, 'b', true),
        ];
        let mut out = Vec::new();
        extract(&text, &events, 1.0, &mut out);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].log_ms, None, "first keypress has no interval");
        assert!(out[1].error);
        assert!((out[1].log_ms.unwrap() - 100f32.ln()).abs() < 1e-6);
    }

    #[test]
    fn fit_learns_a_slow_bigram() {
        let text = chars("the as the as the as the as");
        let events: Vec<Event> = {
            let mut t = 0.0;
            text.iter()
                .enumerate()
                .map(|(i, &c)| {
                    t += if i > 0 && text[i - 1] == 't' { 300.0 } else { 90.0 };
                    Event { t, idx: i, expected: Some(c), key: Key::Char(c), correct: true }
                })
                .collect()
        };
        let runs: Vec<RunLog> =
            (0..20).map(|_| RunLog { text: text.clone(), events: events.clone(), wpm: 100.0, hill: true }).collect();
        let mut model = Model::new(1);
        model.fit(&runs, 30, 1);
        let th = model.predict(&Feat::at(&text, 1)).0;
        let as_ = model.predict(&Feat::at(&text, 5)).0;
        assert!(th > 2.0 * as_, "t→h {th} vs a→s {as_}");
    }

    #[test]
    fn predicted_wpm_is_calibrated_to_scored_wpm() {
        // Clean typing at a known pace, plus runs whose score is dragged down by errors.
        let text = chars("the as the as the as the as");
        let mut t = 0.0;
        let events: Vec<Event> = text
            .iter()
            .enumerate()
            .map(|(i, &c)| {
                t += if i % 3 == 0 { 220.0 } else { 80.0 };
                Event { t, idx: i, expected: Some(c), key: Key::Char(c), correct: i % 7 != 3 }
            })
            .collect();
        let clean_wpm = 12000.0 / (t / (text.len() - 1) as f64);
        let scored = clean_wpm * 0.8;
        let runs: Vec<RunLog> =
            (0..30).map(|_| RunLog { text: text.clone(), events: events.clone(), wpm: scored, hill: true }).collect();
        let mut model = Model::new(2);
        model.fit(&runs, 30, 2);
        let predicted = model.predicted_wpm(&text);
        assert!((predicted / scored - 1.0).abs() < 0.05, "predicted {predicted:.1}, scored {scored:.1}");
    }

    #[test]
    fn old_model_files_load_as_uncalibrated() {
        let dir = std::env::temp_dir().join(format!("sisyphus-v1-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("old.bin");
        let model = Model::new(1);
        let mut bytes = b"SSPH\x01".to_vec();
        bytes.extend(model.error_ms.to_le_bytes());
        model.params.iter().for_each(|p| bytes.extend(p.to_le_bytes()));
        model.seen.iter().for_each(|n| bytes.extend(n.to_le_bytes()));
        std::fs::write(&path, bytes).unwrap();
        let (old, current) = Model::load_versioned(&path).unwrap();
        assert!(!current && old.ms_scale == 1.0);
        old.save(&path).unwrap();
        assert!(Model::load_versioned(&path).unwrap().1);
        std::fs::remove_dir_all(dir).unwrap();
    }
}
