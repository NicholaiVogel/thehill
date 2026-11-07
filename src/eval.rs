//! `--evaluate`: replays the log in order, predicting each run before training on it the way
//! the app does, and compares the model at each context length with n-gram baselines.

use std::collections::HashMap;
use std::path::Path;

use anyhow::Result;

use crate::learn::{self, COLD_RUNS, TRAIN_WINDOW};
use crate::model::{CONTEXTS, Feat, Model, RunLog, Sample, extract, recency};
use crate::session::{CORRECTION_RUNS, Mode, committed, correction};
use crate::storage::Store;

/// Runs predicted before the first evaluated one: enough for every predictor to have a start.
const WARMUP: usize = 20;
/// Pseudo-counts pulling an n-gram toward its shorter context.
const SMOOTHING: f64 = 5.0;
/// Runs a predictor's run-level bias is learned over, as the app's correction is.
const BIAS_RUNS: usize = 20;

/// A run's stored mode, amount, and whether it was a boss or against a checkpoint.
type Kind = (String, i64, bool, bool);

/// Something that predicts (ln interval ms, error probability) for a keystroke.
trait Predictor: Send {
    fn name(&self) -> String;
    fn predict(&self, f: &Feat) -> (f64, f64);
    /// Learns from the log up to and including the newest run, oldest first, as the app
    /// does once that many runs are logged.
    fn learn(&mut self, window: &[RunLog], run_count: usize);
    /// The app's model, for its committed wpm prediction; baselines have none.
    fn model(&self) -> Option<&Model> {
        None
    }
}

impl Predictor for Model {
    fn name(&self) -> String {
        format!("model ctx {}", self.context())
    }
    fn predict(&self, f: &Feat) -> (f64, f64) {
        let (ms, p) = Model::predict(self, f);
        ((ms as f64).ln(), p as f64)
    }
    fn learn(&mut self, window: &[RunLog], run_count: usize) {
        *self = learn::train(self.context(), window, run_count as i64);
    }
    fn model(&self) -> Option<&Model> {
        Some(self)
    }
}

/// Recency-weighted means of ln ms and of errors for the last `n` chars, backing off to
/// shorter contexts. Refit from scratch each run over the same window the model sees.
struct NGram {
    n: usize,
    /// Per context length 0..=n: context -> (Σw·ln ms, Σw timed, Σw·err, Σw).
    tables: Vec<HashMap<Vec<usize>, [f64; 4]>>,
}

impl NGram {
    fn new(n: usize) -> NGram {
        NGram { n, tables: vec![HashMap::new(); n + 1] }
    }
}

impl Predictor for NGram {
    fn name(&self) -> String {
        format!("{}-gram", self.n)
    }
    fn predict(&self, f: &Feat) -> (f64, f64) {
        let toks = f.last(self.n);
        let (mut ms, mut err) = (5.0, 0.05);
        for k in 0..=self.n {
            if let Some(&[sm, wm, se, w]) = self.tables[k].get(&toks[self.n - k..]) {
                ms = (sm + SMOOTHING * ms) / (wm + SMOOTHING);
                err = (se + SMOOTHING * err) / (w + SMOOTHING);
            }
        }
        (ms, err)
    }
    fn learn(&mut self, window: &[RunLog], _: usize) {
        let mut samples = Vec::new();
        for (age, run) in window.iter().rev().enumerate() {
            extract(&run.text, &run.events, recency(age), &mut samples);
        }
        self.tables.iter_mut().for_each(HashMap::clear);
        for s in &samples {
            let toks = s.feat.last(self.n);
            let w = s.weight as f64;
            for k in 0..=self.n {
                let e = self.tables[k].entry(toks[self.n - k..].to_vec()).or_default();
                if let Some(y) = s.log_ms {
                    e[0] += w * y as f64;
                    e[1] += w;
                }
                e[2] += w * f64::from(u8::from(s.error));
                e[3] += w;
            }
        }
    }
}

#[derive(Default)]
struct Score {
    /// Per keystroke: (predicted ln ms, actual ln ms) for timed ones; (p, error) for all.
    timing: Vec<(f64, f64)>,
    errors: Vec<(f64, bool)>,
    /// Per run: (hill, predicted mean ln ms, actual mean ln ms).
    runs: Vec<(bool, f64, f64)>,
    /// Hill runs, app models only, warmup included: (scored, corrected, raw predicted wpm,
    /// scored wpm). `corrected` runs are the ones the app corrects and learns its correction
    /// from: not bosses, not checkpoint opponents.
    wpm: Vec<(bool, bool, f64, f64)>,
}

impl Score {
    /// The prediction the app would have committed before the run, as `session` does.
    fn commit(&mut self, p: &dyn Predictor, run: &RunLog, kind: &Kind, scored: bool) {
        let (Some(model), true, Some(mode)) = (p.model(), run.hill, Mode::of_run(&kind.0, kind.1)) else {
            return;
        };
        // The words on screen when it started; time mode grows the text as you type.
        let opening: Vec<char> = run
            .text
            .iter()
            .collect::<String>()
            .split(' ')
            .take(mode.initial_words())
            .collect::<Vec<_>>()
            .join(" ")
            .chars()
            .collect();
        let raw = committed(model, &opening, mode, 1.0).0;
        self.wpm.push((scored, !kind.2 && !kind.3, raw, run.wpm));
    }

    fn add(&mut self, p: &dyn Predictor, run: &RunLog) {
        let mut samples: Vec<Sample> = Vec::new();
        extract(&run.text, &run.events, 1.0, &mut samples);
        let (mut pred, mut act) = (Vec::new(), Vec::new());
        for s in &samples {
            let (ms, err) = p.predict(&s.feat);
            self.errors.push((err.clamp(1e-6, 1.0 - 1e-6), s.error));
            if let Some(y) = s.log_ms {
                self.timing.push((ms, y as f64));
                pred.push(ms);
                act.push(y as f64);
            }
        }
        if pred.len() >= 10 {
            self.runs.push((run.hill, mean(&pred), mean(&act)));
        }
    }
}

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len().max(1) as f64
}

fn corr(pairs: &[(f64, f64)]) -> f64 {
    let (a, b): (Vec<f64>, Vec<f64>) = pairs.iter().copied().unzip();
    let (ma, mb) = (mean(&a), mean(&b));
    let cov: f64 = pairs.iter().map(|(x, y)| (x - ma) * (y - mb)).sum();
    let va: f64 = a.iter().map(|x| (x - ma).powi(2)).sum();
    let vb: f64 = b.iter().map(|y| (y - mb).powi(2)).sum();
    cov / (va * vb).sqrt()
}

/// Share of (error, clean) pairs where the error got the higher probability.
fn auc(errors: &[(f64, bool)]) -> f64 {
    let mut v = errors.to_vec();
    v.sort_by(|a, b| a.0.total_cmp(&b.0));
    let (mut clean_below, mut wins, mut pairs) = (0.0, 0.0, 0.0);
    let clean_total = v.iter().filter(|e| !e.1).count() as f64;
    let mut i = 0;
    while i < v.len() {
        let j = (i..v.len()).find(|&j| v[j].0 != v[i].0).unwrap_or(v.len());
        let (errs, cleans) = v[i..j].iter().fold((0.0, 0.0), |(e, c), x| if x.1 { (e + 1.0, c) } else { (e, c + 1.0) });
        wins += errs * (clean_below + cleans / 2.0);
        pairs += errs * clean_total;
        clean_below += cleans;
        i = j;
    }
    wins / pairs
}

fn report(name: &str, s: &Score) {
    let ys: Vec<f64> = s.timing.iter().map(|t| t.1).collect();
    let my = mean(&ys);
    let var = mean(&ys.iter().map(|y| (y - my).powi(2)).collect::<Vec<_>>());
    let mse = mean(&s.timing.iter().map(|(p, y)| (p - y).powi(2)).collect::<Vec<_>>());
    let logloss = -mean(&s.errors.iter().map(|&(p, e)| if e { p.ln() } else { (1.0 - p).ln() }).collect::<Vec<_>>());
    let brier = mean(&s.errors.iter().map(|&(p, e)| (p - f64::from(u8::from(e))).powi(2)).collect::<Vec<_>>());
    let p_mean = mean(&s.errors.iter().map(|e| e.0).collect::<Vec<_>>());
    let e_rate = s.errors.iter().filter(|e| e.1).count() as f64 / s.errors.len() as f64;

    // Run pace: corrected by the mean miss over the previous runs, as the app corrects.
    let mut pace_err = Vec::new();
    for (i, &(_, p, a)) in s.runs.iter().enumerate().skip(BIAS_RUNS) {
        let bias = mean(&s.runs[i - BIAS_RUNS..i].iter().map(|r| r.2 - r.1).collect::<Vec<_>>());
        pace_err.push(((p + bias - a).exp() - 1.0).abs());
    }
    let run_corr = corr(&s.runs.iter().map(|r| (r.1, r.2)).collect::<Vec<_>>());
    // Back-to-back runs: you're in about the same form, so what's left is the text.
    let step_corr = corr(&s.runs.windows(2).map(|w| (w[1].1 - w[0].1, w[1].2 - w[0].2)).collect::<Vec<_>>());
    let bias = |hill: bool| {
        let v: Vec<f64> = s.runs.iter().filter(|r| r.0 == hill).map(|r| r.2 - r.1).collect();
        format!("{:+.1}%", (mean(&v).exp() - 1.0) * 100.0)
    };
    print!(
        "{name:<12} {:>6.3} {:>7.4} {:>7.4} {:>5.3}  {:>5.1}%/{:>4.1}%  {:>6.1}% {:>6.2} {:>6.2}  {:>7} {:>7}",
        1.0 - mse / var,
        logloss,
        brier,
        auc(&s.errors),
        p_mean * 100.0,
        e_rate * 100.0,
        mean(&pace_err) * 100.0,
        run_corr,
        step_corr,
        bias(true),
        bias(false)
    );
    if s.wpm.iter().all(|w| !w.0) {
        println!();
    } else {
        // As the app shows them: regular hill runs scaled by the correction learned from the
        // regular hill runs before (newest first); bosses and checkpoints as they are.
        let (mut raw, mut shown) = (Vec::new(), Vec::new());
        for (i, &(scored, corrected, predicted, wpm)) in s.wpm.iter().enumerate() {
            if !scored {
                continue;
            }
            let factor = if corrected {
                let before: Vec<(f64, f64)> = s.wpm[..i]
                    .iter()
                    .rev()
                    .filter(|w| w.1 && w.2 > 0.0 && w.3 > 0.0)
                    .take(CORRECTION_RUNS)
                    .map(|w| (w.3, w.2))
                    .collect();
                correction(&before)
            } else {
                1.0
            };
            raw.push((predicted - wpm).abs());
            shown.push((predicted * factor - wpm).abs());
        }
        println!("  {:>5.1} {:>5.1}", mean(&raw), mean(&shown));
    }
}

/// Difficulty buckets: keystrokes split into fifths by predicted time, with what they took.
fn buckets(name: &str, s: &Score) {
    let mut v = s.timing.clone();
    if v.len() < 5 {
        println!("{name:<12} too few timed keystrokes");
        return;
    }
    v.sort_by(|a, b| a.0.total_cmp(&b.0));
    let fifth = v.len() / 5;
    let cells: Vec<String> = (0..5)
        .map(|q| {
            let part = &v[q * fifth..if q == 4 { v.len() } else { (q + 1) * fifth }];
            format!("{:>4.0}/{:<4.0}", mean(&part.iter().map(|t| t.0.exp()).collect::<Vec<_>>()), {
                let mut a: Vec<f64> = part.iter().map(|t| t.1.exp()).collect();
                a.sort_by(f64::total_cmp);
                a[a.len() / 2]
            })
        })
        .collect();
    println!("{name:<12} {}", cells.join("  "));
}

pub fn evaluate(dir: &Path) -> Result<()> {
    let store = Store::open_existing(dir)?;
    let runs = store.recent_runs(usize::MAX)?;
    let kinds = store.run_kinds()?;
    anyhow::ensure!(kinds.len() == runs.len(), "the log changed while loading it");
    anyhow::ensure!(runs.len() > WARMUP + BIAS_RUNS, "need more than {} runs", WARMUP + BIAS_RUNS);
    let mut predictors: Vec<Box<dyn Predictor>> = vec![Box::new(NGram::new(1)), Box::new(NGram::new(2))];
    predictors.extend(CONTEXTS.iter().map(|&c| Box::new(Model::with_context(c, 0)) as Box<dyn Predictor>));

    let scores: Vec<(String, Score)> = std::thread::scope(|scope| {
        let handles: Vec<_> = predictors
            .into_iter()
            .map(|mut p| {
                let (runs, kinds) = (&runs, &kinds);
                scope.spawn(move || {
                    let mut score = Score::default();
                    for (i, run) in runs.iter().enumerate() {
                        // The app has no opponent for its first few runs.
                        if i >= COLD_RUNS as usize {
                            score.commit(p.as_ref(), run, &kinds[i], i >= WARMUP);
                        }
                        if i >= WARMUP {
                            score.add(p.as_ref(), run);
                        }
                        p.learn(&runs[(i + 1).saturating_sub(TRAIN_WINDOW)..=i], i + 1);
                    }
                    (p.name(), score)
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    let hill = runs[WARMUP..].iter().filter(|r| r.hill).count();
    println!(
        "{} runs, predicted in order after the first {WARMUP} ({hill} against the hill, {} plain)\n",
        runs.len(),
        runs.len() - WARMUP - hill
    );
    println!(
        "{:<12} {:>6} {:>7} {:>7} {:>5}  {:>12}  {:>7} {:>6} {:>6}  {:>7} {:>7}  {:>5} {:>5}",
        "",
        "R² ms",
        "logloss",
        "brier",
        "AUC",
        "p err/actual",
        "pace",
        "corr",
        "Δcorr",
        "hill",
        "plain",
        "raw",
        "shown"
    );
    for (name, s) in &scores {
        report(name, s);
    }
    println!("\nkeystrokes in fifths by predicted time: mean predicted / median actual ms");
    for (name, s) in &scores {
        buckets(name, s);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparse_logs_report_instead_of_crashing() {
        let mut s = Score::default();
        buckets("none", &s);
        s.timing = vec![(4.6, 4.7); 4];
        buckets("four", &s);
        s.timing.push((5.0, 5.1));
        buckets("five", &s);
    }

    #[test]
    fn auc_counts_ties_as_half() {
        assert_eq!(auc(&[(0.1, false), (0.9, true)]), 1.0);
        assert_eq!(auc(&[(0.5, false), (0.5, true)]), 0.5);
        assert_eq!(auc(&[(0.9, false), (0.1, true), (0.5, false)]), 0.0);
    }
}
