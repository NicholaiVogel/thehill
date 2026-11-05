//! A synthetic typist for exercising the whole pipeline headless:
//! generation, logging, training and checkpoints.

use crate::session::{Mode, Session, train};
use anyhow::Result;
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::path::Path;

pub struct Typist {
    slow: Vec<(char, char)>,
    rng: StdRng,
}

impl Typist {
    const BASE_MS: f64 = 90.0;
    const NOISE_MS: f64 = 15.0;
    const SLOW_FACTOR: f64 = 3.0;
    const SLOW_ERROR: f64 = 0.15;
    const ERROR: f64 = 0.01;

    pub fn new(slow: &[String], seed: u64) -> Typist {
        let slow = slow.iter().filter_map(|b| Some((b.chars().next()?, b.chars().nth(1)?))).collect();
        Typist { slow, rng: StdRng::seed_from_u64(seed) }
    }

    fn gaussian(&mut self) -> f64 {
        let (u, v): (f64, f64) = (self.rng.random_range(1e-12..1.0), self.rng.random());
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    }

    /// Types the whole run, fixing every mistake with one backspace.
    pub fn type_run(&mut self, run: &mut crate::session::Run) {
        let mut t = 0.0;
        while !run.test.finished {
            let (w, i) = (run.test.current, run.test.inputs[run.test.current].len());
            let word = &run.test.words[w];
            let c = word.get(i).copied().unwrap_or(' ');
            let prev = if i > 0 { word[i - 1] } else { ' ' };
            let slow = self.slow.contains(&(prev.to_ascii_lowercase(), c.to_ascii_lowercase()));
            let base = Self::BASE_MS * if slow { Self::SLOW_FACTOR } else { 1.0 };
            if run.test.started() {
                t += (base + Self::NOISE_MS * self.gaussian()).max(20.0);
            }
            if self.rng.random_bool(if slow { Self::SLOW_ERROR } else { Self::ERROR }) {
                run.test.type_char(if c == 'z' { 'q' } else { 'z' }, t);
                t += 200.0;
                run.test.backspace(t);
                t += Self::BASE_MS;
            }
            run.test.type_char(c, t);
            run.top_up();
        }
    }
}

fn share_with(words: &[Vec<char>], slow: &[String]) -> f64 {
    let hits = words
        .iter()
        .filter(|w| {
            let w: String = w.iter().collect::<String>().to_ascii_lowercase();
            slow.iter().any(|b| w.contains(b.as_str()))
        })
        .count();
    hits as f64 / words.len() as f64
}

pub struct Report {
    /// Per run: (wpm, predicted wpm, share of words containing a slow bigram).
    pub runs: Vec<(f64, f64, f64)>,
}

pub fn simulate(dir: &Path, n: usize, slow: &[String], quiet: bool) -> Result<Report> {
    let mut session = Session::open(dir)?;
    session.mode = Mode::Words(50);
    let mut typist = Typist::new(slow, 42);
    let mut report = Report { runs: Vec::new() };
    for i in 1..=n {
        let mut run = session.new_run()?;
        let share = share_with(&run.test.words, slow);
        typist.type_run(&mut run);
        let done = session.finish(&run)?;
        let (model, costs, _) = train(dir, session.training_base())?;
        session.install(model, costs);
        report.runs.push((done.score.wpm, done.predicted, share));
        if !quiet && (i == 1 || i == n / 2 || i == n) {
            println!(
                "run {i:>4}: {:>5.1} wpm, predicted {:>5.1}, words with {}: {:>4.1}%",
                done.score.wpm,
                done.predicted,
                slow.join("/"),
                share * 100.0
            );
        }
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learn::COLD_RUNS;
    use crate::model::Feat;

    #[test]
    fn learns_to_target_slow_bigrams() {
        let dir = std::env::temp_dir().join(format!("sisyphus-sim-{}", std::process::id()));
        let slow = vec!["th".to_string(), "er".to_string()];
        let report = simulate(&dir, 60, &slow, true).unwrap();
        let mean = |r: &[(f64, f64, f64)]| r.iter().map(|x| x.2).sum::<f64>() / r.len() as f64;
        let cold = mean(&report.runs[..COLD_RUNS as usize]);
        let last = mean(&report.runs[50..]);
        assert!(last >= 2.0 * cold, "slow-bigram share went from {cold:.3} to {last:.3}");

        let session = Session::open(&dir).unwrap();
        let model = session.live.unwrap();
        let text: Vec<char> = "the as".chars().collect();
        let (th, as_) = (model.predict(&Feat::at(&text, 1)).0, model.predict(&Feat::at(&text, 5)).0);
        assert!(th >= 2.0 * as_, "t→h {th:.0} ms vs a→s {as_:.0} ms");
        println!("share {cold:.3} → {last:.3}; t→h {th:.0} ms, a→s {as_:.0} ms");
        assert_eq!(session.store.checkpoints().unwrap().len(), 6);
        std::fs::remove_dir_all(dir).unwrap();
    }
}
