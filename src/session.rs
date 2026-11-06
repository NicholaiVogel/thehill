//! The run lifecycle shared by the app and the simulator: settings, the opponent,
//! generating a run, scoring it, logging it, and learning from it.

use crate::generator::{Generator, Options, REST_WORDS, WordCosts, typed_words};
use crate::learn::{self, COLD_RUNS};
use crate::model::{Model, wpm_from_costs};
use crate::storage::{RunRecord, Store};
use crate::typing::{Event, Key, Score, Test};
use crate::voice::{self, RunFacts, Voice};
use anyhow::Result;
use chrono::Timelike;
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Mode {
    Words(u32),
    Time(u32),
}

impl Mode {
    pub const ALL: [Mode; 5] = [Mode::Words(10), Mode::Words(50), Mode::Time(10), Mode::Time(15), Mode::Time(30)];

    fn encode(self) -> String {
        match self {
            Mode::Words(n) => format!("words {n}"),
            Mode::Time(n) => format!("time {n}"),
        }
    }

    fn decode(s: &str) -> Option<Mode> {
        Mode::ALL.into_iter().find(|m| m.encode() == s)
    }

    /// Words to generate up front.
    pub fn initial_words(self) -> usize {
        match self {
            Mode::Words(n) => n as usize,
            Mode::Time(_) => 100,
        }
    }
}

pub type Opponent = (Arc<Model>, Arc<WordCosts>);

pub struct Run {
    pub test: Test,
    pub generator: Generator,
    pub mode: Mode,
    pub opponent_id: Option<i64>,
    /// The opponent's top weaknesses (trigram tags), for heckling.
    pub weak: Vec<String>,
    /// What the hill says before the run starts.
    pub preview: String,
    /// Against the hill (targeted words, its voice) or random common words in silence.
    pub hill: bool,
    /// The opponent's wpm prediction, committed and shown before the run starts; the run is
    /// judged against exactly this number. 0 without an opponent.
    pub predicted: f64,
}

impl Run {
    /// Time mode: keep at least this many words ahead of the cursor.
    const LOOKAHEAD: usize = 25;

    pub fn top_up(&mut self) {
        if matches!(self.mode, Mode::Time(_)) && self.test.words.len() - self.test.current < Self::LOOKAHEAD {
            let more = self.generator.take(50);
            self.test.extend(more);
        }
    }

    pub fn score(&self) -> Score {
        let seconds = match self.mode {
            Mode::Words(_) => self.test.elapsed_ms() / 1000.0,
            Mode::Time(s) => s as f64,
        };
        self.test.score(seconds)
    }
}

pub struct Finished {
    pub score: Score,
    pub predicted: f64,
    pub mode: Mode,
    /// Slowest transitions this run: (from, to, ms).
    pub slowest: Vec<(char, char, f64)>,
    /// What the hill has to say about it.
    pub lines: Vec<String>,
    /// The live model's current top weaknesses, and the last checkpoint's that dropped out.
    pub struggling: Vec<String>,
    pub improved: Vec<String>,
}

pub struct Session {
    pub store: Store,
    pub dir: PathBuf,
    pub options: Options,
    pub mode: Mode,
    /// Off: random common words, no voice. The model still learns from every run.
    pub hill: bool,
    /// `None` plays against the live model.
    pub opponent_id: Option<i64>,
    pub opponent: Option<Opponent>,
    pub live: Option<Model>,
    pub voice: Voice,
    /// Recent words, newest first, that rest before the hill uses them again.
    resting: Vec<String>,
    seed: u64,
}

impl Session {
    pub fn open(dir: &Path) -> Result<Session> {
        let store = Store::open(dir)?;
        learn::migrate(&store)?;
        let flag = |k: &str| store.setting(k).as_deref() == Some("1");
        let options = Options { punctuation: flag("punctuation"), numbers: flag("numbers") };
        let mode = store.setting("mode").and_then(|m| Mode::decode(&m)).unwrap_or(Mode::Time(15));
        let hill = store.setting("hill").as_deref() != Some("0");
        let opponent_id = store.setting("opponent").and_then(|s| s.parse().ok());
        let live = learn::load_live(&store);
        let history = store.setting("voice").unwrap_or_default();
        let seed = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_nanos() as u64;
        let mut session = Session {
            store,
            dir: dir.to_path_buf(),
            options,
            mode,
            hill,
            opponent_id: None,
            opponent: None,
            live,
            voice: Voice::new(seed, &history),
            resting: Vec::new(),
            seed,
        };
        session.refresh_resting()?;
        session.set_opponent(opponent_id)?;
        Ok(session)
    }

    pub fn save_settings(&self) -> Result<()> {
        self.store.set_setting("punctuation", if self.options.punctuation { "1" } else { "0" })?;
        self.store.set_setting("numbers", if self.options.numbers { "1" } else { "0" })?;
        self.store.set_setting("mode", &self.mode.encode())?;
        self.store.set_setting("hill", if self.hill { "1" } else { "0" })?;
        let opponent = self.opponent_id.map(|id| id.to_string()).unwrap_or_default();
        self.store.set_setting("opponent", &opponent)
    }

    /// Switches to a checkpoint (or back to live with `None`). Unknown checkpoints fall back to live.
    pub fn set_opponent(&mut self, id: Option<i64>) -> Result<()> {
        let checkpoint = match id {
            Some(id) => learn::load_checkpoint(&self.store, id)?.map(|m| (id, m)),
            None => None,
        };
        self.opponent_id = checkpoint.as_ref().map(|(id, _)| *id);
        let model = checkpoint.map(|(_, m)| m).or_else(|| self.live.clone());
        self.opponent = model.map(|m| {
            let costs = WordCosts::new(&m);
            (Arc::new(m), Arc::new(costs))
        });
        Ok(())
    }

    pub fn new_run(&mut self) -> Result<Run> {
        self.seed = self.seed.wrapping_add(1);
        let runs = self.store.run_count()?;
        let opponent = self.opponent.clone().filter(|_| self.hill && runs >= COLD_RUNS);
        let rest: &[String] = if opponent.is_some() { &self.resting } else { &[] };
        let mut generator = Generator::new(opponent.clone(), self.options, rest, self.seed);
        let test = Test::new(generator.take(self.mode.initial_words()));
        let weak = opponent.as_ref().map(|(m, _)| learn::weaknesses(m, 6)).unwrap_or_default();
        let preview = match &opponent {
            _ if !self.hill => String::new(),
            None if runs < COLD_RUNS => self.voice.cold_preview(COLD_RUNS - runs),
            // Enough runs but no usable model (deleted or corrupt): nothing honest to say yet.
            None => String::new(),
            Some((model, _)) => self.preview(model, &test, &weak),
        };
        self.save_voice();
        let text: Vec<char> = test.text().chars().collect();
        let predicted = match (&self.opponent, self.mode) {
            (Some((m, _)), Mode::Words(_)) => wpm_from_costs(&m.costs(&text)),
            (Some((m, _)), Mode::Time(s)) => wpm_from_costs(expected_reach(&m.costs(&text), s)),
            (None, _) => 0.0,
        };
        Ok(Run {
            predicted,
            test,
            generator,
            mode: self.mode,
            opponent_id: self.opponent_id,
            weak,
            preview,
            hill: self.hill,
        })
    }

    fn preview(&mut self, model: &Model, test: &Test, weak: &[String]) -> String {
        if let Some(id) = self.opponent_id
            && let Some(c) = self.store.checkpoints().ok().and_then(|cs| cs.into_iter().find(|c| c.id == id))
        {
            return voice::opponent_preview(&c.date(), &c.tags);
        }
        let text = format!(" {} ", test.text());
        let planted = weak.first().map(|tag| (tag.as_str(), text.matches(&tag.replace('_', " ")).count()));
        let hardest = test
            .words
            .iter()
            .map(|w| w.iter().collect::<String>())
            .filter(|w| w.len() >= 3)
            .max_by(|a, b| mean_cost(model, a).total_cmp(&mean_cost(model, b)));
        self.voice.preview(planted, hardest.as_deref())
    }

    /// Scores and logs a finished run. Training is separate (`train`) so the app can run it off-thread.
    pub fn finish(&mut self, run: &Run) -> Result<Finished> {
        let score = run.score();
        let text: Vec<char> = run.test.text().chars().collect();
        let predicted = run.predicted;
        let (mode, amount) = match run.mode {
            Mode::Words(n) => ("words", n),
            Mode::Time(n) => ("time", n),
        };
        let (punctuation, numbers) = (self.options.punctuation, self.options.numbers);
        let best_before = self.store.best_wpm(mode, amount, punctuation, numbers)?;
        let previous = self.store.last_wpm()?;
        let text: String = text.iter().collect();
        self.store.save_run(
            &RunRecord {
                mode,
                amount,
                punctuation,
                numbers,
                opponent: run.opponent_id,
                text: &text,
                wpm: score.wpm,
                raw: score.raw,
                accuracy: score.accuracy,
                predicted,
                hill: run.hill,
            },
            &run.test.events,
        )?;
        self.refresh_resting()?;

        let slowest = slowest(&run.test, 5);
        let checkpoint = match run.opponent_id {
            Some(id) => self.store.checkpoints()?.into_iter().find(|c| c.id == id).map(|c| c.date()),
            None => None,
        };
        let facts = RunFacts {
            wpm: score.wpm,
            raw: score.raw,
            previous,
            runs_today: self.store.runs_today()?,
            hour: chrono::Local::now().hour(),
            checkpoint,
            predicted,
            accuracy: score.accuracy,
            errors: score.errors,
            personal_best: best_before.is_some_and(|b| score.wpm > b),
            streak: if predicted > 0.0 { streak(&self.store.recent_results(usize::MAX)?) } else { 0 },
            slowest: slowest.first().copied(),
            worst_word: worst_word(&run.test),
        };
        let lines = if run.hill { self.voice.results(&facts) } else { Vec::new() };
        self.save_voice();
        let (struggling, improved) = match &self.live {
            Some(live) => {
                let now = learn::weaknesses(live, 12);
                let improved = match self.store.checkpoints()?.last() {
                    Some(c) => learn::load_checkpoint(&self.store, c.id)
                        .ok()
                        .flatten()
                        .map(|before| learn::got_past(&before, &c.tags, live, &now))
                        .unwrap_or_default(),
                    None => Vec::new(),
                };
                (now.into_iter().take(4).collect(), improved)
            }
            None => (Vec::new(), Vec::new()),
        };
        Ok(Finished { score, predicted, mode: run.mode, slowest, lines, struggling, improved })
    }

    /// Persists which lines the hill has said, so a restart doesn't bring them straight back.
    pub fn save_voice(&self) {
        if let Err(e) = self.store.set_setting("voice", &self.voice.history()) {
            eprintln!("thehill: could not save the hill's memory: {e:#}");
        }
    }

    /// The live model to continue training from.
    pub fn training_base(&self) -> Model {
        self.live.clone().unwrap_or_else(|| Model::new(0))
    }

    /// Installs a newly trained live model; if playing against live, it becomes the opponent.
    pub fn install(&mut self, model: Model, costs: WordCosts) {
        if self.opponent_id.is_none() {
            self.opponent = Some((Arc::new(model.clone()), Arc::new(costs)));
        }
        self.live = Some(model);
    }

    fn refresh_resting(&mut self) -> Result<()> {
        let mut seen = std::collections::HashSet::new();
        self.resting = self
            .store
            .recent_runs(30)?
            .iter()
            .rev()
            .flat_map(|r| typed_words(&r.text, r.reached().len()))
            .filter(|w| seen.insert(w.clone()))
            .take(REST_WORDS)
            .collect();
        Ok(())
    }
}

/// Trains on the log (opening its own connection so it can run on any thread) and prepares
/// word costs for the new model. Also returns the hill's announcement if a checkpoint was frozen.
pub fn train(dir: &Path, base: Model) -> Result<(Model, WordCosts, Option<String>)> {
    let store = Store::open(dir)?;
    let (model, froze) = learn::after_run(&store, base)?;
    let announcement = if froze {
        let all = store.checkpoints()?;
        let got_past = match all.len().checked_sub(2).map(|i| &all[i]) {
            Some(prev) => match learn::load_checkpoint(&store, prev.id).ok().flatten() {
                Some(before) => learn::got_past(&before, &prev.tags, &model, &learn::weaknesses(&model, 12)),
                None => Vec::new(),
            },
            None => Vec::new(),
        };
        all.last().map(|c| voice::checkpoint(all.len(), &c.tags, &got_past))
    } else {
        None
    };
    let costs = WordCosts::new(&model);
    Ok((model, costs, announcement))
}

fn mean_cost(model: &Model, word: &str) -> f32 {
    let costs = model.costs(&format!(" {word}").chars().collect::<Vec<_>>());
    costs[1..].iter().sum::<f32>() / word.len() as f32
}

/// Time mode: the keystroke costs the model expects you to get through in `seconds`.
fn expected_reach(costs: &[f32], seconds: u32) -> &[f32] {
    let budget = seconds as f32 * 1000.0;
    let mut spent = 0.0;
    // The first keystroke starts the clock, so it's free.
    let n = 1 + costs[1..]
        .iter()
        .take_while(|&&c| {
            spent += c;
            spent <= budget
        })
        .count();
    &costs[..n.min(costs.len())]
}

/// Runs in a row over (positive) or under (negative) the prediction, newest first.
/// A draw (within 3%) is neither, and ends a streak.
fn streak(results: &[(f64, f64)]) -> i32 {
    let side = |(wpm, pred): &(f64, f64)| voice::outcome(*wpm, *pred);
    let Some(first) = results.first().map(side).filter(|s| s.is_ne()) else { return 0 };
    let n = results.iter().take_while(|r| side(r) == first).count() as i32;
    if first.is_gt() { n } else { -n }
}

/// Keypresses typed right immediately after another right keypress, with the interval.
fn clean_intervals(events: &[Event]) -> impl Iterator<Item = (&Event, f64)> {
    events.windows(2).filter_map(|pair| {
        let (a, b) = (&pair[0], &pair[1]);
        let clean = a.correct && b.correct && matches!((a.key, b.key), (Key::Char(_), Key::Char(_)));
        (clean && b.idx > 0).then_some((b, b.t - a.t))
    })
}

/// The slowest clean transitions in a run, one per char pair.
fn slowest(test: &Test, n: usize) -> Vec<(char, char, f64)> {
    let text: Vec<char> = test.text().chars().collect();
    let mut best: Vec<(char, char, f64)> = Vec::new();
    for (e, ms) in clean_intervals(&test.events) {
        let (a, b) = (text[e.idx - 1], text[e.idx]);
        match best.iter_mut().find(|(x, y, _)| (*x, *y) == (a, b)) {
            Some(entry) => entry.2 = entry.2.max(ms),
            None => best.push((a, b, ms)),
        }
    }
    best.sort_by(|a, b| b.2.total_cmp(&a.2));
    best.truncate(n);
    best
}

/// The word (3+ letters, typed without a single mistake) with the most time per letter.
/// Each letter's time is its first clean interval; the run's very first keystroke has none.
fn worst_word(test: &Test) -> Option<(String, f64)> {
    let mut spans = Vec::new();
    let mut start = 0;
    for w in &test.words {
        spans.push(start..start + w.len());
        start += w.len() + 1;
    }
    let word_of = |idx: usize| spans.iter().position(|s| s.contains(&idx));
    let mut flawed = vec![false; spans.len()];
    for e in test.events.iter().filter(|e| !e.correct) {
        if let Some(i) = word_of(e.idx) {
            flawed[i] = true;
        }
    }
    let mut timed: Vec<std::collections::HashMap<usize, f64>> = vec![Default::default(); spans.len()];
    for (e, ms) in clean_intervals(&test.events) {
        if let Some(i) = word_of(e.idx) {
            timed[i].entry(e.idx).or_insert(ms);
        }
    }
    (0..spans.len())
        .filter(|&i| spans[i].len() >= 3 && !flawed[i])
        .filter(|&i| timed[i].len() == spans[i].len() - usize::from(spans[i].start == 0))
        .map(|i| {
            let per_letter = timed[i].values().sum::<f64>() / timed[i].len() as f64;
            (test.words[i].iter().collect::<String>(), per_letter)
        })
        .max_by(|a, b| a.1.total_cmp(&b.1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn old_checkpoints_are_recalibrated_when_selected() {
        let dir = std::env::temp_dir().join(format!("thehill-oldcp-{}", std::process::id()));
        let mut session = Session::open(&dir).unwrap();
        // As if an older binary froze it after this session started.
        let id = session
            .store
            .add_checkpoint(0, &[], 1.0, |path| {
                Model::new(1).save(path)?;
                let mut bytes = std::fs::read(path)?;
                bytes[4] = 2;
                std::fs::write(path, bytes)
            })
            .unwrap();
        session.set_opponent(Some(id)).unwrap();
        assert_eq!(session.opponent_id, Some(id));
        assert!(Model::load_versioned(&session.store.checkpoint_path(id)).unwrap().1, "rewritten as current");
        let c = session.store.checkpoints().unwrap().into_iter().find(|c| c.id == id).unwrap();
        assert!(c.reference_wpm > 1.0, "reference speed repaired");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn hill_off_plays_random_common_words_in_silence() {
        let dir = std::env::temp_dir().join(format!("thehill-hill-{}", std::process::id()));
        let mut session = Session::open(&dir).unwrap();
        session.live = Some(Model::new(1));
        session.set_opponent(None).unwrap();
        session.hill = false;
        let run = session.new_run().unwrap();
        assert!(run.preview.is_empty() && run.weak.is_empty());
        let common = &crate::generator::words()[..1000];
        assert!(run.test.words.iter().all(|w| common.contains(&w.iter().collect::<String>().as_str())));
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn time_mode_predicts_over_the_text_you_would_reach() {
        let costs = [0.0, 100.0, 100.0, 100.0, 100.0, 100.0];
        assert_eq!(expected_reach(&costs[..], 0).len(), 1);
        let costs: Vec<f32> = vec![100.0; 400];
        assert_eq!(expected_reach(&costs, 15).len(), 151);
        assert_eq!(expected_reach(&costs[..50], 15).len(), 50, "clamped to the text");
    }

    #[test]
    fn streak_counts_the_current_run_of_results() {
        assert_eq!(streak(&[(110.0, 100.0), (105.0, 100.0), (90.0, 100.0)]), 2);
        assert_eq!(streak(&[(90.0, 100.0), (95.0, 100.0), (96.0, 100.0)]), -3);
        assert_eq!(streak(&[]), 0);
        assert_eq!(streak(&[(100.0, 100.0), (90.0, 100.0)]), 0, "a tie is not a loss");
        assert_eq!(streak(&[(59.3, 59.2), (70.0, 56.0)]), 0, "a 0.1 wpm edge is a draw, not a win");
        assert_eq!(streak(&[(90.0, 100.0), (100.0, 100.0), (95.0, 100.0)]), -1);
    }

    #[test]
    fn worst_word_finds_the_slow_one() {
        let mut t = Test::new(vec!["the".into(), "jinx".into(), "cat".into()]);
        let mut at = 0.0;
        for (i, c) in "the jinx cat".chars().enumerate() {
            at += if (4..8).contains(&i) { 300.0 } else { 80.0 };
            t.type_char(c, at);
        }
        let (word, ms) = worst_word(&t).unwrap();
        assert_eq!(word, "jinx");
        assert!((ms - 300.0).abs() < 1e-9);
    }

    #[test]
    fn worst_word_counts_the_first_word_and_skips_corrected_ones() {
        let mut t = Test::new(vec!["slow".into(), "cat".into(), "dog".into()]);
        let mut at = 0.0;
        for c in "slow ".chars() {
            at += 500.0;
            t.type_char(c, at);
        }
        t.type_char('c', at + 80.0);
        t.type_char('x', at + 980.0);
        t.backspace(at + 990.0);
        at += 990.0;
        for c in "at dog".chars() {
            at += 80.0;
            t.type_char(c, at);
        }
        assert_eq!(worst_word(&t).unwrap().0, "slow");
    }
}
