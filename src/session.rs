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

/// Boss runs: every `CHECKPOINT_EVERY`th run against the live hill, this many words long.
pub const BOSS_WORDS: u32 = 25;
/// The hill picks the words it thinks are slowest for you, and some of those it has simply
/// overrated, so on its own text it predicts low. Its committed predictions are corrected by
/// how far off they've been lately: a weighted mean of log(actual / predicted) over this many
/// recent ordinary runs, each run counting half as much as the one after it every
/// `CORRECTION_HALF_LIFE` runs.
const CORRECTION_RUNS: usize = 40;
const CORRECTION_HALF_LIFE: f64 = 10.0;
/// No correction until this many runs, and never more than this much either way.
const CORRECTION_MIN_RUNS: usize = 8;
const CORRECTION_LIMIT: (f64, f64) = (0.8, 1.4);

/// The ghost is you as the checkpoint frozen about this many runs ago.
const GHOST_RUNS_BACK: i64 = 50;
/// The hill only brings up a word from a run at least this many runs back.
const MEMORY_RUNS_BACK: i64 = 20;
/// ...and only when your time per letter on it changed by at least this much.
const MEMORY_MIN_CHANGE: f64 = 0.15;

/// A past version of you, typing this run's text at the pace an old checkpoint expects.
pub struct Ghost {
    /// "Nov 5 · run 130".
    pub label: String,
    /// When the ghost types each char, in ms from the first keystroke.
    pub pace: Vec<f32>,
    pub wpm: f64,
}

/// What the results graph plots: cumulative wpm over the run for you, the hill's committed
/// pace, and the ghost; when you made mistakes; and the word that cost you the most.
pub struct Graph {
    pub seconds: f32,
    pub you: Vec<(f32, f32)>,
    pub hill: Vec<(f32, f32)>,
    pub ghost: Vec<(f32, f32)>,
    pub errors: Vec<f32>,
    /// The word you lost the most time on against the hill's pace, and how much (ms).
    pub lost: Option<(String, f64)>,
}

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
    /// The model's own prediction before the correction; what the correction learns from.
    pub model_predicted: f64,
    /// When the opponent expects you to type each char, in ms from the first keystroke.
    pub pace: Vec<f32>,
    pub ghost: Option<Ghost>,
    pub boss: bool,
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
    pub graph: Graph,
    /// The ghost's label and the wpm it would have scored.
    pub ghost: Option<(String, f64)>,
    pub boss: bool,
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
    /// Show a past you racing alongside.
    pub ghost: bool,
    /// The ghost's checkpoint, cached: (id, label, model).
    ghost_model: Option<(i64, String, Arc<Model>)>,
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
        let ghost = store_flag_default_on(&store, "ghost");
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
            ghost,
            ghost_model: None,
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
        self.store.set_setting("ghost", if self.ghost { "1" } else { "0" })?;
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

    /// Whether the next run is a boss: the one that completes each block of
    /// `CHECKPOINT_EVERY`, against the live hill. Retrying or skipping doesn't dodge it.
    fn boss_due(&self, runs: i64) -> bool {
        self.hill && self.opponent_id.is_none() && runs >= COLD_RUNS && (runs + 1) % learn::CHECKPOINT_EVERY == 0
    }

    pub fn new_run(&mut self) -> Result<Run> {
        self.seed = self.seed.wrapping_add(1);
        let runs = self.store.run_count()?;
        let opponent = self.opponent.clone().filter(|_| self.hill && runs >= COLD_RUNS);
        let boss = opponent.is_some() && self.boss_due(runs);
        let mode = if boss { Mode::Words(BOSS_WORDS) } else { self.mode };
        let rest: &[String] = if opponent.is_some() { &self.resting } else { &[] };
        let mut generator = Generator::new(opponent.clone(), self.options, rest, self.seed);
        let mut targets = Vec::new();
        if let (true, Some((m, _))) = (boss, &opponent) {
            generator = generator.boss(&learn::weaknesses(m, 12));
            targets = generator.boss_targets().to_vec();
        }
        let test = Test::new(generator.take(mode.initial_words()));
        let weak = opponent.as_ref().map(|(m, _)| learn::weaknesses(m, 6)).unwrap_or_default();
        let preview = match &opponent {
            _ if !self.hill => String::new(),
            None if runs < COLD_RUNS => self.voice.cold_preview(COLD_RUNS - runs),
            // Enough runs but no usable model (deleted or corrupt): nothing honest to say yet.
            None => String::new(),
            Some(_) if boss => self.voice.boss_preview(&targets),
            Some((model, _)) => self.preview(model, &test, &weak),
        };
        self.save_voice();
        let text: Vec<char> = test.text().chars().collect();
        // Bosses pick words by trigram, not by expected cost, so they don't need correcting;
        // nor do old checkpoints, which are you as you were.
        let factor = if boss || self.opponent_id.is_some() { 1.0 } else { self.correction()? };
        let commit = |m: &Model, factor: f64| {
            let costs: Vec<f32> = m.costs(&text).into_iter().map(|c| c / factor as f32).collect();
            let wpm = match mode {
                Mode::Words(_) => wpm_from_costs(&costs),
                Mode::Time(s) => wpm_from_costs(expected_reach(&costs, s)),
            };
            (wpm, pace(&costs))
        };
        // Only against the hill: with it off, or before it has seen enough, there's nothing
        // honest to predict.
        let (predicted, pace) = opponent.as_ref().map(|(m, _)| commit(m, factor)).unwrap_or_default();
        let model_predicted = opponent.as_ref().map_or(0.0, |(m, _)| commit(m, 1.0).0);
        let ghost = if self.ghost && opponent.is_some() { self.ghost_model(runs)? } else { None };
        let ghost = ghost.map(|(label, m)| {
            let (wpm, pace) = commit(&m, 1.0);
            Ghost { label, pace, wpm }
        });
        Ok(Run {
            predicted,
            model_predicted,
            pace,
            ghost,
            boss,
            test,
            generator,
            mode,
            // A checkpoint picked while the hill is off isn't being played.
            opponent_id: self.opponent_id.filter(|_| opponent.is_some()),
            weak,
            preview,
            hill: self.hill,
        })
    }

    /// How much faster than the model's own prediction you've been typing its text lately.
    fn correction(&self) -> Result<f64> {
        Ok(correction(&self.store.model_results(CORRECTION_RUNS)?))
    }

    /// The checkpoint frozen about `GHOST_RUNS_BACK` runs ago, or the oldest one if there
    /// isn't one that far back yet. `None` until a checkpoint is at least 10 runs old.
    fn ghost_model(&mut self, runs: i64) -> Result<Option<(String, Arc<Model>)>> {
        let checkpoints = self.store.checkpoints()?;
        let old_enough = checkpoints.iter().filter(|c| c.run_count + learn::CHECKPOINT_EVERY <= runs);
        let pick = old_enough
            .clone()
            .filter(|c| c.run_count <= runs - GHOST_RUNS_BACK)
            .next_back()
            .or_else(|| old_enough.min_by_key(|c| c.run_count));
        let Some(c) = pick else { return Ok(None) };
        if self.ghost_model.as_ref().map(|g| g.0) != Some(c.id) {
            let Some(model) = learn::load_checkpoint(&self.store, c.id)? else { return Ok(None) };
            self.ghost_model = Some((c.id, format!("{} · run {}", c.date(), c.run_count), Arc::new(model)));
        }
        Ok(self.ghost_model.as_ref().map(|g| (g.1.clone(), g.2.clone())))
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
                boss: run.boss,
                model_predicted: run.model_predicted,
            },
            &run.test.events,
        )?;
        let run_id = self.store.last_run_id()?;
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
            personal_best: !run.boss && best_before.is_some_and(|b| score.wpm > b),
            boss: run.boss,
            memory: if run.hill { self.memory(&run.test, run_id)? } else { None },
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
        let seconds = match run.mode {
            Mode::Words(_) => run.test.elapsed_ms() / 1000.0,
            Mode::Time(s) => s as f64,
        } as f32;
        let ghost_pace = run.ghost.as_ref().map(|g| g.pace.as_slice());
        let graph = graph(&run.test, &run.pace, ghost_pace, seconds);
        let ghost = run.ghost.as_ref().map(|g| (g.label.clone(), g.wpm));
        Ok(Finished {
            score,
            predicted,
            mode: run.mode,
            slowest,
            lines,
            struggling,
            improved,
            graph,
            ghost,
            boss: run.boss,
        })
    }

    /// A word from this run you also typed long ago, when your time per letter on it has
    /// changed a lot since: the one that changed the most.
    fn memory(&self, test: &Test, run_id: i64) -> Result<Option<voice::Memory>> {
        let now = word_speeds(&test.words, &test.events);
        let mut best: Option<voice::Memory> = None;
        let mut seen = std::collections::HashSet::new();
        for (w, ms) in test.words.iter().zip(now) {
            let word = crate::generator::bare(&w.iter().collect::<String>());
            let Some(now_ms) = ms else { continue };
            if word.len() < 4 || !seen.insert(word.clone()) {
                continue;
            }
            // The earliest past run where you actually typed it cleanly.
            let mut found = None;
            for id in self.store.runs_with(&word, run_id - MEMORY_RUNS_BACK)? {
                let past = self.store.past_run(id)?;
                let text: String = past.log.text.iter().collect();
                let words: Vec<Vec<char>> = text.split(' ').map(|w| w.chars().collect()).collect();
                let then = words
                    .iter()
                    .zip(word_speeds(&words, &past.log.events))
                    .find_map(|(w, ms)| ms.filter(|_| crate::generator::bare(&w.iter().collect::<String>()) == word));
                if let Some(then_ms) = then {
                    found = Some((past, then_ms));
                    break;
                }
            }
            let Some((past, then_ms)) = found else { continue };
            let change = (now_ms / then_ms - 1.0).abs();
            if change >= MEMORY_MIN_CHANGE && best.as_ref().is_none_or(|b| change > (b.now_ms / b.then_ms - 1.0).abs())
            {
                let today = chrono::Local::now().format("%b %-d").to_string();
                let when = if past.date() == today { format!("{} runs ago", run_id - past.id) } else { past.date() };
                best = Some(voice::Memory { word, when, then_ms, now_ms });
            }
        }
        Ok(best)
    }

    /// Persists which lines the hill has said, so a restart doesn't bring them straight back.
    pub fn save_voice(&self) {
        if let Err(e) = self.store.set_setting("voice", &self.voice.history()) {
            eprintln!("thehill: could not save the hill's memory: {e:#}");
        }
    }

    /// The live model to continue training from.
    pub fn training_base(&self) -> Model {
        self.live.clone().unwrap_or_else(|| Model::with_context(crate::profiles::context(&self.store), 0))
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
    let store = Store::open_existing(dir)?;
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

/// Per word: mean time per letter if it was typed without a single mistake and every letter
/// has a clean interval (the run's very first keystroke has none); `None` otherwise.
fn word_speeds(words: &[Vec<char>], events: &[Event]) -> Vec<Option<f64>> {
    let mut spans = Vec::new();
    let mut start = 0;
    for w in words {
        spans.push(start..start + w.len());
        start += w.len() + 1;
    }
    let word_of = |idx: usize| spans.iter().position(|s| s.contains(&idx));
    let mut flawed = vec![false; spans.len()];
    for e in events.iter().filter(|e| !e.correct) {
        if let Some(i) = word_of(e.idx) {
            flawed[i] = true;
        }
    }
    let mut timed: Vec<std::collections::HashMap<usize, f64>> = vec![Default::default(); spans.len()];
    for (e, ms) in clean_intervals(events) {
        if let Some(i) = word_of(e.idx) {
            timed[i].entry(e.idx).or_insert(ms);
        }
    }
    (0..spans.len())
        .map(|i| {
            let complete = !spans[i].is_empty()
                && !flawed[i]
                && timed[i].len() == spans[i].len() - usize::from(spans[i].start == 0);
            complete.then(|| timed[i].values().sum::<f64>() / timed[i].len() as f64)
        })
        .collect()
}

/// The word (3+ letters, typed without a single mistake) with the most time per letter.
fn worst_word(test: &Test) -> Option<(String, f64)> {
    test.words
        .iter()
        .zip(word_speeds(&test.words, &test.events))
        .filter(|(w, _)| w.len() >= 3)
        .filter_map(|(w, ms)| Some((w.iter().collect::<String>(), ms?)))
        .max_by(|a, b| a.1.total_cmp(&b.1))
}

/// `results` are (wpm, model's prediction), newest first.
fn correction(results: &[(f64, f64)]) -> f64 {
    if results.len() < CORRECTION_MIN_RUNS {
        return 1.0;
    }
    let (mut sum, mut weights) = (0.0, 0.0);
    for (i, (wpm, predicted)) in results.iter().enumerate() {
        let w = 0.5f64.powf(i as f64 / CORRECTION_HALF_LIFE);
        sum += w * (wpm / predicted).ln();
        weights += w;
    }
    (sum / weights).exp().clamp(CORRECTION_LIMIT.0, CORRECTION_LIMIT.1)
}

/// Cumulative time at which each char is typed, given per-keystroke costs; the first is free.
fn pace(costs: &[f32]) -> Vec<f32> {
    let mut t = 0.0;
    costs
        .iter()
        .enumerate()
        .map(|(i, c)| {
            if i > 0 {
                t += c;
            }
            t
        })
        .collect()
}

const GRAPH_POINTS: usize = 40;
/// Cumulative wpm is noise before this.
const GRAPH_START_S: f32 = 0.5;

fn graph(test: &Test, hill: &[f32], ghost: Option<&[f32]>, seconds: f32) -> Graph {
    // Your progress: the chars that count toward wpm after each keystroke, exactly as scored.
    let progress: Vec<(f32, usize)> = test.progress().into_iter().map(|(t, c)| (t as f32, c)).collect();
    let you_at = |t: f32| progress.iter().take_while(|r| r.0 <= t).last().map_or(0, |r| r.1);
    let paced = |pace: &[f32], t: f32| pace.partition_point(|&p| p <= t);
    let wpm = |chars: usize, t: f32| chars as f32 / 5.0 / (t / 60_000.0);
    let times: Vec<f32> = (1..=GRAPH_POINTS)
        .map(|k| seconds * 1000.0 * k as f32 / GRAPH_POINTS as f32)
        .filter(|&t| t >= GRAPH_START_S * 1000.0)
        .collect();
    let line = |at: &dyn Fn(f32) -> usize| times.iter().map(|&t| (t / 1000.0, wpm(at(t), t))).collect();
    // A pace that finishes the text early ends there, instead of drooping as if it slowed down.
    let paced_line = |pace: &[f32]| -> Vec<(f32, f32)> {
        let Some(&done) = pace.last() else { return Vec::new() };
        let mut points: Vec<(f32, f32)> =
            times.iter().filter(|&&t| t < done).map(|&t| (t / 1000.0, wpm(paced(pace, t), t))).collect();
        if done <= seconds * 1000.0 && done > 0.0 {
            points.push((done / 1000.0, wpm(pace.len(), done)));
        }
        points
    };
    let errors = test
        .events
        .iter()
        .filter(|e| !e.correct && matches!(e.key, Key::Char(_)))
        .map(|e| e.t as f32 / 1000.0)
        .collect();

    // Time lost per word: from the space before it (or the start) to its last letter, how
    // long it took you minus how long the hill expected. The first keystroke is t = 0 for both.
    let first_reach = |pos: usize| if pos == 0 { Some(0.0) } else { progress.iter().find(|r| r.1 >= pos).map(|r| r.0) };
    let expected_at = |pos: usize| if pos == 0 { Some(0.0) } else { hill.get(pos - 1).copied() };
    let mut lost: Option<(String, f64)> = None;
    let mut start = 0;
    for w in &test.words {
        let end = start + w.len();
        if let (Some(a), Some(b), Some(ea), Some(eb)) =
            (first_reach(start), first_reach(end), expected_at(start), expected_at(end))
        {
            let loss = (b - a - (eb - ea)) as f64;
            if loss > 0.0 && lost.as_ref().is_none_or(|l| loss > l.1) {
                lost = Some((w.iter().collect(), loss));
            }
        }
        start = end + 1;
    }
    Graph {
        seconds,
        you: line(&you_at),
        hill: paced_line(hill),
        ghost: ghost.map(paced_line).unwrap_or_default(),
        errors,
        lost,
    }
}

fn store_flag_default_on(store: &Store, key: &str) -> bool {
    store.setting(key).as_deref() != Some("0")
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
                // v2: the same as v4 without the context length.
                let mut bytes = std::fs::read(path)?;
                bytes[4] = 2;
                bytes.drain(5..9);
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

    fn typed(words: &[&str], ms: impl Fn(usize) -> f64) -> Test {
        let mut t = Test::new(words.iter().map(|w| w.to_string()).collect());
        let mut at = 0.0;
        for (i, c) in words.join(" ").chars().enumerate() {
            if i > 0 {
                at += ms(i);
            }
            t.type_char(c, at);
        }
        t
    }

    #[test]
    fn graph_tracks_you_against_the_pace_and_finds_the_costly_word() {
        // 100 ms a key, except the letters of "jinx", at 400.
        let t = typed(&["the", "jinx", "cat"], |i| if (5..=7).contains(&i) { 400.0 } else { 100.0 });
        let pace: Vec<f32> = (0..12).map(|i| i as f32 * 100.0).collect();
        let g = graph(&t, &pace, None, t.elapsed_ms() as f32 / 1000.0);
        assert_eq!(g.lost.as_ref().map(|l| l.0.as_str()), Some("jinx"));
        assert!((g.lost.unwrap().1 - 900.0).abs() < 1.0, "three letters 300 ms slower than paced");
        let (you, hill) = (g.you.last().unwrap().1, g.hill.last().unwrap().1);
        assert!(you < hill, "slower than the pace: {you} vs {hill}");
        assert!(g.errors.is_empty() && g.ghost.is_empty());
    }

    #[test]
    fn correction_learns_how_far_off_the_hill_has_been() {
        assert_eq!(correction(&[(66.0, 60.0); CORRECTION_MIN_RUNS - 1]), 1.0, "too few runs");
        assert!((correction(&[(66.0, 60.0); 20]) - 1.1).abs() < 1e-9);
        // Recent runs count more: lately on target, earlier 20% fast.
        let mixed: Vec<(f64, f64)> = [(60.0, 60.0); 10].into_iter().chain([(72.0, 60.0); 30]).collect();
        let c = correction(&mixed);
        assert!(c > 1.0 && c < 1.1, "{c}");
        assert_eq!(correction(&[(200.0, 60.0); 20]), CORRECTION_LIMIT.1, "capped");
    }

    #[test]
    fn graph_counts_only_what_the_score_counts() {
        // "hello cat" typed as "h cat": the skipped letters must not count.
        let mut t = Test::new(vec!["hello".into(), "cat".into()]);
        for (i, c) in "h cat".chars().enumerate() {
            t.type_char(c, i as f64 * 250.0);
        }
        let g = graph(&t, &[], None, 1.0);
        let end = g.you.last().unwrap().1 as f64;
        assert!((end - t.score(1.0).wpm).abs() < 0.5, "graph {end} vs score {}", t.score(1.0).wpm);
    }

    #[test]
    fn lost_time_includes_the_pause_before_a_word() {
        // 100 ms a key, but 1100 ms before the "t" of "two".
        let t = typed(&["one", "two", "end"], |i| if i == 4 { 1100.0 } else { 100.0 });
        let pace: Vec<f32> = (0..11).map(|i| i as f32 * 100.0).collect();
        let g = graph(&t, &pace, None, t.elapsed_ms() as f32 / 1000.0);
        let (word, ms) = g.lost.expect("lost time on two");
        assert_eq!(word, "two");
        assert!((ms - 1000.0).abs() < 1.0, "{ms}");
    }

    #[test]
    fn bosses_stay_on_target_even_with_common_trigrams_and_numbers() {
        let model = Arc::new(Model::new(1));
        let costs = Arc::new(WordCosts::new(&model));
        let tags: Vec<String> =
            ["_a_", "_i_", "we_", "ou_", "ing", "_th", "er_", "and"].iter().map(|t| t.to_string()).collect();
        for seed in 0..10 {
            let options = Options { punctuation: false, numbers: true };
            let mut g = Generator::new(Some((model.clone(), costs.clone())), options, &[], seed).boss(&tags);
            let used: Vec<String> = g.boss_targets().iter().map(|t| t.replace('_', " ")).collect();
            assert!(used.len() > 4, "four rare targets get topped up");
            for w in g.take(BOSS_WORDS as usize) {
                let padded = format!(" {w} ");
                assert!(used.iter().any(|t| padded.contains(t.as_str())), "{w} misses {used:?}");
            }
        }
    }

    #[test]
    fn boss_runs_are_built_from_the_target_trigrams() {
        let model = Arc::new(Model::new(1));
        let costs = Arc::new(WordCosts::new(&model));
        let tags = vec!["ing".to_string(), "_th".to_string()];
        let mut g =
            Generator::new(Some((model, costs)), Options { punctuation: false, numbers: false }, &[], 2).boss(&tags);
        for w in g.take(BOSS_WORDS as usize) {
            let padded = format!(" {w} ");
            assert!(padded.contains("ing") || padded.contains(" th"), "{w}");
        }
    }

    #[test]
    fn the_hill_remembers_words_from_long_ago() {
        let dir = std::env::temp_dir().join(format!("thehill-memory-{}", std::process::id()));
        let mut s = Session::open(&dir).unwrap();
        let mut save = |t: &Test| {
            let text = t.text();
            let record = RunRecord {
                mode: "words",
                amount: 10,
                punctuation: false,
                numbers: false,
                opponent: None,
                text: &text,
                wpm: 60.0,
                raw: 60.0,
                accuracy: 1.0,
                predicted: 60.0,
                hill: true,
                boss: false,
                model_predicted: 60.0,
            };
            s.store.save_run(&record, &t.events).unwrap();
        };
        // The earliest mention was never reached, so it can't shadow the run that typed it.
        let mut unreached = Test::new(vec!["the".into(), "pueblo".into()]);
        "the".chars().enumerate().for_each(|(i, c)| unreached.type_char(c, i as f64 * 100.0));
        save(&unreached);
        save(&typed(&["a", "pueblo", "cat"], |_| 250.0));
        for _ in 0..MEMORY_RUNS_BACK {
            save(&typed(&["the", "dog"], |_| 100.0));
        }
        let now = typed(&["a", "pueblo", "cat"], |_| 120.0);
        save(&now);
        let id = s.store.last_run_id().unwrap();
        let m = s.memory(&now, id).unwrap().expect("remembers pueblo");
        assert_eq!(m.word, "pueblo");
        assert!((m.then_ms - 250.0).abs() < 1e-6 && (m.now_ms - 120.0).abs() < 1e-6);
        assert!(s.memory(&now, id - 1).unwrap().is_none(), "nothing from inside the last 20 runs");
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
        assert!(run.predicted == 0.0 && run.pace.is_empty(), "no prediction without the hill");
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
