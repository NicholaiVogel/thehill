//! Training after each run, checkpoints, and what a model says about you.

use crate::generator::words;
use crate::model::{Feat, Model};
use crate::storage::Store;
use anyhow::Result;
use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};
use std::collections::HashMap;
use std::sync::OnceLock;

pub const CHECKPOINT_EVERY: i64 = 10;
/// Older runs carry almost no weight after the recency decay, so they are not loaded.
const TRAIN_WINDOW: usize = 300;
const EPOCHS: usize = 3;
const RETRAIN_EPOCHS: usize = 10;
/// Runs before the generator trusts the model.
pub const COLD_RUNS: i64 = 3;

pub fn load_live(store: &Store) -> Option<Model> {
    Model::load(&store.live_model_path()).ok()
}

/// Recalibrates models saved before the cost calibration existed: the live model on the
/// recent log, each checkpoint only on the runs it had seen when it was frozen.
pub fn migrate(store: &Store) -> Result<()> {
    let path = store.live_model_path();
    if let Ok((mut model, false)) = Model::load_versioned(&path) {
        model.calibrate_cost(&store.recent_runs(TRAIN_WINDOW)?);
        model.save(&path)?;
    }
    for c in store.checkpoints()? {
        load_checkpoint(store, c.id)?;
    }
    Ok(())
}

/// Loads a checkpoint, recalibrating it first if it was saved in an older format (an old
/// binary still running can write one at any time). `None` if the file is missing or bad.
pub fn load_checkpoint(store: &Store, id: i64) -> Result<Option<Model>> {
    let path = store.checkpoint_path(id);
    let Ok((mut model, current)) = Model::load_versioned(&path) else { return Ok(None) };
    if !current {
        let Some(c) = store.checkpoints()?.into_iter().find(|c| c.id == id) else { return Ok(Some(model)) };
        let runs = store.first_runs(c.run_count)?;
        model.calibrate_cost(&runs[runs.len().saturating_sub(TRAIN_WINDOW)..]);
        // Metadata first: if saving the model fails, the file stays old and this retries.
        store.set_reference_wpm(id, reference_wpm(&model))?;
        model.save(&path)?;
    }
    Ok(Some(model))
}

/// Trains the live model on the log after a run and freezes a checkpoint every
/// `CHECKPOINT_EVERY` runs. Returns the new live model and whether it froze a checkpoint.
pub fn after_run(store: &Store, mut model: Model) -> Result<(Model, bool)> {
    let (runs, count) = store.training_snapshot(TRAIN_WINDOW)?;
    model.fit(&runs, EPOCHS, count as u64);
    model.save(&store.live_model_path())?;
    // Compare against the last checkpoint rather than `count % N == 0`: a run can be
    // logged while the previous one is still training, so the exact count can be skipped.
    let due = count / CHECKPOINT_EVERY * CHECKPOINT_EVERY;
    let last = store.checkpoints()?.last().map_or(0, |c| c.run_count);
    let freeze = due > last;
    if freeze {
        store.add_checkpoint(count, &weaknesses(&model, 4), reference_wpm(&model), |path| model.save(path))?;
    }
    Ok((model, freeze))
}

/// Rebuilds the live model from scratch from the whole log.
pub fn retrain(store: &Store) -> Result<Model> {
    let mut model = Model::new(0);
    let loss = model.fit(&store.recent_runs(usize::MAX)?, RETRAIN_EPOCHS, 0);
    println!("retrained on {} runs, final loss {loss:.4}", store.run_count()?);
    model.save(&store.live_model_path())?;
    Ok(model)
}

/// The 300 most common English trigrams, word boundaries included.
fn common_trigrams() -> &'static [[char; 3]] {
    static TRIGRAMS: OnceLock<Vec<[char; 3]>> = OnceLock::new();
    TRIGRAMS.get_or_init(|| {
        let mut freq: HashMap<[char; 3], f64> = HashMap::new();
        for (rank, w) in words().iter().enumerate() {
            let padded: Vec<char> = format!(" {w} ").chars().collect();
            for t in padded.windows(3) {
                *freq.entry([t[0], t[1], t[2]]).or_default() += 1.0 / (rank + 1) as f64;
            }
        }
        let mut all: Vec<_> = freq.into_iter().collect();
        all.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        all.into_iter().take(300).map(|(t, _)| t).collect()
    })
}

/// The common trigrams whose last keystroke the model expects to cost the most,
/// written with `_` for a space.
pub fn weaknesses(model: &Model, n: usize) -> Vec<String> {
    let mut scored: Vec<(f32, &[char; 3])> =
        common_trigrams().iter().map(|t| (model.cost(&Feat::at(t, 2)), t)).collect();
    scored.sort_by(|a, b| b.0.total_cmp(&a.0));
    scored.iter().take(n).map(|(_, t)| t.iter().map(|&c| if c == ' ' { '_' } else { c }).collect()).collect()
}

/// Expected cost of a weakness tag's last keystroke (tags use `_` for a space).
fn tag_cost(model: &Model, tag: &str) -> f32 {
    let chars: Vec<char> = tag.chars().map(|c| if c == '_' { ' ' } else { c }).collect();
    model.cost(&Feat::at(&chars, chars.len() - 1))
}

/// Earlier weaknesses that left the current top list *and* got cheaper since, i.e. that
/// you actually improved on rather than ones that were merely outranked.
pub fn got_past(before: &Model, before_tags: &[String], after: &Model, after_top: &[String]) -> Vec<String> {
    before_tags.iter().filter(|t| !after_top.contains(t) && tag_cost(after, t) < tag_cost(before, t)).cloned().collect()
}

/// The keys drawn on the heatmap, row by row.
pub const KEYBOARD: [&str; 3] = ["qwertyuiop", "asdfghjkl;", "zxcvbnm,./"];

/// Trigrams ending in a key, with how common each is.
type Contexts = HashMap<char, Vec<([char; 3], f64)>>;

/// Typical contexts for each keyboard key: the trigrams ending in it, weighted by how common
/// they are. Punctuation follows the ends of common words.
fn key_contexts() -> &'static Contexts {
    static CONTEXTS: OnceLock<Contexts> = OnceLock::new();
    CONTEXTS.get_or_init(|| {
        let mut all: HashMap<char, HashMap<[char; 3], f64>> = HashMap::new();
        for (rank, w) in words().iter().enumerate() {
            let weight = 1.0 / (rank + 1) as f64;
            let padded: Vec<char> = format!(" {w}").chars().collect();
            for t in padded.windows(3) {
                *all.entry(t[2]).or_default().entry([t[0], t[1], t[2]]).or_default() += weight;
            }
            if rank < 300 && padded.len() >= 3 {
                let (a, b) = (padded[padded.len() - 2], padded[padded.len() - 1]);
                for p in [';', ',', '.', '/'] {
                    *all.entry(p).or_default().entry([a, b, p]).or_default() += weight;
                }
            }
        }
        all.into_iter()
            .map(|(key, contexts)| {
                let mut v: Vec<_> = contexts.into_iter().collect();
                v.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
                v.truncate(30);
                (key, v)
            })
            .collect()
    })
}

/// Keys with fewer training samples than this have no meaningful estimate yet.
const MIN_KEY_SAMPLES: u32 = 20;

/// The model's expected cost of each keyboard key in its usual contexts, in ms, or `None`
/// for keys it has barely seen you type.
pub fn key_costs(model: &Model) -> Vec<(char, Option<f32>)> {
    KEYBOARD
        .concat()
        .chars()
        .map(|key| {
            let contexts = key_contexts().get(&key).map_or(&[][..], |v| v.as_slice());
            let total: f64 = contexts.iter().map(|(_, w)| w).sum();
            let cost: f64 = contexts.iter().map(|(t, w)| model.cost(&Feat::at(t, 2)) as f64 * w).sum();
            let known = total > 0.0 && model.samples_for(key) >= MIN_KEY_SAMPLES;
            (key, known.then_some((cost / total) as f32))
        })
        .collect()
}

/// A fixed text (the 200 most common words in a fixed order) every model is measured on,
/// so checkpoints can be compared.
fn reference_text() -> &'static [char] {
    static TEXT: OnceLock<Vec<char>> = OnceLock::new();
    TEXT.get_or_init(|| {
        let mut top: Vec<&str> = words()[..200].to_vec();
        top.shuffle(&mut StdRng::seed_from_u64(7));
        top.join(" ").chars().collect()
    })
}

pub fn reference_wpm(model: &Model) -> f64 {
    model.predicted_wpm(reference_text())
}
