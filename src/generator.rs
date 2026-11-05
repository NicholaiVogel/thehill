//! Picks the words for a run: mostly the ones the model expects you to be slowest on,
//! plus some it knows little about.
//!
//! Difficulty has to come from the transitions you're slow on, not from familiarity
//! tricks: a word family (yeet, yeeting) appears at most once per run, and words from
//! recent runs, with their families, are rested before they come back.

use crate::model::{Feat, Model};
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, OnceLock};

const COLD_POOL: usize = 1000;
/// Large enough that resting recent families still leaves plenty of hard words.
const SHORTLIST: usize = 1500;
const NO_REPEAT: usize = 15;
const TEMPERATURE: f64 = 0.35;
const EXPLORE_SHARE: f64 = 0.2;
const VARIANTS: usize = 3;
/// Distinct words from recent runs that rest, families included.
pub const REST_WORDS: usize = 150;

pub fn words() -> &'static [&'static str] {
    static WORDS: OnceLock<Vec<&'static str>> = OnceLock::new();
    WORDS.get_or_init(|| include_str!("../assets/words.txt").lines().collect())
}

pub fn index_of(word: &str) -> Option<usize> {
    static INDEX: OnceLock<HashMap<&'static str, usize>> = OnceLock::new();
    INDEX.get_or_init(|| words().iter().enumerate().map(|(i, w)| (*w, i)).collect()).get(word).copied()
}

/// A set of words that also answers "is this a relative of one of them?". Words of 3+
/// letters are related when they agree on their first min(4, both lengths) letters:
/// yeet/yeeting, hydro/hydra, know/knows, bye/byes. Shorter words only match exactly.
#[derive(Default)]
pub struct Family {
    exact: HashSet<String>,
    /// First four letters of members with 4+ letters.
    four: HashSet<String>,
    /// Members with exactly 3 letters.
    three: HashSet<String>,
    /// First three letters of members with 3+ letters.
    prefix3: HashSet<String>,
}

impl Family {
    pub fn insert(&mut self, w: &str) {
        self.exact.insert(w.to_string());
        if w.len() >= 3 {
            self.prefix3.insert(w[..3].to_string());
        }
        match w.len() {
            3 => _ = self.three.insert(w.to_string()),
            4.. => _ = self.four.insert(w[..4].to_string()),
            _ => {}
        }
    }

    pub fn related(&self, w: &str) -> bool {
        self.exact.contains(w)
            || match w.len() {
                3 => self.prefix3.contains(w),
                4.. => self.four.contains(&w[..4]) || self.three.contains(&w[..3]),
                _ => false,
            }
    }
}

/// Per-word costs under one model, computed once and reused for every run against it.
///
/// Scores are excess cost: each keystroke's cost minus your average cost for keystrokes in
/// the same slot (first letter, inner letter, trailing space). Raw costs would favour short
/// words, since word boundaries are slow for everyone.
pub struct WordCosts {
    /// Excess of everything except the first char, which depends on the previous word.
    rest: Vec<f32>,
    /// Average cost of a word's first char, subtracted from the rescored first char.
    first_base: f32,
    /// Mean exploration bonus.
    bonus: Vec<f32>,
    shortlist: Vec<usize>,
}

/// Chars for `word` typed after `prev` and followed by a space.
fn in_context(prev: &str, word: &str) -> Vec<char> {
    format!("{prev} {word} ").chars().collect()
}

impl WordCosts {
    pub fn new(model: &Model) -> WordCosts {
        let per_word: Vec<(Vec<f32>, f32)> = words()
            .iter()
            .map(|w| {
                let text = in_context("the", w);
                let feats: Vec<Feat> = (4..text.len()).map(|i| Feat::at(&text, i)).collect();
                let bonus = feats.iter().map(|f| model.bonus(f)).sum::<f32>() / feats.len() as f32;
                (feats.iter().map(|f| model.cost(f)).collect(), bonus)
            })
            .collect();
        let mean = |xs: &mut dyn Iterator<Item = f32>| {
            let (sum, n) = xs.fold((0.0, 0), |(s, n), x| (s + x, n + 1));
            sum / n.max(1) as f32
        };
        let first_base = mean(&mut per_word.iter().map(|(c, _)| c[0]));
        let inner_base = mean(&mut per_word.iter().flat_map(|(c, _)| c[1..c.len() - 1].iter().copied()));
        let space_base = mean(&mut per_word.iter().map(|(c, _)| c[c.len() - 1]));

        let mut rest = Vec::with_capacity(per_word.len());
        let mut base = Vec::with_capacity(per_word.len());
        for (c, _) in &per_word {
            let last = c.len() - 1;
            let inner: f32 = c[1..last].iter().map(|x| x - inner_base).sum();
            let r = inner + c[last] - space_base;
            rest.push(r);
            base.push((c[0] - first_base + r) / c.len() as f32);
        }
        let mut order: Vec<usize> = (0..per_word.len()).collect();
        order.sort_by(|&a, &b| base[b].total_cmp(&base[a]));
        order.truncate(SHORTLIST);
        let bonus = per_word.into_iter().map(|(_, b)| b).collect();
        WordCosts { rest, first_base, bonus, shortlist: order }
    }

    /// The `n` words with the highest excess cost.
    pub fn hardest(&self, n: usize) -> impl Iterator<Item = &'static str> + '_ {
        self.shortlist.iter().take(n).map(|&i| words()[i])
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Options {
    pub punctuation: bool,
    pub numbers: bool,
}

pub struct Generator {
    opponent: Option<(Arc<Model>, Arc<WordCosts>)>,
    options: Options,
    /// Families from recent runs, rested for this one.
    resting: Family,
    /// Families already used in this run.
    used: Family,
    rng: StdRng,
    /// Recent vocabulary picks and the words actually emitted, for the no-repeat window.
    recent: VecDeque<usize>,
    emitted: VecDeque<String>,
    prev: String,
    capitalize: bool,
}

impl Generator {
    /// `opponent` is `None` during cold start or with the hill off: words are then drawn
    /// uniformly from the most common, like any typing test. `rest` is recent words to avoid.
    pub fn new(opponent: Option<(Arc<Model>, Arc<WordCosts>)>, options: Options, rest: &[String], seed: u64) -> Self {
        let mut resting = Family::default();
        rest.iter().for_each(|w| resting.insert(w));
        Generator {
            opponent,
            options,
            resting,
            used: Family::default(),
            rng: StdRng::seed_from_u64(seed),
            recent: VecDeque::new(),
            emitted: VecDeque::new(),
            prev: String::new(),
            capitalize: options.punctuation,
        }
    }

    pub fn take(&mut self, n: usize) -> Vec<String> {
        (0..n).map(|_| self.next_word()).collect()
    }

    fn next_word(&mut self) -> String {
        let pick = self.pick();
        self.used.insert(words()[pick]);
        self.recent.push_back(pick);
        if self.recent.len() > NO_REPEAT {
            self.recent.pop_front();
        }
        let word = self.decorate(words()[pick]);
        self.prev.clone_from(&word);
        self.emitted.push_back(word.clone());
        if self.emitted.len() > NO_REPEAT {
            self.emitted.pop_front();
        }
        word
    }

    fn pick(&mut self) -> usize {
        let Some((model, costs)) = self.opponent.clone() else {
            return loop {
                let i = self.rng.random_range(0..COLD_POOL);
                if !self.recent.contains(&i) {
                    break i;
                }
            };
        };
        if self.rng.random_bool(EXPLORE_SHARE) {
            // Explore by uncertainty alone, so known weaknesses can't crowd it out.
            let all: Vec<usize> = (0..words().len()).collect();
            let scores: Vec<f64> = costs.bonus.iter().map(|&b| b as f64).collect();
            return self.sample(&all, &scores);
        }
        // Targeted: rescore the shortlist with the real transition out of the previous word.
        let prev_tail: String = self.prev.chars().rev().take(2).collect::<Vec<_>>().into_iter().rev().collect();
        let scores: Vec<f64> = costs
            .shortlist
            .iter()
            .map(|&i| {
                let w = words()[i];
                let text = in_context(&prev_tail, w);
                let first = model.cost(&Feat::at(&text, prev_tail.len() + 1)) - costs.first_base;
                ((first + costs.rest[i]) / (w.len() + 1) as f32) as f64
            })
            .collect();
        self.sample(&costs.shortlist, &scores)
    }

    /// Samples ∝ exp(z / T) over z-scored `scores`, skipping families used in this run and,
    /// while any candidate is left, families resting from recent runs.
    fn sample(&mut self, candidates: &[usize], scores: &[f64]) -> usize {
        let n = scores.len() as f64;
        let mean = scores.iter().sum::<f64>() / n;
        let sd = (scores.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / n).sqrt().max(1e-9);
        let fresh = |rest: bool| -> Vec<bool> {
            let ok = |w: &str| !(self.used.related(w) || rest && self.resting.related(w));
            candidates.iter().map(|&i| ok(words()[i])).collect()
        };
        let mut allowed = fresh(true);
        if !allowed.contains(&true) {
            allowed = fresh(false);
        }
        if !allowed.contains(&true) {
            // Every candidate's family is used: any unused family from the whole list beats a repeat.
            let open: Vec<usize> = (0..words().len()).filter(|&i| !self.used.related(words()[i])).collect();
            if !open.is_empty() {
                return open[self.rng.random_range(0..open.len())];
            }
        }
        let weights: Vec<f64> = allowed
            .iter()
            .zip(scores)
            .map(|(&ok, s)| if ok { ((s - mean) / sd / TEMPERATURE).exp() } else { 0.0 })
            .collect();
        let mut r = self.rng.random::<f64>() * weights.iter().sum::<f64>();
        for (i, w) in candidates.iter().zip(&weights) {
            r -= w;
            if r <= 0.0 && *w > 0.0 {
                return *i;
            }
        }
        candidates[weights.iter().rposition(|w| *w > 0.0).unwrap_or(0)]
    }

    /// Applies numbers and punctuation. Draws a few random variants and keeps the one
    /// the opponent expects to be slowest, so punctuation is targeted too.
    fn decorate(&mut self, word: &str) -> String {
        let mut variants: Vec<(String, bool)> = (0..VARIANTS).map(|_| self.variant(word)).collect();
        // Numbers are drawn outside the vocabulary, so they need their own repeat check.
        variants.retain(|(v, _)| !self.emitted.contains(v));
        if variants.is_empty() {
            let mut plain = word.to_string();
            if self.options.punctuation && self.capitalize {
                plain[..1].make_ascii_uppercase();
            }
            variants.push((plain, false));
        }
        let best = match &self.opponent {
            Some((model, _)) => variants
                .into_iter()
                .max_by(|a, b| self.mean_cost(model, &a.0).total_cmp(&self.mean_cost(model, &b.0)))
                .unwrap(),
            None => variants.into_iter().next().unwrap(),
        };
        self.capitalize = best.1;
        best.0
    }

    fn mean_cost(&self, model: &Model, word: &str) -> f32 {
        let text: Vec<char> = format!("{} {word} ", self.prev).chars().collect();
        let start = text.len() - word.len() - 1;
        let costs: Vec<f32> = (start..text.len()).map(|i| model.cost(&Feat::at(&text, i))).collect();
        costs.iter().sum::<f32>() / costs.len() as f32
    }

    /// One random decoration of `word`, and whether the next word starts a sentence.
    fn variant(&mut self, word: &str) -> (String, bool) {
        let rng = &mut self.rng;
        if self.options.numbers && rng.random_bool(0.1) {
            let digits = rng.random_range(1..=4);
            let n = rng.random_range(10u32.pow(digits - 1)..10u32.pow(digits));
            return (n.to_string(), self.capitalize);
        }
        if !self.options.punctuation {
            return (word.to_string(), false);
        }
        let mut w = word.to_string();
        if self.capitalize {
            w[..1].make_ascii_uppercase();
        }
        let r: f64 = rng.random();
        let mut end_sentence = true;
        match r {
            r if r < 1.0 / 8.0 => w.push('.'),
            r if r < 1.0 / 8.0 + 1.0 / 40.0 => w.push('?'),
            r if r < 1.0 / 8.0 + 2.0 / 40.0 => w.push('!'),
            r => {
                end_sentence = false;
                let r = r - (1.0 / 8.0 + 2.0 / 40.0);
                match r {
                    r if r < 0.1 => w.push(','),
                    r if r < 0.125 => w.push(';'),
                    r if r < 0.15 => w.push(':'),
                    r if r < 0.175 => w = format!("\"{w}\""),
                    r if r < 0.2 => w = format!("({w})"),
                    r if r < 0.225 => w.push_str("'s"),
                    _ => {}
                }
            }
        }
        (w, end_sentence)
    }
}

/// Vocabulary words in the part of a run the user reached, decorations stripped.
pub fn typed_words(text: &[char], reached: usize) -> Vec<String> {
    let s: String = text[..reached.min(text.len())].iter().collect();
    s.split(' ')
        .map(|w| w.trim_matches(|c: char| !c.is_ascii_alphabetic()).trim_end_matches("'s").to_ascii_lowercase())
        .filter(|w| index_of(w).is_some())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trained() -> (Arc<Model>, Arc<WordCosts>) {
        let model = Arc::new(Model::new(4));
        let costs = Arc::new(WordCosts::new(&model));
        (model, costs)
    }

    fn none() -> Options {
        Options { punctuation: false, numbers: false }
    }

    #[test]
    fn word_list_is_clean() {
        assert_eq!(words().len(), 20000);
        assert!(words().iter().all(|w| !w.is_empty() && w.bytes().all(|b| b.is_ascii_lowercase())));
    }

    #[test]
    fn plain_words_without_toggles() {
        let mut g = Generator::new(Some(trained()), none(), &[], 1);
        let out = g.take(200);
        assert_eq!(out.len(), 200);
        assert!(out.iter().all(|w| w.bytes().all(|b| b.is_ascii_lowercase())));
    }

    #[test]
    fn no_repeats_within_window() {
        let numbers = Options { punctuation: false, numbers: true };
        for (opponent, options) in [(None, none()), (Some(trained()), none()), (Some(trained()), numbers)] {
            let mut g = Generator::new(opponent, options, &[], 3);
            let out = g.take(400);
            for (i, w) in out.iter().enumerate() {
                let window = &out[i.saturating_sub(NO_REPEAT)..i];
                assert!(!window.contains(w), "{w} repeated at {i}");
            }
        }
    }

    #[test]
    fn toggles_add_numbers_and_punctuation() {
        let mut g = Generator::new(Some(trained()), Options { punctuation: true, numbers: true }, &[], 3);
        let out = g.take(400);
        assert!(out.iter().any(|w| w.bytes().all(|b| b.is_ascii_digit())));
        assert!(out.iter().any(|w| w.ends_with('.')));
        assert!(out[0].starts_with(|c: char| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '"' || c == '('));
        let mut g = Generator::new(Some(trained()), Options { punctuation: false, numbers: true }, &[], 3);
        assert!(g.take(400).iter().all(|w| w.bytes().all(|b| b.is_ascii_alphanumeric())));
    }

    #[test]
    fn families_are_related() {
        let mut f = Family::default();
        for w in ["yeet", "hydro", "bye", "a"] {
            f.insert(w);
        }
        for w in ["yeeting", "hydra", "byes", "yee", "hyd"] {
            assert!(f.related(w), "{w}");
        }
        for w in ["yes", "hymn", "byte", "by", "an", "about", "hy"] {
            assert!(!f.related(w), "{w}");
        }
    }

    #[test]
    fn one_word_per_family_and_recent_words_rest() {
        let rest: Vec<String> = trained().1.hardest(40).map(String::from).collect();
        let mut g = Generator::new(Some(trained()), none(), &rest, 5);
        let out = g.take(60);
        let mut seen = Family::default();
        for w in &out {
            assert!(!seen.related(w), "{w} is related to an earlier word in {out:?}");
            assert!(w.len() < 3 || !rest.iter().any(|r| r == w), "{w} should be resting");
            seen.insert(w);
        }
    }

    #[test]
    fn exhausted_candidates_fall_back_to_an_unused_family() {
        let mut g = Generator::new(Some(trained()), none(), &[], 6);
        g.used.insert("go");
        let go = index_of("go").unwrap();
        let pick = words()[g.sample(&[go], &[1.0])];
        assert!(pick != "go" && !g.used.related(pick), "{pick}");
    }

    #[test]
    fn typed_words_strip_decorations() {
        let text: Vec<char> = "\"Hello,\" world's (zzqx) never".chars().collect();
        assert_eq!(typed_words(&text, 22), vec!["hello", "world"]);
    }
}
