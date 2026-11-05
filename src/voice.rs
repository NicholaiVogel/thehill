//! The hill's voice: short lines that tell you how you're doing, in the tone of an
//! opponent who would like you to do worse. Every line is built from real numbers.

use rand::{SeedableRng, rngs::StdRng, seq::IndexedRandom};
use std::collections::VecDeque;

/// Templates said recently are skipped, so the same line doesn't come round for a while.
const MEMORY: usize = 120;

/// Shows a trigram tag the way people read it: `_` (space) becomes nothing at the edges.
pub fn show(tag: &str) -> String {
    tag.replace('_', " ").trim().to_string()
}

fn key(c: char) -> String {
    if c == ' ' { "space".into() } else { format!("'{c}'") }
}

/// Fills `{name}` placeholders.
fn fill(template: &str, values: &[(&str, String)]) -> String {
    values.iter().fold(template.to_string(), |s, (k, v)| s.replace(&format!("{{{k}}}"), v))
}

/// Who won a run: `Greater` means you beat the prediction. Within 3% either way is a draw,
/// the same band the hill calls "right where i predicted", so a 0.1 wpm edge isn't a win.
pub fn outcome(wpm: f64, predicted: f64) -> std::cmp::Ordering {
    let delta = wpm / predicted - 1.0;
    if delta >= 0.03 {
        std::cmp::Ordering::Greater
    } else if delta > -0.03 {
        std::cmp::Ordering::Equal
    } else {
        std::cmp::Ordering::Less
    }
}

pub struct RunFacts {
    pub wpm: f64,
    pub raw: f64,
    /// 0 when the model hasn't made a prediction yet.
    pub predicted: f64,
    pub accuracy: f64,
    pub errors: usize,
    pub personal_best: bool,
    /// Runs in a row over (positive) or under (negative) the prediction, this one included.
    pub streak: i32,
    pub slowest: Option<(char, char, f64)>,
    /// The word with the most time per letter, and that time.
    pub worst_word: Option<(String, f64)>,
    /// The previous run's wpm.
    pub previous: Option<f64>,
    /// Runs today, this one included.
    pub runs_today: i64,
    /// Local hour, 0-23.
    pub hour: u32,
    /// Date of the checkpoint played against, if not the live model.
    pub checkpoint: Option<String>,
}

const COLD: &[&str] = &[
    "still watching. keep typing.",
    "taking notes. don't mind me.",
    "every key you press, i keep.",
    "go on. i'm learning your hands.",
    "i'm not judging yet. i'm measuring.",
    "interesting. very interesting.",
];

const BEAT: &[&str] = &[
    "{wpm}. i had you at {pred}. fine. i'll fix that.",
    "{d}% over my prediction. enjoy it. i'm already retraining.",
    "faster than i expected. that won't happen twice.",
    "{d}% over my number. i've written down exactly how.",
    "you beat {pred}. i'll remember that you can.",
    "{wpm}? that wasn't in the plan.",
    "{wpm}. noted, filed, to be used against you.",
    "you got lucky with the words. i'll be less generous.",
    "over by {d}%. somewhere, a boulder slips.",
    "that one's yours. the next one's mine.",
    "{wpm}. i underestimated you. i hate that.",
    "impressive. irritating, but impressive.",
    "fine. take it. there's more hill.",
    "{d}% faster than i thought. i'm updating my opinion of you. slightly.",
    "you did {wpm} on words i picked to stop you. rude.",
    "i said {pred}. you said {wpm}. we'll see who's right next time.",
];

const MATCH: &[&str] = &[
    "{wpm}. i said {pred}. i know you.",
    "right where i predicted. you're very readable.",
    "within {d}% of my guess. no surprises here.",
    "{wpm}. almost exactly what i expected. consistency is a weakness too.",
    "i called that to within a few words a minute.",
    "predictable. that's the nicest thing i can say.",
    "you did what i thought you'd do. again.",
    "{wpm} against my {pred}. we're very well acquainted.",
    "on the line. neither of us learned anything.",
    "textbook. my textbook.",
    "{wpm}. i could have typed that for you.",
    "barely off my prediction. i'd be bored if i could be.",
];

const LOSE: &[&str] = &[
    "{wpm}. i predicted {pred}. the boulder rolls back down.",
    "{d}% under my prediction. i chose those words carefully.",
    "that went about how i planned.",
    "{wpm}. i predicted {pred}. this is going well, for me.",
    "{d}% short. the words were the point.",
    "i expected {pred}. you found a way under it.",
    "downhill from here. for the boulder, i mean.",
    "{wpm}. i'd say better luck next time, but i pick the words.",
    "not quite. i'll keep the same words handy.",
    "the hill thanks you for your cooperation.",
    "{d}% under. i barely had to try.",
    "slower than my guess. it's almost like i studied you.",
    "and down it goes.",
    "{wpm}. you were aiming for {pred}, weren't you.",
    "i built that run out of things you hate. it worked.",
    "{d}% below. i'd apologise, but i wouldn't mean it.",
];

const BEST: &[&str] = &[
    "a new best. annoying. the hill gets steeper now.",
    "personal best. i'll have to start taking you seriously.",
    "new record. enjoy the view. it's temporary.",
    "{wpm}, your best yet. the hill just grew.",
    "best ever, in this mode. i've raised the stakes accordingly.",
];

const WIN_STREAK: &[&str] = &[
    "{n} runs in a row over my prediction. i'm taking this personally.",
    "{n} in a row. i'm rewriting myself as we speak.",
    "that's {n} straight. this is getting embarrassing for me.",
    "{n} wins running. i'm not worried. i'm a hill.",
];

const LOSE_STREAK: &[&str] = &[
    "{n} runs in a row under my prediction. we can keep going.",
    "{n} in a row for me. comfortable?",
    "{n} straight. the hill is very patient.",
    "that's {n} for me. i could do this all day. i literally can.",
];

const CLEAN: &[&str] = &[
    "no mistakes. so the words were too easy. noted.",
    "100% accuracy. i'll find something you can't spell.",
    "a clean run. suspicious.",
    "not one error. careful costs speed, you know.",
];

const SLOPPY: &[&str] = &[
    "{e} mistakes. i counted every one.",
    "{e} errors. i have them all written down.",
    "{acc}% accuracy. the keyboard isn't going anywhere.",
    "{e} mistakes. each one is a word i'll use again.",
];

const RAW_GAP: &[&str] = &[
    "raw {raw}, actual {wpm}. fast, in several directions.",
    "{gap} wpm lost to mistakes. that's where i live.",
    "you typed {raw} raw and kept {wpm}. backspace is not a strategy.",
];

const SLOW_PAIR: &[&str] = &[
    "{b} after {a} cost you {ms}ms. it's two keys.",
    "{a} into {b}: {ms}ms. i saw that.",
    "{a} to {b} took {ms}ms. i could see you thinking.",
    "{ms}ms on {a} into {b}. that one's going in the next run.",
];

const SLOW_WORD: &[&str] = &[
    "'{w}' took {ms}ms a letter. it'll be back.",
    "you slowed right down on '{w}'. i noticed.",
    "'{w}': {ms}ms a letter. it's coming back with friends.",
    "you typed '{w}' like it owed you money.",
    "'{w}' slowed you to a crawl. i've made a note.",
];

const UP_ON_LAST: &[&str] =
    &["{d} up on your last run. don't get used to it.", "{d} faster than last time. i've adjusted."];

const DOWN_ON_LAST: &[&str] =
    &["{d} down on your last run. tired? i'm not.", "{d} slower than last time. the hill notices these things."];

const MANY_TODAY: &[&str] = &[
    "run {n} today. i don't get tired.",
    "{n} runs today. i've enjoyed every one.",
    "that's {n} today. your fingers are filing a complaint.",
];

const LATE: &[&str] =
    &["it's {hh}. your fingers know it.", "typing at {hh}. bold.", "{hh}. the boulder doesn't sleep either."];

const BEAT_OLD: &[&str] = &[
    "you beat me from {date}. that version of me was weaker.",
    "past me from {date} couldn't hold you. present me will.",
];

const LOST_OLD: &[&str] = &["even me from {date} got you.", "i had your number back on {date}. i still do."];

const PLANTED: &[&str] = &[
    "{n} '{t}'s in this one. on purpose.",
    "this one's built around '{t}'.",
    "count the '{t}'s. there are {n}.",
    "'{t}' shows up {n} times. pure coincidence.",
    "i hid {n} '{t}'s in here. some of them aren't hidden.",
];

const HARDEST: &[&str] = &[
    "i put '{w}' in here. you know why.",
    "watch out for '{w}'. or don't.",
    "'{w}' is in this one. good luck.",
    "this one has '{w}'. i'm curious.",
    "save some energy for '{w}'.",
];

const READY: &[&str] =
    &["go on then.", "ready when you are. i'm always ready.", "the hill is waiting.", "let's see it."];

const COLD_PREVIEW: &[&str] = &[
    "first few runs are free. i'm just watching.",
    "{n} more and i start choosing the words.",
    "type normally. i'm only taking notes. for now.",
];

const WEAKNESS: &[&str] = &[
    "'{t}' again.",
    "called it.",
    "there it is.",
    "knew it.",
    "'{t}'. every time.",
    "that's the one.",
    "predictable.",
];
const BURST: &[&str] =
    &["steady.", "falling apart a little?", "breathe. or don't.", "messy.", "deep breath.", "oh no."];
const HESITATION: &[&str] = &["hesitating?", "thinking about it?", "go on.", "lost?", "take your time. it's timed."];
const LAST_SECONDS: &[&str] = &["three seconds.", "not enough time.", "tick.", "almost over.", "hurry."];

pub enum Heckle {
    Weakness(String),
    Burst,
    Hesitation,
    LastSeconds,
}

/// Picks lines and remembers which templates it used, so variety survives across runs
/// and (through `history`) across launches.
pub struct Voice {
    rng: StdRng,
    recent: VecDeque<String>,
}

impl Voice {
    pub fn new(seed: u64, history: &str) -> Voice {
        Voice { rng: StdRng::seed_from_u64(seed), recent: history.lines().map(String::from).collect() }
    }

    pub fn history(&self) -> String {
        self.recent.iter().map(String::as_str).collect::<Vec<_>>().join("\n")
    }

    fn pick(&mut self, lines: &[&'static str]) -> &'static str {
        let fresh: Vec<&'static str> = lines.iter().copied().filter(|l| !self.recent.iter().any(|r| r == l)).collect();
        // When every line was said recently, fall back to the one said longest ago.
        let line = match fresh.choose(&mut self.rng) {
            Some(l) => *l,
            None => lines
                .iter()
                .copied()
                .min_by_key(|l| self.recent.iter().position(|r| r == l).unwrap_or(0))
                .unwrap_or_default(),
        };
        self.recent.retain(|r| r != line);
        self.recent.push_back(line.to_string());
        if self.recent.len() > MEMORY {
            self.recent.pop_front();
        }
        line
    }

    /// A verdict on the run, then one observation.
    pub fn results(&mut self, f: &RunFacts) -> Vec<String> {
        let delta = if f.predicted > 0.0 { (f.wpm / f.predicted - 1.0) * 100.0 } else { 0.0 };
        let n = |v: f64| format!("{v:.0}");
        let vals = [
            ("wpm", n(f.wpm)),
            ("pred", n(f.predicted)),
            ("d", n(delta.abs())),
            ("raw", n(f.raw)),
            ("gap", n(f.raw - f.wpm)),
            ("e", f.errors.to_string()),
            ("acc", n(f.accuracy * 100.0)),
        ];
        let verdict = match outcome(f.wpm, f.predicted) {
            _ if f.predicted <= 0.0 => self.pick(COLD),
            std::cmp::Ordering::Greater => self.pick(BEAT),
            std::cmp::Ordering::Equal => self.pick(MATCH),
            std::cmp::Ordering::Less => self.pick(LOSE),
        };
        let verdict = fill(verdict, &vals);

        // Milestones first; otherwise one of the routine observations at random.
        let milestone = if f.personal_best {
            Some(fill(self.pick(BEST), &vals))
        } else if f.streak >= 3 {
            Some(fill(self.pick(WIN_STREAK), &[("n", f.streak.to_string())]))
        } else if f.streak <= -3 {
            Some(fill(self.pick(LOSE_STREAK), &[("n", (-f.streak).to_string())]))
        } else {
            None
        };
        let note = milestone.or_else(|| {
            let mut kinds: Vec<&'static [&'static str]> = Vec::new();
            if f.errors == 0 && f.accuracy > 0.0 {
                kinds.push(CLEAN);
            } else if f.accuracy < 0.9 {
                kinds.push(SLOPPY);
            }
            if f.raw - f.wpm >= 15.0 {
                kinds.push(RAW_GAP);
            }
            if f.slowest.is_some() {
                kinds.push(SLOW_PAIR);
            }
            if f.worst_word.is_some() {
                kinds.push(SLOW_WORD);
            }
            let change = f.previous.map(|p| f.wpm - p);
            if change.is_some_and(|c| c >= 10.0) {
                kinds.push(UP_ON_LAST);
            } else if change.is_some_and(|c| c <= -10.0) {
                kinds.push(DOWN_ON_LAST);
            }
            if f.runs_today >= 20 {
                kinds.push(MANY_TODAY);
            }
            if f.hour < 5 {
                kinds.push(LATE);
            }
            match outcome(f.wpm, f.predicted) {
                _ if f.checkpoint.is_none() || f.predicted <= 0.0 => {}
                std::cmp::Ordering::Greater => kinds.push(BEAT_OLD),
                std::cmp::Ordering::Less => kinds.push(LOST_OLD),
                std::cmp::Ordering::Equal => {}
            }
            let kind = *kinds.choose(&mut self.rng)?;
            let template = self.pick(kind);
            let (pair, word) = (f.slowest, f.worst_word.as_ref());
            let extra = [
                ("a", pair.map(|p| key(p.0)).unwrap_or_default()),
                ("b", pair.map(|p| key(p.1)).unwrap_or_default()),
                ("ms", n(pair.filter(|_| kind == SLOW_PAIR).map(|p| p.2).or(word.map(|w| w.1)).unwrap_or(0.0))),
                ("w", word.map(|w| w.0.clone()).unwrap_or_default()),
                ("d", n(change.unwrap_or(0.0).abs())),
                ("n", f.runs_today.to_string()),
                ("hh", format!("{}am", if f.hour == 0 { 12 } else { f.hour })),
                ("date", f.checkpoint.clone().unwrap_or_default()),
            ];
            Some(fill(&fill(template, &extra), &vals))
        });
        std::iter::once(verdict).chain(note).collect()
    }

    /// Said before a run starts. `planted` is the opponent's top weakness and how often it
    /// appears in this text; `hardest` is the costliest word in it.
    pub fn preview(&mut self, planted: Option<(&str, usize)>, hardest: Option<&str>) -> String {
        let mut kinds: Vec<&'static [&'static str]> = vec![READY];
        let planted = planted.filter(|(_, n)| *n >= 2);
        if planted.is_some() {
            kinds.extend([PLANTED, PLANTED]);
        }
        if hardest.is_some() {
            kinds.extend([HARDEST, HARDEST]);
        }
        let kind = *kinds.choose(&mut self.rng).unwrap_or(&READY);
        let (tag, count) = planted.unwrap_or_default();
        let vals = [("t", show(tag)), ("n", count.to_string()), ("w", hardest.unwrap_or_default().to_string())];
        fill(self.pick(kind), &vals)
    }

    /// `runs_left` must be positive.
    pub fn cold_preview(&mut self, runs_left: i64) -> String {
        fill(self.pick(COLD_PREVIEW), &[("n", runs_left.to_string())])
    }

    pub fn heckle(&mut self, h: &Heckle) -> String {
        match h {
            Heckle::Weakness(tag) => fill(self.pick(WEAKNESS), &[("t", show(tag))]),
            Heckle::Burst => self.pick(BURST).into(),
            Heckle::Hesitation => self.pick(HESITATION).into(),
            Heckle::LastSeconds => self.pick(LAST_SECONDS).into(),
        }
    }
}

pub fn opponent_preview(label_date: &str, tags: &[String]) -> String {
    let tags: Vec<String> = tags.iter().take(2).map(|t| format!("'{}'", show(t))).collect();
    format!("this is me from {label_date}. back then you couldn't type {}.", tags.join(" or "))
}

/// Announced when training freezes a new checkpoint. Varies by checkpoint number.
/// `got_past` must only hold weaknesses that measurably improved.
pub fn checkpoint(number: usize, tags: &[String], got_past: &[String]) -> String {
    let quote = |ts: &[String]| ts.iter().map(|t| format!("'{}'", show(t))).collect::<Vec<_>>();
    let (now, gone) = (quote(&tags[..2.min(tags.len())]), quote(&got_past[..2.min(got_past.len())]));
    let vals = [("n", number.to_string()), ("now", now.join(" and ")), ("gone", gone.join(" and "))];
    let templates: &[&str] = if gone.is_empty() {
        &[
            "checkpoint {n} saved. still {now}. we're not done with those.",
            "checkpoint {n}. it remembers you struggling with {now}.",
            "checkpoint {n} saved. {now}, as ever.",
        ]
    } else {
        &[
            "checkpoint {n} saved. you got past {gone}. now it's {now}.",
            "checkpoint {n}. {gone} got faster. fine. {now} it is.",
            "checkpoint {n}. you fixed {gone}. i found {now}.",
        ]
    };
    fill(templates[number % templates.len()], &vals)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn facts() -> RunFacts {
        RunFacts {
            wpm: 110.0,
            raw: 130.0,
            predicted: 100.0,
            accuracy: 0.87,
            errors: 3,
            personal_best: false,
            streak: 1,
            slowest: Some(('e', 'x', 340.0)),
            worst_word: Some(("convex".into(), 190.0)),
            previous: Some(95.0),
            runs_today: 25,
            hour: 2,
            checkpoint: Some("Nov 5".into()),
        }
    }

    #[test]
    fn every_line_is_filled_in() {
        let mut voice = Voice::new(0, "");
        for _ in 0..500 {
            let lines = voice.results(&facts());
            assert_eq!(lines.len(), 2);
            assert!(lines.iter().all(|l| !l.contains('{') && !l.contains("''")), "{lines:?}");
            let preview = voice.preview(Some(("ind", 3)), Some("jinx"));
            assert!(!preview.contains('{'), "{preview}");
        }
        let mut f = facts();
        f.personal_best = true;
        assert!(voice.results(&f)[1].contains("best"));
    }

    #[test]
    fn lines_do_not_repeat_until_the_pool_runs_out() {
        let mut voice = Voice::new(1, "");
        let said: Vec<&str> = (0..BEAT.len()).map(|_| voice.pick(BEAT)).collect();
        let mut unique = said.clone();
        unique.sort();
        unique.dedup();
        assert_eq!(unique.len(), BEAT.len());
        // History survives a restart.
        let mut restored = Voice::new(2, &voice.history());
        assert_eq!(restored.pick(BEAT), said[0], "the line said longest ago comes back first");
    }

    #[test]
    fn checkpoint_names_what_you_got_past() {
        let now = vec!["ind".to_string(), "_ac".to_string()];
        for n in 0..3 {
            let line = checkpoint(n, &now, &["_ex".to_string()]);
            assert!(line.contains("'ex'") && line.contains("'ind'"), "{line}");
        }
    }
}
